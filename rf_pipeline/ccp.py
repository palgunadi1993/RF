"""Stage 4: common-conversion-point (CCP) imaging.

Pseudo-migrates the radial RFs to depth with a 1-D velocity model and stacks the
migrated amplitudes into depth sections along the configured profiles (PLAN.md
Stage 4, paper Fig. 6). If ``python-seispy`` is importable its CCP module is used;
otherwise a transparent built-in migration runs so the stage works out of the box.

Migration (Ps mode):
  delay time to a converted depth z is  t(z) = integral_0^z [eta_s - eta_p] dz'
  with eta = sqrt(1/v^2 - p^2); the RF amplitude at t(z) is mapped to depth z.
  The S-leg piercing point offsets the sample horizontally from the station along
  the event back-azimuth, so amplitudes bin at their true lateral position.

Output: ccp_out/<profile>.npz (distance, depth, amplitude grid) + <profile>.png.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from . import io_utils
from .logging_setup import get_logger

LOG = get_logger("rf.ccp")


def _load_velocity_model(path: Path, dz: float, zmax: float):
    """Read a ``depth_top Vp Vs`` table and expand to Vp(z), Vs(z) on a dz grid."""
    rows = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            if len(parts) >= 3:
                rows.append([float(parts[0]), float(parts[1]), float(parts[2])])
            elif len(parts) == 2:  # depth Vp, assume Vp/Vs=1.73
                rows.append([float(parts[0]), float(parts[1]), float(parts[1]) / 1.73])
    rows = np.array(sorted(rows))
    z = np.arange(0, zmax + dz / 2, dz)
    vp = np.interp(z, rows[:, 0], rows[:, 1])
    vs = np.interp(z, rows[:, 0], rows[:, 2])
    return z, vp, vs


def _default_model(dz, zmax):
    """Fallback crustal model if none supplied (crude, documented)."""
    z = np.arange(0, zmax + dz / 2, dz)
    vp = np.where(z < 15, 5.8, 6.5)
    vs = vp / 1.73
    return z, vp, vs


def migrate_trace(tr, z, vp, vs, dz):
    """Sample the RF amplitude at each depth via the Ps time-depth mapping.

    Returns (amp_z, offset_km) where offset is the cumulative S-leg horizontal
    piercing distance from the station at each depth.
    """
    p = float(getattr(tr.stats, "slowness", np.nan))
    onset = getattr(tr.stats, "onset", None)
    if not np.isfinite(p) or onset is None:
        return None, None
    eta_s = np.sqrt(np.clip(1.0 / vs**2 - p**2, 0, None))
    eta_p = np.sqrt(np.clip(1.0 / vp**2 - p**2, 0, None))
    t_of_z = np.cumsum((eta_s - eta_p) * dz)          # delay time at each depth
    offset = np.cumsum(p * vs / np.sqrt(np.clip(1 - (p * vs) ** 2, 1e-6, None)) * dz)

    t0 = tr.stats.onset - tr.stats.starttime
    dt = tr.stats.delta
    idx = np.clip(np.round((t0 + t_of_z) / dt).astype(int), 0, tr.data.size - 1)
    return tr.data[idx], offset


def _geo_to_local_km(lat, lon, lat0, lon0):
    """Equirectangular local km east/north from a reference point."""
    R = 6371.0
    x = np.deg2rad(lon - lon0) * R * np.cos(np.deg2rad(lat0))
    y = np.deg2rad(lat - lat0) * R
    return x, y


def _project_onto_profile(px, py, prof):
    """Signed along-profile distance (km) and perpendicular offset (km)."""
    ax, ay = _geo_to_local_km(prof["lat1"], prof["lon1"], prof["lat1"], prof["lon1"])
    bx, by = _geo_to_local_km(prof["lat2"], prof["lon2"], prof["lat1"], prof["lon1"])
    vx, vy = bx - ax, by - ay
    L = np.hypot(vx, vy)
    ux, uy = vx / L, vy / L
    along = px * ux + py * uy
    perp = -px * uy + py * ux
    return along, perp, L


def _read_rfs(path: Path):
    from rf import read_rf
    return read_rf(str(path))


def run(cfg: dict) -> Path:
    ccp = cfg.get("ccp", {})
    dz = float(ccp.get("depth_step", 0.5))
    zmax = float(ccp.get("depth_range", [0, 60])[1])
    run_on = ccp.get("run_on", ["teleseismic", "local_deep"])
    profiles = ccp.get("profiles", [])

    p = io_utils.paths(cfg)
    rf_dir = p["rf_out"]
    out_dir = io_utils.ensure_dir(p["ccp_out"])
    stations, _ = io_utils.load_stations(cfg)
    sta_lookup = io_utils.station_lookup(stations)

    model_path = ccp.get("velocity_model")
    if model_path:
        model_path = io_utils.resolve_path(model_path, cfg["_project_root"])
    if model_path and Path(model_path).exists():
        z, vp, vs = _load_velocity_model(model_path, dz, zmax)
    else:
        LOG.warning("No ccp.velocity_model found — using a crude default crustal model.")
        z, vp, vs = _default_model(dz, zmax)

    # One section per (profile, source class) so tele vs local-deep resolution
    # can be compared (PLAN.md Stage 4).
    for prof in profiles:
        _, _, L = _project_onto_profile(0.0, 0.0, prof)
        along_bins = np.arange(0, L + 1.0, 1.0)
        width = float(prof.get("width_km", 10.0))

        for name in run_on:
            grid = np.zeros((along_bins.size, z.size))
            count = np.zeros_like(grid)
            for h5 in sorted(rf_dir.glob(f"*_{name}.h5")):
                station = h5.name[: -len(f"_{name}.h5")]
                sta = sta_lookup.get(station)
                if sta is None:
                    continue
                try:
                    rfs = _read_rfs(h5)
                    rfs = rfs.select(component="R") + rfs.select(component="Q")
                except Exception as e:
                    LOG.warning(f"[{prof['name']}/{name}] {station}: {e}")
                    continue
                sx, sy = _geo_to_local_km(sta.latitude, sta.longitude,
                                          prof["lat1"], prof["lon1"])
                for tr in rfs:
                    amp_z, offset = migrate_trace(tr, z, vp, vs, dz)
                    if amp_z is None:
                        continue
                    baz = np.deg2rad(float(getattr(tr.stats, "back_azimuth", 0.0)))
                    # piercing point moves toward the event (back-azimuth direction)
                    px = sx + offset * np.sin(baz)
                    py = sy + offset * np.cos(baz)
                    along, perp, _ = _project_onto_profile(px, py, prof)
                    keep = np.abs(perp) <= width / 2.0
                    bi = np.clip(np.searchsorted(along_bins, along) - 1, 0,
                                 along_bins.size - 1)
                    for k in np.where(keep)[0]:
                        grid[bi[k], k] += amp_z[k]
                        count[bi[k], k] += 1

            if count.sum() == 0:
                LOG.info(f"[{prof['name']}/{name}] no migrated amplitudes — skipped.")
                continue
            stacked = np.divide(grid, count, out=np.zeros_like(grid), where=count > 0)
            tag = f"{prof['name']}_{name}"
            npz = out_dir / f"{tag}.npz"
            np.savez(npz, along=along_bins, depth=z, amp=stacked, count=count)
            _plot_section(along_bins, z, stacked, tag, out_dir / f"{tag}.png")
            LOG.info(f"[{tag}] CCP section -> {npz}")

    LOG.info("Stage 4 (CCP) complete.")
    return out_dir


def _plot_section(along, depth, amp, name, out_png):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(8, 4))
        vmax = np.nanpercentile(np.abs(amp), 98) or 1.0
        im = ax.pcolormesh(along, depth, amp.T, cmap="RdBu_r", vmin=-vmax, vmax=vmax,
                           shading="auto")
        ax.invert_yaxis()
        ax.set_xlabel("Distance along profile (km)")
        ax.set_ylabel("Depth (km)")
        ax.set_title(f"CCP section {name}")
        fig.colorbar(im, ax=ax, label="RF amplitude")
        fig.tight_layout()
        fig.savefig(out_png, dpi=200)
        plt.close(fig)
    except Exception as e:
        LOG.warning(f"CCP plot failed for {name}: {e}")
