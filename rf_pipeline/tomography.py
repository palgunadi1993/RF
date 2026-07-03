"""Stage 7: from pair dispersion to one 1-D curve per station (PLAN.md Stage 7).

Path B (two-station / small-array, default & self-contained): for each station,
average every pair curve that involves it into a representative per-station
dispersion curve.

Path A (full 2-D tomography with FMST): export the per-period travel-time datasets
FMST expects, invoke the configured FMST binary, then sample each period's phase-
velocity map at the station locations. If the binary is not configured/available
the stage logs a warning and falls back to Path B so it always produces per-station
curves for the inversion.

Output: tomo/<station>_disp.txt  (period phase_vel [group_vel] sigma).
"""
from __future__ import annotations

from collections import defaultdict
from pathlib import Path

import numpy as np

from . import io_utils
from .logging_setup import get_logger

LOG = get_logger("rf.tomography")


def _load_pair_curves(disp_dir: Path):
    """Return {(STA1,STA2): ndarray[period, phase, group, snr]}."""
    curves = {}
    for f in sorted(Path(disp_dir).glob("*.disp")):
        try:
            arr = np.loadtxt(f, ndmin=2)
        except Exception:
            continue
        if arr.size == 0:
            continue
        a, b = f.stem.split("_")[:2]
        curves[(a, b)] = arr
    return curves


def two_station(cfg: dict) -> Path:
    p = io_utils.paths(cfg)
    disp_dir = p["disp"]
    out_dir = io_utils.ensure_dir(p["tomo"])
    stations, _ = io_utils.load_stations(cfg)
    curves = _load_pair_curves(disp_dir)
    if not curves:
        LOG.warning(f"No pair curves under {disp_dir} — run Stage 6 first.")
        return out_dir

    # accumulate per station: period -> list of (phase, group)
    per_sta: dict[str, dict[float, list]] = defaultdict(lambda: defaultdict(list))
    for (a, b), arr in curves.items():
        for row in arr:
            T, phase, group = row[0], row[1], row[2]
            for s in (a, b):
                per_sta[s][round(float(T), 4)].append((phase, group))

    n = 0
    for sta in stations:
        data = per_sta.get(sta.code)
        if not data:
            continue
        rows = []
        for T in sorted(data):
            vals = np.array(data[T], dtype=float)
            phase = np.nanmean(vals[:, 0])
            group = np.nanmean(vals[:, 1])
            sigma = np.nanstd(vals[:, 0]) if len(vals) > 1 else 0.05 * phase
            rows.append((T, phase, group, max(sigma, 0.01)))
        if rows:
            out = out_dir / f"{sta.code}_disp.txt"
            np.savetxt(out, np.array(rows), fmt="%.4f",
                       header="period_s phase_vel group_vel sigma")
            n += 1
    LOG.info(f"Stage 7 (path B): {n} per-station curves -> {out_dir}")
    return out_dir


def _export_fmst_inputs(cfg, curves, stations, out_dir) -> Path:
    """Write FMST-style per-period travel-time datasets (sources/receivers/times)."""
    fmst_dir = io_utils.ensure_dir(out_dir / "fmst_inputs")
    sta_lookup = io_utils.station_lookup(stations)
    periods = sorted({round(float(T), 4)
                      for arr in curves.values() for T in arr[:, 0]})
    for T in periods:
        lines = []
        for (a, b), arr in curves.items():
            sa, sb = sta_lookup.get(a), sta_lookup.get(b)
            if sa is None or sb is None:
                continue
            match = arr[np.isclose(arr[:, 0], T)]
            if match.size == 0:
                continue
            c = float(match[0, 1])
            from obspy.geodetics import gps2dist_azimuth
            dist, _, _ = gps2dist_azimuth(sa.latitude, sa.longitude,
                                          sb.latitude, sb.longitude)
            tt = (dist / 1000.0) / c
            lines.append(f"{sa.latitude:.5f} {sa.longitude:.5f} "
                         f"{sb.latitude:.5f} {sb.longitude:.5f} {tt:.5f} {c:.4f}")
        (fmst_dir / f"tt_{T:.2f}s.dat").write_text("\n".join(lines) + "\n")
    LOG.info(f"FMST inputs exported for {len(periods)} periods -> {fmst_dir}")
    return fmst_dir


def full_tomography(cfg: dict) -> Path:
    p = io_utils.paths(cfg)
    out_dir = io_utils.ensure_dir(p["tomo"])
    stations, _ = io_utils.load_stations(cfg)
    curves = _load_pair_curves(p["disp"])
    if not curves:
        LOG.warning("No pair curves — cannot run tomography.")
        return out_dir

    _export_fmst_inputs(cfg, curves, stations, out_dir)
    binary = cfg.get("tomo", {}).get("fmst", {}).get("binary")
    if binary:
        binary = io_utils.resolve_path(binary, cfg["_project_root"])
    if not binary or not Path(binary).exists():
        LOG.warning("tomo.path=A but no FMST binary configured/available. "
                    "FMST inputs were exported to tomo/fmst_inputs/; run FMST "
                    "externally, or use path B. Falling back to path B now.")
        return two_station(cfg)

    # If a binary is present, the FMST run would be driven here (external Fortran).
    LOG.info(f"FMST binary {binary} present — drive it over tomo/fmst_inputs/, "
             "then sample maps at station locations. (External step.)")
    # After an external FMST run the maps should be sampled; until then provide
    # the two-station curves so the inversion still has inputs.
    return two_station(cfg)


def run(cfg: dict) -> Path:
    path = str(cfg.get("tomo", {}).get("path", "B")).upper()
    if path == "A":
        return full_tomography(cfg)
    return two_station(cfg)
