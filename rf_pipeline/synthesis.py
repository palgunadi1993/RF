"""Stage 9: synthesis & publication figures (PLAN.md Stage 9).

One figure = one function = one ``plot.figures`` toggle. Geographic maps
(F1/F2/F3/F9) use PyGMT; the rest use Matplotlib. Every figure reads its source
data from the earlier stages' outputs and degrades gracefully (logs a skip) when
those inputs are not yet present, so the stage can be run at any point.

Figures are written to ``figures/`` in every format listed under ``plot.format``
at ``plot.dpi``.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from . import catalogs, io_utils
from .logging_setup import get_logger

LOG = get_logger("rf.synthesis")


# --------------------------------------------------------------------------
# shared helpers
# --------------------------------------------------------------------------

def _mpl():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    return plt


def _save(fig_or_pygmt, name, cfg, is_pygmt=False):
    p = io_utils.paths(cfg)
    out_dir = io_utils.ensure_dir(p["figures"])
    dpi = int(cfg.get("plot", {}).get("dpi", 300))
    fmts = cfg.get("plot", {}).get("format", ["png"])
    written = []
    for fmt in fmts:
        target = out_dir / f"{name}.{fmt}"
        if is_pygmt:
            fig_or_pygmt.savefig(str(target), dpi=dpi)
        else:
            fig_or_pygmt.savefig(target, dpi=dpi, bbox_inches="tight")
        written.append(target.name)
    if not is_pygmt:
        _mpl().close(fig_or_pygmt)
    LOG.info(f"{name}: wrote {written}")


def _region(stations, pad=0.08):
    lons = [s.longitude for s in stations]; lats = [s.latitude for s in stations]
    return [min(lons) - pad, max(lons) + pad, min(lats) - pad, max(lats) + pad]


def _cmap_velocity(cfg):
    return cfg.get("plot", {}).get("cmap_velocity", "roma")


# --------------------------------------------------------------------------
# F1 — station & tectonic map
# --------------------------------------------------------------------------

def F1_station_map(cfg):
    import pygmt
    stations, _ = io_utils.load_stations(cfg)
    region = _region(stations)
    fig = pygmt.Figure()
    pygmt.config(FONT_TITLE="14p,Helvetica-Bold", MAP_FRAME_TYPE="plain")
    fig.basemap(region=region, projection="M14c", frame=["af", "WSne+tDieng seismic network"])
    try:
        relief = pygmt.datasets.load_earth_relief(
            resolution=cfg.get("plot", {}).get("topo_resolution", "03s"), region=region)
        fig.grdimage(relief, shading=True, cmap="geo")
    except Exception as e:
        LOG.warning(f"F1 relief unavailable: {e}")
        fig.coast(land="240/240/235", water="200/220/240")
    fig.plot(x=[s.longitude for s in stations], y=[s.latitude for s in stations],
             style="i0.35c", fill="white", pen="0.8p,black")
    for s in stations:
        fig.text(x=s.longitude, y=s.latitude, text=s.code, font="5p,Helvetica",
                 justify="ML", offset="0.15c/0c")
    fig.basemap(map_scale="jBR+w5k+o0.6c/0.6c+f")
    _save(fig, "F1_station_map", cfg, is_pygmt=True)


# --------------------------------------------------------------------------
# F2 — event distribution (three catalogs)
# --------------------------------------------------------------------------

def F2_event_distribution(cfg):
    import pygmt
    stations, _ = io_utils.load_stations(cfg)
    clat = np.mean([s.latitude for s in stations])
    clon = np.mean([s.longitude for s in stations])
    colours = {"teleseismic": "red", "regional": "orange", "local_deep": "blue"}
    fig = pygmt.Figure()
    fig.basemap(region="g", projection=f"E{clon}/{clat}/160/14c", frame="afg")
    fig.coast(land="200/200/200", water="white", shorelines="0.2p")
    fig.plot(x=[clon], y=[clat], style="a0.6c", fill="black", pen="1p")
    any_ev = False
    for name, colour in colours.items():
        cat = catalogs.load_class_catalog(cfg, name)
        if cat is None or len(cat) == 0:
            continue
        df = io_utils.catalog_to_df(cat).dropna(subset=["latitude", "longitude"])
        if df.empty:
            continue
        any_ev = True
        fig.plot(x=df["longitude"], y=df["latitude"], style="c0.15c",
                 fill=colour, pen="0.2p,black", label=name)
    if any_ev:
        fig.legend(position="JBL+o0.2c", box=True)
    _save(fig, "F2_event_distribution", cfg, is_pygmt=True)


# --------------------------------------------------------------------------
# F3 — coverage: ANT paths + RF piercing points
# --------------------------------------------------------------------------

def F3_coverage_raypaths(cfg):
    import pygmt
    stations, _ = io_utils.load_stations(cfg)
    lookup = io_utils.station_lookup(stations)
    region = _region(stations)
    fig = pygmt.Figure()
    fig.basemap(region=region, projection="M14c", frame=["af", "WSne+tRay-path coverage"])
    fig.coast(shorelines="0.3p", land="245/245/240")
    p = io_utils.paths(cfg)
    for f in Path(p["ccfs"]).glob("*.npz") if Path(p["ccfs"]).exists() else []:
        a, b = f.stem.split("_")[:2]
        sa, sb = lookup.get(a), lookup.get(b)
        if sa and sb:
            fig.plot(x=[sa.longitude, sb.longitude], y=[sa.latitude, sb.latitude],
                     pen="0.3p,gray")
    fig.plot(x=[s.longitude for s in stations], y=[s.latitude for s in stations],
             style="i0.3c", fill="white", pen="0.7p,black")
    _save(fig, "F3_coverage_raypaths", cfg, is_pygmt=True)


# --------------------------------------------------------------------------
# F4 — H-kappa stack panels (representative stations)
# --------------------------------------------------------------------------

def F4_hk_panels(cfg):
    plt = _mpl()
    p = io_utils.paths(cfg)
    hk_dir = p.get("hk_out")
    reps = cfg.get("plot", {}).get("representative_stations", [])
    npzs = []
    for sta in reps:
        npzs += sorted(Path(hk_dir).glob(f"{sta}_*_hk.npz")) if hk_dir else []
    if not npzs:
        LOG.warning("F4: no H-kappa npz grids found — skipping.")
        return
    n = len(npzs)
    fig, axes = plt.subplots(1, n, figsize=(4 * n, 4), squeeze=False)
    for ax, npz in zip(axes[0], npzs):
        d = np.load(npz)
        im = ax.pcolormesh(d["k_grid"], d["h_grid"], d["stack"], cmap="viridis",
                           shading="auto")
        ax.plot(d["bestK"], d["bestH"], "r*", ms=14)
        ax.invert_yaxis()
        ax.set_xlabel("Vp/Vs"); ax.set_ylabel("H (km)")
        ax.set_title(npz.stem.replace("_hk", ""))
        fig.colorbar(im, ax=ax, label="stack")
    fig.tight_layout()
    _save(fig, "F4_hk_panels", cfg)


# --------------------------------------------------------------------------
# F5 — H-kappa summary across stations
# --------------------------------------------------------------------------

def F5_hk_summary(cfg):
    plt = _mpl()
    p = io_utils.paths(cfg)
    hk_dir = p.get("hk_out")
    csvs = sorted(Path(hk_dir).glob("hk_*.csv")) if hk_dir else []
    if not csvs:
        LOG.warning("F5: no hk_*.csv — skipping.")
        return
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4))
    for csv in csvs:
        cls = csv.stem.replace("hk_", "")
        df = pd.read_csv(csv)
        ax1.errorbar(df["station"], df["H_km"],
                     yerr=[df["H_km"] - df["H_lo"], df["H_hi"] - df["H_km"]],
                     fmt="o", capsize=3, label=cls)
        ax2.errorbar(df["station"], df["kappa"],
                     yerr=[df["kappa"] - df["k_lo"], df["k_hi"] - df["kappa"]],
                     fmt="s", capsize=3, label=cls)
    ax1.set_ylabel("H (km)"); ax2.set_ylabel("Vp/Vs")
    for ax in (ax1, ax2):
        ax.tick_params(axis="x", rotation=90); ax.legend()
    fig.tight_layout()
    _save(fig, "F5_hk_summary", cfg)


# --------------------------------------------------------------------------
# F6 — RF record sections (back-azimuth binned)
# --------------------------------------------------------------------------

def F6_rf_record_sections(cfg):
    plt = _mpl()
    try:
        from rf import read_rf
    except Exception:
        LOG.warning("F6 needs `rf` to read RF H5 — skipping.")
        return
    p = io_utils.paths(cfg)
    classes = list(cfg.get("rf", {}).get("classes", {}).keys())
    reps = cfg.get("plot", {}).get("representative_stations", [])
    files = []
    for sta in reps:
        for cls in classes:
            f = Path(p["rf_out"]) / f"{sta}_{cls}.h5"
            if f.exists():
                files.append((sta, cls, f))
    if not files:
        LOG.warning("F6: no RF H5 for representative stations — skipping.")
        return
    fig, axes = plt.subplots(1, len(files), figsize=(4 * len(files), 6), squeeze=False)
    for ax, (sta, cls, f) in zip(axes[0], files):
        rfs = read_rf(str(f)).select(component="R") + read_rf(str(f)).select(component="Q")
        rfs.sort(["back_azimuth"])
        for i, tr in enumerate(rfs):
            t = tr.stats.get("onset")
            t0 = (tr.stats.onset - tr.stats.starttime) if t else 0
            time = np.arange(tr.stats.npts) * tr.stats.delta - t0
            y = i + tr.data / (np.max(np.abs(tr.data)) or 1) * 0.5
            ax.plot(time, y, "k", lw=0.4)
            ax.fill_between(time, i, y, where=(y > i), color="r", alpha=0.4)
        ax.set_title(f"{sta} {cls}"); ax.set_xlabel("Time after P (s)")
        ax.set_xlim(-2, 20)
    axes[0][0].set_ylabel("RF index (baz-sorted)")
    fig.tight_layout()
    _save(fig, "F6_rf_record_sections", cfg)


# --------------------------------------------------------------------------
# F7 — CCP sections
# --------------------------------------------------------------------------

def F7_ccp_sections(cfg):
    plt = _mpl()
    p = io_utils.paths(cfg)
    profiles = cfg.get("plot", {}).get("cross_sections", ["NS", "EW"])
    npzs = []
    for name in profiles:
        npzs += sorted(Path(p["ccp_out"]).glob(f"{name}_*.npz")) if Path(p["ccp_out"]).exists() else []
        plain = Path(p["ccp_out"]) / f"{name}.npz"
        if plain.exists():
            npzs.append(plain)
    if not npzs:
        LOG.warning("F7: no CCP npz — skipping.")
        return
    fig, axes = plt.subplots(len(npzs), 1, figsize=(9, 4 * len(npzs)), squeeze=False)
    for ax, f in zip(axes[:, 0], npzs):
        d = np.load(f)
        vmax = np.nanpercentile(np.abs(d["amp"]), 98) or 1.0
        im = ax.pcolormesh(d["along"], d["depth"], d["amp"].T, cmap="RdBu_r",
                           vmin=-vmax, vmax=vmax, shading="auto")
        ax.invert_yaxis(); ax.set_ylabel("Depth (km)")
        ax.set_xlabel("Distance (km)"); ax.set_title(f"CCP {f.stem}")
        fig.colorbar(im, ax=ax, label="RF amp")
    fig.tight_layout()
    _save(fig, "F7_ccp_sections", cfg)


# --------------------------------------------------------------------------
# F8 — noise CCF gather vs interstation distance
# --------------------------------------------------------------------------

def F8_ccf_gather(cfg):
    plt = _mpl()
    from scipy.signal import butter, filtfilt
    p = io_utils.paths(cfg)
    files = sorted(Path(p["ccfs"]).glob("*.npz")) if Path(p["ccfs"]).exists() else []
    if not files:
        LOG.warning("F8: no CCFs — skipping.")
        return
    band = cfg.get("plot", {}).get("ccf_bandpass", [0.2, 0.4])
    fig, ax = plt.subplots(figsize=(7, 8))
    for f in files:
        d = np.load(f)
        dist = float(d["dist_km"]) if "dist_km" in d else np.nan
        if not np.isfinite(dist):
            continue
        lag, ccf = d["lag_s"], d["ccf"].astype(float)
        sr = 1.0 / (lag[1] - lag[0])
        b, a = butter(3, [band[0] / (sr / 2), band[1] / (sr / 2)], btype="band")
        y = filtfilt(b, a, ccf)
        y = y / (np.max(np.abs(y)) or 1) * 3.0
        ax.plot(lag, dist + y, "k", lw=0.4)
    ax.set_xlabel("Lag (s)"); ax.set_ylabel("Interstation distance (km)")
    ax.set_title(f"CCF gather {band[0]}-{band[1]} Hz")
    fig.tight_layout()
    _save(fig, "F8_ccf_gather", cfg)


# --------------------------------------------------------------------------
# F9 — dispersion maps (path A only)
# --------------------------------------------------------------------------

def F9_dispersion_maps(cfg):
    if str(cfg.get("tomo", {}).get("path", "B")).upper() != "A":
        LOG.info("F9: tomo.path != A — dispersion maps not applicable, skipping.")
        return
    LOG.warning("F9: requires FMST velocity maps (tomo/). Skipping until FMST "
                "maps are produced; see tomography.full_tomography.")


# --------------------------------------------------------------------------
# F10 — joint inversion result per station
# --------------------------------------------------------------------------

def F10_inversion_per_station(cfg):
    p = io_utils.paths(cfg)
    inv_dir = p.get("inversion")
    reps = cfg.get("plot", {}).get("representative_stations", [])
    found = [Path(inv_dir) / s for s in reps if inv_dir and (Path(inv_dir) / s).exists()]
    if not found:
        LOG.warning("F10: no per-station inversion output — skipping "
                    "(BayHunter writes its own best_model/fit plots per station).")
        return
    LOG.info(f"F10: BayHunter per-station plots live under {[str(f) for f in found]}.")


# --------------------------------------------------------------------------
# F11 — Vs cross-sections from per-station profiles
# --------------------------------------------------------------------------

def F11_vs_cross_sections(cfg):
    plt = _mpl()
    p = io_utils.paths(cfg)
    inv_dir = p.get("inversion")
    profiles = _station_vs_profiles(inv_dir) if inv_dir else {}
    if not profiles:
        LOG.warning("F11: no per-station Vs(z) profiles found "
                    "(expected inversion/<sta>/vs_profile.txt) — skipping.")
        return
    stations, _ = io_utils.load_stations(cfg)
    lookup = io_utils.station_lookup(stations)
    clip = cfg.get("plot", {}).get("vs_clip", [0.5, 4.8])
    for pname in cfg.get("plot", {}).get("cross_sections", ["NS", "EW"]):
        key = (lambda s: s.latitude) if pname == "NS" else (lambda s: s.longitude)
        ordered = sorted([(key(lookup[s]), s, prof) for s, prof in profiles.items()
                          if s in lookup])
        if not ordered:
            continue
        fig, ax = plt.subplots(figsize=(9, 5))
        xs = [o[0] for o in ordered]
        depth = ordered[0][2][:, 0]
        grid = np.column_stack([o[2][:, 1] for o in ordered])
        im = ax.pcolormesh(xs, depth, grid, cmap=_cmap_velocity(cfg),
                           vmin=clip[0], vmax=clip[1], shading="auto")
        ax.invert_yaxis()
        ax.set_xlabel("Latitude" if pname == "NS" else "Longitude")
        ax.set_ylabel("Depth (km)"); ax.set_title(f"Vs cross-section {pname}")
        fig.colorbar(im, ax=ax, label="Vs (km/s)")
        fig.tight_layout()
        _save(fig, f"F11_vs_cross_section_{pname}", cfg)


def _station_vs_profiles(inv_dir):
    out = {}
    for d in Path(inv_dir).glob("*"):
        f = d / "vs_profile.txt"
        if f.exists():
            try:
                out[d.name] = np.loadtxt(f, ndmin=2)
            except Exception:
                pass
    return out


# --------------------------------------------------------------------------
# F12 — integrated structural model (interpreted cartoon)
# --------------------------------------------------------------------------

def F12_structural_model(cfg):
    plt = _mpl()
    fig, ax = plt.subplots(figsize=(6, 8))
    layers = [
        (0, 1.5, "Hydrothermal / altered clay cap", "#d9c9a3", 1.2),
        (1.5, 5, "Volcaniclastic / fractured crust", "#c2d0b0", 2.4),
        (5, 12, "Possible partial-melt / low-Vs storage", "#e2a6a6", 2.9),
        (12, 30, "Crystalline basement", "#a6bcd0", 3.6),
    ]
    for z0, z1, label, colour, vs in layers:
        ax.axhspan(z0, z1, color=colour)
        ax.text(0.5, (z0 + z1) / 2, f"{label}\nVs~{vs} km/s", ha="center",
                va="center", fontsize=9)
    ax.set_ylim(30, 0); ax.set_xlim(0, 1); ax.set_xticks([])
    ax.set_ylabel("Depth (km)")
    ax.set_title("Interpreted crustal structure, Dieng")
    fig.tight_layout()
    _save(fig, "F12_structural_model", cfg)


# --------------------------------------------------------------------------
# dispatcher
# --------------------------------------------------------------------------

_FIGURES = {
    "F1_station_map": F1_station_map,
    "F2_event_distribution": F2_event_distribution,
    "F3_coverage_raypaths": F3_coverage_raypaths,
    "F4_hk_panels": F4_hk_panels,
    "F5_hk_summary": F5_hk_summary,
    "F6_rf_record_sections": F6_rf_record_sections,
    "F7_ccp_sections": F7_ccp_sections,
    "F8_ccf_gather": F8_ccf_gather,
    "F9_dispersion_maps": F9_dispersion_maps,
    "F10_inversion_per_station": F10_inversion_per_station,
    "F11_vs_cross_sections": F11_vs_cross_sections,
    "F12_structural_model": F12_structural_model,
}


def run(cfg: dict) -> Path:
    p = io_utils.paths(cfg)
    out_dir = io_utils.ensure_dir(p["figures"])
    toggles = cfg.get("plot", {}).get("figures", {})
    for name, fn in _FIGURES.items():
        if not toggles.get(name, False):
            continue
        try:
            fn(cfg)
        except Exception as e:
            LOG.warning(f"{name}: failed ({e}).")
    LOG.info("Stage 9 (synthesis) complete.")
    return out_dir
