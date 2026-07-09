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
    # PyGMT's savefig can't write SVG; fall back to PDF (also vector, GMT-native).
    pygmt_ok = {"png", "pdf", "jpg", "bmp", "eps", "tif", "tiff", "kml", "ppm"}
    pygmt_alt = {"svg": "pdf"}
    written = []
    for fmt in fmts:
        out_fmt = fmt
        if is_pygmt and fmt not in pygmt_ok:
            out_fmt = pygmt_alt.get(fmt)
            if out_fmt is None:
                LOG.warning(f"{name}: pygmt cannot write .{fmt} — skipping that format")
                continue
        target = out_dir / f"{name}.{out_fmt}"
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
        # H on the x-axis, k (Vp/Vs) on the y-axis: transpose the (nH, nK) stack.
        # Map the stack linearly onto 0..1 (min->0, max->1) so the colour axis is a
        # true 0-1 range like paper Fig. 3, then paint invalid cells at 0 (dark).
        s = d["stack"].T
        mn, mx = np.nanmin(s), np.nanmax(s)
        rng = mx - mn
        s = (s - mn) / rng if (np.isfinite(rng) and rng > 0) else np.zeros_like(s)
        s = np.nan_to_num(s, nan=0.0)
        im = ax.pcolormesh(d["h_grid"], d["k_grid"], s, cmap="viridis",
                           shading="auto", vmin=0, vmax=1)
        # white crosshair at the maximum (paper Fig. 3 convention)
        ax.axvline(float(d["bestH"]), color="w", lw=1.0)
        ax.axhline(float(d["bestK"]), color="w", lw=1.0)
        ax.set_xlabel("H (km)"); ax.set_ylabel("k (Vp/Vs)")
        ax.set_title(npz.stem.replace("_hk", ""))
        fig.colorbar(im, ax=ax, label="amplitude")
    fig.tight_layout()
    _save(fig, "F4_hk_panels", cfg)


# --------------------------------------------------------------------------
# F5 — H-kappa summary across stations
# --------------------------------------------------------------------------

def F5_hk_summary(cfg):
    plt = _mpl()
    p = io_utils.paths(cfg)
    hk_dir = p.get("hk_out")
    # Per-class H-kappa summaries are hk_<class>.csv. The glob also matches
    # hk_corrected.csv (the kappa-correction table, different schema with no
    # H_km/bounds columns) — exclude it, and defensively skip any csv missing
    # the columns this figure plots, so one odd file never fails the whole figure.
    need = {"station", "H_km", "H_lo", "H_hi", "kappa", "k_lo", "k_hi"}
    csvs = [c for c in sorted(Path(hk_dir).glob("hk_*.csv"))
            if c.name != "hk_corrected.csv"] if hk_dir else []
    csvs = [c for c in csvs if need.issubset(pd.read_csv(c, nrows=0).columns)]
    if not csvs:
        LOG.warning("F5: no per-class hk_<class>.csv summaries — skipping.")
        return
    # QC thresholds (config-overridable): a solution is flagged unreliable if its
    # kappa sits on the search-grid edge (the maximum ran off the grid -> noise or
    # wrong band) or its H 95%-contour half-width is too wide (poorly constrained).
    hkp = cfg.get("plot", {}).get("hk_qc", {})
    k_lo_edge, k_hi_edge = cfg.get("hk", {}).get("k_range", [1.6, 2.5])
    k_edge_tol = float(hkp.get("kappa_edge_tol", 0.03))
    h_err_max = float(hkp.get("h_halfwidth_max_km", 12.0))

    def _bad(row):
        rails = (row["kappa"] <= k_lo_edge + k_edge_tol
                 or row["kappa"] >= k_hi_edge - k_edge_tol)
        h_hw = max(row["H_km"] - row["H_lo"], row["H_hi"] - row["H_km"])
        return rails or (np.isfinite(h_hw) and h_hw > h_err_max)

    # H-k crossplot: k (Vp/Vs) on the x-axis, H (km) on the y-axis (depth down).
    # One point per station/class, with x/y error bars from the 95%-contour bounds
    # and station labels; QC-flagged solutions drawn faint with an x.
    fig, ax = plt.subplots(figsize=(7.5, 6))
    flagged = []
    for csv in csvs:
        cls = csv.stem.replace("hk_", "")
        df = pd.read_csv(csv).reset_index(drop=True)
        bad = df.apply(_bad, axis=1).to_numpy()
        flagged += [f"{s}/{cls}" for s in df["station"][bad]]
        colour = None
        for mask, alpha, marker in ((~bad, 1.0, "o"), (bad, 0.3, "x")):
            if not mask.any():
                continue
            sub = df[mask]
            eb = ax.errorbar(sub["kappa"], sub["H_km"],
                             xerr=[sub["kappa"] - sub["k_lo"], sub["k_hi"] - sub["kappa"]],
                             yerr=[sub["H_km"] - sub["H_lo"], sub["H_hi"] - sub["H_km"]],
                             fmt=marker, ms=7, capsize=2, lw=0.8, alpha=alpha,
                             color=(colour if colour else None),
                             label=(cls if colour is None else None))
            colour = colour or eb.lines[0].get_color()
            for _, r in sub.iterrows():
                ax.annotate(r["station"], (r["kappa"], r["H_km"]), fontsize=6,
                            xytext=(3, 3), textcoords="offset points", alpha=alpha)
    # shade the k grid-edge bands so railed (unreliable) solutions are obvious
    ax.axvspan(k_hi_edge - k_edge_tol, k_hi_edge, color="0.88", zorder=0)
    ax.axvspan(k_lo_edge, k_lo_edge + k_edge_tol, color="0.88", zorder=0)
    ax.set_xlabel("k (Vp/Vs)"); ax.set_ylabel("H (km)")
    ax.set_xlim(k_lo_edge, k_hi_edge)
    ax.invert_yaxis()                         # depth increases downward
    ax.legend(title="source class")
    ax.set_title(f"H-k stacking summary  (grey = k grid edge; "
                 f"faint x = QC-flagged, {len(flagged)})")
    fig.tight_layout()
    _save(fig, "F5_hk_summary", cfg)
    if flagged:
        LOG.info(f"F5 QC: {len(flagged)} unreliable H-kappa solutions "
                 f"(kappa on grid edge or H poorly constrained): {flagged}")


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
    # Station selection. An explicit override wins — either run_synthesis.py
    # --rf-stations (cfg["_rf_stations"]) or plot.rf_stations in the config —
    # so any station can be plotted by name; otherwise fall back to the shared
    # representative_stations used by the other per-station panels.
    override = cfg.get("_rf_stations") or cfg.get("plot", {}).get("rf_stations")
    reps = override or cfg.get("plot", {}).get("representative_stations", [])
    reps = [str(s).strip().upper() for s in reps if str(s).strip()]
    files = []
    for sta in reps:
        for cls in classes:
            f = Path(p["rf_out"]) / f"{sta}_{cls}.h5"
            if f.exists():
                files.append((sta, cls, f))
    if not files:
        LOG.warning(f"F6: no RF H5 found for stations {reps} in {p['rf_out']} "
                    f"— skipping.")
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
    # Keep the default (representative-station) figure under its canonical name so
    # the pipeline overwrites it in place; a by-name request gets a station-tagged
    # name so it never clobbers that shared figure.
    name = "F6_rf_record_sections"
    if override:
        name += "_" + "_".join(reps)
    _save(fig, name, cfg)


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
    # Paper Fig. 6 style: the section is continuous because the CCP stack itself
    # uses dense, overlapping along-profile bins (ccp.along_step / ccp.bin_km) and
    # each RF is unit-peak normalized before stacking (ccp) — NOT image smoothing.
    # The stored `amp` is already a mean normalized amplitude; here we only blank
    # never-sampled bins and plot it on a fixed colour range.
    clim = float(cfg.get("plot", {}).get("ccp_clim", 0.2))
    # optional depth crop of the plotted section (km); None -> full depth_range.
    ccp_dmax = cfg.get("plot", {}).get("ccp_max_depth", None)

    # Profile geometry (endpoints) keyed by name, so the map can draw each line and
    # mark its 0-km end (= endpoint 1 = lon1/lat1, where the section's x-axis starts).
    geom = {pr["name"]: pr for pr in cfg.get("ccp", {}).get("profiles", [])}
    try:
        stations, _ = io_utils.load_stations(cfg)
    except Exception as e:
        LOG.debug(f"F7 map: stations unavailable ({e}).")
        stations = []

    from matplotlib.gridspec import GridSpec
    from matplotlib.patches import Rectangle
    n = len(npzs)
    fig = plt.figure(figsize=(17, 3.8 * n))
    # LEFT column: a network-context map (top) + a zoomed profile map (bottom).
    # RIGHT column: the section panels stacked. Split the left column in two.
    gs = GridSpec(n, 2, width_ratios=[1.0, 2.3], figure=fig,
                  wspace=0.22, hspace=0.45)
    half = max(1, n // 2)
    mlat = np.mean([s.latitude for s in stations]) if stations else \
        np.mean([pr["lat1"] for pr in geom.values()] or [0.0])
    aspect = 1.0 / max(np.cos(np.deg2rad(mlat)), 1e-3)      # ~true-scale lon/lat
    cyc = plt.rcParams["axes.prop_cycle"].by_key()["color"]

    # zoom window = profile bounding box + padding
    plons = [pr[k] for pr in geom.values() for k in ("lon1", "lon2")] or [0.0]
    plats = [pr[k] for pr in geom.values() for k in ("lat1", "lat2")] or [0.0]
    padx = max((max(plons) - min(plons)) * 0.35, 0.03)
    pady = max((max(plats) - min(plats)) * 0.35, 0.03)
    zx = (min(plons) - padx, max(plons) + padx)
    zy = (min(plats) - pady, max(plats) + pady)

    def _draw_profiles(ax, labels):
        if stations:
            ax.scatter([s.longitude for s in stations], [s.latitude for s in stations],
                       marker="^", s=26, c="0.35", zorder=2, label="stations")
        for i, (nm, pr) in enumerate(geom.items()):
            c = cyc[i % len(cyc)]
            ax.plot([pr["lon1"], pr["lon2"]], [pr["lat1"], pr["lat2"]], "-", lw=2.2,
                    color=c, zorder=3)
            ax.plot(pr["lon1"], pr["lat1"], "o", color=c, ms=8, zorder=4)       # 0 km
            ax.plot(pr["lon2"], pr["lat2"], "o", mfc="white", mec=c, mew=1.6,   # far end
                    ms=8, zorder=4)
            if not labels:
                continue
            # offset each label OUTWARD from the line centre so the two 0-km ends
            # (which nearly coincide) don't collide.
            mlon = 0.5 * (pr["lon1"] + pr["lon2"]); mla = 0.5 * (pr["lat1"] + pr["lat2"])
            for lon, lat, txt in ((pr["lon1"], pr["lat1"], f"{nm} 0 km"),
                                  (pr["lon2"], pr["lat2"], f"{nm}'")):
                ox = 9 * (1 if lon >= mlon else -1)
                oy = 9 * (1 if lat >= mla else -1)
                ax.annotate(txt, (lon, lat), color=c, fontsize=9, fontweight="bold",
                            xytext=(ox, oy), textcoords="offset points",
                            ha="left" if ox >= 0 else "right",
                            va="bottom" if oy >= 0 else "top", zorder=5)

    # context map (full network) with the zoom window drawn on it
    axc = fig.add_subplot(gs[:half, 0])
    _draw_profiles(axc, labels=False)
    axc.add_patch(Rectangle((zx[0], zy[0]), zx[1] - zx[0], zy[1] - zy[0],
                            fill=False, ec="k", lw=1.2, ls="--", zorder=6))
    axc.set_aspect(aspect)
    axc.set_xlabel("Longitude (deg E)"); axc.set_ylabel("Latitude (deg N)")
    axc.set_title("Network context (dashed box = zoom)")
    axc.grid(True, ls=":", alpha=0.4)
    axc.legend(loc="best", fontsize=8)

    # zoomed map (profiles + labels)
    axz = fig.add_subplot(gs[half:, 0])
    _draw_profiles(axz, labels=True)
    axz.set_xlim(*zx); axz.set_ylim(*zy)
    axz.set_aspect(aspect)
    axz.set_xlabel("Longitude (deg E)"); axz.set_ylabel("Latitude (deg N)")
    axz.set_title("CCP profiles (filled dot = 0 km end)")
    axz.grid(True, ls=":", alpha=0.4)

    for j, f in enumerate(npzs):
        ax = fig.add_subplot(gs[j, 1])
        d = np.load(f)
        amp = d["amp"].astype(float)               # (along, depth), mean norm amp
        if "count" in d.files:                     # blank bins that never sampled
            amp = np.where(d["count"] > 0, amp, np.nan)
        amp = np.ma.masked_invalid(amp)
        im = ax.pcolormesh(d["along"], d["depth"], amp.T, cmap="RdBu_r",
                           vmin=-clim, vmax=clim, shading="nearest")
        # depth axis increases downward; crop to ccp_max_depth if set.
        dbot = float(ccp_dmax) if ccp_dmax else float(np.max(d["depth"]))
        ax.set_ylim(dbot, 0); ax.set_ylabel("Depth (km)")
        ax.set_xlabel("Distance along profile (km)"); ax.set_title(f"CCP {f.stem}")
        fig.colorbar(im, ax=ax, label="Mean normalized amplitude")
    # the spanning map axes aren't tight_layout-compatible; spacing is already set
    # via the GridSpec wspace/hspace, so skip tight_layout (and its warning) here.
    _save(fig, "F7_ccp_sections", cfg)


# --------------------------------------------------------------------------
# F8 — noise CCF gather vs interstation distance
# --------------------------------------------------------------------------

def F8_ccf_gather(cfg):
    """Noise CCF gather vs interstation distance.

    The raw CCFs span the full ±(cc_len/2) correlation lag (±1800 s here), but the
    surface-wave signal only occupies |t| < dist/v_min. Plotting the full lag drowns
    the signal in ~2 % of the panel, so we crop to a physically-motivated window,
    optionally fold to the symmetric (causal+acausal) component to gain SNR, and
    stack into distance bins so 300 overlapping pairs become a legible moveout.
    Dashed reference lines mark constant apparent velocities.
    """
    plt = _mpl()
    from scipy.signal import butter, filtfilt, hilbert
    pl = cfg.get("plot", {})
    p = io_utils.paths(cfg)
    files = sorted(Path(p["ccfs"]).glob("*.npz")) if Path(p["ccfs"]).exists() else []
    if not files:
        LOG.warning("F8: no CCFs — skipping.")
        return
    band = pl.get("ccf_bandpass", [0.2, 0.4])
    symmetric = pl.get("ccf_symmetric", True)     # fold causal+acausal (SNR)
    bin_km = pl.get("ccf_dist_bin_km", 0.5)        # distance-bin stacking; 0 = per pair (MSNoise style)
    ref_vels = pl.get("ccf_ref_vels", [1.5, 2.5, 3.5])  # km/s moveout guides
    v_min = pl.get("ccf_v_min", 1.0)               # sets the auto lag window
    fill = pl.get("ccf_fill", True)                # fill positive lobes (variable-area; False = wiggle only)
    fill_color = pl.get("ccf_fill_color", "k")     # variable-area fill colour (black is the seismic standard)
    ampli = pl.get("ccf_ampli")                    # km per unit amplitude; None = auto (row spacing)
    gain = pl.get("ccf_gain", 1.0)                 # multiplies the wiggle amplitude (make moveout pop)
    clip = pl.get("ccf_clip", 1.0)                 # clip normalized amplitude to +/-clip before scaling
    min_pairs = pl.get("ccf_min_pairs", 1)         # drop distance bins with fewer pairs (undersampled)
    annotate_n = pl.get("ccf_annotate_n", False)   # label each row with its pair count
    zero_lag_mute = pl.get("ccf_zero_lag_mute", 0.0)  # s; Gaussian taper of the stationary
                                                    # zero-lag lobe that otherwise masks the moveout
    pws = pl.get("ccf_pws", 0.0)                    # phase-weighted-stack power (nu); 0 = linear stack.
                                                    # nu~1 suppresses incoherent scatter within each bin
    row_norm = pl.get("ccf_row_norm", True)         # renormalize each bin to unit amplitude before
                                                    # drawing (equalizes rows; needed so PWS bins stay visible)

    # ---- load, filter, normalize each pair on its own signal window -------------
    traces = []                                     # (dist_km, lag[>=0], amp)
    dmax = 0.0
    for f in files:
        d = np.load(f)
        dist = float(d["dist_km"]) if "dist_km" in d else np.nan
        if not np.isfinite(dist) or dist <= 0:
            continue
        lag, ccf = d["lag_s"].astype(float), d["ccf"].astype(float)
        sr = 1.0 / (lag[1] - lag[0])
        b, a = butter(3, [band[0] / (sr / 2), band[1] / (sr / 2)], btype="band")
        y = filtfilt(b, a, ccf)
        dt = 1.0 / sr
        # resample onto a common positive-lag grid via causal/acausal interpolation
        tmax_data = float(np.abs(lag).max())
        tpos = np.arange(0.0, tmax_data, dt)
        causal = np.interp(tpos, lag, y)
        acausal = np.interp(tpos, -lag[::-1], y[::-1])
        if zero_lag_mute > 0:
            # taper -> 0 at lag 0, -> 1 by ~2*mute; kills the stationary lobe near t=0
            taper = 1.0 - np.exp(-(tpos / zero_lag_mute) ** 2)
            causal = causal * taper
            acausal = acausal * taper
        amp = (causal + acausal) / 2.0 if symmetric else causal
        # (for the two-sided view we keep both branches below)
        traces.append((dist, tpos, causal, acausal, amp))
        dmax = max(dmax, dist)

    if not traces:
        LOG.warning("F8: no finite-distance CCFs — skipping.")
        return

    lag_max = pl.get("ccf_lag_max") or (dmax / v_min + 5.0)   # s, cropped window

    # ---- distance binning + linear stack of per-pair-normalized traces ----------
    tpos = traces[0][1]
    win = tpos <= lag_max
    tpos = tpos[win]

    def _norm(sig):
        m = np.max(np.abs(sig[win])) or 1.0
        return sig[win] / m

    def _stack(members):
        # linear mean, optionally phase-weighted (Schimmel & Paulssen 1997): the linear
        # stack is multiplied by the inter-trace instantaneous-phase coherence^pws, which
        # suppresses scattered energy that is incoherent across the pairs in this bin.
        A = np.asarray(members)
        lin = A.mean(axis=0)
        if pws and pws > 0 and A.shape[0] >= 2:
            phase = np.angle(hilbert(A, axis=1))
            coh = np.abs(np.mean(np.exp(1j * phase), axis=0))
            return lin * coh ** pws
        return lin

    if bin_km and bin_km > 0:
        acc = {}                                     # bin -> [amp[], cau[], aca[]]
        for dist, _t, cau, aca, amp in traces:
            k = int(round(dist / bin_km))
            s = acc.setdefault(k, [[], [], []])
            s[0].append(_norm(amp))
            s[1].append(_norm(cau))
            s[2].append(_norm(aca))
        rows = [(k * bin_km, len(s[0]), _stack(s[0]), _stack(s[1]), _stack(s[2]))
                for k, s in sorted(acc.items()) if len(s[0]) >= min_pairs]
        spacing = bin_km
    else:
        rows = [(dist, 1, _norm(amp), _norm(cau), _norm(aca))
                for dist, _t, cau, aca, amp in sorted(traces)]
        spacing = np.median(np.diff(sorted(t[0] for t in traces))) or 0.2

    if not rows:
        LOG.warning("F8: all distance bins culled by ccf_min_pairs — skipping.")
        return
    dmax_show = max(r[0] for r in rows)          # top of the *populated* range (after culling)
    scale = (float(ampli) if ampli else spacing) * gain   # wiggle amplitude in km units
    lw = 0.4 if len(rows) < 60 else 0.3

    # ---- draw -------------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(7, 9))
    for dist, n, sym, cau, aca in rows:
        if symmetric:
            branches = [(tpos, sym)]
            rn = np.max(np.abs(sym)) if row_norm else 1.0
        else:
            branches = [(tpos, cau), (-tpos, aca)]   # causal right, acausal left
            # one factor for both branches keeps their relative amplitude (= noise-source asymmetry)
            rn = max(np.max(np.abs(cau)), np.max(np.abs(aca))) if row_norm else 1.0
        rn = rn or 1.0
        for x, s in branches:
            s = np.clip(s / rn, -clip, clip)
            w = dist + s * scale
            ax.plot(x, w, "k", lw=lw, zorder=3)
            if fill:
                ax.fill_between(x, dist, w, where=(s > 0), color=fill_color,
                                lw=0, zorder=2)
        if annotate_n and (bin_km and bin_km > 0):
            ax.annotate(str(int(n)), xy=(lag_max, dist), xytext=(2, 0),
                        textcoords="offset points", fontsize=5.5, color="0.5",
                        va="center", clip_on=False)

    for v in ref_vels:
        tv = np.array([0.0, lag_max])
        ax.plot(tv, tv * v, "--", color="0.35", lw=0.8, zorder=4)
        if not symmetric:
            ax.plot(-tv, tv * v, "--", color="0.35", lw=0.8, zorder=4)
        # label each guide where it exits the panel (top edge, else right edge)
        if dmax_show / v <= lag_max:
            lx, ly, va, ha = dmax_show / v, dmax_show, "bottom", "center"
        else:
            lx, ly, va, ha = lag_max, lag_max * v, "center", "right"
        ax.annotate(f"{v:g} km/s", xy=(lx, ly), xytext=(3, 3),
                    textcoords="offset points", ha=ha, va=va,
                    fontsize=7, color="0.35")

    ax.set_xlim(0 if symmetric else -lag_max, lag_max)
    ax.set_ylim(-spacing, dmax_show + spacing)
    ax.set_xlabel("Lag (s)")
    ax.set_ylabel("Interstation distance (km)")
    tag = "symmetric" if symmetric else "two-sided"
    grouping = f"{len(rows)} bins" if (bin_km and bin_km > 0) else f"{len(rows)} pairs"
    ax.set_title(f"CCF gather {band[0]}–{band[1]} Hz  ({tag}, {grouping})")
    fig.tight_layout()
    _save(fig, "F8_ccf_gather", cfg)


# --------------------------------------------------------------------------
# F13 — pair dispersion curves (Stage 6 output: ant/disp/*.disp)
# --------------------------------------------------------------------------

def _load_disp(f: Path):
    """Load a ``period phase_vel group_vel sigma`` table; None if empty/unreadable."""
    try:
        arr = np.loadtxt(f, ndmin=2)
    except Exception:
        return None
    if arr.size == 0 or arr.shape[1] < 2:
        return None
    return arr[np.argsort(arr[:, 0])]           # ascending period


def F13_dispersion_pair_curves(cfg):
    """Every station-pair Rayleigh phase-velocity curve + a median/16-84 envelope."""
    plt = _mpl()
    p = io_utils.paths(cfg)
    disp_dir = p.get("disp")
    files = sorted(Path(disp_dir).glob("*.disp")) if disp_dir else []
    if not files:
        LOG.warning("F13: no pair .disp curves (run Stage 6) — skipping.")
        return
    from collections import defaultdict
    by_T = defaultdict(list)
    fig, ax = plt.subplots(figsize=(7, 5))
    n_pairs = 0
    for f in files:
        arr = _load_disp(f)
        if arr is None:
            continue
        T, c = arr[:, 0], arr[:, 1]
        ax.plot(T, c, color="0.7", lw=0.6, alpha=0.6, zorder=1)
        for Ti, ci in zip(T, c):
            by_T[round(float(Ti), 4)].append(float(ci))
        n_pairs += 1
    if n_pairs == 0:
        LOG.warning("F13: pair .disp files present but empty — skipping.")
        return
    Ts = sorted(by_T)
    med = [np.median(by_T[T]) for T in Ts]
    lo = [np.percentile(by_T[T], 16) for T in Ts]
    hi = [np.percentile(by_T[T], 84) for T in Ts]
    ax.fill_between(Ts, lo, hi, color="tab:blue", alpha=0.2, zorder=2,
                    label="16–84th percentile")
    ax.plot(Ts, med, color="tab:blue", lw=2.2, zorder=3, label="median")
    ax.set_xlabel("Period (s)"); ax.set_ylabel("Phase velocity (km/s)")
    ax.set_title(f"Rayleigh phase-velocity dispersion — {n_pairs} station pairs")
    ax.legend()
    fig.tight_layout()
    _save(fig, "F13_dispersion_pair_curves", cfg)


# --------------------------------------------------------------------------
# F14 — per-station dispersion curves (Stage 7 output: tomo/*_disp.txt)
# --------------------------------------------------------------------------

def F14_dispersion_station_curves(cfg):
    """Two-station-average curve per station (σ band), over the faint pair cloud."""
    plt = _mpl()
    p = io_utils.paths(cfg)
    tomo_dir = p.get("tomo")
    files = sorted(Path(tomo_dir).glob("*_disp.txt")) if tomo_dir else []
    if not files:
        LOG.warning("F14: no per-station curves (run Stage 7) — skipping.")
        return
    fig, ax = plt.subplots(figsize=(7, 5))
    # faint backdrop: the raw pair curves this average was built from (context).
    disp_dir = p.get("disp")
    if disp_dir and Path(disp_dir).exists():
        for f in sorted(Path(disp_dir).glob("*.disp")):
            arr = _load_disp(f)
            if arr is not None:
                ax.plot(arr[:, 0], arr[:, 1], color="0.85", lw=0.5, zorder=1)
    cmap = plt.get_cmap("turbo")
    n = 0
    for i, f in enumerate(files):
        arr = _load_disp(f)
        if arr is None:
            continue
        T, c = arr[:, 0], arr[:, 1]
        colour = cmap(i / max(1, len(files) - 1))
        ax.plot(T, c, color=colour, lw=1.5, marker="o", ms=3, zorder=3,
                label=f.stem.replace("_disp", ""))
        if arr.shape[1] > 3:
            sig = arr[:, 3]
            ax.fill_between(T, c - sig, c + sig, color=colour, alpha=0.15, zorder=2)
        n += 1
    if n == 0:
        LOG.warning("F14: per-station curve files present but empty — skipping.")
        return
    ax.set_xlabel("Period (s)"); ax.set_ylabel("Phase velocity (km/s)")
    ax.set_title(f"Per-station dispersion (two-station average) — {n} stations")
    ax.legend(fontsize=6, ncol=2 if n > 12 else 1, loc="best")
    fig.tight_layout()
    _save(fig, "F14_dispersion_station_curves", cfg)


# --------------------------------------------------------------------------
# F9 — dispersion maps (path A only)
# --------------------------------------------------------------------------

def F9_dispersion_maps(cfg):
    """DSurfTomo 3-D Vs horizontal depth slices (replaces the old FMST maps)."""
    plt = _mpl()
    p = io_utils.paths(cfg)
    d_out = io_utils.resolve_path(cfg.get("dsurftomo", {}).get("output_dir", "dsurftomo"),
                                  cfg["_project_root"])
    npz = Path(d_out) / "vs3d.npz"
    if not npz.exists():
        LOG.info("F9: no DSurfTomo vs3d.npz — run the DSurfTomo stage first, skipping.")
        return
    d = np.load(npz)
    lon, lat, depth, vs = d["lon"], d["lat"], d["depth"], d["vs"]
    slices = cfg.get("plot", {}).get("vs_slice_depths_km", [2, 5, 10, 20])
    avail = np.unique(depth)
    slices = [min(avail, key=lambda z: abs(z - s)) for s in slices]
    clip = cfg.get("plot", {}).get("vs_clip", [0.5, 4.8])
    fig, axes = plt.subplots(1, len(slices), figsize=(4 * len(slices), 4), squeeze=False)
    for ax, zt in zip(axes[0], slices):
        m = np.isclose(depth, zt)
        sc = ax.scatter(lon[m], lat[m], c=vs[m], cmap=_cmap_velocity(cfg),
                        vmin=clip[0], vmax=clip[1], s=14, marker="s")
        ax.set_title(f"Vs @ {zt:g} km"); ax.set_xlabel("Lon"); ax.set_ylabel("Lat")
        ax.set_aspect("equal")
        fig.colorbar(sc, ax=ax, label="Vs (km/s)")
    fig.tight_layout()
    _save(fig, "F9_vs_depth_slices", cfg)


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
                    "(run Stage 8; it exports vs_profile.txt and BayHunter's "
                    "posterior figures per station).")
        return
    LOG.info(f"F10: per-station inversion outputs live under "
             f"{[str(f) for f in found]}.")


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
    "F13_dispersion_pair_curves": F13_dispersion_pair_curves,
    "F14_dispersion_station_curves": F14_dispersion_station_curves,
}

# Which figures become drawable the moment a given stage's outputs land. Used to
# render incrementally (a figure right after its data exists) instead of deferring
# every plot to Stage 9. Keyed by the orchestrator's stage keys; a figure is only
# ever rendered once its own inputs are present (each figure self-skips otherwise),
# so listing it here is safe even if the stage produced nothing.
STAGE_FIGURES: dict[str, list[str]] = {
    "prep":       ["F1_station_map", "F2_event_distribution"],
    "rf":         ["F6_rf_record_sections"],
    "hk":         ["F4_hk_panels", "F5_hk_summary"],
    "ccp":        ["F7_ccp_sections"],
    "ant":        ["F3_coverage_raypaths", "F8_ccf_gather"],
    "dispersion": ["F13_dispersion_pair_curves"],
    "tomo":       ["F14_dispersion_station_curves"],
    "dsurftomo":  ["F9_dispersion_maps"],
    "inversion":  ["F10_inversion_per_station", "F11_vs_cross_sections",
                   "F12_structural_model"],
    "synthesis":  [],                       # Stage 9 renders the full set below
}


def _fig_mtimes(fig_dir: Path) -> dict[str, float]:
    return {f.name: f.stat().st_mtime for f in Path(fig_dir).glob("*") if f.is_file()}


def _render(cfg: dict, names) -> list[str]:
    """Render the named figures that are toggled on; return those actually drawn.

    Each figure degrades gracefully (logs a skip and writes nothing) when its
    inputs are missing, so this is safe to call after any stage, in any order.
    "Actually drawn" is detected by a change in the ``figures/`` directory around
    each call — not by the function returning — so a figure that self-skips is
    honestly reported as *not* made, even though it raised no error. (Detection is
    by output file, since some figures save under a different name than their
    toggle, e.g. F9_dispersion_maps -> F9_vs_depth_slices.)
    """
    p = io_utils.paths(cfg)
    fig_dir = io_utils.ensure_dir(p["figures"])
    toggles = cfg.get("plot", {}).get("figures", {})
    made: list[str] = []
    for name in names:
        fn = _FIGURES.get(name)
        if fn is None or not toggles.get(name, False):
            continue
        before = _fig_mtimes(fig_dir)
        try:
            fn(cfg)
        except Exception as e:
            LOG.warning(f"{name}: failed ({e}).")
            continue
        after = _fig_mtimes(fig_dir)
        if any(after.get(k) != before.get(k) for k in set(after) | set(before)):
            made.append(name)
    return made


def plot_for_stage(cfg: dict, stage_key: str) -> list[str]:
    """Render just the figures whose data becomes available after ``stage_key``.

    Called from the pipeline's per-stage hook (``progress.run_stage``) so plots
    appear incrementally. Returns the names actually drawn (may be empty).
    """
    made = _render(cfg, STAGE_FIGURES.get(stage_key, []))
    if made:
        LOG.info(f"[{stage_key}] rendered {made}")
    return made


def run(cfg: dict) -> Path:
    p = io_utils.paths(cfg)
    out_dir = io_utils.ensure_dir(p["figures"])
    _render(cfg, list(_FIGURES))            # Stage 9: (re)render the complete set
    LOG.info("Stage 9 (synthesis) complete.")
    return out_dir
