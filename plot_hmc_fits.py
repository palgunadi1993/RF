#!/usr/bin/env python3
"""Per-station joint-inversion summary figure (Stage 8, rfsurfhmc):

  (a) depth vs Vs           — posterior model density (all sampled trials),
  (b) receiver function fit — posterior-PREDICTIVE density of the synthetic
                              RFs (every accepted sample's forward response,
                              stored per sample in the chain HDF5) with the
                              observed stack overlaid, one row per RF class,
  (c) Rayleigh phase velocity fit — posterior-predictive density per period
                              with the observed curve and its uncertainty.

All three use the same row/column-normalized probability heat convention.

Outputs inversion/<sta>/joint_fit.png per station.
Usage: python plot_hmc_fits.py [--config config.yaml] [--stations ST01,ST02]
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import h5py
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from plot_hmc_posterior import _CHAIN, _DENSITY_CMAP, _INK, station_figure  # noqa: E402
from rf_pipeline.io_utils import load_config, paths  # noqa: E402


def rf_time_axis(cfg, cls):
    """Fitted-window time axis for a class, mirroring inversion._build_joint."""
    rf_cfg = cfg.get("rf", {})
    hp = cfg.get("inversion", {}).get("rfsurfhmc", {}) or {}
    params = {**rf_cfg.get("defaults", {}), **rf_cfg.get("classes", {}).get(cls, {})}
    win = params.get("window", [-10, 35])
    hz = float(cfg.get("inversion", {}).get("rf_resample_hz", 25.0)) or 250.0
    t = win[0] + np.arange(int(round((win[1] - win[0]) * hz)) + 1) / hz
    fw = hp.get("fit_window", [-5.0, 25.0])
    if fw:
        t = t[(t >= float(fw[0])) & (t <= float(fw[1]))]
    return t


def load_ensemble(sta_dir: Path):
    """(obs, syn_matrix) from the chain HDF5 stores."""
    obs, syn = None, []
    for f5 in sorted(sta_dir.glob("hmc.*.h5")):
        try:
            with h5py.File(f5, "r") as f:
                if obs is None and "obs" in f:
                    obs = f["obs"][:]
                keys = sorted((k for k in f.keys() if k.isdigit()), key=int)
                syn.append(np.array([f[f"{k}/syn"][:] for k in keys]))
        except OSError:
            continue
    if obs is None or not syn:
        return None, None
    return obs, np.vstack(syn)


def _column_density(ax, x_edges, values, y_bins):
    """values (nsamples, nx) -> column-normalized 2D density on ax."""
    dens = np.zeros((y_bins.size - 1, x_edges.size - 1))
    for j in range(values.shape[1]):
        h, _ = np.histogram(values[:, j], bins=y_bins)
        dens[:, j] = h / max(h.max(), 1)
    ax.pcolormesh(x_edges, y_bins, dens, cmap=_DENSITY_CMAP, vmin=0, vmax=1,
                  shading="flat", rasterized=True)


def station_fit_figure(cfg, sta: str, sta_dir: Path) -> Path | None:
    p = paths(cfg)
    obs, syn = load_ensemble(sta_dir)
    if obs is None or not (sta_dir / "vs_profile.txt").exists():
        return None
    hp = cfg.get("inversion", {}).get("rfsurfhmc", {}) or {}
    classes = hp.get("rf_classes") or [hp.get("rf_class", "local_deep")]
    if isinstance(classes, str):
        classes = [classes]
    # classes actually present at build time = those whose stack existed; the
    # data-vector length is the ground truth, so trim the class list until the
    # remainder equals this station's dispersion-point count (variable per
    # station — never assume a fixed number of periods).
    axes_t = {c: rf_time_axis(cfg, c) for c in classes}
    disp = np.loadtxt(p["tomo"] / f"{sta}_disp.txt", ndmin=2)
    cand = list(classes)
    while cand and sum(axes_t[c].size for c in cand) + disp.shape[0] != obs.size:
        cand = cand[:-1]
    classes = cand or classes
    n_rf = sum(axes_t[c].size for c in classes)
    n_swd = obs.size - n_rf
    periods = disp[:n_swd, 0]
    sig_swd = disp[:n_swd, 3] if disp.shape[1] > 3 else np.full(n_swd, 0.05)

    nrows = len(classes) + 1
    fig = plt.figure(figsize=(11, 2.1 * nrows + 1.2), dpi=200)
    gs = fig.add_gridspec(nrows, 2, width_ratios=[1, 2.1],
                          hspace=0.55, wspace=0.22)

    # (a) model density — reuse the posterior plot on the left column
    ax_a = fig.add_subplot(gs[:, 0])
    vs_lim = cfg.get("plot", {}).get("vs_clip", [0.5, 4.8])
    depth_max = float(cfg.get("inversion", {}).get("depth_max", 60))
    station_figure(sta, sta_dir, vs_lim, depth_max, ax=ax_a)
    ax_a.set_xlabel("$V_S$ (km/s)", fontsize=9)
    ax_a.set_ylabel("Depth (km)", fontsize=9)
    ax_a.legend(loc="lower left", fontsize=7, framealpha=0.9)
    ax_a.set_title(f"(a) {sta} — posterior Vs "
                   f"({syn.shape[0]} models)", fontsize=9, color=_INK)

    # (b) one RF-fit density panel per class
    i0 = 0
    for r, cls in enumerate(classes):
        t = axes_t[cls]
        block = syn[:, i0:i0 + t.size]
        o = obs[i0:i0 + t.size]
        i0 += t.size
        ax = fig.add_subplot(gs[r, 1])
        lo = min(np.percentile(block, 0.5), o.min())
        hi = max(np.percentile(block, 99.5), o.max())
        pad = 0.08 * (hi - lo)
        ybins = np.linspace(lo - pad, hi + pad, 130)
        tedges = np.concatenate([t - (t[1] - t[0]) / 2,
                                 [t[-1] + (t[1] - t[0]) / 2]])
        _column_density(ax, tedges, block, ybins)
        ax.plot(t, o, color=_INK, lw=1.0, label="observed")
        ax.set_xlim(t[0], t[-1])
        ax.tick_params(labelsize=7)
        ax.set_title(f"(b{r + 1}) {cls} RF — posterior-predictive density",
                     fontsize=9, color=_INK, loc="left")
        if r == 0:
            ax.legend(loc="upper right", fontsize=7, framealpha=0.9)
        if r == len(classes) - 1:
            pass
        ax.set_ylabel("amplitude", fontsize=8)
        ax.set_xlabel("Time after P (s)", fontsize=8)

    # (c) dispersion fit density
    ax = fig.add_subplot(gs[len(classes), 1])
    dsyn = syn[:, n_rf:]
    dobs = obs[n_rf:]
    lo = min(dsyn.min(), (dobs - sig_swd).min()) - 0.05
    hi = max(dsyn.max(), (dobs + sig_swd).max()) + 0.05
    ybins = np.linspace(lo, hi, 100)
    # one density column per period, drawn as a narrow band around it
    for j, T in enumerate(periods):
        w = 0.055 * T
        h, _ = np.histogram(dsyn[:, j], bins=ybins)
        dens = (h / max(h.max(), 1))[:, None]
        ax.pcolormesh([T - w, T + w], ybins, dens, cmap=_DENSITY_CMAP,
                      vmin=0, vmax=1, shading="flat", rasterized=True)
    ax.errorbar(periods, dobs, yerr=sig_swd, fmt="o", color=_INK, ms=4,
                lw=1.0, capsize=2.5, label="observed ± σ")
    ax.set_xlim(periods.min() * 0.85, periods.max() * 1.1)
    ax.tick_params(labelsize=7)
    ax.set_xlabel("Period (s)", fontsize=8)
    ax.set_ylabel("Rayleigh phase\nvelocity (km/s)", fontsize=8)
    ax.set_title("(c) dispersion — posterior-predictive density",
                 fontsize=9, color=_INK, loc="left")
    ax.legend(loc="lower right", fontsize=7, framealpha=0.9)

    out = sta_dir / "joint_fit.png"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=str(ROOT / "config.yaml"))
    ap.add_argument("--stations", default=None)
    args = ap.parse_args()
    cfg = load_config(args.config)
    inv_root = Path(paths(cfg)["inversion"])
    stas = sorted(d.name for d in inv_root.iterdir()
                  if (d / "vs_profile.txt").exists())
    if args.stations:
        want = {s.strip().upper() for s in args.stations.split(",")}
        stas = [s for s in stas if s.upper() in want]
    for sta in stas:
        out = station_fit_figure(cfg, sta, inv_root / sta)
        print(f"{sta}: {out or 'skipped'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
