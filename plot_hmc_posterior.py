#!/usr/bin/env python3
"""Per-station Bayesian posterior plots for the Stage-8 rfsurfhmc inversion.

For every station under inversion/ this renders the full HMC model ensemble
(all chains x all samples) as a row-normalized probability density in
Vs-depth space — the classic "all trials" posterior view — overlaid with the
posterior median, the 16/84% credible bounds (both from vs_profile.txt, i.e.
exactly what Stage 9 consumes) and each chain's nbest-mean model (their spread
is a visual convergence check).

Outputs:
  inversion/<sta>/vs_posterior.png       one figure per station
  figures/hmc_posterior_overview.png     contact sheet of all stations

Usage: python plot_hmc_posterior.py [--config config.yaml] [--stations ST01,ST02]
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
from matplotlib.colors import LinearSegmentedColormap

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from rf_pipeline.inversion import _steps_to_profile  # noqa: E402
from rf_pipeline.io_utils import load_config, paths  # noqa: E402

# single-hue sequential ramp (white surface -> deep blue), density is magnitude
_DENSITY_CMAP = LinearSegmentedColormap.from_list(
    "posterior", ["#ffffff", "#c6d8ef", "#6f9fd8", "#2f6bb0", "#123f73"])
_INK = "#1f2430"          # median line + text
_CHAIN = "#d9772a"        # per-chain mean models (identity: 'chains')


def load_models(sta_dir: Path):
    """(models, chain_means) from the chain HDF5 stores; models in x-space."""
    models, means = [], []
    for h5 in sorted(sta_dir.glob("hmc.*.h5")):
        try:
            with h5py.File(h5, "r") as f:
                keys = [k for k in f.keys() if k.isdigit()]
                models.append(np.array([f[f"{k}/model"][:] for k in keys]))
                if "mean" in f:
                    means.append(f["mean/model"][:])
        except OSError:
            continue
    if not models:
        return None, None
    return np.vstack(models), means


def station_figure(sta: str, sta_dir: Path, vs_lim, depth_max, ax=None):
    models, means = load_models(sta_dir)
    prof_file = sta_dir / "vs_profile.txt"
    if models is None or not prof_file.exists():
        return None
    dep = np.arange(0.0, depth_max + 0.25, 0.25)
    nl = models.shape[1] // 2
    profs = np.array([_steps_to_profile(m[:nl], m[nl:], dep) for m in models])

    # row-normalized 2D density: each depth row becomes the marginal PDF of Vs
    vbins = np.linspace(vs_lim[0], vs_lim[1], 120)
    dens = np.zeros((dep.size, vbins.size - 1))
    for i in range(dep.size):
        h, _ = np.histogram(profs[:, i], bins=vbins)
        dens[i] = h / max(h.max(), 1)

    ref = np.loadtxt(prof_file)
    rdep, med, p16, p84 = ref.T

    standalone = ax is None
    if standalone:
        fig, ax = plt.subplots(figsize=(3.6, 5.2), dpi=200)
    dep_edges = np.concatenate([dep - 0.125, [dep[-1] + 0.125]])
    ax.pcolormesh(vbins, dep_edges, dens, cmap=_DENSITY_CMAP, rasterized=True,
                  vmin=0, vmax=1, shading="flat")
    for m in means:
        ax.plot(_steps_to_profile(m[:nl], m[nl:], dep), dep,
                color=_CHAIN, lw=0.8, alpha=0.6, zorder=3)
    ax.plot(med, rdep, color=_INK, lw=1.6, zorder=4, label="median")
    ax.plot(p16, rdep, color=_INK, lw=0.9, ls="--", zorder=4, label="16/84%")
    ax.plot(p84, rdep, color=_INK, lw=0.9, ls="--", zorder=4)
    ax.plot([], [], color=_CHAIN, lw=1.2, label="chain means")

    ax.set_xlim(vs_lim)
    ax.set_ylim(depth_max, 0)
    ax.grid(color="0.85", lw=0.4, zorder=1)
    ax.set_axisbelow(False)
    ax.tick_params(labelsize=8)
    ax.set_title(f"{sta}  ({len(profs)} models, {len(means)} chains)",
                 fontsize=9, color=_INK)
    if standalone:
        ax.set_xlabel("$V_S$ (km/s)", fontsize=9)
        ax.set_ylabel("Depth (km)", fontsize=9)
        ax.legend(loc="lower left", fontsize=7, framealpha=0.9)
        fig.tight_layout()
        out = sta_dir / "vs_posterior.png"
        fig.savefig(out)
        plt.close(fig)
        return out
    return ax


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=str(ROOT / "config.yaml"))
    ap.add_argument("--stations", default=None,
                    help="comma-separated subset (default: all with output)")
    args = ap.parse_args()
    cfg = load_config(args.config)
    p = paths(cfg)
    inv_root = p["inversion"]
    vs_lim = cfg.get("plot", {}).get("vs_clip", [0.5, 4.8])
    depth_max = float(cfg.get("inversion", {}).get("depth_max", 60))

    stas = sorted(d.name for d in Path(inv_root).iterdir()
                  if (d / "vs_profile.txt").exists())
    if args.stations:
        want = {s.strip().upper() for s in args.stations.split(",")}
        stas = [s for s in stas if s.upper() in want]
    done = []
    for sta in stas:
        out = station_figure(sta, Path(inv_root) / sta, vs_lim, depth_max)
        if out:
            done.append(sta)
            print(f"{sta}: {out}")

    # contact sheet
    if len(done) > 1:
        ncol = 5
        nrow = int(np.ceil(len(done) / ncol))
        fig, axes = plt.subplots(nrow, ncol,
                                 figsize=(2.4 * ncol, 3.2 * nrow), dpi=150,
                                 sharex=True, sharey=True)
        for ax in np.ravel(axes):
            ax.set_visible(False)
        for sta, ax in zip(done, np.ravel(axes)):
            ax.set_visible(True)
            station_figure(sta, Path(inv_root) / sta, vs_lim, depth_max, ax=ax)
        for ax in np.ravel(axes)[::ncol]:
            ax.set_ylabel("Depth (km)", fontsize=8)
        for ax in np.ravel(axes)[-ncol:]:
            ax.set_xlabel("$V_S$ (km/s)", fontsize=8)
        handles = [plt.Line2D([], [], color=_INK, lw=1.6, label="median"),
                   plt.Line2D([], [], color=_INK, lw=0.9, ls="--", label="16/84%"),
                   plt.Line2D([], [], color=_CHAIN, lw=1.2, label="chain means")]
        fig.legend(handles=handles, loc="lower right", fontsize=8, ncol=3,
                   frameon=False)
        fig.suptitle("Stage 8 (rfsurfhmc) posterior Vs models — all sampled trials",
                     fontsize=11, color=_INK)
        fig.tight_layout(rect=(0, 0.015, 1, 0.98))
        out = Path(p["figures"]) / "hmc_posterior_overview.png"
        out.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out)
        plt.close(fig)
        print(f"overview: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
