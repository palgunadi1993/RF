#!/usr/bin/env python3
"""Per-station H-kappa stack panels (one figure per station, all classes side by side).

Reads the fine-grid stacks written by Stage 3 (hk_out/<sta>_<class>_hk.npz) and
draws a smooth filled contour of the H-kappa stack for each source class, with the
best (H, Vp/Vs) marked and its 95%-contour bounds. Saved to figures/ as
F4b_hk_<station>.png. Mirrors run_synthesis.py --rf-stations (F6) but for H-kappa.

  python plot_hk_per_station.py                 # every station with a stack
  python plot_hk_per_station.py ST12 ST17       # just these
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from rf_pipeline.io_utils import load_config, paths  # noqa: E402
from rf_pipeline.logging_setup import get_logger      # noqa: E402

LOG = get_logger("rf.hk_per_station")


def _bounds(stack, h_grid, k_grid, frac=0.95):
    """95%-of-max contour extent -> crude 1-sigma-like H/kappa bounds."""
    mx = np.nanmax(stack)
    if not np.isfinite(mx) or mx <= 0:
        return None
    mask = stack >= frac * mx
    hi = h_grid[np.any(mask, axis=1)]; ki = k_grid[np.any(mask, axis=0)]
    if hi.size == 0 or ki.size == 0:
        return None
    return hi.min(), hi.max(), ki.min(), ki.max()


def render(cfg, stations=None):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    p = paths(cfg)
    hk_dir = Path(p["hk_out"]); fig_dir = Path(p["figures"]); fig_dir.mkdir(exist_ok=True)
    classes = list(cfg.get("hk", {}).get("run_on", []))
    k_lo_edge, k_hi_edge = cfg.get("hk", {}).get("k_range", [1.6, 2.5])
    fmts = cfg.get("plot", {}).get("format", ["png"])
    dpi = int(cfg.get("plot", {}).get("dpi", 300))

    all_sta = sorted({f.name.split("_")[0] for f in hk_dir.glob("*_hk.npz")})
    stations = [s.upper() for s in (stations or all_sta)]
    made = []
    for sta in stations:
        npzs = [(c, hk_dir / f"{sta}_{c}_hk.npz") for c in classes]
        npzs = [(c, f) for c, f in npzs if f.exists()]
        if not npzs:
            LOG.warning(f"{sta}: no H-kappa npz — skipping.")
            continue
        fig, axes = plt.subplots(1, len(npzs), figsize=(4.4 * len(npzs), 4.4),
                                 squeeze=False)
        for ax, (cls, f) in zip(axes[0], npzs):
            d = np.load(f)
            stack, H, K = d["stack"], d["h_grid"], d["k_grid"]
            bH, bK = float(d["bestH"]), float(d["bestK"])
            # H on x-axis, k (Vp/Vs) on y-axis: transpose the (nH, nK) stack.
            # Map the stack linearly onto 0..1 (min->0, max->1) so the colour axis
            # is a true 0-1 range like paper Fig. 3.
            mn, mx = np.nanmin(stack), np.nanmax(stack)
            rng = mx - mn
            norm = (stack - mn) / rng if (np.isfinite(rng) and rng > 0) \
                else np.zeros_like(stack)
            im = ax.contourf(H, K, np.nan_to_num(norm, nan=0.0).T,
                             levels=np.linspace(0, 1, 21), cmap="viridis",
                             vmin=0, vmax=1)
            b = _bounds(stack, H, K)
            rails = bK <= k_lo_edge + 0.03 or bK >= k_hi_edge - 0.03
            wide = b is not None and max(bH - b[0], b[1] - bH) > 12.0
            # white crosshair at the maximum (paper Fig. 3 convention)
            ax.axvline(bH, color="w", lw=1.0); ax.axhline(bK, color="w", lw=1.0)
            ax.set_xlabel("H (km)"); ax.set_ylabel("k (Vp/Vs)")
            flag = "  [QC-flagged]" if (rails or wide) else ""
            ax.set_title(f"{sta} {cls}\nH={bH:.1f} km  k={bK:.2f}{flag}",
                         color=("firebrick" if flag else "black"), fontsize=10)
            fig.colorbar(im, ax=ax, label="amplitude", shrink=0.85)
        fig.tight_layout()
        for fmt in fmts:
            fig.savefig(fig_dir / f"F4b_hk_{sta}.{fmt}", dpi=dpi, bbox_inches="tight")
        plt.close(fig)
        made.append(sta)
        LOG.info(f"{sta}: wrote F4b_hk_{sta} ({[c for c, _ in npzs]})")
    LOG.info(f"H-kappa per-station panels: {len(made)} stations -> {fig_dir}")
    return made


if __name__ == "__main__":
    cfg = load_config(str(ROOT / "config.yaml"))
    args = [a for a in sys.argv[1:] if a.strip()]
    render(cfg, args or None)
