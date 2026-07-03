"""Stage 3: H-kappa stacking (Zhu & Kanamori 2000).

Per station and source class, grid-search crustal thickness ``H`` and ``Vp/Vs``
(kappa) by stacking the radial RF amplitudes at the predicted Ps, PpPs and
PpSs+PsPs delay times. Reads the individual RFs written by Stage 2 (each carries
its per-event ray parameter/slowness), so the moveout-consistent multiples line
up at the true (H, kappa).

The stack is a standard, well-defined operation (not a bespoke inversion); it is
implemented here directly to avoid a heavyweight optional dependency, and mirrors
what ``seispy hk`` / ``rf``'s H-kappa routine compute.

Output: hk_out/hk_<class>.csv (station, H, kappa, Vp, bounds) and, per station,
hk_out/<station>_<class>_hk.npz (the full stack grid, for figure F4).
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from . import io_utils
from .logging_setup import get_logger

LOG = get_logger("rf.hk_stacking")


def _read_rfs(path: Path):
    """Load individual RFs from a Stage-2 H5 file (needs rf/obspyh5)."""
    try:
        from rf import read_rf
        return read_rf(str(path))
    except Exception as e:
        raise ImportError(
            f"Reading {path} needs `rf`+`obspyh5` (pip install rf obspyh5): {e}"
        )


def _phase_times(H, k, vp, p):
    """Predicted Ps, PpPs, PpSs+PsPs delay times (s) for thickness H, kappa k.

    ``p`` is the ray parameter (slowness) in s/km; ``vp`` in km/s; ``H`` in km.
    """
    vs = vp / k
    term_s = np.sqrt(np.clip(1.0 / vs**2 - p**2, 0, None))
    term_p = np.sqrt(np.clip(1.0 / vp**2 - p**2, 0, None))
    t_ps = H * (term_s - term_p)
    t_ppps = H * (term_s + term_p)
    t_ppss = H * 2.0 * term_s
    return t_ps, t_ppps, t_ppss


def hk_stack(rfs, vp, h_grid, k_grid, weights, t_ps_min=1.0):
    """Return the (nH, nK) H-kappa stack and (best_H, best_k).

    Cells whose predicted Ps delay falls below ``t_ps_min`` (s) are excluded so
    the direct-P peak at zero lag cannot masquerade as a converted phase — the
    classic degenerate ``H~0`` maximum. This restricts H-kappa to resolvable
    (deeper) interfaces; shallow structure is recovered by the joint inversion.
    """
    w1, w2, w3 = weights
    stack = np.zeros((h_grid.size, k_grid.size), dtype=float)
    valid = np.zeros_like(stack)
    n_used = 0
    HH, KK = np.meshgrid(h_grid, k_grid, indexing="ij")
    for tr in rfs:
        p = float(getattr(tr.stats, "slowness", np.nan))
        onset = getattr(tr.stats, "onset", None)
        if not np.isfinite(p) or onset is None:
            continue
        t0 = tr.stats.onset - tr.stats.starttime  # seconds from trace start to onset
        dt = tr.stats.delta
        data = tr.data
        n = data.size

        def amp(tsec):
            idx = np.round((t0 + tsec) / dt).astype(int)
            idx = np.clip(idx, 0, n - 1)
            return data[idx]

        t_ps, t_ppps, t_ppss = _phase_times(HH, KK, vp, p)
        mask = t_ps >= t_ps_min
        contrib = (w1 * amp(t_ps) + w2 * amp(t_ppps) - w3 * amp(t_ppss)) * mask
        stack += contrib
        valid += mask
        n_used += 1
    if n_used == 0:
        return stack, (np.nan, np.nan), 0
    stack = np.divide(stack, valid, out=np.full_like(stack, -np.inf), where=valid > 0)
    i, j = np.unravel_index(np.argmax(stack), stack.shape)
    return stack, (float(h_grid[i]), float(k_grid[j])), n_used


def _bounds(stack, h_grid, k_grid, frac=0.95):
    """Extent of the >= frac*max contour, as a crude 1-sigma-like bound."""
    mx = np.nanmax(stack)
    if not np.isfinite(mx) or mx <= 0:
        return (np.nan, np.nan, np.nan, np.nan)
    mask = stack >= frac * mx
    hi = h_grid[np.any(mask, axis=1)]
    ki = k_grid[np.any(mask, axis=0)]
    if hi.size == 0 or ki.size == 0:
        return (np.nan, np.nan, np.nan, np.nan)
    return (float(hi.min()), float(hi.max()), float(ki.min()), float(ki.max()))


def run(cfg: dict) -> Path:
    hk = cfg.get("hk", {})
    vp = float(hk.get("vp_crust", 6.0))
    h_range = hk.get("h_range", [0, 70]); h_step = float(hk.get("h_step", 2.0))
    k_range = hk.get("k_range", [1.6, 2.5]); k_step = float(hk.get("k_step", 0.05))
    weights = hk.get("weights", [0.6, 0.3, 0.1])
    run_on = hk.get("run_on", ["teleseismic", "local_deep"])

    h_grid = np.arange(h_range[0], h_range[1] + h_step / 2, h_step)
    k_grid = np.arange(k_range[0], k_range[1] + k_step / 2, k_step)

    if hk.get("depth_correct_k"):
        LOG.warning("hk.depth_correct_k=true: the bottom-up recursive kappa "
                    "correction (paper App. A) is not applied in this build; "
                    "reported kappa is the direct single-layer stack value.")

    p = io_utils.paths(cfg)
    rf_dir = p["rf_out"]
    out_dir = io_utils.ensure_dir(p["hk_out"])

    for name in run_on:
        rows = []
        for h5 in sorted(rf_dir.glob(f"*_{name}.h5")):
            station = h5.name[: -len(f"_{name}.h5")]
            try:
                rfs = _read_rfs(h5)
                rfs = rfs.select(component="R") + rfs.select(component="Q")
            except Exception as e:
                LOG.warning(f"[{name}] {station}: {e}")
                continue
            stack, (bestH, bestK), n_used = hk_stack(rfs, vp, h_grid, k_grid, weights)
            if n_used == 0:
                LOG.info(f"[{name}] {station}: no RFs with slowness — skipped.")
                continue
            h_lo, h_hi, k_lo, k_hi = _bounds(stack, h_grid, k_grid)
            rows.append({"station": station, "H_km": bestH, "kappa": bestK,
                         "vp": vp, "n_rf": n_used,
                         "H_lo": h_lo, "H_hi": h_hi, "k_lo": k_lo, "k_hi": k_hi})
            stack_plot = np.where(np.isfinite(stack), stack, np.nan)
            np.savez(out_dir / f"{station}_{name}_hk.npz",
                     stack=stack_plot, h_grid=h_grid, k_grid=k_grid,
                     bestH=bestH, bestK=bestK)
            LOG.info(f"[{name}] {station}: H={bestH:.1f} km  kappa={bestK:.2f} "
                     f"(n={n_used})")
        if rows:
            csv = out_dir / f"hk_{name}.csv"
            pd.DataFrame(rows).to_csv(csv, index=False)
            LOG.info(f"[{name}] H-kappa summary -> {csv}")
    LOG.info("Stage 3 (H-kappa) complete.")
    return out_dir
