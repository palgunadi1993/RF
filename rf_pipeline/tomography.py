"""Stage 7: from pair dispersion to one 1-D curve per station (PLAN.md Stage 7).

This stage produces the per-station dispersion curves the joint inversion (Stage 8)
consumes, by two-station averaging: for each station, average every pair curve that
involves it into one representative dispersion curve.

Full 3-D tomography is handled by the separate **DSurfTomo** stage
(``rf_pipeline.dsurftomo``, ``run_dsurftomo.py``) — it inverts the same pair curves
directly for a 3-D Vs model. When ``tomo.path: A`` (the "also do full tomography"
setting), this stage additionally triggers that DSurfTomo run. There is no FMST path
(Rawlinson's FMST is not publicly distributable); DSurfTomo replaces it.

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


def run(cfg: dict) -> Path:
    """Produce per-station curves (always); if path A, also run DSurfTomo for 3-D.

    The joint inversion needs the two-station per-station curves regardless, so
    they are always written. ``tomo.path: A`` additionally launches the DSurfTomo
    3-D inversion (which replaces the old FMST full-tomography path).
    """
    out_dir = two_station(cfg)
    path = str(cfg.get("tomo", {}).get("path", "B")).upper()
    if path == "A":
        from . import dsurftomo
        LOG.info("tomo.path=A -> full 3-D tomography via DSurfTomo.")
        # honour path A even if dsurftomo.enabled wasn't set explicitly
        cfg.setdefault("dsurftomo", {})
        if not cfg["dsurftomo"].get("enabled"):
            cfg["dsurftomo"] = {**cfg["dsurftomo"], "enabled": True}
        dsurftomo.run(cfg)
    return out_dir
