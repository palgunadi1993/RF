"""Stage 6: Rayleigh phase-velocity dispersion via amb_noise_tools (Kaestle).

The dispersion picking — velocity filtering + the zero-crossing / kernel-density
phase-velocity extraction (which properly resolves the 2*pi branch against a
reference curve) — is done by the published ``amb_noise_tools``
(``noise.velocity_filter`` + ``noise.get_smooth_pv``), NOT by hand-written code.
This module is glue: load each pair's stacked CC spectrum from Stage 5, build the
reference curve, call the picker, and resample the picked phase velocity onto the
configured target periods so the joint inversion gets consistent periods per pair.

Output: ant/disp/<STA1>_<STA2>.disp  (period_s phase_vel group_vel sigma).
Group velocity is not produced by get_smooth_pv, so its column is NaN (the joint
inversion uses phase velocity by default).
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from . import io_utils, parallel
from .logging_setup import get_logger

LOG = get_logger("rf.dispersion")

# Default Rayleigh reference dispersion curve for the Dieng crust (period_s, c_kms).
# Only used to guide the branch picking; override with dispersion.ref_curve.
_DEFAULT_REF = [[8, 3.2], [6, 3.0], [5, 2.9], [4, 2.7], [3, 2.5],
                [2, 2.2], [1.5, 2.0], [1, 1.8], [0.75, 1.65], [0.5, 1.5]]


def _ref_curve(cfg) -> np.ndarray:
    """Reference curve as ascending-frequency [[freq, vel], ...] for get_smooth_pv."""
    ref = cfg.get("dispersion", {}).get("ref_curve", _DEFAULT_REF)
    arr = np.array([[1.0 / float(T), float(v)] for T, v in ref], dtype=float)
    return arr[np.argsort(arr[:, 0])]


def _pick_pair(cfg: dict, f: Path, out_dir: Path, prm: dict) -> bool:
    """Pick + resample ONE pair's dispersion curve: the parallel work unit.

    Reads one Stage-5 .npz, writes one .disp — no shared state. Returns True
    if a curve was written.
    """
    noise = io_utils.import_amb_noise_tools(cfg)

    d = np.load(f)
    if "corr_spectrum" not in d or "freq" not in d:
        LOG.warning(f"{f.name}: not an amb_noise_tools CC spectrum — re-run Stage 5.")
        return False
    dist = float(d["dist_km"]) if "dist_km" in d else np.nan
    if not np.isfinite(dist) or dist <= 0:
        return False
    freq, spectrum = d["freq"], d["corr_spectrum"]
    try:
        smoothed = noise.velocity_filter(freq, spectrum, dist,
                                         velband=prm["velband"])
        crossings, phase_vel = noise.get_smooth_pv(
            freq, smoothed, dist, prm["ref"],
            freqmin=prm["freqmin"], freqmax=prm["freqmax"],
            min_vel=prm["min_vel"], max_vel=prm["max_vel"],
            horizontal_polarization=False,
            smooth_spectrum=False, plotting=False)
    except Exception as e:
        LOG.debug(f"{f.stem}: get_smooth_pv failed ({e})")
        return False
    phase_vel = np.asarray(phase_vel)
    if phase_vel.ndim != 2 or phase_vel.shape[0] < 2:
        return False
    pv_freq, pv_c = phase_vel[:, 0], phase_vel[:, 1]
    order = np.argsort(pv_freq)
    pv_freq, pv_c = pv_freq[order], pv_c[order]

    rows = []
    for T in prm["periods"]:
        fq = 1.0 / T
        if fq < pv_freq.min() or fq > pv_freq.max():
            continue
        c = float(np.interp(fq, pv_freq, pv_c))
        if not (prm["min_vel"] <= c <= prm["max_vel"]):
            continue
        if dist < prm["min_wl"] * c * T:                   # min-wavelength gate
            continue
        rows.append((T, c, np.nan, 0.05 * c))              # group=NaN, sigma~5%
    if not rows:
        return False
    np.savetxt(out_dir / f"{f.stem}.disp", np.array(rows), fmt="%.4f",
               header="period_s phase_vel group_vel sigma")
    return True


def run(cfg: dict) -> Path:
    io_utils.import_amb_noise_tools(cfg)   # fail fast if the picker is missing

    disp = cfg.get("dispersion", {})
    periods = np.array(disp.get("periods", [0.5, 1, 2, 3, 4, 5, 6, 8]), dtype=float)
    min_vel = float(disp.get("min_vel", 1.5))
    max_vel = float(disp.get("max_vel", 5.0))
    # outer taper corners must stay positive even for small min_vel
    velband = tuple(disp.get("velband", [max_vel, max_vel - 0.5, min_vel,
                                         max(0.1, min_vel - 0.4)]))
    min_wl = float(disp.get("min_wavelengths", 2))
    ref = _ref_curve(cfg)
    freqmin = float(disp.get("freqmin", 1.0 / periods.max()))
    freqmax = float(disp.get("freqmax", 1.0 / periods.min()))

    p = io_utils.paths(cfg)
    out_dir = io_utils.ensure_dir(p["disp"])
    ccfs = sorted(Path(p["ccfs"]).glob("*.npz"))
    if not ccfs:
        LOG.warning(f"No CCFs under {p['ccfs']} — run Stage 5 first.")
        return out_dir

    # Clear stale .disp from earlier runs: a pair that fails to pick this run must NOT
    # keep an old curve made with different periods/min_vel/ref_curve — that silently
    # contaminates F13 and the inversion with mixed-parameter picks.
    stale = list(Path(out_dir).glob("*.disp"))
    for old in stale:
        old.unlink()
    if stale:
        LOG.info(f"Cleared {len(stale)} stale .disp from a previous run")

    prm = {"periods": periods, "min_vel": min_vel, "max_vel": max_vel,
           "velband": velband, "min_wl": min_wl, "ref": ref,
           "freqmin": freqmin, "freqmax": freqmax}
    n_jobs = parallel.resolve_n_jobs(cfg, n_tasks=len(ccfs))
    tasks = [(cfg, f, out_dir, prm) for f in ccfs]
    results = parallel.pmap(_pick_pair, tasks, n_jobs, desc="dispersion")
    n_written = sum(1 for r in results if r)
    LOG.info(f"Stage 6 (dispersion via amb_noise_tools): {n_written}/{len(ccfs)} "
             f"pair curves -> {out_dir}")
    return out_dir
