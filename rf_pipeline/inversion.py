"""Stage 8: joint inversion of RF + Rayleigh dispersion, per station (BayHunter).

For every station we assemble two (or more) targets describing the same 1-D
column (PLAN.md Stage 8):
  * the stacked radial RF(s) from Stage 2 (with slowness, dt, Gaussian a), and
  * the per-station dispersion curve from Stage 7 (period, phase/group velocity).

BayHunter is transdimensional (it solves for the number of layers) and returns a
posterior Vs(z) with uncertainty. Priors come from the config; the Vp/Vs prior is
informed by the Stage-3 H-kappa result when available. ``rfsurfhmc`` is offered as
the paper-reproduction alternative (same inputs).

Output: inversion/<station>/ — BayHunter storage, posterior models and fits.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from . import io_utils
from .logging_setup import get_logger

LOG = get_logger("rf.inversion")


def _require_bayhunter():
    try:
        from BayHunter import Targets, MCMC_Optimizer  # noqa: F401
        import BayHunter  # noqa: F401
        return BayHunter
    except Exception as e:  # pragma: no cover
        raise ImportError(
            "Stage 8 (engine=bayhunter) needs BayHunter "
            "(git clone https://github.com/jenndrei/BayHunter && pip install -e .). "
            f"Import failed: {e}"
        )


def _read_stacked_rf(sac_path: Path, win_start: float):
    """Return (time_rel_onset, amplitude) for a Stage-2 stacked radial RF.

    Stage 2 trims each RF to the class window anchored at the P onset, so the
    trace's first sample sits at ``win_start`` seconds relative to P.
    """
    import obspy

    tr = obspy.read(str(sac_path))[0]
    n = tr.stats.npts
    t = win_start + np.arange(n) * tr.stats.delta
    return t, tr.data.astype(float)


def _hk_vpvs_prior(cfg, station) -> float | None:
    """Mean kappa for this station from any H-kappa CSV, else None."""
    p = io_utils.paths(cfg)
    hk_dir = p.get("hk_out")
    if hk_dir is None:
        return None
    vals = []
    for csv in Path(hk_dir).glob("hk_*.csv"):
        try:
            df = pd.read_csv(csv)
        except Exception:
            continue
        row = df[df["station"] == station]
        if not row.empty:
            vals.append(float(row["kappa"].iloc[0]))
    return float(np.mean(vals)) if vals else None


def _build_targets(cfg, station, rf_cfg):
    """Assemble the BayHunter JointTarget list for one station (or None)."""
    from BayHunter import Targets

    p = io_utils.paths(cfg)
    inv_cfg = cfg.get("inversion", {})
    targets = []

    # --- RF targets (one per requested source class) ---
    for cls in inv_cfg.get("rf_targets", ["local_deep"]):
        sac = p["rf_out"] / f"{station}_{cls}_stack.sac"
        if not sac.exists():
            continue
        class_params = {**rf_cfg.get("defaults", {}),
                        **rf_cfg.get("classes", {}).get(cls, {})}
        win_start = class_params.get("window", [-10, 35])[0]
        gauss = float(class_params.get("gauss", 1.0))
        # config slowness is s/km (rf/moveout convention); BayHunter's rfmini
        # plugin expects angular slowness in sec/deg (verified: rfmini_modrf.py
        # compute_rf docstring "p: angular slowness in sec/deg").
        slow_skm = float(class_params.get("moveout_ref_slowness",
                         rf_cfg.get("defaults", {}).get("moveout_ref_slowness", 0.06)))
        p_sdeg = slow_skm * 111.19492664455873
        t, y = _read_stacked_rf(sac, win_start)
        rf_target = Targets.PReceiverFunction(t, y)
        try:
            rf_target.moddata.plugin.set_modelparams(
                gauss=gauss, water=0.01, p=p_sdeg)
        except Exception as e:
            LOG.debug(f"{station}/{cls}: set_modelparams failed ({e}); using defaults.")
        targets.append(rf_target)

    # --- dispersion target(s) ---
    disp_file = p["tomo"] / f"{station}_disp.txt"
    if disp_file.exists():
        arr = np.loadtxt(disp_file, ndmin=2)
        if arr.size:
            periods, phase, group = arr[:, 0], arr[:, 1], arr[:, 2]
            swd = inv_cfg.get("swd_targets", ["phase"])
            if "phase" in swd or "both" in swd:
                targets.append(Targets.RayleighDispersionPhase(periods, phase))
            if "group" in swd or "both" in swd:
                targets.append(Targets.RayleighDispersionGroup(periods, group))

    if not targets:
        return None
    return Targets.JointTarget(targets=targets)


def invert_station(cfg, station, rf_cfg, out_root) -> Path | None:
    from BayHunter import MCMC_Optimizer

    joint = _build_targets(cfg, station, rf_cfg)
    if joint is None:
        LOG.info(f"{station}: no RF+SWD inputs — skipped.")
        return None

    inv_cfg = cfg.get("inversion", {})
    pri = inv_cfg.get("priors", {})
    vpvs_range = tuple(pri.get("vpvs", [1.6, 2.0]))
    # Inform the Vp/Vs prior with the Stage-3 H-kappa result when available: keep
    # BayHunter's expected (min, max) form, narrowed around the measured kappa but
    # clipped to the configured bounds (BayHunter samples vpvs within this range).
    hk_vpvs = _hk_vpvs_prior(cfg, station)
    if hk_vpvs is not None:
        lo = max(vpvs_range[0], hk_vpvs - 0.1)
        hi = min(vpvs_range[1], hk_vpvs + 0.1)
        vpvs_prior = (lo, hi) if lo < hi else vpvs_range
    else:
        vpvs_prior = vpvs_range
    priors = {
        "vs": tuple(pri.get("vs", [0.5, 4.8])),
        "z": (0.0, float(inv_cfg.get("depth_max", 60))),
        "layers": tuple(pri.get("n_layers", [3, 12])),
        "vpvs": vpvs_prior,
    }
    mcmc = inv_cfg.get("mcmc", {})
    station_dir = io_utils.ensure_dir(out_root / station)
    initparams = {
        "nchains": int(mcmc.get("chains", 6)),
        "iter_burnin": int(mcmc.get("burnin", 40000)),
        "iter_main": int(mcmc.get("iterations", 100000)) - int(mcmc.get("burnin", 40000)),
        "propdist": (0.015, 0.015, 0.015, 0.005, 0.005),
        "acceptance": (40, 45),
        "thickmin": 0.1,
        "rcond": 1e-5,
        "savepath": str(station_dir),   # BayHunter writes storage/plots here
        "station": station,
    }
    try:
        optimizer = MCMC_Optimizer(joint, initparams=initparams, priors=priors)
        optimizer.mp_inversion(nthreads=initparams["nchains"], baywatch=False, dtsend=1)
    except Exception as e:
        LOG.warning(f"{station}: BayHunter inversion failed ({e}).")
        return None
    LOG.info(f"{station}: inversion complete -> {station_dir}")
    return station_dir


def run(cfg: dict) -> Path:
    engine = str(cfg.get("inversion", {}).get("engine", "bayhunter")).lower()
    p = io_utils.paths(cfg)
    out_root = io_utils.ensure_dir(p["inversion"])
    if engine == "rfsurfhmc":
        LOG.error("engine=rfsurfhmc: build github.com/nqdu/RfSurfHmc and drive it "
                  "from the exported RF stacks + tomo/<sta>_disp.txt. Not wired in "
                  "this build — set inversion.engine=bayhunter to run here.")
        return out_root

    _require_bayhunter()
    stations, _ = io_utils.load_stations(cfg)
    rf_cfg = cfg.get("rf", {})
    only = cfg.get("inversion", {}).get("stations")  # optional subset
    for sta in stations:
        if only and sta.code not in only:
            continue
        invert_station(cfg, sta.code, rf_cfg, out_root)
    LOG.info("Stage 8 (joint inversion) complete.")
    return out_root
