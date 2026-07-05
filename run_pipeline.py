#!/usr/bin/env python3
"""End-to-end orchestrator for the Dieng RF + ANT joint-inversion pipeline.

Runs all stages in order, or a selected subset. Every stage takes only the master
config (config.yaml). The earthquake side (rf, hk, ccp) and the noise side (ant,
dispersion, tomo) are independent and converge at the inversion (PLAN.md).

    python run_pipeline.py --config config.yaml
    python run_pipeline.py --stages rf,hk,ccp
    python run_pipeline.py --stages ant,dispersion,tomo
    python run_pipeline.py --status              # print where you're standing, run nothing
    python run_pipeline.py --resume              # skip stages already marked done
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from rf_pipeline import (  # noqa: E402
    ambient_noise, ccp, data_prep, dispersion, dsurftomo, hk_stacking, inversion,
    progress, receiver_functions, synthesis, tomography,
)
from rf_pipeline.io_utils import load_config  # noqa: E402
from rf_pipeline.logging_setup import get_logger  # noqa: E402

# stage key -> (module, human label). Order defines default run order.
STAGES = {
    "prep": (data_prep, "Stage 1 - data prep + catalogs"),
    "rf": (receiver_functions, "Stage 2 - receiver functions"),
    "hk": (hk_stacking, "Stage 3 - H-kappa stacking"),
    "ccp": (ccp, "Stage 4 - CCP imaging"),
    "ant": (ambient_noise, "Stage 5 - ambient-noise cross-correlation"),
    "dispersion": (dispersion, "Stage 6 - dispersion measurement"),
    "tomo": (tomography, "Stage 7 - per-station dispersion curves"),
    "dsurftomo": (dsurftomo, "Stage 7-alt - DSurfTomo 3-D ANT inversion (opt-in)"),
    "inversion": (inversion, "Stage 8 - joint inversion"),
    "synthesis": (synthesis, "Stage 9 - synthesis & figures"),
}


def main() -> int:
    ap = argparse.ArgumentParser(description="Dieng RF + ANT pipeline")
    ap.add_argument("--config", default=str(ROOT / "config.yaml"))
    ap.add_argument("--stages", default=",".join(STAGES),
                    help=f"Comma-separated subset of: {','.join(STAGES)}")
    ap.add_argument("--status", action="store_true",
                    help="Print the current per-stage standing and exit (runs nothing).")
    ap.add_argument("--resume", action="store_true",
                    help="Skip stages already marked 'done' in logs/progress.json.")
    args = ap.parse_args()

    log = get_logger("orchestrator", log_file=ROOT / "logs" / "pipeline.log")
    cfg = load_config(args.config)

    selected = [s.strip() for s in args.stages.split(",") if s.strip()]
    invalid = [s for s in selected if s not in STAGES]
    if invalid:
        log.error(f"Unknown stage(s): {invalid}. Valid: {list(STAGES)}")
        return 2

    # --status: just show where things stand, in canonical order, and exit.
    if args.status:
        progress.print_status(cfg, selected=[k for k in STAGES if k in selected])
        return 0

    tracker = progress.ProgressTracker.for_config(cfg)
    run_order = [k for k in STAGES if k in selected]   # canonical order
    if args.resume:
        skipped = [k for k in run_order if tracker.is_done(k)]
        for k in skipped:
            log.info(f"--resume: {STAGES[k][1]} already done — skipping.")
        run_order = [k for k in run_order if k not in skipped]

    total = len(run_order)
    log.info(f"Running {total} stage(s): {run_order}")
    t_start = time.time()
    for i, key in enumerate(run_order, 1):
        module, _ = STAGES[key]
        elapsed = time.time() - t_start
        if i > 1:
            eta = elapsed / (i - 1) * (total - (i - 1))
            log.info(f"progress: {i - 1}/{total} stages done, "
                     f"{progress._fmt_dur(elapsed)} elapsed, ~{progress._fmt_dur(eta)} left")
        progress.run_stage(cfg, key, module.run, position=i, total=total)

    log.info(f"Pipeline complete in {progress._fmt_dur(time.time() - t_start)}.")
    progress.print_status(cfg, selected=run_order)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
