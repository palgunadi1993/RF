#!/usr/bin/env python3
"""End-to-end orchestrator for the Dieng RF + ANT joint-inversion pipeline.

Runs all stages in order, or a selected subset. Every stage takes only the master
config (config.yaml). The earthquake side (rf, hk, ccp) and the noise side (ant,
dispersion, tomo) are independent and converge at the inversion (PLAN.md).

    python run_pipeline.py --config config.yaml
    python run_pipeline.py --stages rf,hk,ccp
    python run_pipeline.py --stages ant,dispersion,tomo
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from rf_pipeline import (  # noqa: E402
    ambient_noise, ccp, data_prep, dispersion, dsurftomo, hk_stacking, inversion,
    receiver_functions, synthesis, tomography,
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
    args = ap.parse_args()

    log = get_logger("orchestrator", log_file=ROOT / "logs" / "pipeline.log")
    cfg = load_config(args.config)

    selected = [s.strip() for s in args.stages.split(",") if s.strip()]
    invalid = [s for s in selected if s not in STAGES]
    if invalid:
        log.error(f"Unknown stage(s): {invalid}. Valid: {list(STAGES)}")
        return 2

    log.info(f"Running stages: {selected}")
    for key in STAGES:                       # preserve canonical order
        if key not in selected:
            continue
        module, label = STAGES[key]
        log.info(f"=== {label} ===")
        module.run(cfg)
    log.info("Pipeline complete.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
