#!/usr/bin/env python3
"""Stage 8 - joint RF+dispersion inversion entrypoint. Reads all parameters from config.yaml."""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

# BayHunter's chain processes are FORKED from this process and inherit its
# BLAS thread pools, so this must be set before numpy is first imported:
# with multi-threaded BLAS, workers x chains x blas-threads oversubscribes
# the machine (parallel.executor does the same for the spawn-pool stages).
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
           "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_v, "1")

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from rf_pipeline import inversion, progress  # noqa: E402
from rf_pipeline.io_utils import load_config  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description="Stage 8 - joint RF+dispersion inversion")
    ap.add_argument("--config", default=str(ROOT / "config.yaml"))
    args = ap.parse_args()
    progress.run_stage(load_config(args.config), "inversion", inversion.run)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
