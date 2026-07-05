#!/usr/bin/env python3
"""Stage 2 - receiver functions entrypoint. Reads all parameters from config.yaml."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from rf_pipeline import progress, receiver_functions  # noqa: E402
from rf_pipeline.io_utils import load_config  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description="Stage 2 - receiver functions")
    ap.add_argument("--config", default=str(ROOT / "config.yaml"))
    args = ap.parse_args()
    progress.run_stage(load_config(args.config), "rf", receiver_functions.run)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
