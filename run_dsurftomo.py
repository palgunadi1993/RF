#!/usr/bin/env python3
"""Stage 7-alt - DSurfTomo direct 3-D Vs inversion. Reads all params from config.yaml."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from rf_pipeline import dsurftomo, progress  # noqa: E402
from rf_pipeline.io_utils import load_config  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description="Stage 7-alt - DSurfTomo 3-D inversion")
    ap.add_argument("--config", default=str(ROOT / "config.yaml"))
    args = ap.parse_args()
    progress.run_stage(load_config(args.config), "dsurftomo", dsurftomo.run)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
