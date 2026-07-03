#!/usr/bin/env python3
"""Stage 5 - ambient-noise cross-correlation entrypoint. Reads all parameters from config.yaml."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from rf_pipeline import ambient_noise  # noqa: E402
from rf_pipeline.io_utils import load_config  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description="Stage 5 - ambient-noise cross-correlation")
    ap.add_argument("--config", default=str(ROOT / "config.yaml"))
    args = ap.parse_args()
    ambient_noise.run(load_config(args.config))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
