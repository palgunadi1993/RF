#!/usr/bin/env python3
"""Stage 8 - joint RF+dispersion inversion entrypoint. Reads all parameters from config.yaml."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from rf_pipeline import inversion  # noqa: E402
from rf_pipeline.io_utils import load_config  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description="Stage 8 - joint RF+dispersion inversion")
    ap.add_argument("--config", default=str(ROOT / "config.yaml"))
    args = ap.parse_args()
    inversion.run(load_config(args.config))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
