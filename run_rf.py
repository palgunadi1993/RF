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
    ap = argparse.ArgumentParser(
        description="Stage 2 - receiver functions",
        epilog="Examples:\n"
               "  python run_rf.py                        # all stations, all classes\n"
               "  python run_rf.py --stations ST01        # just ST01 (quick test)\n"
               "  python run_rf.py --stations ST01,ST05   # two stations\n"
               "  python run_rf.py --stations ST01 --classes teleseismic  # one station, one class",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("--config", default=str(ROOT / "config.yaml"))
    ap.add_argument("--stations", default=None,
                    help="Comma-separated station codes to process (e.g. ST01 or "
                         "ST01,ST05). Default: every station in the station file.")
    ap.add_argument("--classes", default=None,
                    help="Comma-separated source classes to run (teleseismic,"
                         "regional,local_deep). Default: all enabled classes.")
    args = ap.parse_args()

    cfg = load_config(args.config)
    if args.stations:
        cfg["_only_stations"] = [s.strip() for s in args.stations.split(",") if s.strip()]
    if args.classes:
        cfg["_only_classes"] = [c.strip() for c in args.classes.split(",") if c.strip()]
    progress.run_stage(cfg, "rf", receiver_functions.run)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
