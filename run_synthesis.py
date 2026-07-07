#!/usr/bin/env python3
"""Stage 9 - synthesis & figures entrypoint. Reads all parameters from config.yaml."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from rf_pipeline import progress, synthesis  # noqa: E402
from rf_pipeline.io_utils import load_config  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Stage 9 - synthesis & figures",
        epilog="Examples:\n"
               "  python run_synthesis.py                       # render the full figure set\n"
               "  python run_synthesis.py --stages dispersion   # only Stage 6's figure (F13)\n"
               "  python run_synthesis.py --stages dispersion,tomo   # Stages 6 & 7 (F13,F14)\n"
               "  python run_synthesis.py --rf-stations ST12    # RF record section for ST12 (F6)\n"
               "  python run_synthesis.py --rf-stations ST12,ST17  # F6 for two named stations\n"
               "  python run_synthesis.py --list                # show the stage->figure map",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("--config", default=str(ROOT / "config.yaml"))
    ap.add_argument("--stages", default=None,
                    help="Render only the figures mapped to these pipeline stages "
                         "(comma-separated, e.g. dispersion,tomo). Reads existing "
                         "stage outputs from disk — does NOT recompute the science. "
                         "Omit to render the complete set.")
    ap.add_argument("--rf-stations", default=None,
                    help="Plot the F6 RF record section for these station code(s) "
                         "(comma-separated, e.g. ST12 or ST12,ST17) instead of the "
                         "config's representative_stations. Saved as "
                         "F6_rf_record_sections_<codes>. Renders only F6.")
    ap.add_argument("--list", action="store_true",
                    help="Print the stage->figure mapping and exit.")
    args = ap.parse_args()

    if args.list:
        for key, figs in synthesis.STAGE_FIGURES.items():
            if figs:
                print(f"  {key:12s} {', '.join(figs)}")
        return 0

    cfg = load_config(args.config)
    if args.rf_stations:
        stations = [s.strip() for s in args.rf_stations.split(",") if s.strip()]
        cfg["_rf_stations"] = stations
        # Force F6 on so the request is honoured even if its config toggle is off,
        # and render only that figure.
        cfg.setdefault("plot", {}).setdefault("figures", {})["F6_rf_record_sections"] = True
        synthesis._render(cfg, ["F6_rf_record_sections"])
        return 0

    if args.stages:
        keys = [s.strip() for s in args.stages.split(",") if s.strip()]
        unknown = [k for k in keys if k not in synthesis.STAGE_FIGURES]
        if unknown:
            print(f"Unknown stage(s): {unknown}. "
                  f"Valid: {list(synthesis.STAGE_FIGURES)}", file=sys.stderr)
            return 2
        for key in keys:
            synthesis.plot_for_stage(cfg, key)
        return 0

    progress.run_stage(cfg, "synthesis", synthesis.run)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
