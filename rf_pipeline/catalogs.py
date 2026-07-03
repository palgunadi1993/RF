"""Stage 1b: assemble the three event catalogs (teleseismic / regional / local-deep).

Each class is either supplied as a local QuakeML (``catalogs.<class>.file``, used
verbatim if it exists) or fetched from an FDSN provider (``catalogs.<class>.fetch``)
over the deployment time window, relative to the network centroid. The three
source classes are the whole point of the RF design (PLAN.md Stage 2): local-deep
+ regional are the workhorses for 5 Hz nodes, teleseismic a cross-check.
"""
from __future__ import annotations

from datetime import date
from pathlib import Path

from . import io_utils
from .logging_setup import get_logger

LOG = get_logger("rf.catalogs")

# Map friendly provider names to ObsPy FDSN base URLs / keys.
_PROVIDERS = {"USGS": "USGS", "ISC": "ISC", "IRIS": "IRIS", "EMSC": "EMSC"}


def _centroid(stations) -> tuple[float, float]:
    lats = [s.latitude for s in stations]
    lons = [s.longitude for s in stations]
    return sum(lats) / len(lats), sum(lons) / len(lons)


def _deployment_window(cfg: dict) -> tuple[date, date]:
    """Deployment window from ``data.date_range`` or, failing that, the data itself."""
    dr = cfg.get("data", {}).get("date_range") or cfg.get("input", {}).get("date_range")
    if dr and dr[0] and dr[1]:
        return io_utils.parse_iso_date(dr[0]), io_utils.parse_iso_date(dr[1])
    p = io_utils.paths(cfg)
    src = _source_waveform_dir(cfg, p)
    files = io_utils.discover_waveforms(src)
    if not files:
        raise ValueError(
            "Cannot determine the deployment window: set data.date_range in the "
            "config, or make the continuous data discoverable."
        )
    days = sorted(wf.date for wf in files)
    return days[0], days[-1]


def _source_waveform_dir(cfg: dict, p: dict) -> Path:
    src = cfg.get("data", {}).get("source_waveform_dir")
    if src:
        return io_utils.resolve_path(src, cfg["_project_root"])
    return p.get("continuous", p["root"])


def _fetch_class(name: str, spec: dict, window, centroid, out_path: Path):
    """Fetch one class from its FDSN provider and write QuakeML."""
    from obspy import UTCDateTime
    from obspy.clients.fdsn import Client

    fetch = spec.get("fetch", {})
    provider = _PROVIDERS.get(str(fetch.get("provider", "USGS")).upper(), "USGS")
    lat, lon = centroid
    start = UTCDateTime(window[0].isoformat())
    end = UTCDateTime(window[1].isoformat()) + 86400  # inclusive last day

    kwargs = dict(starttime=start, endtime=end, latitude=lat, longitude=lon,
                  minmagnitude=float(fetch.get("min_magnitude", 0.0)),
                  orderby="time")
    if "min_distance_deg" in fetch:
        kwargs["minradius"] = float(fetch["min_distance_deg"])
    if "max_distance_deg" in fetch:
        kwargs["maxradius"] = float(fetch["max_distance_deg"])
    if "min_depth_km" in fetch:
        kwargs["mindepth"] = float(fetch["min_depth_km"])
    if "max_depth_km" in fetch:
        kwargs["maxdepth"] = float(fetch["max_depth_km"])

    LOG.info(f"[{name}] querying {provider}: {kwargs}")
    client = Client(provider)
    cat = client.get_events(**kwargs)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cat.write(str(out_path), format="QUAKEML")
    LOG.info(f"[{name}] {len(cat)} events -> {out_path}")
    return cat


def build_catalog(cfg: dict, name: str, spec: dict, window, centroid) -> Path | None:
    """Resolve one class to a QuakeML path (local file preferred, else fetch)."""
    if not spec.get("enabled", True):
        LOG.info(f"[{name}] disabled — skipping")
        return None
    p = io_utils.paths(cfg)
    catalogs_dir = p.get("catalogs", p["root"] / "data" / "catalogs")
    local = spec.get("file")
    out_path = io_utils.resolve_path(local, cfg["_project_root"]) if local \
        else catalogs_dir / f"{name}.xml"

    if out_path.exists():
        LOG.info(f"[{name}] using existing catalog {out_path}")
        return out_path
    try:
        _fetch_class(name, spec, window, centroid, out_path)
        return out_path
    except Exception as e:  # network/provider failure should not kill the stage
        LOG.warning(f"[{name}] fetch failed ({e}); provide {out_path} manually.")
        return None


def run(cfg: dict) -> Path:
    """Assemble all enabled catalogs; return the catalogs directory."""
    stations, _ = io_utils.load_stations(cfg)
    centroid = _centroid(stations)
    window = _deployment_window(cfg)
    LOG.info(f"Deployment window {window[0]}..{window[1]}  centroid={centroid}")

    p = io_utils.paths(cfg)
    catalogs_dir = io_utils.ensure_dir(p.get("catalogs", p["root"] / "data" / "catalogs"))
    produced = {}
    for name, spec in (cfg.get("catalogs", {}) or {}).items():
        path = build_catalog(cfg, name, spec, window, centroid)
        if path is not None:
            produced[name] = path
    LOG.info(f"Catalogs ready: {sorted(produced)}")
    return catalogs_dir


def load_class_catalog(cfg: dict, name: str):
    """Load one class's catalog as an ObsPy Catalog (or None if unavailable)."""
    spec = (cfg.get("catalogs", {}) or {}).get(name, {})
    p = io_utils.paths(cfg)
    catalogs_dir = p.get("catalogs", p["root"] / "data" / "catalogs")
    local = spec.get("file")
    path = io_utils.resolve_path(local, cfg["_project_root"]) if local \
        else catalogs_dir / f"{name}.xml"
    if not Path(path).exists():
        LOG.warning(f"[{name}] catalog not found at {path}")
        return None
    return io_utils.load_catalog(path)
