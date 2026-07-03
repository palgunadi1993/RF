"""I/O helpers: config, paths, station metadata, 3C waveform discovery/reading.

The waveform-reading layer intentionally mirrors ``repeater_pipeline/pipeline/
io_utils.py`` so the two Dieng pipelines share one convention for the SmartSolo
3-component daily files (``[NET.]STA.YYYY-MM-DD.mseed`` — one 3C file per station
per day, all of Z/N/E in a single file). See PLAN.md Stage 1.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Iterable

import pandas as pd
import yaml


# ===========================================================================
# config
# ===========================================================================

def load_config(path: str | Path) -> dict:
    """Load the master YAML and stamp resolved config path + project root.

    The project root is taken from ``project.root`` if present (the RF config
    sets it explicitly); otherwise it falls back to the config file's directory.
    """
    with open(path, "r") as f:
        cfg = yaml.safe_load(f)
    cfg_path = Path(path).resolve()
    cfg["_config_path"] = str(cfg_path)
    root = cfg.get("project", {}).get("root")
    cfg["_project_root"] = str(Path(root).expanduser().resolve()) if root else str(cfg_path.parent)
    return cfg


def resolve_path(p: str | Path, project_root: str | Path) -> Path:
    """Expand ``~`` and resolve relative paths against the project root."""
    p = Path(str(p)).expanduser()
    if not p.is_absolute():
        p = Path(project_root) / p
    return p


def paths(cfg: dict) -> dict[str, Path]:
    """Resolve every entry of ``project.paths`` against the project root.

    Returns a dict of absolute :class:`~pathlib.Path` keyed by the logical name
    (``continuous``, ``stationxml``, ``rf_out`` ...). Missing keys are simply
    absent from the returned dict.
    """
    root = cfg["_project_root"]
    out = {"root": Path(root)}
    for name, rel in (cfg.get("project", {}).get("paths", {}) or {}).items():
        out[name] = resolve_path(rel, root)
    return out


def ensure_dir(p: str | Path) -> Path:
    p = Path(p)
    p.mkdir(parents=True, exist_ok=True)
    return p


# ===========================================================================
# station metadata
# ===========================================================================

@dataclass(frozen=True)
class Station:
    network: str
    code: str            # e.g. "ST09"
    latitude: float
    longitude: float
    elevation_m: float

    @property
    def full_id(self) -> str:
        return f"{self.network}.{self.code}"


def parse_station_file(path: str | Path) -> list[Station]:
    """Parse a whitespace-separated Dieng ``station.txt``.

        NET.STA.   lat   lon   elev_m   [unused] [unused]

    Tolerant of CRLF, comments (``#``) and a trailing dot on ``NET.STA.``.
    """
    stations: list[Station] = []
    with open(path, "r", newline="") as f:
        for raw in f:
            line = raw.strip()
            if not line or line.startswith("#"):
                continue
            parts = [p for p in line.split() if p]
            if len(parts) < 4:
                continue
            netsta = parts[0].rstrip(".")
            net, sta = netsta.split(".", 1) if "." in netsta else ("", netsta)
            try:
                lat, lon, elev = float(parts[1]), float(parts[2]), float(parts[3])
            except ValueError:
                continue
            stations.append(Station(net, sta, lat, lon, elev))
    return stations


def stations_from_inventory(inv) -> list[Station]:
    """Extract one :class:`Station` per station code from an ObsPy Inventory."""
    out: list[Station] = []
    seen: set[str] = set()
    for net in inv:
        for sta in net:
            if sta.code in seen:
                continue
            seen.add(sta.code)
            out.append(Station(net.code, sta.code, float(sta.latitude),
                               float(sta.longitude), float(sta.elevation or 0.0)))
    return out


def load_stations(cfg: dict):
    """Return ``(stations, inventory_or_None)`` from whatever the config points to.

    Preference order: an explicit ``data.station_file``; else the StationXML
    inventory in ``project.paths.stationxml``. Callers that need response info
    should use the returned inventory.
    """
    import obspy

    p = paths(cfg)
    station_file = cfg.get("data", {}).get("station_file")
    inv = None
    # Prefer a StationXML already staged under data/stationxml; else fall back to
    # the explicit data.inventory path (so response removal works pre-prep).
    inv_path = _first_inventory(p.get("stationxml"))
    if inv_path is None:
        cfg_inv = cfg.get("data", {}).get("inventory")
        if cfg_inv:
            cand = resolve_path(cfg_inv, cfg["_project_root"])
            inv_path = cand if cand.exists() else None
    if inv_path is not None:
        inv = obspy.read_inventory(str(inv_path))
    if station_file:
        stations = parse_station_file(resolve_path(station_file, cfg["_project_root"]))
    elif inv is not None:
        stations = stations_from_inventory(inv)
    else:
        raise FileNotFoundError(
            "No station metadata found: set data.station_file or place a "
            "StationXML under project.paths.stationxml."
        )
    return stations, inv


def _first_inventory(stationxml_dir: Path | None) -> Path | None:
    if stationxml_dir is None or not Path(stationxml_dir).exists():
        return None
    d = Path(stationxml_dir)
    if d.is_file():
        return d
    xmls = sorted(d.glob("*.xml"))
    return xmls[0] if xmls else None


def station_lookup(stations: Iterable[Station]) -> dict[str, Station]:
    return {s.code: s for s in stations}


def stations_to_df(stations: Iterable[Station]) -> pd.DataFrame:
    return pd.DataFrame([
        {"id": s.full_id, "network": s.network, "station": s.code,
         "latitude": s.latitude, "longitude": s.longitude,
         "elevation": s.elevation_m}
        for s in stations
    ])


# ===========================================================================
# waveform file discovery (mirrors repeater_pipeline)
# ===========================================================================

# Legacy single-file-per-component and 3C combined:
#   ST09_2025.07.27_00.00.00.z.mseed   (legacy Z-only)
#   SAPSI_2026.06.14_00.00.00.mseed    (3-component)
_FNAME_RE = re.compile(
    r"^(?P<station>[A-Z0-9]+)_"
    r"(?P<year>\d{4})\.(?P<month>\d{2})\.(?P<day>\d{2})_"
    r"(?P<hour>\d{2})\.(?P<minute>\d{2})\.(?P<second>\d{2})"
    r"(?:\.[a-z])?\.mseed$",
    re.IGNORECASE,
)

# The Dieng broadband/short-period 3C daily layout:
#   DG.ST01.2025-07-26.mseed  ->  [NET.]STA.YYYY-MM-DD.mseed
_FNAME_RE_DASHDATE = re.compile(
    r"^(?:(?P<network>[A-Z0-9]+)\.)?(?P<station>[A-Z0-9]+)\."
    r"(?P<year>\d{4})-(?P<month>\d{2})-(?P<day>\d{2})"
    r"\.mseed$",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class WaveformFile:
    path: Path
    station: str
    date: date

    @property
    def date_str(self) -> str:
        return self.date.isoformat()


def parse_waveform_filename(path: str | Path) -> WaveformFile | None:
    p = Path(path)
    m = _FNAME_RE.match(p.name) or _FNAME_RE_DASHDATE.match(p.name)
    if not m:
        return None
    d = date(int(m["year"]), int(m["month"]), int(m["day"]))
    return WaveformFile(path=p, station=m["station"].upper(), date=d)


def discover_waveforms(
    waveform_dir: str | Path,
    start: date | None = None,
    end: date | None = None,
) -> list[WaveformFile]:
    """All parseable 3C daily files under ``waveform_dir`` in an optional window."""
    out: list[WaveformFile] = []
    wdir = Path(waveform_dir)
    if not wdir.exists():
        return out
    for p in sorted(wdir.iterdir()):
        if not p.is_file():
            continue
        wf = parse_waveform_filename(p)
        if wf is None:
            continue
        if start is not None and wf.date < start:
            continue
        if end is not None and wf.date > end:
            continue
        out.append(wf)
    return out


def index_by_station_date(files: Iterable[WaveformFile]) -> dict[tuple[str, date], WaveformFile]:
    return {(wf.station, wf.date): wf for wf in files}


def parse_iso_date(s: str | None) -> date | None:
    return None if s is None else datetime.fromisoformat(s).date()


# ===========================================================================
# 3C waveform reading
# ===========================================================================

def read_day_3c(wf: WaveformFile, station: Station | None = None):
    """Read one 3C daily file into an ObsPy Stream, re-stamping net/sta.

    Mirrors ``repeater_pipeline`` ``_load_day_stream`` for a single file: the
    mseed already carries channel codes (DH?/DP?), we only overwrite net/sta so
    unknown stations still get a consistent trace id.
    """
    from obspy import read

    st = read(str(wf.path))
    if station is not None:
        for tr in st:
            tr.stats.network = station.network or tr.stats.network
            tr.stats.station = station.code
    st.merge(method=1, fill_value=0)
    return st


def read_event_window_3c(
    origin_time,
    station: Station,
    index: dict[tuple[str, date], WaveformFile],
    pre_s: float,
    post_s: float,
):
    """Cut a 3C window ``[origin - pre_s, origin + post_s]`` for one station.

    Handles windows that straddle midnight by reading every daily file the
    window intersects and merging. Returns an ObsPy Stream (possibly empty).
    """
    from obspy import Stream, UTCDateTime

    t0 = UTCDateTime(origin_time)
    t1 = t0 - float(pre_s)
    t2 = t0 + float(post_s)

    st = Stream()
    d = t1.date
    last = t2.date
    while d <= last:
        wf = index.get((station.code.upper(), d))
        if wf is not None:
            try:
                st += read_day_3c(wf, station)
            except Exception:
                pass
        d = d + timedelta(days=1)
    if len(st):
        st.merge(method=1, fill_value=0)
        st.trim(t1, t2)
    return st


# ===========================================================================
# catalogs
# ===========================================================================

def load_catalog(path: str | Path):
    """Read a QuakeML catalog (returns an ObsPy Catalog)."""
    import obspy

    return obspy.read_events(str(path))


def catalog_to_df(cat) -> pd.DataFrame:
    """Flatten an ObsPy Catalog to a tidy events DataFrame."""
    rows = []
    for ev in cat:
        try:
            o = ev.preferred_origin() or ev.origins[0]
        except (IndexError, AttributeError):
            continue
        mag = None
        try:
            m = ev.preferred_magnitude() or (ev.magnitudes[0] if ev.magnitudes else None)
            mag = float(m.mag) if m is not None else None
        except (IndexError, AttributeError):
            pass
        rows.append({
            "event_id": str(ev.resource_id),
            "time": pd.Timestamp(str(o.time), tz="UTC"),
            "latitude": float(o.latitude) if o.latitude is not None else None,
            "longitude": float(o.longitude) if o.longitude is not None else None,
            "depth_km": float(o.depth) / 1000.0 if o.depth is not None else None,
            "magnitude": mag,
        })
    return pd.DataFrame(rows)
