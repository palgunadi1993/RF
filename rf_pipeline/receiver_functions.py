"""Stage 2: receiver functions for all three source classes, using ``rf``.

One code path, looped over ``rf.classes``. ``rf.defaults`` apply to every class;
each class block overrides only what differs (rotation, Gaussian width, slowness
range, travel-time model). That is how local-deep, regional and teleseismic are
all driven from the one YAML (PLAN.md Stage 2).

For each (station, class) we:
  1. cut a wide 3C window around the theoretical P arrival for every catalog event,
  2. remove instrument response and attach per-event ray geometry (``rf.rfstats``),
  3. rotate + iterative time-domain deconvolution -> radial/transverse RF (``rf()``),
  4. move-out correct to a reference slowness,
  5. QC by SNR, then linear-stack the radial RF used downstream by the inversion.

Outputs per station/class:
  rf_out/<station>_<class>.h5           individual QC-passed RFs (obspyh5)
  rf_out/<station>_<class>_stack.sac     linear-stacked radial RF (inversion input)
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from . import catalogs, io_utils
from .logging_setup import get_logger

LOG = get_logger("rf.receiver_functions")


def _require_rf():
    try:
        import rf  # noqa: F401
        from rf import RFStream, rfstats  # noqa: F401
        return rf
    except Exception as e:  # pragma: no cover - depends on optional install
        raise ImportError(
            "Stage 2 needs the `rf` package (`pip install rf obspyh5`). "
            f"Import failed: {e}"
        )


# Config rotation strings -> rf's rotate keyword.
_ROTATE_MAP = {
    "ZNE->RT": "NE->RT",
    "NE->RT": "NE->RT",
    "ZNE->LQT": "ZNE->LQT",
    "LQT": "ZNE->LQT",
    "RT": "NE->RT",
}

# Radial-equivalent component letter after each rotation.
_RADIAL_COMP = {"NE->RT": "R", "ZNE->LQT": "Q"}

# Config deconvolution value -> rf method name. The paper (and PLAN.md Stage 2)
# use *iterative* time-domain deconvolution; rf's 'time' method is the damped
# variant, so config 'time' maps to rf 'iterative'.
_DECONV_MAP = {
    "time": "iterative", "iterative": "iterative",
    "freqattr": "waterlevel", "waterlevel": "waterlevel", "freq": "waterlevel",
    "multitaper": "multitaper",
}

_KM_PER_DEG = 111.19492664455873   # great-circle km per degree (for slowness units)


def _deconv_kwargs(method: str, gauss: float, dp: dict) -> dict:
    """Kwargs for rf's deconvolution, matched to each method's real signature.

    Verified against rf.deconvolve: deconv_iterative(tshift, gauss, itmax,
    minderr, ...); deconv_waterlevel(waterlevel, gauss, ...).
    """
    if method == "iterative":
        return {"gauss": gauss,
                "itmax": int(dp.get("max_iters", 400)),
                "minderr": float(dp.get("min_deltaE", dp.get("min_deltae", 0.001)))}
    if method == "waterlevel":
        return {"gauss": gauss, "waterlevel": float(dp.get("waterlevel", 0.05))}
    return {"gauss": gauss}


def _class_params(cfg: dict, name: str) -> dict:
    """Merge ``rf.defaults`` with the class override block."""
    rf_cfg = cfg.get("rf", {})
    params = dict(rf_cfg.get("defaults", {}) or {})
    params.update(rf_cfg.get("classes", {}).get(name, {}) or {})
    return params


def _dist_range(params: dict, name: str) -> tuple[float, float]:
    """Epicentral distance range (deg) used by rfstats to accept/reject events."""
    spec = params  # class params already merged
    # explicit override wins
    if "dist_range_deg" in spec:
        return tuple(spec["dist_range_deg"])
    # sensible defaults per class name
    return {
        "teleseismic": (30.0, 90.0),
        "regional": (2.0, 30.0),
        "local_deep": (0.0, 5.0),
    }.get(name, (0.0, 180.0))


def _resolve_tt_model(cfg, params, name) -> str:
    """Return the travel-time model name/path rfstats should use for this class.

    PLAN.md Stage 2: regional/local-deep ray parameters must come from a *local*
    1-D model, not the global default. If ``local_model`` (a ``.nd`` file) is set
    for the class it is compiled once with obspy.taup into a cached ``.npz`` and
    its path is handed to rfstats; otherwise the class ``tt_model`` (default
    ``iasp91``) is used.
    """
    nd = params.get("local_model")
    if not nd:
        return params.get("tt_model", "iasp91")
    nd_path = io_utils.resolve_path(nd, cfg["_project_root"])
    if not nd_path.exists():
        LOG.warning(f"[{name}] local_model {nd_path} missing — using "
                    f"{params.get('tt_model', 'iasp91')} instead.")
        return params.get("tt_model", "iasp91")
    try:
        from obspy.taup.taup_create import build_taup_model
        cache = io_utils.ensure_dir(io_utils.paths(cfg)["root"] / "data" / "taup")
        npz = cache / (nd_path.stem + ".npz")
        if not npz.exists():
            build_taup_model(str(nd_path), output_folder=str(cache), verbose=False)
        LOG.info(f"[{name}] using local travel-time model {npz.name}")
        return str(npz)
    except Exception as e:
        LOG.warning(f"[{name}] could not build local model ({e}); using "
                    f"{params.get('tt_model', 'iasp91')}.")
        return params.get("tt_model", "iasp91")


def _snr(tr, onset, signal=(-2.0, 8.0), noise=(-25.0, -5.0)) -> float:
    """RMS SNR of an RF trace: signal window vs pre-onset noise, about ``onset``."""
    from obspy import UTCDateTime

    t0 = UTCDateTime(onset)
    sig = tr.slice(t0 + signal[0], t0 + signal[1]).data
    noi = tr.slice(t0 + noise[0], t0 + noise[1]).data
    if sig.size == 0 or noi.size == 0:
        return 0.0
    nrms = float(np.sqrt(np.mean(noi ** 2)))
    if nrms == 0:
        return np.inf
    return float(np.sqrt(np.mean(sig ** 2)) / nrms)


def compute_class(cfg, name, stations, inv, index, out_dir) -> dict[str, Path]:
    """Compute + stack RFs for one source class across all stations."""
    rf = _require_rf()
    from rf import RFStream, rfstats

    params = _class_params(cfg, name)
    cat = catalogs.load_class_catalog(cfg, name)
    if cat is None or len(cat) == 0:
        LOG.warning(f"[{name}] no catalog events — skipping class.")
        return {}

    rotate = _ROTATE_MAP.get(params.get("rotate", "ZNE->RT"), "NE->RT")
    radial = _RADIAL_COMP[rotate]
    gauss = float(params.get("gauss", 1.0))
    deconv = _DECONV_MAP.get(str(params.get("deconvolve", "time")).lower(), "iterative")
    dp = params.get("deconv_params", {}) or {}
    tt_model = _resolve_tt_model(cfg, params, name)
    dist_range = _dist_range(params, name)
    slow_range = params.get("slowness_range")
    win = params.get("window", [-10, 35])
    snr_min = float(params.get("snr_min", 1.5))
    resp_out = cfg.get("data", {}).get("response_output", "VEL")
    pre_filt = cfg.get("data", {}).get("pre_filt", [0.05, 0.1, 40, 45])

    produced: dict[str, Path] = {}
    for sta in stations:
        stream = RFStream()
        n_events = 0
        for ev in cat:
            try:
                origin = ev.preferred_origin() or ev.origins[0]
            except (IndexError, AttributeError):
                continue
            coords = {"latitude": sta.latitude, "longitude": sta.longitude,
                      "elevation": sta.elevation_m}
            try:
                stats = rfstats(station=coords, event=ev, phase=params.get("phase", "P"),
                                dist_range=dist_range, tt_model=tt_model)
            except Exception:
                stats = None
            if stats is None:
                continue
            if slow_range and not (slow_range[0] <= stats.slowness <= slow_range[1]):
                continue
            # cut a wide 3C window around the theoretical onset; rf trims later.
            st = io_utils.read_event_window_3c(stats.onset, sta, index,
                                               pre_s=60.0, post_s=120.0)
            if len(st) < 3:
                continue
            if inv is not None:
                try:
                    st.remove_response(inventory=inv, output=resp_out,
                                       pre_filt=pre_filt, water_level=60)
                except Exception as e:
                    LOG.debug(f"[{name}] {sta.code} response removal failed: {e}")
            for tr in st:
                tr.stats.update(stats)
            stream.extend(st)
            n_events += 1

        if len(stream) == 0:
            LOG.info(f"[{name}] {sta.code}: no usable events.")
            continue

        # rotate + deconvolve -> RFs. The paper's f1/f2 are a bandpass applied
        # before deconvolution (rf's `filter` kwarg); the iterative time-domain
        # deconvolution takes gauss/itmax/minderr (verified against rf source:
        # rf.deconvolve.deconv_iterative). See rf.RFStream.rf docstring.
        bandpass = {"type": "bandpass",
                    "freqmin": float(dp.get("f1", 0.03)),
                    "freqmax": float(dp.get("f2", 20.0)),
                    "corners": 2, "zerophase": True}
        deconv_kwargs = _deconv_kwargs(deconv, gauss, dp)
        phase = params.get("phase", "P")
        stream.rf(method=phase, rotate=rotate, filter=bandpass,
                  deconvolve=deconv, **deconv_kwargs)
        # rf.moveout ref is slowness in s/deg; config value is s/km -> convert.
        ref_sdeg = float(params.get("moveout_ref_slowness", 0.06)) * _KM_PER_DEG
        stream.moveout(ref=ref_sdeg)
        stream.trim2(win[0], win[1], reftime="onset")

        # QC on the radial RF by SNR
        rad = stream.select(component=radial)
        kept = RFStream()
        for tr in rad:
            if _snr(tr, tr.stats.onset) >= snr_min:
                kept.append(tr)
        if len(kept) == 0:
            LOG.info(f"[{name}] {sta.code}: {len(rad)} RFs, none passed SNR>= {snr_min}.")
            continue

        # persist individual RFs + linear stack of the radial component
        h5 = out_dir / f"{sta.code}_{name}.h5"
        try:
            kept.write(str(h5), "H5")
        except Exception as e:
            LOG.warning(f"[{name}] {sta.code}: could not write H5 ({e}).")
        stack = kept.copy().stack()
        sac = out_dir / f"{sta.code}_{name}_stack.sac"
        stack.write(str(sac), "SAC")
        produced[sta.code] = sac
        LOG.info(f"[{name}] {sta.code}: {n_events} events -> {len(kept)} RFs, "
                 f"stacked -> {sac.name}")
    return produced


def run(cfg: dict) -> Path:
    _require_rf()
    stations, inv = io_utils.load_stations(cfg)
    p = io_utils.paths(cfg)
    out_dir = io_utils.ensure_dir(p["rf_out"])

    # index continuous data once for fast per-event windowing
    src = cfg.get("data", {}).get("source_waveform_dir")
    scan_dir = io_utils.resolve_path(src, cfg["_project_root"]) if src else p["continuous"]
    if not Path(scan_dir).exists():
        scan_dir = p["continuous"]
    index = io_utils.index_by_station_date(io_utils.discover_waveforms(scan_dir))
    LOG.info(f"Indexed {len(index)} station-days of continuous data from {scan_dir}.")

    classes = cfg.get("rf", {}).get("classes", {}) or {}
    for name, class_cfg in classes.items():
        # honour the per-class catalog toggle
        if not (cfg.get("catalogs", {}).get(name, {}).get("enabled", True)):
            LOG.info(f"[{name}] catalog disabled — skipping RF class.")
            continue
        compute_class(cfg, name, stations, inv, index, out_dir)

    LOG.info("Stage 2 (receiver functions) complete.")
    return out_dir
