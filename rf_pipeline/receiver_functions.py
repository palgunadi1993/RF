"""Stage 2: receiver functions for all three source classes, using ``rf``.

One code path, looped over ``rf.classes``. ``rf.defaults`` apply to every class;
each class block overrides only what differs (rotation, Gaussian width, slowness
range, travel-time model). That is how local-deep, regional and teleseismic are
all driven from the one YAML (PLAN.md Stage 2).

For each (station, class) we:
  1. cut a wide 3C window around the theoretical P arrival for every catalog event,
  2. remove instrument response and attach per-event ray geometry (``rf.rfstats``),
  3. rotate + deconvolution -> radial/transverse RF (``rf()``),
  4. QC by SNR, write the individual RFs *without* moveout correction (H-kappa and
     CCP need each RF's own per-event slowness and un-distorted multiples),
  5. moveout-correct a copy to a reference slowness and stack it -> inversion input.

Note on deconvolution: Criado-Sutti et al. (2026) used *water-level* deconvolution
(Gaussian a=0.5, water level c=0.1, band 0.01-2 Hz). This pipeline defaults to
iterative time-domain (Ligorria & Ammon 1999) with wider bands tuned for the Dieng
nodes — a deliberate difference, switchable via ``deconvolve: waterlevel``.

Outputs per station/class:
  rf_out/<station>_<class>.h5           individual QC-passed RFs (no moveout)
  rf_out/<station>_<class>_stack.sac     moveout-corrected stacked radial RF
"""
from __future__ import annotations

import warnings
from pathlib import Path

import numpy as np

from . import catalogs, io_utils, parallel
from .logging_setup import get_logger

LOG = get_logger("rf.receiver_functions")

# rf.rfstats looks up the P onset with TauP. In the upper-mantle triplication band
# (~14-24 deg — regional events off the 410/660 discontinuities) TauP returns
# several P branches, and rf warns before taking arrivals[0]. That first arrival is
# the earliest = direct P, which IS the correct RF onset (and its ray parameter),
# so the choice is right; only the per-event warning is noise. Silence just that one
# message (matched on text) so it doesn't flood the log — every other warning still
# shows. The filter is set at import, so it also applies inside the spawn workers,
# which re-import this module. See the regional class notes in config.yaml.
warnings.filterwarnings(
    "ignore",
    message="TauPy returns more than one arrival",
    category=UserWarning,
)


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

# Config deconvolution value -> rf method name. rf's 'time' method is the damped
# variant; config 'time'/'iterative' map to rf 'iterative' (Ligorria & Ammon).
# (The paper itself used water-level deconvolution — see module docstring.)
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


def _is_dead(tr) -> bool:
    """True if a trace can't be deconvolved: empty, non-finite, or flat.

    A flat (all-zero / constant) component — a dead sensor or a zero-filled gap
    from ``read_event_window_3c``'s ``merge(fill_value=0)`` — deconvolves to an
    all-zero RF. ``rf`` then normalizes every component by ``1/max(|parent RF|)``,
    so a dead parent gives ``norm = inf`` and turns the whole event's RFs into
    NaN (the ``divide by zero`` / ``invalid value`` RuntimeWarnings from
    rf.deconvolve). Dropping such events up front avoids the warning, the wasted
    deconvolution, and any chance of a NaN RF reaching the stack.
    """
    d = getattr(tr, "data", None)
    if d is None or d.size == 0:
        return True
    if not np.all(np.isfinite(d)):
        return True
    return float(np.ptp(d)) == 0.0            # exactly flat -> dead/zero-filled


def _snr(tr, onset, signal=(-2.0, 8.0), noise=(-25.0, -5.0)) -> float:
    """RMS SNR of an RF trace: signal window vs pre-onset noise, about ``onset``."""
    from obspy import UTCDateTime

    t0 = UTCDateTime(onset)
    sig = tr.slice(t0 + signal[0], t0 + signal[1]).data
    noi = tr.slice(t0 + noise[0], t0 + noise[1]).data
    if sig.size == 0 or noi.size == 0:
        return 0.0
    # A NaN/Inf RF (e.g. a deconvolution that blew up) must never pass QC; reject
    # it outright rather than let a zero noise-RMS return inf below.
    if not (np.all(np.isfinite(sig)) and np.all(np.isfinite(noi))):
        return 0.0
    nrms = float(np.sqrt(np.mean(noi ** 2)))
    if nrms == 0:
        return np.inf
    return float(np.sqrt(np.mean(sig ** 2)) / nrms)


def _stack_rfs(kept, method: str = "linear", nu: float = 2.0):
    """Stack an RFStream of equal-length radial RFs: linear or phase-weighted.

    Phase-weighted stack (Schimmel & Paulssen 1997): the linear stack is scaled
    sample-by-sample by ``|mean(exp(i*phi_k))|**nu`` where phi_k is each trace's
    instantaneous phase (Hilbert transform). Coherent arrivals keep their
    amplitude; incoherent noise is down-weighted.
    """
    stack = kept.copy().stack()
    if str(method).lower() in ("phase_weighted", "pws"):
        from scipy.signal import hilbert

        nmin = min(tr.data.size for tr in kept)
        arr = np.array([tr.data[:nmin] for tr in kept], dtype=float)
        analytic = hilbert(arr, axis=1)
        mag = np.abs(analytic)
        mag[mag == 0] = 1.0
        coherence = np.abs(np.mean(analytic / mag, axis=0)) ** float(nu)
        stack[0].data = arr.mean(axis=0)[: stack[0].data.size] \
            * coherence[: stack[0].data.size]
    return stack


def _station_task(ctx: dict, sta, cat, inv, index, out_dir: Path):
    """Compute + stack the RFs of one (station, class): the parallel work unit.

    Module-level (and all arguments picklable) so it can run in a spawned
    worker process; every input is read-only and both outputs are per-station
    files, so tasks never contend. Returns
    ``(code, n_events, n_rfs, n_kept, sac_path | None)`` for the parent to log.
    """
    from rf import RFStream, rfstats

    name = ctx["name"]
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
            stats = rfstats(station=coords, event=ev, phase=ctx["phase"],
                            dist_range=ctx["dist_range"], tt_model=ctx["tt_model"])
        except Exception as e:
            LOG.debug(f"[{name}] {sta.code}: rfstats failed for event "
                      f"{getattr(origin, 'time', '?')}: {e}")
            stats = None
        if stats is None:
            continue
        # rfstats stores slowness in s/deg; config slowness_range is s/km.
        p_km = stats.slowness / _KM_PER_DEG
        slow_range = ctx["slow_range"]
        if slow_range and not (slow_range[0] <= p_km <= slow_range[1]):
            continue
        # cut a wide 3C window around the theoretical onset; rf trims later.
        st = io_utils.read_event_window_3c(stats.onset, sta, index,
                                           pre_s=60.0, post_s=120.0)
        if len(st) < 3:
            continue
        if inv is not None:
            try:
                st.remove_response(inventory=inv, output=ctx["resp_out"],
                                   pre_filt=ctx["pre_filt"], water_level=60)
            except Exception as e:
                LOG.debug(f"[{name}] {sta.code} response removal failed: {e}")
        # skip events with a dead/flat component — they would make rf's iterative
        # deconvolution divide by zero and NaN out the whole event (see _is_dead).
        if any(_is_dead(tr) for tr in st):
            LOG.debug(f"[{name}] {sta.code}: dead/flat component at "
                      f"{getattr(origin, 'time', '?')} — skipping event.")
            continue
        for tr in st:
            tr.stats.update(stats)
        stream.extend(st)
        n_events += 1

    if len(stream) == 0:
        return (sta.code, 0, 0, 0, None)

    # rotate + deconvolve -> RFs. f1/f2 are a bandpass applied before
    # deconvolution (rf's `filter` kwarg); the iterative time-domain
    # deconvolution takes gauss/itmax/minderr (verified against rf source:
    # rf.deconvolve.deconv_iterative). See rf.RFStream.rf docstring.
    dp = ctx["dp"]
    bandpass = {"type": "bandpass",
                "freqmin": float(dp.get("f1", 0.03)),
                "freqmax": float(dp.get("f2", 20.0)),
                "corners": 2, "zerophase": True}
    deconv_kwargs = _deconv_kwargs(ctx["deconv"], ctx["gauss"], dp)
    # rf's method must be 'P' or 'S'; the TauP phase may be lowercase
    # (e.g. 'p' for the upgoing leg of deep local events).
    stream.rf(method=ctx["phase"][-1].upper(), rotate=ctx["rotate"],
              filter=bandpass, deconvolve=ctx["deconv"], **deconv_kwargs)

    # QC on the radial RF by SNR *before* trimming, so the configured
    # pre-onset noise window (which may lie outside the final RF window)
    # is still available on the deconvolved trace.
    rad = stream.select(component=ctx["radial"])
    kept = RFStream()
    for tr in rad:
        if _snr(tr, tr.stats.onset, signal=ctx["sig_win"],
                noise=ctx["noi_win"]) >= ctx["snr_min"]:
            kept.append(tr)
    if len(kept) == 0:
        return (sta.code, n_events, len(rad), 0, None)
    win = ctx["win"]
    kept.trim2(win[0], win[1], reftime="onset")

    # persist individual RFs WITHOUT moveout correction: H-kappa and CCP
    # need each trace's own slowness and unshifted multiples (rf.moveout
    # overwrites stats.slowness with the reference and applies the Ps
    # operator, which mis-times PpPs/PpSs).
    h5 = out_dir / f"{sta.code}_{name}.h5"
    try:
        kept.write(str(h5), "H5")
    except Exception as e:
        LOG.warning(f"[{name}] {sta.code}: could not write H5 ({e}).")

    # moveout-correct a copy to the reference slowness, then stack it for
    # the joint inversion (rf.moveout ref is s/deg; config is s/km).
    mo = kept.copy()
    # NOTE: rf.moveout's `model` kwarg wants a SimpleModel depth/vp/vs .dat
    # table, NOT the TauP .npz used by rfstats, so the default (iasp91) is
    # used here; the near-surface timing effect on the stretch is small.
    mo.moveout(ref=ctx["ref_sdeg"])
    stack = _stack_rfs(mo, method=ctx["stack"], nu=ctx["pws_power"])
    sac = out_dir / f"{sta.code}_{name}_stack.sac"
    stack.write(str(sac), "SAC")
    return (sta.code, n_events, len(rad), len(kept), str(sac))


def compute_class(cfg, name, stations, inv, index, out_dir,
                  n_jobs: int = 1) -> dict[str, Path]:
    """Compute + stack RFs for one source class across all stations."""
    _require_rf()

    params = _class_params(cfg, name)
    cat = catalogs.load_class_catalog(cfg, name)
    if cat is None or len(cat) == 0:
        LOG.warning(f"[{name}] no catalog events — skipping class.")
        return {}

    # Everything a worker needs, resolved once in the parent. In particular
    # _resolve_tt_model must run here so the TauP .npz cache is built exactly
    # once, not raced by several fresh workers.
    ctx = {
        "name": name,
        "phase": params.get("phase", "P"),
        "rotate": _ROTATE_MAP.get(params.get("rotate", "ZNE->RT"), "NE->RT"),
        "gauss": float(params.get("gauss", 1.0)),
        "deconv": _DECONV_MAP.get(str(params.get("deconvolve", "time")).lower(),
                                  "iterative"),
        "dp": params.get("deconv_params", {}) or {},
        "tt_model": _resolve_tt_model(cfg, params, name),
        "dist_range": _dist_range(params, name),
        "slow_range": params.get("slowness_range"),
        "win": params.get("window", [-10, 35]),
        "sig_win": tuple(params.get("signal_window", [-2.0, 8.0])),
        "noi_win": tuple(params.get("noise_window", [-25.0, -5.0])),
        "snr_min": float(params.get("snr_min", 1.5)),
        "resp_out": cfg.get("data", {}).get("response_output", "VEL"),
        "pre_filt": cfg.get("data", {}).get("pre_filt", [0.05, 0.1, 40, 45]),
        "ref_sdeg": float(params.get("moveout_ref_slowness", 0.06)) * _KM_PER_DEG,
        "stack": str(params.get("stack", "linear")),
        "pws_power": float(params.get("pws_power", 2.0)),
    }
    ctx["radial"] = _RADIAL_COMP[ctx["rotate"]]

    tasks = [(ctx, sta, cat, inv, index, out_dir) for sta in stations]
    produced: dict[str, Path] = {}
    for res in parallel.pmap(_station_task, tasks, n_jobs, desc=f"RF[{name}]"):
        if res is None:
            continue
        code, n_events, n_rfs, n_kept, sac = res
        if sac is None:
            if n_rfs == 0:
                LOG.info(f"[{name}] {code}: no usable events.")
            else:
                LOG.info(f"[{name}] {code}: {n_rfs} RFs, none passed "
                         f"SNR>= {ctx['snr_min']}.")
            continue
        produced[code] = Path(sac)
        LOG.info(f"[{name}] {code}: {n_events} events -> {n_kept} RFs, "
                 f"stacked ({ctx['stack']}) -> {Path(sac).name}")
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

    n_jobs = parallel.resolve_n_jobs(cfg, n_tasks=len(stations))
    classes = cfg.get("rf", {}).get("classes", {}) or {}
    for name, class_cfg in classes.items():
        # honour the per-class catalog toggle
        if not (cfg.get("catalogs", {}).get(name, {}).get("enabled", True)):
            LOG.info(f"[{name}] catalog disabled — skipping RF class.")
            continue
        compute_class(cfg, name, stations, inv, index, out_dir, n_jobs=n_jobs)

    LOG.info("Stage 2 (receiver functions) complete.")
    return out_dir
