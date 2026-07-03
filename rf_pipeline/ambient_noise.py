"""Stage 5: ambient-noise cross-correlation via amb_noise_tools (Kaestle).

The scientific core — windowed frequency-domain cross-correlation with spectral
whitening — is done by the published, validated ``amb_noise_tools`` ``noise.noisecorr``
(Kaestle et al.; Bensen et al. 2007 procedure), NOT by hand-written code here. This
module is glue only: discover the daily 3C files, preprocess Z (detrend, response
removal, resample), align overlapping time spans (``noise.adapt_timespan``), call
``noise.noisecorr`` per pair per day, stack the resulting cross-correlation spectra,
and save them for the dispersion stage.

Output: ant/ccfs/<STA1>_<STA2>.npz with
  freq          frequency axis of the stacked CC spectrum
  corr_spectrum complex stacked cross-correlation spectrum (Z-Z, Rayleigh)
  n_days        number of daily spectra stacked
  dist_km       inter-station distance
  lag_s, ccf    time-domain CCF (via noise.freq_to_time_domain) for the F8 gather
"""
from __future__ import annotations

from datetime import date
from itertools import combinations
from pathlib import Path

import numpy as np

from . import io_utils
from .logging_setup import get_logger

LOG = get_logger("rf.ambient_noise")


_RESP_WARNED: set[str] = set()


def _preprocess_z(tr, sr, inv, resp_out, pre_filt):
    """Detrend, (optionally) remove response, resample. Whitening/one-bit are left
    to noise.noisecorr, so we do NOT normalize the spectrum here."""
    tr = tr.copy()
    tr.detrend("demean"); tr.detrend("linear"); tr.taper(0.02)
    if inv is not None:
        try:
            tr.remove_response(inventory=inv, output=resp_out, pre_filt=pre_filt,
                               water_level=60)
        except Exception as e:
            # An uncorrected instrument PHASE response biases inter-station
            # phase velocities — never swallow this silently.
            if tr.id not in _RESP_WARNED:
                _RESP_WARNED.add(tr.id)
                LOG.warning(f"response removal failed for {tr.id} ({e}); "
                            f"trace used WITHOUT response correction.")
    if abs(tr.stats.sampling_rate - sr) > 1e-6:
        tr.resample(sr)
    return tr


def run(cfg: dict) -> Path:
    noise = io_utils.import_amb_noise_tools(cfg)   # validated CC/dispersion core

    ant = cfg.get("ant", {})
    sr = float(cfg.get("data", {}).get("sampling_rate", 100.0))
    target_sr = float(ant.get("target_sampling_rate", min(sr, 20.0)))
    cc_len = float(ant.get("cc_len", 3600))
    cc_step = float(ant.get("cc_step", cc_len / 2))
    overlap = float(ant.get("overlap", max(0.0, min(0.95, 1.0 - cc_step / cc_len))))
    onebit = str(ant.get("time_norm", "")).lower() in ("one_bit", "onebit")
    whiten = str(ant.get("freq_norm", "rma")).lower() not in ("", "none", "false")
    water_level = float(ant.get("water_level", 60))
    resp_out = cfg.get("data", {}).get("response_output", "VEL")
    pre_filt = cfg.get("data", {}).get("pre_filt", [0.05, 0.1, 40, 45])

    stations, inv = io_utils.load_stations(cfg)
    sta_lookup = io_utils.station_lookup(stations)
    p = io_utils.paths(cfg)
    out_dir = io_utils.ensure_dir(p["ccfs"])

    src = cfg.get("data", {}).get("source_waveform_dir")
    scan_dir = io_utils.resolve_path(src, cfg["_project_root"]) if src else p["continuous"]
    if not Path(scan_dir).exists():
        scan_dir = p["continuous"]
    files = io_utils.discover_waveforms(scan_dir)
    by_day: dict[date, list] = {}
    for wf in files:
        by_day.setdefault(wf.date, []).append(wf)
    if not by_day:
        LOG.warning(f"No continuous data under {scan_dir} — nothing to correlate.")
        return out_dir

    # per pair: window-count-weighted sum of daily CC spectra + freq axis + counts
    spec_sum: dict[tuple[str, str], np.ndarray] = {}
    freq_axis: dict[tuple[str, str], np.ndarray] = {}
    ndays: dict[tuple[str, str], int] = {}
    nwins_sum: dict[tuple[str, str], int] = {}

    for day in sorted(by_day):
        zt = {}
        for wf in by_day[day]:
            sta = sta_lookup.get(wf.station)
            try:
                st = io_utils.read_day_3c(wf, sta)
            except Exception:
                continue
            zsel = st.select(component="Z")
            if len(zsel):
                zt[wf.station] = _preprocess_z(zsel[0], target_sr, inv, resp_out, pre_filt)
        avail = sorted(zt)
        for a, b in combinations(avail, 2):
            key = (a, b)
            try:
                # noisecorr takes two Traces and cuts them to the common time
                # range internally (calls adapt_timespan), so pass Z directly.
                freq, spectrum, nwins = noise.noisecorr(
                    zt[a], zt[b], window_length=cc_len, overlap=overlap,
                    onebit=onebit, whiten=whiten, water_level=water_level)
            except Exception as e:
                LOG.debug(f"[{day}] {a}-{b}: noisecorr failed ({e})")
                continue
            nw = max(1, int(nwins))
            if key not in spec_sum:
                spec_sum[key] = spectrum.astype(complex) * nw
                freq_axis[key] = freq
                ndays[key] = 1; nwins_sum[key] = nw
            elif spec_sum[key].shape == spectrum.shape:
                # weight each day by its surviving window count -> true
                # all-window average, not day-mean-of-means
                spec_sum[key] += spectrum * nw
                ndays[key] += 1; nwins_sum[key] += nw
            else:
                # never RESET an accumulated stack on a stray malformed day
                LOG.warning(f"[{day}] {a}-{b}: spectrum shape {spectrum.shape} != "
                            f"stack {spec_sum[key].shape} — day skipped.")
                continue
        LOG.info(f"[{day}] correlated {len(avail)} stations "
                 f"({len(list(combinations(avail, 2)))} pairs)")

    for key, ssum in spec_sum.items():
        a, b = key
        spectrum = ssum / max(1, nwins_sum[key])      # window-weighted mean CC spectrum
        freq = freq_axis[key]
        sa, sb = sta_lookup.get(a), sta_lookup.get(b)
        dist = _dist_km(sa, sb) if sa and sb else np.nan
        lag, ccf = noise.freq_to_time_domain(spectrum, freq)   # for the F8 gather
        np.savez(out_dir / f"{a}_{b}.npz",
                 freq=freq, corr_spectrum=spectrum, n_days=ndays[key],
                 dist_km=dist, lag_s=lag, ccf=np.real(ccf))
    LOG.info(f"Stage 5 (ANT via amb_noise_tools): {len(spec_sum)} pair CCFs -> {out_dir}")
    return out_dir


def _dist_km(sa, sb) -> float:
    from obspy.geodetics import gps2dist_azimuth
    d, _, _ = gps2dist_azimuth(sa.latitude, sa.longitude, sb.latitude, sb.longitude)
    return d / 1000.0
