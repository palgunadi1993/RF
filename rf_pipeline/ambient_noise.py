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

from . import io_utils, parallel
from .logging_setup import get_logger

LOG = get_logger("rf.ambient_noise")


_RESP_WARNED: set[str] = set()


def _preprocess_z(tr, sr, inv, resp_out, pre_filt, ram_win=0.0):
    """Detrend, (optionally) remove response, resample, and (optionally) apply
    running-absolute-mean time normalization. Whitening is left to noise.noisecorr,
    so we do NOT normalize the spectrum here."""
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
    if ram_win and ram_win > 0:
        # Running-absolute-mean time normalization (Bensen et al. 2007): divide by
        # the smoothed amplitude envelope so earthquakes/tremor bursts stop dominating
        # the correlation windows. Without it, teleseisms crossing the small aperture
        # as near-vertical plane waves stamp a zero-lag lobe on EVERY pair CCF that
        # out-amplitudes the interstation surface wave (no "V" in the gather).
        n = int(round(ram_win * tr.stats.sampling_rate))
        if n >= 2:
            w = np.convolve(np.abs(tr.data), np.ones(n) / n, mode="same")
            pos = w[w > 0]
            if pos.size:
                # floor keeps zero-filled gaps from exploding; data there is 0 anyway
                w = np.maximum(w, 0.01 * float(np.mean(pos)))
                tr.data = tr.data / w
    return tr


def _day_task(cfg: dict, day, wfs, sta_lookup, inv, prm: dict):
    """Correlate all station pairs of ONE day: the parallel work unit.

    Days are independent (each reads only its own files); the parent reduces
    the returned per-pair daily spectra into the running stacks in day order.
    Returns ``(n_stations, {(a, b): (freq, spectrum, nwins)})``.
    """
    noise = io_utils.import_amb_noise_tools(cfg)

    zt = {}
    for wf in wfs:
        sta = sta_lookup.get(wf.station)
        try:
            st = io_utils.read_day_3c(wf, sta)
        except Exception:
            continue
        zsel = st.select(component="Z")
        if len(zsel):
            ztr = zsel[0]
            del st, zsel  # drop the two unused horizontals before the resp-removal FFT
            zt[wf.station] = _preprocess_z(ztr, prm["target_sr"], inv,
                                           prm["resp_out"], prm["pre_filt"],
                                           prm.get("ram_window", 0.0))
    out = {}
    for a, b in combinations(sorted(zt), 2):
        try:
            # noisecorr takes two Traces and cuts them to the common time
            # range internally (calls adapt_timespan), so pass Z directly.
            freq, spectrum, nwins = noise.noisecorr(
                zt[a], zt[b], window_length=prm["cc_len"], overlap=prm["overlap"],
                onebit=prm["onebit"], whiten=prm["whiten"],
                water_level=prm["water_level"])
        except Exception as e:
            LOG.debug(f"[{day}] {a}-{b}: noisecorr failed ({e})")
            continue
        out[(a, b)] = (freq, spectrum, max(1, int(nwins)))
    return len(zt), out


def run(cfg: dict) -> Path:
    io_utils.import_amb_noise_tools(cfg)   # fail fast if the CC core is missing

    ant = cfg.get("ant", {})
    sr = float(cfg.get("data", {}).get("sampling_rate", 100.0))
    target_sr = float(ant.get("target_sampling_rate", min(sr, 20.0)))
    cc_len = float(ant.get("cc_len", 3600))
    cc_step = float(ant.get("cc_step", cc_len / 2))
    overlap = float(ant.get("overlap", max(0.0, min(0.95, 1.0 - cc_step / cc_len))))
    time_norm = str(ant.get("time_norm", "")).lower()
    onebit = time_norm in ("one_bit", "onebit")
    # 'ram' = running-absolute-mean normalization (Bensen et al. 2007), done in our
    # own _preprocess_z (the upstream one-bit is flagged "not well implemented")
    ram_window = float(ant.get("ram_window_s", 20.0)) if time_norm in ("ram", "running_mean") else 0.0
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

    def _fold(day, n_sta, day_spectra):
        """Reduce one day's spectra into the running per-pair stacks."""
        for key, (freq, spectrum, nw) in day_spectra.items():
            a, b = key
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
        LOG.info(f"[{day}] correlated {n_sta} stations ({len(day_spectra)} pairs)")

    prm = {"target_sr": target_sr, "resp_out": resp_out, "pre_filt": pre_filt,
           "cc_len": cc_len, "overlap": overlap, "onebit": onebit,
           "whiten": whiten, "water_level": water_level, "ram_window": ram_window}
    days = sorted(by_day)
    # An ANT day-worker holds a full day of many-station 3C at the native rate
    # plus a response-removal FFT spike; measured peak is ~4 GB, so tell the RAM
    # cap that rather than let it assume the generic 2 GB and oversubscribe.
    mem_gb = float(ant.get("mem_per_worker_gb", 4.5))
    n_jobs = parallel.resolve_n_jobs(cfg, n_tasks=len(days), mem_per_task_gb=mem_gb)
    if n_jobs <= 1:
        for day in days:
            n_sta, day_spectra = _day_task(cfg, day, by_day[day], sta_lookup, inv, prm)
            _fold(day, n_sta, day_spectra)
    else:
        # Fold in day order, releasing each result as it is consumed, so the
        # whole deployment's daily spectra are never all held in memory.
        LOG.info(f"ANT day correlation: {len(days)} days on {n_jobs} processes")
        failed: list = []
        with parallel.executor(n_jobs) as ex:
            futures = [ex.submit(_day_task, cfg, day, by_day[day],
                                 sta_lookup, inv, prm) for day in days]
            for i, day in enumerate(days):
                try:
                    n_sta, day_spectra = futures[i].result()
                except Exception as e:
                    # A worker OOM/crash poisons the ENTIRE ProcessPoolExecutor
                    # (BrokenProcessPool), so every still-pending day fails here
                    # too. Don't discard them — collect and retry serially below.
                    LOG.warning(f"[{day}] correlation failed in worker ({e!r}) "
                                f"— queued for serial retry")
                    failed.append(day)
                    continue
                finally:
                    futures[i] = None
                _fold(day, n_sta, day_spectra)

        # Re-run the poisoned days in-process, one at a time: no concurrent
        # memory pressure, so a pool that died on a single OOM still produces a
        # complete stack instead of silently dropping the tail of the deployment.
        if failed:
            LOG.info(f"ANT: retrying {len(failed)} day(s) serially after pool failure")
            for day in failed:
                try:
                    n_sta, day_spectra = _day_task(cfg, day, by_day[day],
                                                   sta_lookup, inv, prm)
                except Exception as e:
                    LOG.warning(f"[{day}] serial retry also failed ({e!r})")
                    continue
                _fold(day, n_sta, day_spectra)

    noise = io_utils.import_amb_noise_tools(cfg)
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
