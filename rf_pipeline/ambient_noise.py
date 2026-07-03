"""Stage 5: ambient-noise cross-correlation (empirical Green's functions).

Vertical-vertical cross-correlation of the continuous 3C data for every station
pair, to recover the Rayleigh-wave EGF (PLAN.md Stage 5). This is a transparent,
self-contained implementation of the standard Bensen et al. (2007) workflow
(preprocess -> temporal normalization -> spectral whitening -> segmented
cross-correlation -> daily then full stack). NoisePy is a drop-in alternative for
this stage; the outputs (ant/ccfs) and downstream dispersion code are identical.

Output: ant/ccfs/<STA1>_<STA2>.npz  (lag_s, ccf, n_stack, dist_km).
"""
from __future__ import annotations

from datetime import date
from itertools import combinations
from pathlib import Path

import numpy as np

from . import io_utils
from .logging_setup import get_logger

LOG = get_logger("rf.ambient_noise")


def _preprocess_trace(tr, sr, freqmin, freqmax, time_norm, inv, resp_out, pre_filt):
    tr = tr.copy()
    tr.detrend("demean"); tr.detrend("linear"); tr.taper(0.02)
    if inv is not None:
        try:
            tr.remove_response(inventory=inv, output=resp_out, pre_filt=pre_filt,
                               water_level=60)
        except Exception:
            pass
    if abs(tr.stats.sampling_rate - sr) > 1e-6:
        tr.resample(sr)
    tr.filter("bandpass", freqmin=freqmin, freqmax=freqmax, corners=4, zerophase=True)
    data = tr.data.astype(float)
    if time_norm in ("one_bit", "onebit"):
        data = np.sign(data)
    elif time_norm in ("rma", "running_mean"):
        w = max(1, int(sr / freqmin / 2))
        env = np.convolve(np.abs(data), np.ones(w) / w, mode="same")
        data = np.divide(data, env, out=np.zeros_like(data), where=env > 0)
    tr.data = data
    return tr


def _whiten(spec, freq_norm):
    if freq_norm in ("rma", "whiten", "phase"):
        amp = np.abs(spec)
        amp[amp == 0] = 1.0
        return spec / amp
    return spec


def _xcorr_pair(za, zb, sr, maxlag, freq_norm):
    """Whitened cross-correlation of two equal-length vertical windows."""
    n = min(za.size, zb.size)
    if n < 2:
        return None
    nfft = 2 ** int(np.ceil(np.log2(2 * n)))
    fa = _whiten(np.fft.rfft(za[:n], nfft), freq_norm)
    fb = _whiten(np.fft.rfft(zb[:n], nfft), freq_norm)
    cc = np.fft.irfft(fa * np.conj(fb), nfft)
    cc = np.fft.fftshift(cc)
    mid = nfft // 2
    maxsamp = int(maxlag * sr)
    return cc[mid - maxsamp: mid + maxsamp + 1]


def _windows(data, wlen, wstep):
    for start in range(0, max(1, data.size - wlen + 1), wstep):
        seg = data[start:start + wlen]
        if seg.size == wlen:
            yield seg


def run(cfg: dict) -> Path:
    ant = cfg.get("ant", {})
    sr = float(cfg.get("data", {}).get("sampling_rate", 100.0))
    target_sr = float(ant.get("target_sampling_rate", min(sr, 20.0)))
    cc_len = float(ant.get("cc_len", 3600)); cc_step = float(ant.get("cc_step", 1800))
    maxlag = float(ant.get("maxlag", 100))
    freqmin = float(ant.get("freqmin", 0.1)); freqmax = float(ant.get("freqmax", 5.0))
    freq_norm = ant.get("freq_norm", "rma"); time_norm = ant.get("time_norm", "one_bit")
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

    wlen = int(cc_len * target_sr); wstep = int(cc_step * target_sr)
    pair_stack: dict[tuple[str, str], np.ndarray] = {}
    pair_n: dict[tuple[str, str], int] = {}

    for day in sorted(by_day):
        zt: dict[str, np.ndarray] = {}
        for wf in by_day[day]:
            sta = sta_lookup.get(wf.station)
            try:
                st = io_utils.read_day_3c(wf, sta)
            except Exception:
                continue
            zsel = st.select(component="Z")
            if len(zsel) == 0:
                continue
            tr = _preprocess_trace(zsel[0], target_sr, freqmin, freqmax, time_norm,
                                   inv, resp_out, pre_filt)
            zt[wf.station] = tr.data
        avail = sorted(zt)
        for a, b in combinations(avail, 2):
            key = tuple(sorted((a, b)))
            acc = None; nn = 0
            for wa, wb in zip(_windows(zt[a], wlen, wstep), _windows(zt[b], wlen, wstep)):
                cc = _xcorr_pair(wa, wb, target_sr, maxlag, freq_norm)
                if cc is None:
                    continue
                acc = cc if acc is None else acc + cc
                nn += 1
            if acc is None:
                continue
            if key in pair_stack:
                pair_stack[key] += acc; pair_n[key] += nn
            else:
                pair_stack[key] = acc; pair_n[key] = nn
        LOG.info(f"[{day}] correlated {len(avail)} stations "
                 f"({len(list(combinations(avail, 2)))} pairs)")

    lag = np.arange(-int(maxlag * target_sr), int(maxlag * target_sr) + 1) / target_sr
    for key, acc in pair_stack.items():
        a, b = key
        sa, sb = sta_lookup.get(a), sta_lookup.get(b)
        dist = _dist_km(sa, sb) if sa and sb else np.nan
        np.savez(out_dir / f"{a}_{b}.npz", lag_s=lag, ccf=acc / max(1, pair_n[key]),
                 n_stack=pair_n[key], dist_km=dist)
    LOG.info(f"Stage 5 (ANT): {len(pair_stack)} pair CCFs -> {out_dir}")
    return out_dir


def _dist_km(sa, sb) -> float:
    from obspy.geodetics import gps2dist_azimuth
    d, _, _ = gps2dist_azimuth(sa.latitude, sa.longitude, sb.latitude, sb.longitude)
    return d / 1000.0
