"""Stage 6: Rayleigh dispersion measurement from the noise CCFs (FTAN).

Frequency-time analysis (Bensen et al. 2007) of each station-pair EGF to extract
Rayleigh **group** and **phase** velocity vs period (PLAN.md Stage 6). Group
velocity is measured from the envelope peak of narrow Gaussian-filtered signals;
phase velocity from the filtered phase with the 2*pi*N ambiguity resolved against a
reference velocity. Path-quality gating uses the min-wavelength and SNR criteria.

``amb_noise_tools`` is a drop-in alternative for this stage; it also performs the
correlation, so it can replace Stage 5 too if preferred.

Output: ant/disp/<STA1>_<STA2>.disp  (period phase_vel group_vel snr).
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
from scipy.signal import hilbert

from . import io_utils
from .logging_setup import get_logger

LOG = get_logger("rf.dispersion")


def _symmetric(lag, ccf):
    """Fold the CCF into its symmetric (causal+acausal average) component."""
    mid = lag.size // 2
    pos = ccf[mid:]
    neg = ccf[:mid + 1][::-1]
    n = min(pos.size, neg.size)
    sym = 0.5 * (pos[:n] + neg[:n])
    t = lag[mid:mid + n]
    return t, sym


def _gaussian_filter(sym, dt, fc, alpha):
    """Narrow Gaussian bandpass about ``fc`` (FTAN kernel)."""
    n = sym.size
    freqs = np.fft.rfftfreq(n, dt)
    spec = np.fft.rfft(sym)
    g = np.exp(-alpha * ((freqs - fc) / fc) ** 2)
    return np.fft.irfft(spec * g, n)


def ftan(t, sym, dist, periods, alpha=20.0, ref_vel=3.0):
    """Return arrays (period, group_vel, phase_vel, snr) via FTAN."""
    dt = t[1] - t[0]
    out = []
    analytic_full = None
    for T in periods:
        fc = 1.0 / T
        band = _gaussian_filter(sym, dt, fc, alpha)
        env = np.abs(hilbert(band))
        # restrict to a physical velocity window (1.5 - 5 km/s)
        with np.errstate(divide="ignore"):
            vel = np.where(t > 0, dist / t, np.inf)
        win = (vel >= 1.5) & (vel <= 5.0)
        if not np.any(win):
            out.append((T, np.nan, np.nan, 0.0)); continue
        idx_win = np.where(win)[0]
        ipk = idx_win[np.argmax(env[idx_win])]
        tg = t[ipk]
        u = dist / tg if tg > 0 else np.nan                    # group velocity
        # SNR: envelope peak vs late-coda noise rms
        noise = env[t > dist / 1.5] if np.any(t > dist / 1.5) else env[-env.size // 4:]
        nrms = np.sqrt(np.mean(noise ** 2)) if noise.size else 0.0
        snr = float(env[ipk] / nrms) if nrms > 0 else np.inf
        # phase velocity: phase of the analytic signal at the group arrival
        phase = np.angle(hilbert(band))[ipk]
        c = _phase_velocity(dist, tg, fc, phase, ref_vel)
        out.append((T, u, c, snr))
    arr = np.array(out)
    return arr[:, 0], arr[:, 1], arr[:, 2], arr[:, 3]


def _phase_velocity(dist, tg, fc, phase, ref_vel):
    """Resolve the 2*pi*N branch by picking the phase velocity nearest ref_vel."""
    # travel time from phase: t_phase = tg - (phase + 2*pi*N + pi/4) / (2*pi*fc)
    best_c, best_d = np.nan, np.inf
    for N in range(-5, 6):
        tphase = tg - (phase + 2 * np.pi * N + np.pi / 4) / (2 * np.pi * fc)
        if tphase <= 0:
            continue
        c = dist / tphase
        if 1.5 <= c <= 5.0 and abs(c - ref_vel) < best_d:
            best_c, best_d = c, abs(c - ref_vel)
    return best_c


def run(cfg: dict) -> Path:
    disp = cfg.get("dispersion", {})
    periods = np.array(disp.get("periods", [0.5, 1, 2, 3, 4, 5, 6, 8]), dtype=float)
    min_wl = float(disp.get("min_wavelengths", 2))
    snr_min = float(disp.get("snr_min", 5.0))
    ref_vel = float(disp.get("ref_velocity", 3.0))
    alpha = float(disp.get("ftan_alpha", 20.0))

    p = io_utils.paths(cfg)
    ccf_dir = p["ccfs"]
    out_dir = io_utils.ensure_dir(p["disp"])
    ccfs = sorted(Path(ccf_dir).glob("*.npz"))
    if not ccfs:
        LOG.warning(f"No CCFs under {ccf_dir} — run Stage 5 first.")
        return out_dir

    n_written = 0
    for f in ccfs:
        d = np.load(f)
        dist = float(d["dist_km"]) if "dist_km" in d else np.nan
        if not np.isfinite(dist) or dist <= 0:
            continue
        t, sym = _symmetric(d["lag_s"], d["ccf"])
        T, u, c, snr = ftan(t, sym, dist, periods, alpha=alpha, ref_vel=ref_vel)
        # gate: SNR and min-wavelength (dist >= N * wavelength = N * c * T)
        rows = []
        for i in range(T.size):
            wl_ok = np.isfinite(c[i]) and dist >= min_wl * c[i] * T[i]
            if snr[i] >= snr_min and wl_ok:
                rows.append((T[i], c[i], u[i], snr[i]))
        if not rows:
            continue
        out = out_dir / f"{f.stem}.disp"
        np.savetxt(out, np.array(rows), fmt="%.4f",
                   header="period_s phase_vel group_vel snr")
        n_written += 1
    LOG.info(f"Stage 6 (dispersion): {n_written}/{len(ccfs)} pair curves -> {out_dir}")
    return out_dir
