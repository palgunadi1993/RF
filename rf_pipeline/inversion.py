"""Stage 8: joint inversion of RF + Rayleigh dispersion, per station (BayHunter).

For every station we assemble two (or more) targets describing the same 1-D
column (PLAN.md Stage 8):
  * the stacked radial RF(s) from Stage 2 (with slowness, dt, Gaussian a), and
  * the per-station dispersion curve from Stage 7 (period, phase/group velocity).

BayHunter is transdimensional (it solves for the number of layers) and returns a
posterior Vs(z) with uncertainty. Priors come from the config; the Vp/Vs prior is
informed by the Stage-3 H-kappa result when available. ``rfsurfhmc`` is offered as
the paper-reproduction alternative (same inputs).

Output: inversion/<station>/ — BayHunter chain storage (data/c*_p2*.npy), the
aggregated posterior median profile ``vs_profile.txt`` (read by figure F11),
and BayHunter's own posterior figures when its plotting succeeds.
"""
from __future__ import annotations

import logging
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from . import io_utils, parallel
from .logging_setup import get_logger

LOG = get_logger("rf.inversion")

# Stage 2 (rf package) and BayHunter's rfmini define the Gaussian width
# differently: rf uses exp(-0.5*(f/f0)**2) with f0 = std dev in Hz
# (rf.deconvolve._gauss_filter), rfmini uses the Ligorria & Ammon convention
# exp(-(pi*f/a)**2). Equating the exponents gives a = pi*sqrt(2)*f0; passing
# the config gauss through unconverted makes the synthetics ~4.4x smoother
# than the data they must fit (verified on the stack spectra).
_GAUSS_RF_TO_RFMINI = math.pi * math.sqrt(2.0)


def _require_bayhunter():
    # BayHunter 2.1 still uses np.float (SingleChain.py), removed in numpy>=1.24;
    # without this shim MCMC_Optimizer construction raises AttributeError on
    # every station. Restore it as the plain builtin alias it used to be.
    if not hasattr(np, "float"):
        np.float = float  # type: ignore[attr-defined]
    try:
        from BayHunter import Targets, MCMC_Optimizer  # noqa: F401
        import BayHunter  # noqa: F401
        return BayHunter
    except Exception as e:  # pragma: no cover
        raise ImportError(
            "Stage 8 (engine=bayhunter) needs BayHunter "
            "(git clone https://github.com/jenndrei/BayHunter && pip install -e .). "
            f"Import failed: {e}"
        )


def _read_stacked_rf(sac_path: Path, win_start: float,
                     resample_hz: float | None = None):
    """Return (time_rel_onset, amplitude) for a Stage-2 stacked radial RF.

    Stage 2 trims each RF to the class window anchored at the P onset, so the
    trace's first sample sits at ``win_start`` seconds relative to P.

    ``resample_hz`` decimates the raw-rate stack (250 Hz for the Dieng nodes)
    before it reaches BayHunter: the likelihood builds an npts x npts noise
    covariance (1 GB and O(n^3) per chain at 250 Hz) and rfmini's forward call
    scales with npts, so full rate makes Stage 8 ~100x slower while the widest
    class gauss (3 Hz corner) leaves no fittable signal above ~8 Hz. Zero-phase
    lowpass then integer decimation; the first sample stays at ``win_start``.
    """
    import obspy

    tr = obspy.read(str(sac_path))[0]
    if resample_hz and tr.stats.sampling_rate > resample_hz * 1.001:
        factor = int(round(tr.stats.sampling_rate / resample_hz))
        tr.filter("lowpass", freq=0.4 * tr.stats.sampling_rate / factor,
                  corners=8, zerophase=True)
        tr.decimate(factor, no_filter=True)
    n = tr.stats.npts
    t = win_start + np.arange(n) * tr.stats.delta
    return t, tr.data.astype(float)


def _hk_vpvs_prior(cfg, station) -> float | None:
    """Mean kappa for this station from any H-kappa CSV, else None."""
    p = io_utils.paths(cfg)
    hk_dir = p.get("hk_out")
    if hk_dir is None:
        return None
    vals = []
    for csv in Path(hk_dir).glob("hk_*.csv"):
        try:
            df = pd.read_csv(csv)
        except Exception:
            continue
        # hk_corrected.csv uses a different schema (kappa_meas_*/kappa_intrinsic_* per
        # class) with no plain 'kappa' column — skip it rather than KeyError.
        if "station" not in df.columns or "kappa" not in df.columns:
            continue
        row = df[df["station"] == station]
        if not row.empty:
            vals.append(float(row["kappa"].iloc[0]))
    return float(np.mean(vals)) if vals else None


def _build_targets(cfg, station, rf_cfg):
    """Assemble the BayHunter JointTarget list for one station (or None)."""
    from BayHunter import Targets

    p = io_utils.paths(cfg)
    inv_cfg = cfg.get("inversion", {})
    targets = []

    # --- RF targets (one per requested source class) ---
    resample_hz = float(inv_cfg.get("rf_resample_hz", 25.0)) or None
    for cls in inv_cfg.get("rf_targets", ["local_deep"]):
        sac = p["rf_out"] / f"{station}_{cls}_stack.sac"
        if not sac.exists():
            continue
        class_params = {**rf_cfg.get("defaults", {}),
                        **rf_cfg.get("classes", {}).get(cls, {})}
        win_start = class_params.get("window", [-10, 35])[0]
        gauss = float(class_params.get("gauss", 1.0)) * _GAUSS_RF_TO_RFMINI
        # config slowness is s/km (rf/moveout convention); BayHunter's rfmini
        # plugin expects angular slowness in sec/deg (verified: rfmini_modrf.py
        # compute_rf docstring "p: angular slowness in sec/deg").
        slow_skm = float(class_params.get("moveout_ref_slowness",
                         rf_cfg.get("defaults", {}).get("moveout_ref_slowness", 0.06)))
        p_sdeg = slow_skm * 111.19492664455873
        t, y = _read_stacked_rf(sac, win_start, resample_hz)
        rf_target = Targets.PReceiverFunction(t, y)
        try:
            rf_target.moddata.plugin.set_modelparams(
                gauss=gauss, water=0.01, p=p_sdeg)
        except Exception as e:
            LOG.debug(f"{station}/{cls}: set_modelparams failed ({e}); using defaults.")
        targets.append(rf_target)

    # --- dispersion target(s) ---
    disp_file = p["tomo"] / f"{station}_disp.txt"
    if disp_file.exists():
        arr = np.loadtxt(disp_file, ndmin=2)
        if arr.size:
            periods, phase, group = arr[:, 0], arr[:, 1], arr[:, 2]
            # col 3 (per-period sigma from the pair scatter) -> BayHunter yerr
            yerr = arr[:, 3] if arr.shape[1] > 3 else None
            swd = inv_cfg.get("swd_targets", ["phase"])
            if "phase" in swd or "both" in swd:
                targets.append(Targets.RayleighDispersionPhase(periods, phase,
                                                               yerr=yerr))
            if "group" in swd or "both" in swd:
                targets.append(Targets.RayleighDispersionGroup(periods, group,
                                                               yerr=yerr))

    if not targets:
        return None
    return Targets.JointTarget(targets=targets)


def invert_station(cfg, station, rf_cfg, out_root,
                   mp_children_max: int | None = None) -> Path | None:
    from BayHunter import MCMC_Optimizer

    joint = _build_targets(cfg, station, rf_cfg)
    if joint is None:
        LOG.info(f"{station}: no RF+SWD inputs — skipped.")
        return None

    inv_cfg = cfg.get("inversion", {})
    pri = inv_cfg.get("priors", {})
    vpvs_range = tuple(pri.get("vpvs", [1.6, 2.0]))
    # Inform the Vp/Vs prior with the Stage-3 H-kappa result when available: keep
    # BayHunter's expected (min, max) form, narrowed around the measured kappa but
    # clipped to the configured bounds (BayHunter samples vpvs within this range).
    hk_vpvs = _hk_vpvs_prior(cfg, station)
    if hk_vpvs is not None:
        lo = max(vpvs_range[0], hk_vpvs - 0.1)
        hi = min(vpvs_range[1], hk_vpvs + 0.1)
        vpvs_prior = (lo, hi) if lo < hi else vpvs_range
    else:
        vpvs_prior = vpvs_range
    priors = {
        "vs": tuple(pri.get("vs", [0.5, 4.8])),
        "z": (0.0, float(inv_cfg.get("depth_max", 60))),
        "layers": tuple(pri.get("n_layers", [3, 12])),
        "vpvs": vpvs_prior,
    }
    # Weighting: fix BayHunter's data-noise sigma to the paper's guide values so the
    # RF-vs-SWD balance matches (likelihood ~ 1/sigma^2). BayHunter's fix mechanism
    # is a SCALAR prior (SingleChain.draw_initnoiseparams checks `type(...) in
    # [int, float]`); a (x, x) tuple keeps the sigma in the proposal pool and every
    # proposal is auto-rejected. Verified keys: rfnoise_sigma / swdnoise_sigma
    # (BayHunter defaults.ini [modelpriors]). Drop misfit_sigma to let it estimate
    # noise hierarchically instead.
    sigma = inv_cfg.get("misfit_sigma") or {}
    if "rf" in sigma:
        priors["rfnoise_sigma"] = float(sigma["rf"])
    if "swd" in sigma:
        priors["swdnoise_sigma"] = float(sigma["swd"])
    mcmc = inv_cfg.get("mcmc", {})
    station_dir = io_utils.ensure_dir(out_root / station)
    initparams = {
        "nchains": int(mcmc.get("chains", 6)),
        "iter_burnin": int(mcmc.get("burnin", 40000)),
        "iter_main": int(mcmc.get("iterations", 100000)) - int(mcmc.get("burnin", 40000)),
        "propdist": (0.015, 0.015, 0.015, 0.005, 0.005),
        "acceptance": (40, 45),
        "thickmin": 0.1,
        "rcond": 1e-5,
        "savepath": str(station_dir),   # BayHunter writes chain .npy storage here
        "station": station,
    }
    ntargets = len(joint.targets)
    LOG.info(f"{station}: starting BayHunter — {initparams['nchains']} chains x "
             f"{initparams['iter_burnin'] + initparams['iter_main']} iterations, "
             f"{ntargets} targets")
    try:
        optimizer = MCMC_Optimizer(joint, initparams=initparams, priors=priors)
        # nthreads is only BayHunter's dispatch throttle: it sleeps while
        # len(mp.active_children()) > nthreads. active_children() is global to
        # this process, so with several stations running concurrently (threads)
        # each mp_inversion must tolerate the WHOLE fleet's children (chains +
        # one Manager per station) or stations starve each other and most
        # chains are never launched.
        optimizer.mp_inversion(
            nthreads=mp_children_max or initparams["nchains"] + 1,
            baywatch=False, dtsend=1)
    except Exception as e:
        LOG.warning(f"{station}: BayHunter inversion failed ({e!r}).", exc_info=True)
        return None
    _export_posterior(cfg, station_dir, station)
    LOG.info(f"{station}: inversion complete -> {station_dir}")
    return station_dir


def _export_posterior(cfg, station_dir: Path, station: str) -> None:
    """Aggregate BayHunter's saved chains into ``vs_profile.txt`` (+ its plots).

    BayHunter only writes per-chain ``data/c*_p2models.npy`` etc.; nothing else
    in the pipeline would otherwise turn those into the median Vs(z) profile
    that Stage 9 (figure F11) reads.

    Order matters: ``save_final_distribution`` runs FIRST because it detects
    outlier chains (median main-phase likelihood deviating >5% from the best
    chain — with 40k iterations some chains do get stuck in local minima) and
    writes the filtered posterior to ``data/c_models.npy``; the Vs profile is
    then computed from that, not from the raw all-chain stack.
    """
    data_dir = station_dir / "data"
    model_files = sorted(data_dir.glob("c*_p2models.npy"))
    if not model_files:
        LOG.warning(f"{station}: no posterior chain files under {data_dir} — "
                    f"vs_profile.txt not written.")
        return
    # BayHunter's outlier filtering + its summary figures. Plots must be
    # headless: this runs inside station worker THREADS (Agg), and BayHunter
    # 2.1 still calls cm.get_cmap, removed in matplotlib 3.9.
    try:
        import matplotlib
        matplotlib.use("Agg", force=True)
        import matplotlib.cm as _cm
        if not hasattr(_cm, "get_cmap"):
            _cm.get_cmap = lambda name=None, lut=None: (
                matplotlib.colormaps[name] if lut is None
                else matplotlib.colormaps[name].resampled(lut))
        from BayHunter import PlotFromStorage

        cfile = next(data_dir.glob("*_config.pkl"))
        plotter = PlotFromStorage(str(cfile))
        plotter.save_final_distribution(maxmodels=100000, dev=0.05)
        plotter.save_plots()
    except Exception as e:
        LOG.warning(f"{station}: BayHunter outlier filter/plots incomplete "
                    f"({e!r}); vs_profile.txt falls back to ALL chains only "
                    f"if data/c_models.npy is missing.")
    try:
        from BayHunter import ModelMatrix

        final = data_dir / "c_models.npy"
        if final.exists():
            models = np.load(final)
            outliers = _read_outliers(data_dir)
            src = (f"{len(models)} models, outlier chains excluded: "
                   f"{outliers if outliers else 'none'}")
        else:
            models = np.vstack([np.load(f) for f in model_files])
            src = f"{len(models)} models from ALL {len(model_files)} chains"
        depth_max = float(cfg.get("inversion", {}).get("depth_max", 60))
        dep = np.arange(0.0, depth_max + 0.25, 0.25)
        vss, _ = ModelMatrix.get_interpmodels(models, dep)
        med = np.median(vss, axis=0)
        p16, p84 = np.percentile(vss, [16, 84], axis=0)
        out = station_dir / "vs_profile.txt"
        np.savetxt(out, np.column_stack([dep, med, p16, p84]), fmt="%10.4f",
                   header="depth_km vs_median_km_s vs_p16 vs_p84")
        LOG.info(f"{station}: posterior median Vs(z) -> {out} ({src})")
    except Exception as e:
        LOG.warning(f"{station}: could not export vs_profile.txt ({e!r}).")


def _read_outliers(data_dir: Path) -> list[int]:
    """Chain indices BayHunter flagged in data/outliers.dat (may be absent)."""
    f = data_dir / "outliers.dat"
    if not f.exists():
        return []
    try:
        arr = np.loadtxt(f, ndmin=2)
        return [int(i) for i in arr[:, 0]] if arr.size else []
    except Exception:
        return []


def _invert_station_task(cfg, station, rf_cfg, out_root_str, mp_children_max):
    """Per-station work unit for pmap: re-apply the np.float shim (a spawned
    worker starts with a fresh numpy) before running the inversion."""
    _require_bayhunter()
    return invert_station(cfg, station, rf_cfg, Path(out_root_str),
                          mp_children_max)


def _ensure_root_log_handler() -> None:
    """Make BayHunter's own progress visible.

    BayHunter logs a chain status line every 5000 iterations — but to the ROOT
    logger (``logging.getLogger()`` in SingleChain/mcmcOptimizer). The pipeline
    only configures named ``rf.*`` loggers (propagate=False), so without a root
    handler Python's last-resort handler silently drops everything below
    WARNING and Stage 8 looks frozen for hours. Chain processes are forked and
    inherit this handler. rf.* loggers don't propagate, so no duplicates.
    """
    root = logging.getLogger()
    if not any(isinstance(h, logging.StreamHandler) for h in root.handlers):
        h = logging.StreamHandler(sys.stdout)
        h.setFormatter(logging.Formatter(
            "%(asctime)s | %(levelname)-7s | %(processName)s | %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S"))
        root.addHandler(h)
    if root.getEffectiveLevel() > logging.INFO:
        root.setLevel(logging.INFO)


# --------------------------------------------------------------------------
# engine: rfsurfhmc  (github.com/nqdu/RfSurfHmc — HMC with analytic gradients)
# --------------------------------------------------------------------------

_RFSURFHMC_DEFAULT_PATH = "/home/kadek/Documents/software/RfSurfHmc"


def _import_rfsurfhmc(cfg):
    """Make the RfSurfHmc clone importable and return its modules.

    The package is a plain source tree (model/, pyhmc/) with compiled
    pybind11 modules in model/lib — not pip-installed, so we extend sys.path.
    """
    hp = cfg.get("inversion", {}).get("rfsurfhmc", {}) or {}
    pkg = Path(hp.get("path", _RFSURFHMC_DEFAULT_PATH)).expanduser()
    if not (pkg / "model" / "model_rf.py").exists():
        raise ImportError(
            f"RfSurfHmc not found at {pkg} — clone github.com/nqdu/RfSurfHmc, "
            "build it (cmake + make install), or set inversion.rfsurfhmc.path.")
    if str(pkg) not in sys.path:
        sys.path.insert(0, str(pkg))
    from model import model_rf, model_surf, model_rf_swd_vs_thk  # noqa: F401
    from pyhmc import hmc as hmc_mod, hmcda as hmcda_mod  # noqa: F401
    return model_rf, model_surf, model_rf_swd_vs_thk, hmc_mod, hmcda_mod


def _make_scaled_rf(model_rf_mod):
    """ReceiverFunc subclass whose misfit uses an optimal amplitude scale.

    Absolute RF amplitude in the stacks depends on the deconvolution
    normalization, water-level damping and (for LQT) rotation leakage — none
    of which the forward operator models (verified: clean radial synthetic
    direct-P ~1.85 vs 0.62 observed for the same gauss). s* = <obs,syn>/
    <syn,syn> fits the waveform SHAPE (timing + polarity + relative
    amplitudes), which carries the structural information. At s = s* the
    misfit is stationary in s, so by the envelope theorem the model gradient
    is exactly s* * K^T (s*·syn - obs) — no approximation.
    """
    ReceiverFunc = model_rf_mod.ReceiverFunc
    librf = model_rf_mod.librf

    class ScaledReceiverFunc(ReceiverFunc):
        last_scale = 1.0
        noise_sigma = 0.03      # overwritten from the stack's pre-onset noise

        def _optimal_scale(self, d):
            den = float(d @ d)
            s = float(self.dobs @ d) / den if den > 0 else 1.0
            self.last_scale = s
            return s

        def misfit(self, x):
            d = self.forward(x)
            s = self._optimal_scale(d)
            return 0.5 * float(np.sum((s * d - self.dobs) ** 2)) / self.noise_sigma ** 2

        def misfit_and_grad(self, x):
            n = int(x.shape[0] / 2)
            vs, thk = x[:n], x[n:]
            vp, rho, drda, dadb = self.empirical_relation(vs, True)
            drda = drda.reshape(n, 1)
            dadb = dadb.reshape(n, 1)
            qa = thk * 0 + 9999.
            qb = thk * 0 + 9999.
            d, kl = librf.kernel_all(thk, rho, vp, vs, qa, qb, self.ray_p,
                                     self.nt, self.dt, self.gauss,
                                     self.time_shift, self.method,
                                     self.water_level, self.rf_type)
            kernel = kl[2, ...] + dadb * kl[1, ...] + drda * dadb * kl[0, ...]
            s = self._optimal_scale(d)
            r = (s * d - self.dobs) / self.noise_sigma ** 2
            grad = np.hstack((kernel @ r, kl[3, ...] @ r)) * s
            misfit = 0.5 * float((s * d - self.dobs) @ r)
            return misfit, grad, s * d

    return ScaledReceiverFunc


def _steps_to_profile(vs, thk, dep):
    """Vs at each depth in `dep` for a step model (thk[-1]=0 halfspace)."""
    interfaces = np.cumsum(thk[:-1])
    return vs[np.searchsorted(interfaces, dep, side="right")]


class _ParamScaledModel:
    """Sample in z = x / scale instead of x (diagonal mass matrix by proxy),
    and keep every evaluation finite.

    Scaling: the pyhmc samplers use an identity mass matrix and a single
    global step size, but the joint model mixes Vs (posterior scale ~0.1 km/s)
    and layer thickness (~1 km) parameters — no single dt integrates both.
    Dividing each parameter by its expected scale makes the identity mass
    appropriate; gradients transform as g_z = g_x * scale.

    Safety box: a diverging leapfrog trajectory sends x towards +-inf, where
    upstream's boundary mirroring loops forever (reflecting infinities) or the
    C++ kernels spin on absurd models — observed as chains frozen inside
    _find_initial_dt. Physics is only ever evaluated at x clipped into
    [xlo, xhi]; outside, a quadratic penalty (finite, smooth) pushes the
    trajectory back, so H stays finite and the proposal is simply rejected.
    """

    def __init__(self, inner, scales, xlo, xhi, penalty=1e3):
        self.inner = inner
        self.scales = np.asarray(scales, float)
        self.xlo = np.asarray(xlo, float)
        self.xhi = np.asarray(xhi, float)
        self.k = float(penalty)
        self.dobs = inner.dobs

    def _boxed(self, z):
        x = np.nan_to_num(z * self.scales, nan=0.0,
                          posinf=1e30, neginf=-1e30)
        return x, np.clip(x, self.xlo, self.xhi)

    def forward(self, z):
        _, xc = self._boxed(z)
        return self.inner.forward(xc)

    def misfit_and_grad(self, z):
        x, xc = self._boxed(z)
        U, g, d, flag = self.inner.misfit_and_grad(xc)
        dx = x - xc
        if dx.any():
            U = U + 0.5 * self.k * float(dx @ dx)
            g = g + self.k * dx
        return U, g * self.scales, d, flag


class _JointMultiRFSWD:
    """N receiver functions + dispersion, for the pyhmc samplers.

    Mirrors upstream Joint_RF_SWD (which supports exactly one RF) but sums any
    number of RF datasets. The sampler consumes only ``.dobs`` and
    ``.misfit_and_grad(x) -> (misfit, grad, dsyn, flag)``.

    Every term is a NOISE-NORMALIZED chi^2/2 — the upstream code never divides
    by an absolute sigma (only the sigma1/sigma2 ratio), which leaves the
    Hamiltonian's energy scale arbitrary; with real noisy data that made
    delta-U ~ O(100) between neighbouring models, the acceptance collapse to
    0% and the dual-averaging step size shrink towards zero. RF sigmas live
    inside each ScaledReceiverFunc (from the stack's pre-onset noise); the
    dispersion chi^2 is divided by sigma_swd^2 here. The dispersion block is
    additionally up-weighted by swd_weight * n_rf/n_swd (upstream convention)
    so 4 points can balance ~3000 correlated RF samples.
    """

    def __init__(self, rfmodels, swdmodel, sigma_swd, swd_weight=1.0):
        self.rfmodels = rfmodels
        self.swdmodel = swdmodel
        self.sigma_swd = float(sigma_swd)
        self.swd_weight = float(swd_weight)
        self.n_rf = int(sum(m.nt for m in rfmodels))
        self.dobs = np.concatenate([m.dobs for m in rfmodels]
                                   + [swdmodel.dobs])

    def forward(self, x):
        parts = [m.forward(x) for m in self.rfmodels]
        ds, flag = self.swdmodel.forward(x)
        return np.concatenate(parts + [ds]), flag

    def misfit_and_grad(self, x):
        misfit = 0.0
        grad = None
        parts = []
        for m in self.rfmodels:
            mis, g, d = m.misfit_and_grad(x)
            misfit += mis
            grad = g if grad is None else grad + g
            parts.append(d)
        mis_s, g_s, d_s, flag = self.swdmodel.misfit_and_grad(x)
        if not flag:
            return 0.0, np.zeros_like(grad), self.dobs, False
        wt = (self.swd_weight * self.n_rf / d_s.size) / self.sigma_swd ** 2
        misfit += wt * mis_s
        grad = grad + wt * g_s
        return misfit, grad, np.concatenate(parts + [d_s]), True


def _build_joint(cfg, station, rf_cfg):
    """Assemble the multi-RF + SWD joint model for one station (or None).

    Returns (joint, used_classes, rf_models, thk) — shared by every chain of
    the station, rebuilt per worker (construction is a few ms of file I/O).
    """
    mrf, msurf, mjoint, hmc_mod, hmcda_mod = _import_rfsurfhmc(cfg)

    inv_cfg = cfg.get("inversion", {})
    hp = inv_cfg.get("rfsurfhmc", {}) or {}
    p = io_utils.paths(cfg)

    # ---- RF observations (the forward operator is a radial P-RF; the
    # optimal-scale misfit absorbs per-class amplitude/polarity issues) ----
    classes = hp.get("rf_classes") or [hp.get("rf_class", "local_deep")]
    if isinstance(classes, str):
        classes = [classes]
    disp_file = p["tomo"] / f"{station}_disp.txt"
    if not disp_file.exists():
        LOG.info(f"{station}: no dispersion curve — skipped.")
        return None
    resample_hz = float(inv_cfg.get("rf_resample_hz", 25.0)) or None
    thk = np.asarray(hp.get("thk", [2., 4., 6., 8., 10., 30., 0.]), float)
    nl = thk.size
    ScaledRF = _make_scaled_rf(mrf)
    rf_models, used_classes = [], []
    for cls in classes:
        sac = p["rf_out"] / f"{station}_{cls}_stack.sac"
        if not sac.exists():
            continue
        class_params = {**rf_cfg.get("defaults", {}),
                        **rf_cfg.get("classes", {}).get(cls, {})}
        win_start = class_params.get("window", [-10, 35])[0]
        t, rf_obs = _read_stacked_rf(sac, win_start, resample_hz)
        # fit a sub-window of the stack: the structural signal (direct P, Ps,
        # multiples) lives in the first ~20 s; the kernel cost per HMC step is
        # linear in nt, so the tail is pure overhead.
        fw = hp.get("fit_window", [-5.0, 25.0])
        if fw:
            sel = (t >= float(fw[0])) & (t <= float(fw[1]))
            t, rf_obs = t[sel], rf_obs[sel]
        # librf verified to use the Ligorria convention exp(-(pi f/a)^2); the
        # stacks were deconvolved with the rf package's exp(-0.5 (f/f0)^2).
        m = ScaledRF(ray_p=float(class_params.get("moveout_ref_slowness", 0.06)),
                     nt=len(rf_obs), dt=float(t[1] - t[0]),
                     gauss=float(class_params.get("gauss", 1.0)) * _GAUSS_RF_TO_RFMINI,
                     time_shift=float(-t[0]),
                     water_level=float(hp.get("water_level", 0.001)),
                     type_="P", method=str(hp.get("method", "freq")))
        m.set_thk(thk)
        m.set_obsdata(rf_obs)
        # chi^2 noise level: rms of the pre-onset part of the stack, inflated
        # for correlated samples (25 Hz oversamples the <~3 Hz signal band, so
        # the effective number of independent points is ~4x lower), floored so
        # a suspiciously quiet stack cannot dominate the joint misfit.
        pre = rf_obs[t < -2.0]
        rms = float(np.sqrt(np.mean(pre ** 2))) if pre.size else 0.0
        m.noise_sigma = max(rms, 0.02 * float(np.abs(rf_obs).max())) \
            * float(hp.get("noise_inflation", 2.0))
        rf_models.append(m)
        used_classes.append(cls)
    if not rf_models:
        LOG.info(f"{station}: no stacked RFs for classes {classes} — skipped.")
        return None

    # ---- dispersion observation ----
    arr = np.loadtxt(disp_file, ndmin=2)
    periods, phase, group = arr[:, 0], arr[:, 1], arr[:, 2]
    swd = inv_cfg.get("swd_targets", ["phase"])
    use_ph = "phase" in swd or "both" in swd
    use_gr = "group" in swd or "both" in swd
    swd_obs = np.concatenate([phase if use_ph else [], group if use_gr else []])
    model_swd = msurf.SurfWD(tRc=list(periods) if use_ph else [],
                             tRg=list(periods) if use_gr else [],
                             tLc=[], tLg=[])
    model_swd.set_thk(thk)
    model_swd.set_obsdata(swd_obs)
    # dispersion noise: the per-period scatter column of the disp file when
    # present, else 0.05 km/s; config override wins.
    sigma_swd = hp.get("noise_swd")
    if sigma_swd is None:
        sigma_swd = float(np.median(arr[:, 3])) if arr.shape[1] > 3 else 0.05
        sigma_swd = max(float(sigma_swd), 0.01)

    joint = _JointMultiRFSWD(rf_models, model_swd, sigma_swd,
                             float(hp.get("swd_weight", 1.0)))
    return joint, used_classes, rf_models, thk


def _rfsurfhmc_chain_task(cfg, station, rf_cfg, out_root_str, ci):
    """One HMC chain of one station (module-level: runs in a spawn worker).

    Chains are the parallel unit (25 stations x nchains tasks): a chain takes
    tens of minutes while the joint-model build is milliseconds, so rebuilding
    the model per chain buys near-perfect load balance across the pool.
    Returns (station, ci, misfit_array, scales) — models land in the chain's
    HDF5 file, read back during aggregation.
    """
    built = _build_joint(cfg, station, rf_cfg)
    if built is None:
        return None
    joint, used_classes, rf_models, thk = built
    nl = thk.size
    inv_cfg = cfg.get("inversion", {})
    hp = inv_cfg.get("rfsurfhmc", {}) or {}
    hmc_cfg = inv_cfg.get("hmc", {}) or {}

    vs_lo, vs_hi = inv_cfg.get("priors", {}).get("vs", [0.5, 4.8])
    thk_frac = float(hp.get("thk_frac", 0.3))
    bounds = np.ones((2 * nl, 2))
    bounds[:nl, 0] = vs_lo
    bounds[:nl, 1] = vs_hi
    bounds[nl:, 0] = thk * (1 - thk_frac)
    bounds[nl:, 1] = thk * (1 + thk_frac)
    bounds[-1, :] = (0.0, 2.0)          # halfspace: thickness is meaningless

    station_dir = io_utils.ensure_dir(Path(out_root_str) / station)
    hmc_kwargs = {
        "dt": float(hmc_cfg.get("dt", 0.005)),
        "nbest": int(hmc_cfg.get("nbest", 10)),
        "seed": int(hmc_cfg.get("seed", 0)),
        "nsamples": int(hmc_cfg.get("nsamples", 600)),
        "ndraws": int(hmc_cfg.get("ndraws", 200)),
        "name": "hmc",
        "OUTPUT_DIR": str(station_dir),
    }
    use_da = bool(hmc_cfg.get("dual_averaging", True))
    if use_da:
        hmc_kwargs.update({"L0": int(hmc_cfg.get("L0", 10)),
                           "target_ratio": float(hmc_cfg.get("target_ratio", 0.65))})
        sampler_cls = _sampler_mods(cfg)[1]
    else:
        hmc_kwargs.update({"Lrange": list(hmc_cfg.get("Lrange", [5, 20]))})
        sampler_cls = _sampler_mods(cfg)[0]

    # Parameter rescaling (see _ParamScaledModel): sample in z = x/s so the
    # identity mass matrix fits both Vs and thickness axes. The safety box is
    # the sampler bounds themselves — the mirror should keep z inside anyway;
    # the box only catches diverging trajectories mid-leapfrog.
    span = bounds[:, 1] - bounds[:, 0]
    s = np.maximum(0.03 * span, 1e-3)
    zjoint = _ParamScaledModel(joint, s, bounds[:, 0], bounds[:, 1])
    zbounds = bounds / s[:, None]

    # Warm start: HMC is a local sampler — a uniform-over-the-prior init lands
    # at chi^2 ~ 1e5 where gradients are ~1e4-1e5 and the dual-averaging step
    # size collapses to zero without ever accepting (observed on ST01). Each
    # chain starts from the jittered reference model and runs a short
    # backtracking gradient descent (~seconds) into its basin first.
    init_vs = np.asarray(hp.get(
        "init_vs", [2.0, 2.2, 2.5, 2.8, 3.1, 3.8, 4.3][:nl]
        + [4.3] * max(0, nl - 7)), float)
    z0 = np.hstack([init_vs, thk]) / s
    jitter = float(hp.get("init_jitter", 0.05))
    lmax = int(hmc_cfg.get("Lmax", 60))

    def _descend(model, z, zlo, zhi, max_evals=150):
        U, g, _, flag = model.misfit_and_grad(z)
        if not flag:
            return z
        lr = 1.0
        evals = 0
        while evals < max_evals and lr > 1e-3:
            step = g / max(float(np.linalg.norm(g)), 1e-12)
            znew = np.clip(z - lr * step, zlo, zhi)
            Unew, gnew, _, flag = model.misfit_and_grad(znew)
            evals += 1
            if flag and np.isfinite(Unew) and Unew < U:
                z, U, g = znew, Unew, gnew
                lr *= 1.3
            else:
                lr *= 0.5
        return z

    class _WarmStartSampler(sampler_cls):
        def set_initial_model(self):
            zlo, zhi = self.boundaries[:, 0], self.boundaries[:, 1]
            for _ in range(20):
                z = np.clip(z0 + jitter * (zhi - zlo)
                            * np.random.uniform(-1.0, 1.0, z0.size), zlo, zhi)
                mis, _, _, flag = self.model.misfit_and_grad(z)
                if flag and np.isfinite(mis):
                    return _descend(self.model, z, zlo, zhi)
            return z

        def _leapfrog(self, xcur, dt, L):
            # dual averaging sets L = lambda/dt with no ceiling: after a few
            # early rejections dt shrinks 100x and one trajectory balloons to
            # 1e4+ forward calls (minutes) — the chain looks frozen. Capping L
            # keeps proposals bounded; DA still adapts dt toward the target.
            return super()._leapfrog(xcur, dt, min(int(L), lmax))

        # the HDF5 store must hold PHYSICAL models, not z-space ones
        def save_results(self, x, dsyn, idx):
            super().save_results(np.asarray(x) * s, dsyn, idx)

        def _save_init(self, x):
            super()._save_init(np.asarray(x) * s)

    if ci == 0:
        LOG.info(f"{station}: rfsurfhmc — chain 0 of "
                 f"{int(hmc_cfg.get('nchains', 6))}, "
                 f"{hmc_kwargs['nsamples']}+{hmc_kwargs['ndraws']} samples, "
                 f"RF classes {used_classes}, {nl} layers")
    # NOT .init(): the upstream classmethod hard-codes the base class
    # (`return HMCDualAveraging(...)`, not `cls(...)`), silently discarding
    # every override in _WarmStartSampler (warm start, L cap, x-space save).
    if use_da:
        chain = _WarmStartSampler(
            zjoint, zbounds, hmc_kwargs["dt"], hmc_kwargs["L0"],
            hmc_kwargs["nbest"], hmc_kwargs["target_ratio"],
            hmc_kwargs["seed"], hmc_kwargs["nsamples"], hmc_kwargs["ndraws"],
            ci, hmc_kwargs["name"], hmc_kwargs["OUTPUT_DIR"])
    else:
        chain = _WarmStartSampler(
            zjoint, zbounds, hmc_kwargs["dt"], hmc_kwargs["Lrange"],
            hmc_kwargs["nbest"], hmc_kwargs["seed"],
            hmc_kwargs["nsamples"], hmc_kwargs["ndraws"],
            ci, hmc_kwargs["name"], hmc_kwargs["OUTPUT_DIR"])
    try:
        mis = chain.sample()
    finally:
        try:
            chain.fio.close()       # sampler never closes its HDF5 handle
        except Exception:
            pass
    scales = {c: round(m.last_scale, 3)
              for c, m in zip(used_classes, rf_models)}
    return (station, ci, np.asarray(mis), scales)


def _sampler_mods(cfg):
    """(HamitonianMC, HMCDualAveraging) from the RfSurfHmc clone."""
    _, _, _, hmc_mod, hmcda_mod = _import_rfsurfhmc(cfg)
    return hmc_mod.HamitonianMC, hmcda_mod.HMCDualAveraging


def _export_hmc_posterior(cfg, station_dir: Path, station: str,
                          misfits, models, rf_scales) -> None:
    """vs_profile.txt (same format as the BayHunter path) + chain summary.

    Chains stuck in a bad mode are dropped the same way BayHunter's outlier
    filter works: median misfit more than `outlier_factor` x the best chain's
    median disqualifies the chain.
    """
    hp = cfg.get("inversion", {}).get("rfsurfhmc", {}) or {}
    med = np.median(misfits, axis=1)
    keep = med <= float(hp.get("outlier_factor", 2.0)) * med.min()
    kept_models = models[keep].reshape(-1, models.shape[-1])
    nl = kept_models.shape[1] // 2
    depth_max = float(cfg.get("inversion", {}).get("depth_max", 60))
    dep = np.arange(0.0, depth_max + 0.25, 0.25)
    profs = np.array([_steps_to_profile(m[:nl], m[nl:], dep)
                      for m in kept_models])
    med_prof = np.median(profs, axis=0)
    p16, p84 = np.percentile(profs, [16, 84], axis=0)
    out = station_dir / "vs_profile.txt"
    np.savetxt(out, np.column_stack([dep, med_prof, p16, p84]), fmt="%10.4f",
               header="depth_km vs_median_km_s vs_p16 vs_p84")
    dropped = [int(i) for i in np.where(~keep)[0]]
    LOG.info(f"{station}: posterior median Vs(z) -> {out} "
             f"({len(kept_models)} models from {int(keep.sum())}/{len(keep)} "
             f"chains; dropped {dropped or 'none'}; best misfit "
             f"{misfits.min():.3f}; RF scales s* {rf_scales})")
    with open(station_dir / "hmc_summary.txt", "w") as f:
        f.write(f"# rf_scales: {rf_scales}\n")
        f.write(f"# chain median_misfit min_misfit kept\n")
        for i in range(len(med)):
            f.write(f"{i} {med[i]:.4f} {misfits[i].min():.4f} {bool(keep[i])}\n")


def _run_rfsurfhmc(cfg, out_root: Path) -> Path:
    _import_rfsurfhmc(cfg)      # fail fast in the parent if not built
    stations, _ = io_utils.load_stations(cfg)
    rf_cfg = cfg.get("rf", {})
    only = cfg.get("inversion", {}).get("stations")
    nchains = int(cfg.get("inversion", {}).get("hmc", {}).get("nchains", 6))
    codes = [s.code for s in stations if not only or s.code in only]
    # chain-level fan-out: a chain runs for tens of minutes, so station x chain
    # tasks keep every core busy instead of serializing chains per station.
    tasks = [(cfg, code, rf_cfg, str(out_root), ci)
             for code in codes for ci in range(nchains)]
    n_jobs = parallel.resolve_n_jobs(cfg, n_tasks=len(tasks))
    # process backend: each chain is one single-threaded HMC (python + C++
    # forward), so the GIL would serialize a thread pool.
    results = parallel.pmap(_rfsurfhmc_chain_task, tasks, n_jobs,
                            desc="inversion")
    # aggregate chains per station and export the posterior profile
    import h5py

    by_sta: dict[str, list] = {}
    for res in results:
        if res is None:
            continue
        sta, ci, mis, scales = res
        by_sta.setdefault(sta, []).append((ci, mis, scales))
    nsamples = int(cfg.get("inversion", {}).get("hmc", {}).get("nsamples", 600))
    for sta, chains in sorted(by_sta.items()):
        chains.sort()
        station_dir = out_root / sta
        misfits, models = [], []
        for ci, mis, scales in chains:
            h5 = station_dir / f"hmc.{ci}.h5"
            try:
                with h5py.File(h5, "r") as f:
                    models.append(np.array([f[f"{i}/model"][:]
                                            for i in range(nsamples)]))
                misfits.append(mis)
            except Exception as e:
                LOG.warning(f"{sta}: chain {ci} unreadable ({e!r}) — dropped.")
        if not models:
            LOG.warning(f"{sta}: no readable chains — nothing exported.")
            continue
        _export_hmc_posterior(cfg, station_dir, sta, np.asarray(misfits),
                              np.asarray(models), chains[0][2])
    LOG.info("Stage 8 (joint inversion, rfsurfhmc) complete.")
    return out_root


def run(cfg: dict) -> Path:
    engine = str(cfg.get("inversion", {}).get("engine", "bayhunter")).lower()
    p = io_utils.paths(cfg)
    out_root = io_utils.ensure_dir(p["inversion"])
    if engine == "rfsurfhmc":
        return _run_rfsurfhmc(cfg, out_root)

    _require_bayhunter()
    _ensure_root_log_handler()
    stations, _ = io_utils.load_stations(cfg)
    rf_cfg = cfg.get("rf", {})
    only = cfg.get("inversion", {}).get("stations")  # optional subset

    # BayHunter already runs its chains as parallel processes (mp_inversion
    # with nthreads=nchains), so the station-level width is what is left of
    # the n_jobs core target after each station's own chains are accounted
    # for — running n_jobs stations x nchains chains would oversubscribe.
    nchains = int(cfg.get("inversion", {}).get("mcmc", {}).get("chains", 6))
    n_stations = len([s for s in stations if not only or s.code in only])
    n_jobs = parallel.resolve_n_jobs(cfg, n_tasks=n_stations)
    workers = max(1, n_jobs // max(1, nchains))
    # every concurrent mp_inversion must tolerate the whole fleet's child
    # count (chains + one Manager each) — see invert_station.
    mp_children_max = workers * (nchains + 1)
    tasks = [(cfg, sta.code, rf_cfg, str(out_root), mp_children_max)
             for sta in stations if not only or sta.code in only]
    if workers > 1:
        LOG.info(f"Stage 8: {workers} stations concurrently "
                 f"x {nchains} BayHunter chains each")
    LOG.info("BayHunter chain status lines below read: last-accepted-iteration "
             "| n_layers | RMS misfit | log-likelihood | elapsed s | acceptance %"
             " (one line per chain per 5000 iterations)")
    # thread backend: BayHunter's mp_inversion spawns its own chain PROCESSES, so the
    # station level must not go through the spawn process pool (can't pickle BayHunter's
    # local chain fn). Threads launch-and-wait while the chains do the real work.
    parallel.pmap(_invert_station_task, tasks, workers, desc="inversion", backend="thread")
    LOG.info("Stage 8 (joint inversion) complete.")
    return out_root
