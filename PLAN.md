# Dieng Crustal Structure — RF + ANT Joint-Inversion Plan

A step-by-step runbook to reproduce the multi-method workflow of Criado-Sutti et al.
(2026, *Solid Earth* 17, 711–733) on the **Dieng 3C continuous node dataset**, using
**robust, maintained community software** (no algorithms written from scratch — only
glue/config).

**Goal:** a station-by-station 1-D shear-wave velocity model `Vs(z)` of the crust beneath
Dieng, obtained by **joint inversion of receiver functions (RF) + ambient-noise Rayleigh
dispersion**, supported by H–κ stacking and CCP imaging.

> **Config-driven (single source of truth):** the **only** file a user edits is
> [`config.yaml`](config.yaml). Every stage is a thin runner invoked the same way —
> `python run_<stage>.py --config config.yaml` — and reads *all* of its parameters from
> the matching section of that YAML. No parameters are passed on the command line and
> none are hard-coded in the scripts. Each stage below names its config section.

> **Design requirement (explicit):** every RF stage must accept **local-deep, regional,
> and teleseismic** earthquakes. The chosen tools (`rf`, `seispy`) are distance-agnostic —
> they deconvolve P-coda regardless of source distance; only the *event selection*,
> *frequency band*, and *ray-parameter/moveout* settings change per source class
> (see Stage 2). This is the same strategy the paper used (teleseismic + Jujuy ~200 km
> deep-slab local RFs).

---

## 0. Tool stack & one-time install

| Stage | Tool | Install | Needs |
|---|---|---|---|
| RF, H–κ, stacking | **`rf`** | `pip install rf obspyh5` | Python/ObsPy |
| RF GUI, CCP, H–κ | **`seispy`** | `pip install python-seispy` | Python/ObsPy |
| ANT cross-correlation | **NoisePy** | `pip install noisepy-seis` | Python (MPI optional) |
| Dispersion curves (FTAN, phase+group) | **amb_noise_tools** | `git clone https://github.com/ekaestle/amb_noise_tools && pip install -e .` | Python + f2py |
| 2-D tomography (maps) | **FMST** | compile Fortran (Rawlinson) | gfortran |
| *(alt)* direct 3-D inversion | **DSurfTomo** | git clone https://github.com/HongjianFang/DSurfTomo | gfortran |
| **Joint inversion (RF+SWD)** | **BayHunter** | `git clone https://github.com/jenndrei/BayHunter && pip install -e .` | gfortran (surf96 kernel) |
| *(alt)* authors' inversion | **RfSurfHmc** | `git clone https://github.com/nqdu/RfSurfHmc` + build | C/Python |

> Recommendation: create one conda env (`conda create -n dieng_rf python=3.11`) and install
> all of the above into it. Install `gfortran`/`gcc` first (`sudo apt install gfortran build-essential`).

**Open decisions (set before running):**
- Inversion engine: **BayHunter** (recommended — transdimensional, robust, maintained,
  live monitoring) vs **RfSurfHmc** (exact paper reproduction). Inputs are identical
  (1 stacked RF + 1 dispersion curve per station), so upstream stages are unchanged.
- ANT path: **full tomography** (NoisePy + FMST, needs ~10+ stations) vs **two-station
  dispersion** (amb_noise_tools, fine for small/sparse arrays).

---

## Directory layout (create up front)

```
RF/
├── data/
│   ├── continuous/        # 3C continuous mseed (per station, per day)
│   ├── stationxml/        # inventory + instrument response
│   └── catalogs/
│       ├── teleseismic.xml    # 30–90°, M>5.5 (from ISC/USGS)
│       ├── regional.xml       # ~2–30°, M>4
│       └── local_deep.xml     # Java slab Wadati–Benioff (your catalog / BMKG)
├── rf_out/                # per-source-class RFs + stacks
├── hk_out/                # H–κ results
├── ccp_out/               # CCP sections
├── ant/
│   ├── ccfs/              # NoisePy cross-correlations
│   └── disp/              # dispersion curves per station/pair
├── tomo/                  # FMST maps OR per-station curves
├── inversion/             # BayHunter/RfSurfHmc inputs+outputs per station
└── figures/
```

---

## Stage 1 — Data preparation  · config: `data:`, `catalogs:`

**Input:** raw 3C node data (SmartSolo BD3C-5), station inventory with response.

1. Convert/organize continuous data to day-long 3C MiniSEED per station
   (you already have `scripts/sac2mseed.py`, `dld2mseed.py`).
2. Build/clean `StationXML` with full instrument response (`scripts/remove_response.py`).
3. Assemble **three event catalogs** (QuakeML), covering all source classes:
   - **Teleseismic** 30–90°, M≥5.5 — fetch from USGS/ISC for the deployment window.
   - **Regional** ~2–30°, M≥4 (Sunda arc).
   - **Local-deep** — Java slab earthquakes beneath Dieng (your local catalog / BMKG / ISC),
     the high-frequency analog to the paper's Jujuy cluster.

> ⚠ **Node-bandwidth caveat:** the 5 Hz geophone rolls off below ~1 Hz, so **teleseismic
> RFs (low-frequency) will be marginal**. Expect the workhorse RF sources to be
> **local-deep and regional** events (high-frequency, suits 5 Hz). Keep teleseismic in the
> workflow but treat it as a cross-check, not the backbone.

**QC checkpoint:** plot data availability matrix (station × day); confirm response removal
restores physical units; confirm ≥ a handful of events per source class fall in usable bands.

---

## Stage 2 — Receiver functions (all three source classes) — `rf`  · config: `rf:`

One script, looped over source classes. `rf.defaults` apply to every class; each
`rf.classes.<name>` block overrides only what differs — that is how local-deep, regional,
and teleseismic are all driven from the one YAML.

```python
from rf import read_rf, RFStream
# 1. cut event windows from continuous data using each catalog + inventory
# 2. rf_stream.rotate('NE->RT')  (or 'ZNE->LQT' for steep local-deep rays)
# 3. rf_stream.rf(method='P', deconvolve='time', gauss=<a>)   # iterative time-domain
# 4. rf_stream.moveout()          # normalize to reference slowness
# 5. QC by SNR, then rf_stream.trim2 / bin by back-azimuth & slowness, stack
```

**Per-source-class parameters** (mirrors paper; tune in Appendix-B style ±10% tests):

| Source class | Distance | Phase | Rotation | Gaussian `a` (band) | Typical slowness `p` (s/km) |
|---|---|---|---|---|---|
| Teleseismic | 30–90° | P | ZRT | 1.0–1.5 (≤1 Hz) | 0.04–0.08 |
| Regional | ~2–30° | P/Pn | ZRT or LQT | 2.5 (≈0.01–2 Hz) | 0.06–0.10 |
| Local-deep | <~5°, h>100 km | direct P | **LQT** (steep) | 3.0 (high-freq) | 0.00–0.05 |

> For regional/local-deep events, compute the per-event ray parameter from a **local 1-D
> model** (ObsPy TauPy or pyrocko `cake`) rather than the global default — this is the only
> source-class-specific subtlety in `rf`.

**Deconvolution:** iterative **time-domain** (Ligorria & Ammon 1999; `f1=0.03,
f2=20, max_iters=400`) is this pipeline's default, chosen for the 5 Hz nodes.
NOTE the paper itself used **water-level** deconvolution (Gaussian `a=0.5`,
water level `c=0.1`, band 0.01–2 Hz) — set `deconvolve: freqattr` to match it.

**Outputs:** `rf_out/<station>_<class>.h5` (individual RFs) + `rf_out/<station>_<class>_stack.sac`
(the stacked radial RF used later by the inversion).

**QC checkpoint:** back-azimuth-binned RF panels (like paper Fig. 4); clear Ps at expected
delay; reject incoherent traces; check transverse-component energy (azimuthal anisotropy/dip).

---

## Stage 3 — H–κ stacking — `rf` or `seispy`  · config: `hk:`

Per station, per source class: grid search over crustal thickness `H` and `Vp/Vs` (κ) using
Ps, PpPs, PsPs amplitudes (Zhu & Kanamori 2000).

- `seispy hk` or `rf`'s H–κ routine; fix `Vp≈6 km/s` (or local value).
- Apply the bottom-up κ depth-correction (paper Appendix A) if you want depth-dependent κ.

**Output:** `hk_out/hk_<class>.csv` (H, κ, 95% bounds per station) — direct cross-check on
discontinuity depths and a prior for the inversion.

---

## Stage 4 — CCP imaging — `seispy`  · config: `ccp:`

Pseudo-migrate RFs to depth and stack along profiles (paper Fig. 6).

- `seispy` CCP module: define profiles (N–S, E–W across Dieng), 1-D migration velocity model.
- Run separately for local-deep and teleseismic RFs to compare resolution.

**Output:** `ccp_out/<profile>.png` depth sections — lateral continuity of interfaces.

---

## Stage 5 — Ambient-noise cross-correlation — NoisePy  · config: `ant:`

Uses the same continuous 3C data (independent of earthquakes).

1. Preprocess: downsample, detrend, taper, **spectral whitening + temporal normalization**,
   remove response.
2. Cross-correlate all station pairs (vertical–vertical for Rayleigh), stack over the whole
   deployment (NoisePy config: 1–2 h windows, daily then full stack).

**Output:** `ant/ccfs/` — stacked cross-correlation functions (empirical Green's functions).

**QC checkpoint:** CCF section vs interstation distance shows a clear moveout (surface-wave
arrival); symmetric causal/acausal lobes.

---

## Stage 6 — Dispersion measurement — amb_noise_tools  · config: `dispersion:`

Extract **Rayleigh phase and group velocity vs period** from each CCF (FTAN), resolving the
2πN phase ambiguity against the group curve (Bensen et al. 2007).

- Target band realistic for node aperture: ~**0.5–8 s** (depth ~ c·T/3 ≈ top ~1–10 km).
- amb_noise_tools also does the correlation, so it can replace NoisePy if you prefer a single
  tool for the ANT side.

**Output:** `ant/disp/<pair>.disp` (period, phase_vel, group_vel, uncertainty).

---

## Stage 7 — From dispersion to per-station curves  · config: `tomo:`

**Path A — Full tomography (recommended if ≥~10 nodes):**
- Invert all pair dispersion measurements → 2-D phase/group velocity maps per period (**FMST**).
- Sample each map at every station location → **one 1-D dispersion curve per station**
  (this is exactly the paper's Fig. 8 → per-station curve step).

**Path B — Two-station / regional average (small array):**
- Use path-average dispersion curves directly (amb_noise_tools), one representative curve per
  station neighborhood.

**Output:** `tomo/<station>_disp.txt` (period, phase[, group], σ) per station.

> *(Alternative end-to-end:* **DSurfTomo** inverts dispersion directly to a 3-D `Vs` model
> without maps — useful as an independent ANT-only check against the joint result.)

---

## Stage 8 — Joint inversion RF + dispersion — BayHunter (rec.) / RfSurfHmc  · config: `inversion:`

Per station, feed **two datasets** describing the same 1-D column:
1. the **stacked radial RF** from Stage 2 (with its slowness, dt, Gaussian `a`), and
2. the **dispersion curve** from Stage 7 (period, phase/group).

```python
# BayHunter: targets = [RFTarget(time, rf_amp), SWDTarget(period, c)]
#   - set ref slowness = representative p of the stacked RF source class
#   - priors: Vs range, n-layers range, Vp/Vs (use Stage-3 H–κ as prior)
#   - run multiple chains; monitor live with BayWatch
```

- BayHunter is **transdimensional** — it solves for the number of layers (no need to fix 5)
  and returns posterior `Vs(z)` with uncertainty.
- Weighting: balance RF vs SWD misfit (paper used σ_rf=1e-3, σ_swd=0.7e-2 as a guide).
- **Multi-source-class option:** include teleseismic *and* local-deep stacked RFs as separate
  targets in the same inversion (each with its own slowness) — they jointly constrain the
  column at different frequencies/depths. This is a strength of using the deep+regional+tele
  data together.

**Output:** `inversion/<station>/` — best & mean `Vs(z)`, posterior, synthetic-vs-observed
fits (RF and dispersion), per-station model.

**QC checkpoint:** chains converged (R̂); synthetic RF reproduces Ps timing/amplitude;
synthetic dispersion fits within σ; Moho/interface depths consistent with Stage-3 H–κ and
Stage-4 CCP.

---

## Stage 9 — Synthesis & publication figures  · config: `plot:`

Each figure is produced by `run_synthesis.py --config config.yaml`, toggled individually
under `plot.figures` in the YAML, and written to `figures/` in every format in
`plot.format` (png + svg) at `plot.dpi`. The set mirrors the paper and adds Dieng-specific
panels. **One figure = one function = one config toggle.**

### Figure set (publication-ready)

| # | Figure | Shows | Source data | Library |
|---|---|---|---|---|
| **F1** | **Station & tectonic map** | Dieng edifice, node positions, faults/craters, regional setting inset | stationxml, DEM | PyGMT |
| **F2** | **Event distribution** | Azimuthal-equidistant maps of teleseismic + regional + local-deep events used for RFs | the 3 catalogs | PyGMT |
| **F3** | **Ray-path / coverage map** | ANT inter-station paths + RF piercing points | ccfs, rf_out | PyGMT |
| **F4** | **H–κ stacking** | κ–H stack panels for 2 representative stations (max marked, 95% contour) | hk_out | Matplotlib |
| **F5** | **H–κ summary** | measured vs depth-corrected κ vs depth, all stations, by source class | hk_out | Matplotlib |
| **F6** | **RF record sections** | radial + transverse, back-azimuth-binned, linear stack on top; one column per source class | rf_out | rf / Matplotlib |
| **F7** | **CCP sections** | pseudo-migrated depth sections along NS & EW profiles; tele vs local-deep rows | ccp_out | seispy / Matplotlib |
| **F8** | **Noise CCF gather** | stacked cross-correlations vs inter-station distance (moveout), bandpassed | ant/ccfs | Matplotlib |
| **F9** | **Dispersion maps** | phase & group velocity maps at selected periods (only if `tomo.path: A`) | tomo | PyGMT |
| **F10** | **Joint-inversion result (per station)** | posterior `Vs(z)` ensemble + best/mean, with observed-vs-synthetic RF and dispersion fit | inversion | BayHunter plots / Matplotlib |
| **F11** | **Vs cross-sections** | NS & EW 2-D `Vs` built from per-station profiles, interfaces overlaid (H–κ, CCP, inversion) | inversion + hk + ccp | Matplotlib (extend `scripts/plot_cross_sections.py`) |
| **F12** | **Integrated structural model** | interpreted layer cartoon (sediment/hydrothermal/crystalline/Moho) with velocities | all | Matplotlib |

> **Dieng-specific framing for F10–F12:** annotate shallow low-`Vs` as the hydrothermal/
> altered/clay-cap zone, mid-crustal low-`Vs` as possible magma/partial-melt storage, and
> any deep high-`Vs` step as crystalline basement / Moho. These interpretive overlays are
> what differentiate a Dieng volcano paper from a generic crustal study.

> **All geographic maps use PyGMT** (F1, F2, F3, F9). PyGMT gives the relief/topography
> shading, projections, scale bars and north arrows expected for publication maps; the
> non-map panels (record sections, H–κ, dispersion, inversion, cross-sections) use
> Matplotlib. The mapping backend is fixed in config as `plot.map_backend: pygmt`.

### Publication styling standards (enforced via `plot:` config)
- Vector output (**SVG/PDF**) + 300 dpi PNG; never bitmap-only line art.
- Consistent, **colorblind-safe** velocity colormap (e.g. `roma`/`vik`); shared `vs_clip`
  range across all `Vs` panels so colors are comparable.
- Uniform font family/size, scale bars + north arrows on maps, lettered subpanels (a, b, …).
- Every axis labeled with units; perceptually-uniform colormaps only (no `jet`).
- Figure widths preset to journal columns (single ≈ 8.3 cm, double ≈ 17 cm) via `plot.width_cm`.

---

## End-to-end run order (smooth path)

Every stage takes only `--config config.yaml`:

```
python run_prep.py        --config config.yaml   # Stage 1  -> data/ + 3 catalogs
python run_rf.py          --config config.yaml   # Stage 2  -> rf_out/   (all 3 classes)
python run_hk.py          --config config.yaml   # Stage 3  -> hk_out/
python run_ccp.py         --config config.yaml   # Stage 4  -> ccp_out/
python run_ant.py         --config config.yaml   # Stage 5  -> ant/ccfs/
python run_dispersion.py  --config config.yaml   # Stage 6  -> ant/disp/
python run_tomo.py        --config config.yaml   # Stage 7  -> tomo/
python run_inversion.py   --config config.yaml   # Stage 8  -> inversion/
python run_synthesis.py   --config config.yaml   # Stage 9  -> figures/
```

Or run all in order: `python run_pipeline.py --config config.yaml`.

Stages 2–4 (earthquake side) and 5–7 (noise side) are **independent** and can run in
parallel; they converge at Stage 8.

---

## Decisions still needed from you
All three are just config keys — set them once in `config.yaml`:
1. **Inversion engine** → `inversion.engine`: `bayhunter` (recommended) or `rfsurfhmc` (paper).
2. **ANT path** → `tomo.path`: `A` (full tomography) or `B` (two-station). Depends on node
   count + array aperture — tell me both so I can set the default.
3. **Primary depth target** → drives `dispersion.periods` and whether `teleseismic` stays
   enabled in `catalogs:`/`rf.classes`: shallow volcanic (≤5 km) / mid-crust / whole crust+Moho.
