# Dieng RF + ANT joint-inversion pipeline

Config-driven implementation of the multi-method crustal-structure workflow in
[`PLAN.md`](PLAN.md) — receiver functions (RF) + ambient-noise tomography (ANT)
jointly inverted for a per-station `Vs(z)` model beneath Dieng, after
Criado-Sutti et al. (2026, *Solid Earth* **17**, 711–733).

Everything is driven from a single file, [`config.yaml`](config.yaml). Each stage
is a thin runner:

```bash
python run_prep.py       --config config.yaml   # Stage 1  data + 3 catalogs + QC
python run_rf.py         --config config.yaml   # Stage 2  receiver functions
python run_hk.py         --config config.yaml   # Stage 3  H-kappa stacking
python run_ccp.py        --config config.yaml   # Stage 4  CCP imaging
python run_ant.py        --config config.yaml   # Stage 5  noise cross-correlation
python run_dispersion.py --config config.yaml   # Stage 6  dispersion (FTAN)
python run_tomo.py       --config config.yaml   # Stage 7  per-station curves
python run_dsurftomo.py  --config config.yaml   # Stage 7-alt  DSurfTomo 3-D (opt-in)
python run_inversion.py  --config config.yaml   # Stage 8  joint RF+SWD inversion
python run_synthesis.py  --config config.yaml   # Stage 9  publication figures
# or the whole thing (Stages 2-4 and 5-7 are independent, converge at Stage 8):
python run_pipeline.py   --config config.yaml
python run_pipeline.py   --config config.yaml --stages rf,hk,ccp
```

## Data model (shared with `repeater_pipeline`)

The 3C reading layer mirrors `repeater_pipeline/pipeline/io_utils.py` so both Dieng
pipelines treat the SmartSolo BD3C-5 data identically: one 3-component daily file
per station, `DG.STxx.YYYY-MM-DD.mseed`, all of Z/N/E in a single file
(`rf_pipeline/io_utils.py::discover_waveforms`, `read_event_window_3c`). Station
coordinates come from `station.txt`; the full DH?/DP? instrument responses come
from the merged StationXML built by the repeater pipeline's
`scripts/build_dieng_inventory.py`. All three are wired in `config.yaml` under
`data:` — edit those paths for your machine (the external-drive waveform path is
symlinked into `data/continuous` by `run_prep`).

## Which code is a community tool vs. a verified built-in

Per PLAN.md's rule ("no algorithms from scratch — only glue/config"), external
tools are used where they are the standard. Their APIs were **verified against the
upstream source** (cloned from GitHub), not written from memory:

| Stage | Tool used | API verified against | Status |
|---|---|---|---|
| 2 RF | **`rf`** (trichter/rf) | `rf.RFStream.rf`, `rf.rfstats`, `deconv_iterative`, `moveout`, `trim2` | wired |
| 8 inversion | **BayHunter** (jenndrei/BayHunter) | `Targets.*`, `MCMC_Optimizer`, `rfmini_modrf.set_modelparams` | wired |

The following stages use **transparent, self-contained implementations of standard
published methods** (installed dependencies: ObsPy/NumPy/SciPy only), with the
PLAN-named tool documented as a drop-in alternative:

| Stage | Method (citation) | Named alternative |
|---|---|---|
| 3 H-kappa | Zhu & Kanamori (2000) grid stack + Appendix-A bottom-up kappa correction | `seispy hk` / `rf` |
| 4 CCP | 1-D Ps time-to-depth migration + piercing-point stack | `python-seispy` (`ccpprofile.py`) |
| 5 ANT | Bensen et al. (2007) whitened CC; phase-weighted stack (Schimmel & Paulssen 1997) | NoisePy |
| 6 dispersion | FTAN group + phase (Bensen et al. 2007) | `amb_noise_tools` |
| 7 tomography | two-station average -> per-station curves | (3-D handled by DSurfTomo below) |

Full 3-D tomography is an external Fortran tool wired as *glue only* (writes verified
input files, runs the binary if configured) — **it replaces FMST**, which is not
publicly distributable:

| Stage | Tool used | Input formats verified against | Status |
|---|---|---|---|
| 7-alt / path A | **DSurfTomo** (HongjianFang/DSurfTomo) | `src/main.f90` reader, `GenerateDSurfTomoInputFile.py`, `GenerateIniMOD.py` | wired (needs compiled binary) |

Cited built-ins were chosen over blind-wiring uninstalled tools because an inspectable
implementation is more verifiable. Every non-trivial core is unit-tested on synthetics:
H-kappa recovers a known `H=30 km, Vp/Vs=1.75`; the Appendix-A correction round-trips
intrinsic Vp/Vs exactly; FTAN recovers a known group velocity; PWS improves S/N over a
linear stack; DSurfTomo input files match the reference format byte-for-structure.

## Known limitations / not-yet-wired (honest list)

- **Full 3-D tomography (`tomo.path: A` / the DSurfTomo stage)** needs the DSurfTomo
  Fortran binary compiled and set in `dsurftomo.binary` (see Install step 4). Without
  it, the stage writes the (verified) input files but does not run; the joint inversion
  still gets its per-station curves from the two-station average.
- **`inversion.engine: rfsurfhmc`** is not wired; use the default `bayhunter` here, or
  drive `nqdu/RfSurfHmc` externally from the same exported RF stacks + `tomo/<sta>_disp.txt`.
  (BayHunter already covers Stage 8; RfSurfHmc is only for exact paper reproduction.)
- **Real data has not been run** end-to-end yet (external drive not mounted at build
  time). Code imports cleanly and every numerical core passes synthetic tests; the
  `rf`/BayHunter stages need those packages installed and the drive attached to run.

Now closed (previously listed here): the Appendix-A depth-dependent kappa correction is
implemented and unit-tested; ANT phase-weighted stacking (`stack_method: pws`) is
implemented; `inversion.misfit_sigma` is wired into BayHunter's noise priors; FMST is
replaced by DSurfTomo.

## Install

The pipeline is Python 3.11+; several stages need a Fortran/C toolchain. The tools
split into two kinds, and are installed differently:

- **Standalone binaries** (DSurfTomo) — compiled once, kept in a shared tools
  directory, and **called by absolute path** from `config.yaml`. They are
  env-independent: any project, any conda env, calls the same file.
- **Python libraries** (`rf`, `BayHunter`, `seispy`, `noisepy`, `amb_noise_tools`) —
  cannot be "called as a file"; the pipeline `import`s them, so they must live in a
  Python environment. Their **source** still sits in the shared tools directory
  (editable install), but they register into a conda env.

### Recommended layout (what this project assumes)

- **Shared tools dir:** `/home/kadek/Documents/software/` — all external source +
  compiled binaries live here, reusable across projects.
- **Dedicated conda env `dieng_rf`** for the Python libraries. **Do not install these
  into your shared `obspy` env** — `BayHunter` builds a Fortran extension and `noisepy`
  pins specific numpy/scipy versions, which can disturb an env other work depends on.
  "Use from another project" = just `conda activate dieng_rf` there; the env is
  tool-specific, not project-specific.

### 0. System build tools (once)

```bash
# Debian/Ubuntu
sudo apt update && sudo apt install -y build-essential gfortran gcc g++ git cmake
```

### 1. Dedicated env + the always-needed core

Runs Stages 1, 3, 4, 5, 6, 7(pathB) and all figures (the transparent built-in stages
depend on nothing beyond this core).

```bash
conda create -n dieng_rf python=3.11 -y
conda activate dieng_rf
# ObsPy + PyGMT come cleanest from conda-forge (PyGMT pulls the GMT C library):
conda install -c conda-forge obspy pygmt numpy scipy pandas pyyaml matplotlib h5py -y
python -c "import obspy, pygmt, numpy, scipy, pandas, yaml, h5py; print('core OK')"
```

### 2. Stage 2 — receiver functions (`rf`, into `dieng_rf`)

```bash
pip install rf obspyh5        # obspyh5 lets Stage 2 write rf_out/*.h5
python -c "import rf, obspyh5; print('rf', rf.__version__)"
```

### 3. Stage 8 — joint inversion (BayHunter, source in the shared dir)

Builds a Fortran kernel (`surfdisp96`) at install, so `gfortran` from step 0 must exist.

```bash
cd /home/kadek/Documents/software
git clone https://github.com/jenndrei/BayHunter
pip install -e /home/kadek/Documents/software/BayHunter   # editable: source stays here
python -c "from BayHunter import Targets, MCMC_Optimizer; print('BayHunter OK')"
```

### 4. Stage 7-alt — DSurfTomo (compiled binary, called by path)

```bash
cd /home/kadek/Documents/software
git clone https://github.com/HongjianFang/DSurfTomo
cd DSurfTomo/src && make          # -> /home/kadek/Documents/software/DSurfTomo/src/DSurfTomo
```

Then wire the binary into `config.yaml` (no Python install — the pipeline `subprocess`-calls it):

```yaml
dsurftomo:
  enabled: true
  binary: /home/kadek/Documents/software/DSurfTomo/src/DSurfTomo
```

### 5. Optional extras (only if you need that path)

```bash
cd /home/kadek/Documents/software
# 8-alt: RfSurfHmc — the paper's exact joint engine (C + Python; build per its README)
git clone https://github.com/nqdu/RfSurfHmc

# Drop-in alternatives to the built-in Stages 4-6 (not required):
pip install python-seispy          # Stage 4 CCP / H-kappa
pip install noisepy-seis           # Stage 5 ANT cross-correlation
git clone https://github.com/ekaestle/amb_noise_tools && pip install -e ./amb_noise_tools  # Stage 6 FTAN
# (Full 3-D tomography is DSurfTomo from step 4 — FMST is not used.)
```

### 6. Point the config at your data

Edit the `data:` block in [`config.yaml`](config.yaml) for your machine:

- `source_waveform_dir` — the SmartSolo 3C daily files (`DG.STxx.YYYY-MM-DD.mseed`);
  `run_prep` symlinks them into `data/continuous/`.
- `station_file` — `station.txt` (reused from `repeater_pipeline`).
- `inventory` — merged StationXML with full DH?/DP? responses. If you don't have it,
  build it once with the repeater pipeline: `python scripts/build_dieng_inventory.py`.

### What each stage needs at a glance

| Want to run | Install steps needed | Where it lives |
|---|---|---|
| Stages 1,3,4,5,6,7(B), figures | 0 + 1 | `dieng_rf` env |
| Stage 2 (RF), H-kappa on real RFs | + 2 | `dieng_rf` env |
| Stage 8 (joint inversion) | + 3 | source in `software/`, env `dieng_rf` |
| Stage 7-alt (DSurfTomo 3-D) | + 4 | binary in `software/`, called by path |
| Paper-exact engine (RfSurfHmc) | + 5 | RfSurfHmc in `software/` |

## Layout

```
rf_pipeline/           importable package (one module per stage)
run_*.py               thin stage runners (Stage N -> rf_pipeline.<module>.run)
config.yaml            single source of truth
data/                  continuous/, stationxml/, catalogs/, qc/
rf_out/ hk_out/ ccp_out/ ant/ tomo/ inversion/ figures/   stage outputs
dsurftomo/             DSurfTomo inputs (surfdataTB.dat, MOD, DSurfTomo.in) + vs3d
```
