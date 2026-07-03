"""Stage 7-alt: direct 3-D Vs inversion from noise dispersion — DSurfTomo.

Independent, ANT-only cross-check on the RF+SWD joint result (PLAN.md Stage 7 note):
DSurfTomo (Fang et al. 2015, github.com/HongjianFang/DSurfTomo) inverts the pair
dispersion curves *directly* for a 3-D shear-velocity model, with no intermediate
phase-velocity maps and no per-station 1-D curves.

This module is glue only: it writes DSurfTomo's three inputs from the Stage-6 pair
curves + config, runs the compiled Fortran binary if one is configured, and tidies
the output model. **All file formats were verified against the DSurfTomo source**
(src/main.f90 input reader, scripts/GenerateDSurfTomoInputFile.py, GenerateIniMOD.py):

  * DSurfTomo.in  — fixed line order (3 comment lines, then values) read by main.f90
  * surfdataTB.dat — per source: ``# lat lon period_index wavetype veltype`` header,
                     then ``lat lon velocity`` receiver lines. wavetype 2=Rayleigh,
                     veltype 0=phase/1=group; period_index is 1-based into the tRc list.
  * MOD           — first line nz depths, then nz*ny lines of nx velocities (vs[k,j,i]).

Output model (real-data run): ``DSurfTomo.inMeasure.dat`` with columns
``lon lat depth Vs`` — copied to ``dsurftomo/vs3d.xyz`` and an .npz for plotting.
"""
from __future__ import annotations

import shutil
import subprocess
from collections import defaultdict
from pathlib import Path

import numpy as np

from . import io_utils
from .logging_setup import get_logger

LOG = get_logger("rf.dsurftomo")


def _period_list(disp_dir: Path, want_periods):
    """Sorted unique measured periods, restricted to the configured set."""
    seen = set()
    for f in Path(disp_dir).glob("*.disp"):
        try:
            arr = np.loadtxt(f, ndmin=2)
        except Exception:
            continue
        for T in arr[:, 0]:
            seen.add(round(float(T), 4))
    if want_periods:
        want = {round(float(T), 4) for T in want_periods}
        seen &= want
    return sorted(seen)


def _collect_measurements(disp_dir, periods, sta_lookup, col):
    """Group pair measurements by (period_index, source station).

    Returns dict[(pidx, src_code)] -> list[(recv_code, velocity)] and the set of
    stations that actually carry data.
    """
    pindex = {round(T, 4): i + 1 for i, T in enumerate(periods)}
    groups: dict[tuple, list] = defaultdict(list)
    used = set()
    for f in Path(disp_dir).glob("*.disp"):
        parts = f.stem.split("_")
        if len(parts) < 2:
            continue
        a, b = parts[0], parts[1]
        if a not in sta_lookup or b not in sta_lookup:
            continue
        try:
            arr = np.loadtxt(f, ndmin=2)
        except Exception:
            continue
        for row in arr:
            T = round(float(row[0]), 4)
            if T not in pindex:
                continue
            vel = float(row[col])
            if not np.isfinite(vel) or vel <= 0:
                continue
            groups[(pindex[T], a)].append((b, vel))
            used.update((a, b))
    return groups, used


def _write_surfdata(path, groups, periods, sta_lookup, wavetype, veltype):
    lines = []
    # order by period index then source, as DSurfTomo groups sequentially
    for (pidx, src) in sorted(groups):
        s = sta_lookup[src]
        lines.append(f"# {s.latitude:.6f} {s.longitude:.6f} {pidx} {wavetype} {veltype}")
        for recv, vel in groups[(pidx, src)]:
            r = sta_lookup[recv]
            lines.append(f" {r.latitude:.6f} {r.longitude:.6f} {vel:.4f}")
    Path(path).write_text("\n".join(lines) + "\n")
    return len(lines)


def _grid_from_stations(stations, spacing, pad):
    """Return (nx, ny, goxd, gozd, dvxd, dvzd) covering the array.

    goxd/gozd is the UPPER-LEFT node (max lat, min lon); lat decreases with i,
    lon increases with j (matches main.f90's lon=gozd+(j-1)dvzd, lat=goxd-(i-1)dvxd).
    """
    lats = [s.latitude for s in stations]; lons = [s.longitude for s in stations]
    lat_max, lat_min = max(lats) + pad, min(lats) - pad
    lon_min, lon_max = min(lons) - pad, max(lons) + pad
    nx = int(np.ceil((lat_max - lat_min) / spacing)) + 1   # lat direction
    ny = int(np.ceil((lon_max - lon_min) / spacing)) + 1   # lon direction
    return nx, ny, lat_max, lon_min, spacing, spacing


def _write_mod(path, nx, ny, depth_nodes, minvel, velgrad):
    """Linear-gradient starting model (GenerateIniMOD.py format)."""
    nz = len(depth_nodes)
    with open(path, "w") as fp:
        fp.write("".join("%5.1f" % d for d in depth_nodes) + "\n")
        for k in range(nz):
            vs = minvel + depth_nodes[k] * velgrad
            for _ in range(ny):
                fp.write("".join("%7.3f" % vs for _ in range(nx)) + "\n")


def _write_control(path, cfg_d, datafile, nx, ny, nz, goxd, gozd, dvxd, dvzd,
                   nsrc, periods, veltype):
    kmaxRc = len(periods) if veltype == 0 else 0
    kmaxRg = len(periods) if veltype == 1 else 0
    lines = [
        "c" * 69,
        "c INPUT PARAMETERS",
        "c" * 69,
        f"{datafile}",
        f"{nx} {ny} {nz}",
        f"{goxd:.4f} {gozd:.4f}",
        f"{dvxd:.4f} {dvzd:.4f}",
        f"{nsrc}",
        f"{cfg_d.get('weight', 4.0)} {cfg_d.get('damp', 1.0)}",
        f"{cfg_d.get('sublayers', 3)}",
        f"{cfg_d.get('minvel', 0.5)} {cfg_d.get('maxvel', 4.5)}",
        f"{cfg_d.get('maxiter', 10)}",
        f"{cfg_d.get('sparsity', 0.2)}",
        f"{kmaxRc}",
    ]
    if kmaxRc:
        lines.append(" ".join(f"{T:g}" for T in periods))
    lines.append(f"{kmaxRg}")
    if kmaxRg:
        lines.append(" ".join(f"{T:g}" for T in periods))
    lines += ["0", "0",                       # kmaxLc, kmaxLg (Love unused)
              "0",                             # ifsyn (real data)
              f"{cfg_d.get('noiselevel', 0.02)}",
              f"{cfg_d.get('threshold', 3.0)}"]
    Path(path).write_text("\n".join(lines) + "\n")


def run(cfg: dict) -> Path:
    d = cfg.get("dsurftomo", {}) or {}
    if not d.get("enabled", False):
        LOG.info("dsurftomo.enabled=false — skipping 3-D ANT inversion.")
        return io_utils.paths(cfg)["root"]

    stations, _ = io_utils.load_stations(cfg)
    sta_lookup = io_utils.station_lookup(stations)
    p = io_utils.paths(cfg)
    disp_dir = p["disp"]
    out_dir = io_utils.ensure_dir(io_utils.resolve_path(
        d.get("output_dir", "dsurftomo"), cfg["_project_root"]))

    measure = str(d.get("measure", "phase")).lower()
    veltype = 0 if measure == "phase" else 1
    col = 1 if measure == "phase" else 2      # .disp columns: period phase group snr

    periods = _period_list(disp_dir, d.get("periods") or cfg.get("dispersion", {}).get("periods"))
    if not periods:
        LOG.warning("No measured dispersion periods found — run Stage 6 first.")
        return out_dir
    groups, used = _collect_measurements(disp_dir, periods, sta_lookup, col)
    if not groups:
        LOG.warning("No usable pair measurements for DSurfTomo.")
        return out_dir

    depth_nodes = d.get("depth_nodes_km", [0, 0.5, 1, 2, 3, 5, 8, 12, 20])
    nz = len(depth_nodes)
    spacing = float(d.get("grid_spacing_deg", 0.02))
    pad = float(d.get("pad_deg", spacing))
    nx, ny, goxd, gozd, dvxd, dvzd = _grid_from_stations(stations, spacing, pad)
    nsrc = len(stations)

    datafile = "surfdataTB.dat"
    nrec = _write_surfdata(out_dir / datafile, groups, periods, sta_lookup, 2, veltype)
    _write_mod(out_dir / "MOD", nx, ny, depth_nodes,
               float(d.get("init_minvel", 0.8)), float(d.get("init_velgrad", 0.15)))
    ctrl = out_dir / "DSurfTomo.in"
    _write_control(ctrl, d, datafile, nx, ny, nz, goxd, gozd, dvxd, dvzd,
                   nsrc, periods, veltype)
    LOG.info(f"DSurfTomo inputs written to {out_dir} "
             f"(grid {nx}x{ny}x{nz}, {len(periods)} periods, {nrec} data lines, "
             f"{len(used)} stations).")

    binary = d.get("binary")
    if binary:
        binary = io_utils.resolve_path(binary, cfg["_project_root"])
    if not binary or not Path(binary).exists():
        LOG.warning("dsurftomo.binary not configured/available — inputs written but "
                    "not run. Compile DSurfTomo (src/), then run it in "
                    f"{out_dir} (echo DSurfTomo.in | ./DSurfTomo).")
        return out_dir

    try:
        LOG.info(f"Running {binary} in {out_dir} ...")
        subprocess.run([str(binary)], input="DSurfTomo.in\n", text=True,
                       cwd=str(out_dir), check=True,
                       timeout=int(d.get("timeout_s", 7200)))
    except Exception as e:
        LOG.warning(f"DSurfTomo run failed ({e}); inputs remain in {out_dir}.")
        return out_dir

    model = out_dir / "DSurfTomo.inMeasure.dat"
    if model.exists():
        arr = np.loadtxt(model, ndmin=2)      # columns: lon lat depth vs
        shutil.copy2(model, out_dir / "vs3d.xyz")
        np.savez(out_dir / "vs3d.npz", lon=arr[:, 0], lat=arr[:, 1],
                 depth=arr[:, 2], vs=arr[:, 3])
        LOG.info(f"3-D Vs model -> {out_dir/'vs3d.xyz'} ({len(arr)} nodes).")
    else:
        LOG.warning(f"Expected output {model.name} not found after run.")
    LOG.info("Stage 7-alt (DSurfTomo 3-D) complete.")
    return out_dir
