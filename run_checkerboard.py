#!/usr/bin/env python3
"""DSurfTomo checkerboard resolution test (ifsyn = 1).

Builds MOD.true = the production starting model perturbed by +/-AMP in
CHECKER_NODES x CHECKER_NODES lateral checkers (sign also flips across the
DEPTH_BANDS boundaries), lets DSurfTomo synthesize travel times for the REAL
path geometry (surfdataTB.dat) with its configured noise level, inverts with
the production settings, and plots input vs recovered perturbation maps.

Everything runs in dsurftomo/checkerboard/ — the real-run outputs are
untouched. Requires a prior real run (run_dsurftomo.py) to supply
surfdataTB.dat / MOD / DSurfTomo.in.
"""
from __future__ import annotations

import argparse
import shutil
import subprocess
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent
SRC = ROOT / "dsurftomo"                 # real-run inputs live here
BINARY = Path("/home/kadek/Documents/software/DSurfTomo/bin/DSurfTomo")
FIGDIR = ROOT / "figures"

PLOT_DEPTHS = [0.3, 0.6, 1.0, 1.5, 2.0, 3.0]   # km slices to display

# CAUTION on depth-band sign flips (--bands): the 0.75-2 s Rayleigh kernels
# integrate over roughly the top 0-2 km, so alternating the checker sign at
# ~1 km cancels most of the travel-time signal and the test then understates
# lateral resolution. Default is therefore NO vertical alternation.


def _read_mod(path: Path):
    lines = path.read_text().splitlines()
    depths = np.array(lines[0].split(), float)
    rows = [np.array(l.split(), float) for l in lines[1:] if l.strip()]
    nx = len(rows[0])
    nz = len(depths)
    ny = len(rows) // nz
    return depths, np.array(rows).reshape(nz, ny, nx)   # vs[k, j(lon), i(lat)]


def _stations(datafile: Path):
    """Unique station coords (lat, lon) from both source (#) and receiver rows."""
    pts = set()
    for line in datafile.read_text().splitlines():
        f = line.split()
        if not f:
            continue
        lat, lon = (f[1], f[2]) if f[0] == "#" else (f[0], f[1])
        pts.add((round(float(lat), 5), round(float(lon), 5)))
    return np.array(sorted(pts))


def main() -> int:
    ap = argparse.ArgumentParser(description="DSurfTomo checkerboard test")
    ap.add_argument("--nodes", type=int, default=3,
                    help="lateral checker size in grid nodes (0.02 deg ~ 2.2 km each)")
    ap.add_argument("--amp", type=float, default=0.10,
                    help="perturbation amplitude (fraction of starting Vs)")
    ap.add_argument("--bands", type=float, nargs="*", default=[],
                    help="depths (km) where the checker sign flips; default none "
                         "(see CAUTION above)")
    args = ap.parse_args()

    for req in ("surfdataTB.dat", "MOD", "DSurfTomo.in"):
        if not (SRC / req).exists():
            raise SystemExit(f"{SRC/req} missing — run run_dsurftomo.py first.")
    tag = f"{args.nodes}node" + ("_zflip" if args.bands else "")
    work = SRC / f"checkerboard_{tag}"
    work.mkdir(exist_ok=True)
    shutil.copy(SRC / "surfdataTB.dat", work)
    shutil.copy(SRC / "MOD", work)

    # control file: ifsyn is the 3rd line from the end (…, ifsyn, noiselevel, threshold)
    ctrl = (SRC / "DSurfTomo.in").read_text().splitlines()
    ctrl[-3] = "1"
    (work / "DSurfTomo.in").write_text("\n".join(ctrl) + "\n")

    depths, vs = _read_mod(SRC / "MOD")
    nz, ny, nx = vs.shape
    band = np.searchsorted(np.array(args.bands), depths, side="right") if args.bands \
        else np.zeros(nz, dtype=int)
    ci = np.arange(nx) // args.nodes
    cj = np.arange(ny) // args.nodes
    sign = (-1.0) ** (band[:, None, None] + cj[None, :, None] + ci[None, None, :])
    vs_true = vs * (1.0 + args.amp * sign)

    # MOD.true carries no depth-header line: main.f90 reads data blocks only
    with open(work / "MOD.true", "w") as f:
        for k in range(nz):
            for j in range(ny):
                f.write(" ".join(f"{v:.4f}" for v in vs_true[k, j]) + "\n")

    print(f"Checkerboard: {args.nodes} nodes (~{args.nodes*2.2:.0f} km), "
          f"+/-{args.amp*100:.0f}%, sign flips at {args.bands or 'none'}")
    print("Running DSurfTomo (synthetic)…")
    with open(work / "checkerboard.log", "w") as log:
        subprocess.run([str(BINARY)], input="DSurfTomo.in\n", text=True, cwd=work,
                       stdout=log, stderr=subprocess.STDOUT, timeout=3600, check=True)

    # both files: lon lat depth vs on the inner-node grid (nz-1 depth layers)
    tru = np.loadtxt(work / "Vs_model.real")
    rec = np.loadtxt(work / "DSurfTomo.inSyn.dat")
    start = dict(zip(depths, vs[:, 0, 0]))          # 1-D start model: depth -> Vs
    sta = _stations(SRC / "surfdataTB.dat")

    fig, axes = plt.subplots(2, len(PLOT_DEPTHS),
                             figsize=(3.1 * len(PLOT_DEPTHS), 6.4),
                             sharex=True, sharey=True, constrained_layout=True)
    for col, z in enumerate(PLOT_DEPTHS):
        for row, (arr, lab) in enumerate([(tru, "input"), (rec, "recovered")]):
            ax = axes[row, col]
            m = np.isclose(arr[:, 2], z)
            if not m.any():
                ax.set_axis_off()
                continue
            lon, lat, v = arr[m, 0], arr[m, 1], arr[m, 3]
            pert = 100.0 * (v / start[z] - 1.0)
            lons, lats = np.unique(lon), np.unique(lat)
            grid = np.full((len(lats), len(lons)), np.nan)
            grid[np.searchsorted(lats, lat), np.searchsorted(lons, lon)] = pert
            im = ax.pcolormesh(lons, lats, grid, cmap="RdBu_r",
                               vmin=-100 * args.amp, vmax=100 * args.amp,
                               shading="nearest")
            ax.plot(sta[:, 1], sta[:, 0], "k^", ms=3, mew=0)
            ax.set_title(f"{lab}  z={z:g} km", fontsize=9)
            if row == 1:
                ax.set_xlabel("Lon")
            if col == 0:
                ax.set_ylabel("Lat")
    fig.colorbar(im, ax=axes, shrink=0.8, label="dVs from starting model (%)")
    flips = f", z-flips at {args.bands} km" if args.bands else ""
    fig.suptitle(f"DSurfTomo checkerboard test — {args.nodes}-node "
                 f"(~{args.nodes*2.2:.0f} km) checkers, ±{args.amp*100:.0f}%"
                 f"{flips}, real path geometry", fontsize=11)
    for ext in ("png", "svg"):
        fig.savefig(FIGDIR / f"F9b_checkerboard_{tag}.{ext}", dpi=200)
    print(f"Wrote {FIGDIR}/F9b_checkerboard_{tag}.png")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
