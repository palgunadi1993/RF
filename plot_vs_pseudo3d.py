#!/usr/bin/env python3
"""Pseudo-3D (1.5D) Vs volume from the Stage-8 per-station profiles.

Each station's posterior-median Vs(z) is an INDEPENDENT 1-D column; ordinary
kriging interpolates them laterally per depth level. Lateral continuity is
therefore geostatistical, not physical (unlike the DSurfTomo stage) — treat
the result as a visualization of the station ensemble, valid near stations
only (cells farther than `MASK_KM` from the nearest station are blanked).

Outputs (figures/):
  vs_slices_pseudo3d.png     kriged Vs map slices at selected depths
  vs_sections_pseudo3d.png   two kriged vertical sections (W-E and N-S)

Usage: python plot_vs_pseudo3d.py [--config config.yaml]
"""
from __future__ import annotations

import argparse
import glob
import os
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from cmcrameri import cm as cmc

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from rf_pipeline.io_utils import load_config, load_stations, paths  # noqa: E402

KM_PER_DEG = 111.19492664455873
SLICES_KM = [0.5, 1.5, 2.0, 3.0, 5.0, 7.0, 9.0, 10.0, 15.0]   # map-view depth slices
SECT_ZMAX = 20.0                                # section depth extent
MASK_KM = 5.0                                   # blank cells > this from a station
VRANGE_KM = 6.0                                 # exponential variogram range
NUGGET_FRAC = 0.05                              # stabilizing nugget (of sill)


def ordinary_krige(xy, val, xi):
    """Ordinary kriging with an exponential variogram; returns (est, std).

    25 support points -> dense solve is trivial. gamma(h) = sill*(1-exp(-h/r))
    + nugget; sill from the data variance. Lagrange multiplier enforces the
    unbiasedness constraint.
    """
    n = len(val)
    sill = max(np.var(val), 1e-6)
    nug = NUGGET_FRAC * sill

    def gamma(h):
        return nug + sill * (1.0 - np.exp(-h / VRANGE_KM))

    d = np.linalg.norm(xy[:, None, :] - xy[None, :, :], axis=2)
    A = np.empty((n + 1, n + 1))
    A[:n, :n] = gamma(d)
    np.fill_diagonal(A[:n, :n], 0.0)
    A[n, :] = 1.0
    A[:, n] = 1.0
    A[n, n] = 0.0

    dq = np.linalg.norm(xi[:, None, :] - xy[None, :, :], axis=2)
    B = np.empty((len(xi), n + 1))
    B[:, :n] = gamma(dq)
    B[:, n] = 1.0
    W = np.linalg.solve(A, B.T).T                   # (nq, n+1)
    est = W[:, :n] @ val
    var = np.maximum(np.einsum("ij,ij->i", W, B), 0.0)
    return est, np.sqrt(var)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=str(ROOT / "config.yaml"))
    args = ap.parse_args()
    cfg = load_config(args.config)
    p = paths(cfg)

    # station columns
    stas, _ = load_stations(cfg)
    coords = {s.code: (s.longitude, s.latitude) for s in stas}
    codes, profs = [], []
    for f in sorted(glob.glob(str(p["inversion"] / "*" / "vs_profile.txt"))):
        code = os.path.basename(os.path.dirname(f))
        if code not in coords:
            continue
        d = np.loadtxt(f)
        codes.append(code)
        profs.append(d[:, 1])
    dep = np.loadtxt(f)[:, 0]
    V = np.array(profs)                             # (nsta, ndep)
    lon = np.array([coords[c][0] for c in codes])
    lat = np.array([coords[c][1] for c in codes])
    lat0 = lat.mean()
    xy = np.column_stack([(lon - lon.mean()) * KM_PER_DEG * np.cos(np.radians(lat0)),
                          (lat - lat0) * KM_PER_DEG])

    # lateral grid in km (extended slightly beyond the network)
    pad = 2.0
    gx = np.arange(xy[:, 0].min() - pad, xy[:, 0].max() + pad, 0.25)
    gy = np.arange(xy[:, 1].min() - pad, xy[:, 1].max() + pad, 0.25)
    GX, GY = np.meshgrid(gx, gy)
    gpts = np.column_stack([GX.ravel(), GY.ravel()])
    dist_near = np.min(np.linalg.norm(
        gpts[:, None, :] - xy[None, :, :], axis=2), axis=1)
    mask = (dist_near > MASK_KM).reshape(GX.shape)

    def to_lonlat(x, y):
        return (x / (KM_PER_DEG * np.cos(np.radians(lat0))) + lon.mean(),
                y / KM_PER_DEG + lat0)

    # ---------------- figure 1: depth slices ----------------
    ncol = 3
    nrow = int(np.ceil(len(SLICES_KM) / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(12.5, 3.75 * nrow), dpi=200,
                             sharex=True, sharey=True)
    axes = np.atleast_2d(axes)
    for ax in np.ravel(axes)[len(SLICES_KM):]:
        ax.set_visible(False)
    for z, ax in zip(SLICES_KM, np.ravel(axes)):
        iz = np.argmin(np.abs(dep - z))
        est, _ = ordinary_krige(xy, V[:, iz], gpts)
        est = est.reshape(GX.shape)
        est[mask] = np.nan
        glon, glat = to_lonlat(GX, GY)
        vmin, vmax = np.nanpercentile(est, [2, 98])
        im = ax.pcolormesh(glon, glat, est, cmap=cmc.roma, vmin=vmin,
                           vmax=vmax, shading="auto", rasterized=True)
        ax.scatter(lon, lat, marker="^", s=28, c="k", edgecolors="w",
                   linewidths=0.6, zorder=5)
        ax.set_title(f"z = {dep[iz]:.2g} km", fontsize=10)
        ax.set_aspect(1.0 / np.cos(np.radians(lat0)))
        ax.tick_params(labelsize=7)
        cb = fig.colorbar(im, ax=ax, shrink=0.85, pad=0.02)
        cb.set_label("$V_S$ (km/s)", fontsize=8)
        cb.ax.tick_params(labelsize=7)
    for ax in axes[-1, :]:
        ax.set_xlabel("Longitude", fontsize=9)
    for ax in axes[:, 0]:
        ax.set_ylabel("Latitude", fontsize=9)
    fig.suptitle("Pseudo-3D Vs — ordinary kriging of independent 1-D posterior"
                 f" medians (masked > {MASK_KM:.0f} km from a station)",
                 fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    out1 = Path(p["figures"]) / "vs_slices_pseudo3d.png"
    fig.savefig(out1)
    plt.close(fig)
    print(f"slices  : {out1}")

    # ---------------- figure 2: vertical sections ----------------
    zlevels = dep[dep <= SECT_ZMAX]
    lines = {
        "W–E (through network center)":
            np.column_stack([np.linspace(gx[0], gx[-1], 220),
                             np.zeros(220)]),
        "S–N (through network center)":
            np.column_stack([np.zeros(160),
                             np.linspace(gy[0], gy[-1], 160)]),
    }
    fig, axes = plt.subplots(2, 1, figsize=(10, 7.2), dpi=200)
    for (title, line), ax in zip(lines.items(), axes):
        along = np.linalg.norm(np.diff(line, axis=0), axis=1).cumsum()
        along = np.concatenate([[0.0], along])
        sec = np.full((len(zlevels), len(line)), np.nan)
        dline = np.min(np.linalg.norm(
            line[:, None, :] - xy[None, :, :], axis=2), axis=1)
        ok = dline <= MASK_KM
        for i, z in enumerate(zlevels):
            iz = np.argmin(np.abs(dep - z))
            est, _ = ordinary_krige(xy, V[:, iz], line[ok])
            sec[i, ok] = est
        vmin, vmax = np.nanpercentile(sec, [2, 98])
        im = ax.pcolormesh(along, zlevels, sec, cmap=cmc.roma, vmin=vmin,
                           vmax=vmax, shading="auto", rasterized=True)
        # project nearby stations onto the line; stagger labels on two rows and
        # skip any label that would land within 0.8 km of the previous one
        axis = 0 if "W–E" in title else 1
        near = np.abs(xy[:, 1 - axis]) < 3.0
        xpos_all = xy[near, axis] - line[0, axis]
        order = np.argsort(xpos_all)
        ax.plot(xpos_all, np.zeros(near.sum()) + 0.2,
                "v", color="k", mec="w", ms=7, clip_on=False, zorder=5)
        lastx, row = -1e9, 0
        for c, xpos in zip(np.array(codes)[near][order], xpos_all[order]):
            if xpos - lastx < 0.8:
                continue
            ax.text(xpos, -0.7 - 0.9 * row, c, fontsize=6, ha="center",
                    color="#1f2430", clip_on=False)
            lastx, row = xpos, 1 - row
        ax.set_ylim(SECT_ZMAX, 0)
        ax.set_title(title, fontsize=10, pad=26)
        ax.set_ylabel("Depth (km)", fontsize=9)
        ax.tick_params(labelsize=8)
        cb = fig.colorbar(im, ax=ax, shrink=0.9, pad=0.015)
        cb.set_label("$V_S$ (km/s)", fontsize=8)
        cb.ax.tick_params(labelsize=7)
    axes[-1].set_xlabel("Distance along profile (km)", fontsize=9)
    fig.suptitle("Pseudo-3D Vs sections — kriged 1-D columns "
                 "(lateral continuity is geostatistical, not tomographic)",
                 fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    out2 = Path(p["figures"]) / "vs_sections_pseudo3d.png"
    fig.savefig(out2)
    plt.close(fig)
    print(f"sections: {out2}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
