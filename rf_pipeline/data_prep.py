"""Stage 1a: data preparation & QC.

Organises the SmartSolo 3C daily files into ``data/continuous`` (via symlink so no
bytes are copied), makes sure a StationXML with full response is present under
``data/stationxml``, and writes a station x day availability matrix as the QC
checkpoint required by PLAN.md Stage 1. Catalog assembly lives in ``catalogs.py``
and is invoked here so ``run_prep.py`` produces everything Stage 1 promises.
"""
from __future__ import annotations

import shutil
from pathlib import Path

import pandas as pd

from . import catalogs, io_utils
from .logging_setup import get_logger

LOG = get_logger("rf.data_prep")


def _link_or_copy(src: Path, dst: Path, copy: bool = False) -> None:
    if dst.exists() or dst.is_symlink():
        return
    if copy:
        shutil.copy2(src, dst)
    else:
        try:
            dst.symlink_to(src)
        except OSError:
            shutil.copy2(src, dst)


def organise_continuous(cfg: dict) -> Path:
    """Populate ``data/continuous`` with the discoverable 3C daily files.

    If ``data.source_waveform_dir`` is set and differs from the continuous dir,
    the daily files are symlinked in. If it is unset, the continuous dir is
    assumed already populated and is used as-is.
    """
    p = io_utils.paths(cfg)
    cont = io_utils.ensure_dir(p["continuous"])
    src = cfg.get("data", {}).get("source_waveform_dir")
    if not src:
        LOG.info(f"No data.source_waveform_dir set — using {cont} as-is.")
        return cont
    src = io_utils.resolve_path(src, cfg["_project_root"])
    if src.resolve() == cont.resolve():
        return cont
    if not src.exists():
        LOG.warning(f"source_waveform_dir {src} not present (drive not mounted?) — "
                    f"skipping link step; existing {cont} will be used.")
        return cont
    copy = bool(cfg.get("data", {}).get("copy_instead_of_link", False))
    files = io_utils.discover_waveforms(src)
    for wf in files:
        _link_or_copy(wf.path, cont / wf.path.name, copy=copy)
    LOG.info(f"Linked {len(files)} daily files into {cont}")
    return cont


def ensure_inventory(cfg: dict) -> Path | None:
    """Ensure a StationXML lives under ``data/stationxml``.

    Uses ``data.inventory`` if given (copied in), otherwise leaves any existing
    StationXML in place. Does not fabricate a response — build one beforehand
    with the repeater pipeline's ``build_dieng_inventory.py`` if needed.
    """
    p = io_utils.paths(cfg)
    sxml_dir = io_utils.ensure_dir(p["stationxml"])
    inv_src = cfg.get("data", {}).get("inventory")
    if inv_src:
        inv_src = io_utils.resolve_path(inv_src, cfg["_project_root"])
        if inv_src.exists():
            dst = sxml_dir / inv_src.name
            _link_or_copy(inv_src, dst, copy=True)
            LOG.info(f"Inventory staged: {dst}")
            return dst
        LOG.warning(f"data.inventory {inv_src} not found.")
    existing = io_utils._first_inventory(sxml_dir)
    if existing is None:
        LOG.warning(f"No StationXML under {sxml_dir}. Response removal will fail until "
                    "you place one there (see build_dieng_inventory.py).")
    return existing


def availability_matrix(cfg: dict) -> pd.DataFrame:
    """station x day availability (1 = daily 3C file present)."""
    p = io_utils.paths(cfg)
    src = cfg.get("data", {}).get("source_waveform_dir")
    scan_dir = io_utils.resolve_path(src, cfg["_project_root"]) if src else p["continuous"]
    if not Path(scan_dir).exists():
        scan_dir = p["continuous"]
    files = io_utils.discover_waveforms(scan_dir)
    if not files:
        LOG.warning(f"No waveforms discovered under {scan_dir}.")
        return pd.DataFrame()
    rows = [{"station": wf.station, "date": wf.date_str, "present": 1} for wf in files]
    df = pd.DataFrame(rows)
    matrix = df.pivot_table(index="station", columns="date", values="present",
                            aggfunc="max", fill_value=0)
    return matrix


def run(cfg: dict) -> Path:
    """Full Stage 1: organise data, ensure inventory, QC matrix, build catalogs."""
    organise_continuous(cfg)
    ensure_inventory(cfg)

    p = io_utils.paths(cfg)
    qc_dir = io_utils.ensure_dir(p["root"] / "data" / "qc")
    matrix = availability_matrix(cfg)
    if not matrix.empty:
        out_csv = qc_dir / "availability_matrix.csv"
        matrix.to_csv(out_csv)
        n_sta, n_day = matrix.shape
        LOG.info(f"Availability: {n_sta} stations x {n_day} days -> {out_csv}")
        _plot_availability(matrix, qc_dir / "availability_matrix.png")

    catalogs.run(cfg)
    LOG.info("Stage 1 (data prep) complete.")
    return p["continuous"]


def _plot_availability(matrix: pd.DataFrame, out_png: Path) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(min(0.25 * matrix.shape[1] + 3, 20),
                                        0.3 * matrix.shape[0] + 1))
        ax.imshow(matrix.values, aspect="auto", cmap="Greens", vmin=0, vmax=1,
                  interpolation="nearest")
        ax.set_yticks(range(matrix.shape[0]))
        ax.set_yticklabels(matrix.index, fontsize=6)
        step = max(1, matrix.shape[1] // 15)
        ax.set_xticks(range(0, matrix.shape[1], step))
        ax.set_xticklabels(matrix.columns[::step], rotation=90, fontsize=6)
        ax.set_title("Data availability (station x day)")
        fig.tight_layout()
        fig.savefig(out_png, dpi=150)
        plt.close(fig)
        LOG.info(f"Availability plot -> {out_png}")
    except Exception as e:
        LOG.warning(f"Could not plot availability matrix: {e}")
