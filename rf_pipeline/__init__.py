"""Dieng RF + ANT joint-inversion pipeline.

A config-driven, community-software workflow that reproduces the multi-method
crustal-structure study of Criado-Sutti et al. (2026, *Solid Earth* 17, 711-733)
on the Dieng 3C SmartSolo node dataset.

Every stage is a module exposing ``run(cfg: dict) -> Path`` and is driven entirely
from ``config.yaml`` (see PLAN.md). The thin ``run_<stage>.py`` scripts in the
project root are the entry points.
"""
from __future__ import annotations

__all__ = [
    "io_utils",
    "logging_setup",
    "progress",
    "catalogs",
    "data_prep",
    "receiver_functions",
    "hk_stacking",
    "ccp",
    "ambient_noise",
    "dispersion",
    "tomography",
    "dsurftomo",
    "inversion",
    "synthesis",
]
