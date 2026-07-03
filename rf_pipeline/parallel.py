"""Process-level parallelism for the per-station / per-day / per-pair stages.

Stdlib multiprocessing (ProcessPoolExecutor), NOT MPI: every parallel unit in
this pipeline is an independent task on one machine writing its own output
files, which is exactly the process-pool model. MPI would only pay off across
multiple cluster nodes and would add an mpi4py dependency plus an ``mpirun``
launcher; if that is ever needed, the task lists built for :func:`pmap` are the
natural MPI work units.

The pool uses the ``spawn`` start method so each worker imports numpy/obspy
fresh and picks up the single-threaded BLAS environment set in
:func:`executor` — with ``fork`` the workers would inherit the parent's
already-initialised BLAS thread pools and oversubscribe the CPU (n_jobs
processes x BLAS threads each). Spawn also means worker functions and all
their arguments must be module-level / picklable.

Workers count comes from ``project.n_jobs`` in the config (-1 = all cores).
"""
from __future__ import annotations

import os
from concurrent.futures import ProcessPoolExecutor

from .logging_setup import get_logger

LOG = get_logger("rf.parallel")

# BLAS/OpenMP thread-count variables honoured by numpy/scipy backends.
_BLAS_ENV = ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
             "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS")


def resolve_n_jobs(cfg: dict, n_tasks: int | None = None) -> int:
    """``project.n_jobs``: -1/0/unset = all cores; clamped to [1, n_tasks]."""
    try:
        n = int(cfg.get("project", {}).get("n_jobs", -1))
    except (TypeError, ValueError):
        n = -1
    if n <= 0:
        n = os.cpu_count() or 1
    if n_tasks is not None:
        n = min(n, max(1, int(n_tasks)))
    return max(1, n)


def executor(n_jobs: int) -> ProcessPoolExecutor:
    """A spawn-context pool whose workers run single-threaded BLAS.

    The env vars are set (if the user has not set them) before spawning, so
    the fresh worker interpreters load numpy with one BLAS thread each; the
    parent's numpy is already initialised and is unaffected.
    """
    import multiprocessing as mp

    for var in _BLAS_ENV:
        os.environ.setdefault(var, "1")
    return ProcessPoolExecutor(max_workers=n_jobs,
                               mp_context=mp.get_context("spawn"))


def pmap(fn, tasks: list[tuple], n_jobs: int, desc: str = "") -> list:
    """Run ``fn(*task)`` for every task; results in task order.

    Serial (in-process) when ``n_jobs == 1`` or there is only one task, so the
    single-core path is exactly the old sequential behaviour. A task that
    raises is logged and yields None instead of killing the stage — callers
    already treat missing per-unit results as "skipped".
    """
    label = desc or getattr(fn, "__name__", "task")
    if n_jobs <= 1 or len(tasks) <= 1:
        out = []
        for t in tasks:
            try:
                out.append(fn(*t))
            except Exception as e:
                LOG.warning(f"{label}: task failed ({e!r})")
                out.append(None)
        return out

    LOG.info(f"{label}: {len(tasks)} tasks on {n_jobs} processes")
    out = []
    with executor(n_jobs) as ex:
        futures = [ex.submit(fn, *t) for t in tasks]
        for fut in futures:
            try:
                out.append(fut.result())
            except Exception as e:
                LOG.warning(f"{label}: task failed in worker ({e!r})")
                out.append(None)
    return out
