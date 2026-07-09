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
import time
from concurrent.futures import ProcessPoolExecutor, as_completed

from .logging_setup import get_logger

LOG = get_logger("rf.parallel")

# BLAS/OpenMP thread-count variables honoured by numpy/scipy backends.
_BLAS_ENV = ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
             "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS")


def _available_ram_gb() -> float | None:
    """Best-effort free RAM in GiB, or None if it can't be determined.

    Prefers Linux ``/proc/meminfo`` (``MemAvailable`` — the kernel's own estimate
    of what a new workload can grab, which already accounts for reclaimable cache),
    then ``psutil`` if it happens to be installed. Returns None on anything else so
    the memory cap simply doesn't apply rather than guessing wrong.
    """
    try:
        with open("/proc/meminfo") as f:
            for line in f:
                if line.startswith("MemAvailable:"):
                    return int(line.split()[1]) / (1024.0 ** 2)   # kB -> GiB
    except OSError:
        pass
    try:
        import psutil  # type: ignore
        return psutil.virtual_memory().available / (1024.0 ** 3)
    except Exception:  # noqa: BLE001 — psutil optional; absence is fine
        return None


def resolve_n_jobs(cfg: dict, n_tasks: int | None = None) -> int:
    """Worker count from ``project.n_jobs``, clamped by tasks *and* free RAM.

    - ``project.n_jobs``: -1/0/unset = all cores; a positive value is a hard ceiling.
    - clamped to ``[1, n_tasks]`` so we never spawn more workers than there is work.
    - clamped so ``workers * project.mem_per_worker_gb`` fits in currently-available
      RAM (minus ``project.mem_headroom_gb`` reserved for the parent + OS). This is
      what stops an ``n_jobs: -1`` all-cores RF run from spawning 18 ObsPy workers on
      a box that's already deep in swap and getting OOM-killed. Defaults: 2.0 GB per
      worker, 2.0 GB headroom. Set ``mem_per_worker_gb: 0`` to disable the RAM cap.
    """
    proj = cfg.get("project", {}) or {}
    try:
        n = int(proj.get("n_jobs", -1))
    except (TypeError, ValueError):
        n = -1
    if n <= 0:
        n = os.cpu_count() or 1
    if n_tasks is not None:
        n = min(n, max(1, int(n_tasks)))

    # memory-aware cap
    try:
        per_worker = float(proj.get("mem_per_worker_gb", 2.0))
    except (TypeError, ValueError):
        per_worker = 2.0
    try:
        headroom = float(proj.get("mem_headroom_gb", 2.0))
    except (TypeError, ValueError):
        headroom = 2.0
    if per_worker > 0:
        avail = _available_ram_gb()
        if avail is not None:
            usable = max(0.0, avail - headroom)
            mem_cap = max(1, int(usable // per_worker))
            if mem_cap < n:
                LOG.info(
                    f"n_jobs capped {n} -> {mem_cap} by free RAM "
                    f"({avail:.1f} GiB avail - {headroom:.0f} reserved, "
                    f"{per_worker:.1f} GiB/worker). Tune project.mem_per_worker_gb."
                )
                n = mem_cap

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


def pmap(fn, tasks: list[tuple], n_jobs: int, desc: str = "",
         backend: str = "process") -> list:
    """Run ``fn(*task)`` for every task; results in task order.

    Serial (in-process) when ``n_jobs == 1`` or there is only one task, so the
    single-core path is exactly the old sequential behaviour. A task that
    raises is logged and yields None instead of killing the stage — callers
    already treat missing per-unit results as "skipped".

    ``backend='thread'`` uses a ThreadPoolExecutor instead of the spawn process
    pool. Required when ``fn`` itself spawns processes (e.g. BayHunter's
    mp_inversion): nesting that inside the spawn pool fails to pickle its local
    worker fn. Threads just launch-and-wait while the inner processes do the work.
    """
    label = desc or getattr(fn, "__name__", "task")
    total = len(tasks)
    if n_jobs <= 1 or total <= 1:
        out = []
        for i, t in enumerate(tasks, 1):
            try:
                out.append(fn(*t))
            except Exception as e:
                LOG.warning(f"{label}: task failed ({e!r})")
                out.append(None)
            _log_tick(label, i, total)
        return out

    if backend == "thread":
        from concurrent.futures import ThreadPoolExecutor
        pool = ThreadPoolExecutor(max_workers=n_jobs)
        LOG.info(f"{label}: {total} tasks on {n_jobs} threads")
    else:
        pool = executor(n_jobs)
        LOG.info(f"{label}: {total} tasks on {n_jobs} processes")
    # Preserve task order in the result while still reporting completions as they
    # arrive: submit in order (remember each future's index), fill results by that
    # index, and tick the counter every time *any* worker finishes.
    out: list = [None] * total
    with pool as ex:
        fut_index = {ex.submit(fn, *t): i for i, t in enumerate(tasks)}
        done = 0
        for fut in as_completed(fut_index):
            i = fut_index[fut]
            try:
                out[i] = fut.result()
            except Exception as e:
                LOG.warning(f"{label}: task failed in worker ({e!r})")
                out[i] = None
            done += 1
            _log_tick(label, done, total)
    return out


# Emit an in-stage progress line at ~every 10% (and always the last), so a long
# per-station/per-pair stage shows "37/126 (29%)" without flooding the log.
_TICK_STATE: dict[str, tuple[int, float]] = {}


def _log_tick(label: str, done: int, total: int) -> None:
    if total <= 1:
        return
    step = max(1, total // 10)
    last_done, last_t = _TICK_STATE.get(label, (0, 0.0))
    now = time.time()
    if done == total or done - last_done >= step or now - last_t >= 30:
        pct = 100.0 * done / total
        LOG.info(f"  {label}: {done}/{total} ({pct:.0f}%)")
        _TICK_STATE[label] = (done, now)
        if done == total:
            _TICK_STATE.pop(label, None)
