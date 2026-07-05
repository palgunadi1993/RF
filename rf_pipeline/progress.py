"""Cross-stage progress tracking — "where am I standing in the pipeline?".

The pipeline is long: on real Dieng data a full run (Stages 1–9) is hours of
receiver-function deconvolution, noise cross-correlation and Bayesian inversion.
This module gives the operator a single, always-current answer to *which stage is
done, which is running, and how long each took*, without scrolling the log.

It does that with one JSON file, ``logs/progress.json`` (the source of truth), and
a rendered ``logs/progress.md`` mirror for at-a-glance viewing. Every stage flips
its own record to ``running`` on entry and to ``done``/``failed`` on exit, stamping
wall-clock start/end and a short note (usually the output path). Because the state
lives on disk, a *second* shell can print the current standing at any moment:

    python run_pipeline.py --status            # print the table and exit

The tracker is deliberately best-effort: a failure to write the status file is
logged and swallowed, never propagated, so progress bookkeeping can never be the
thing that kills a scientific run.
"""
from __future__ import annotations

import json
import time
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path

from .logging_setup import get_logger

LOG = get_logger("rf.progress")

# Canonical stage order + human labels, shared by the orchestrator and by every
# thin run_*.py entrypoint so both write into the same nine (+1) slots. Keyed by
# the same short keys the orchestrator's --stages flag accepts.
STAGE_META: dict[str, str] = {
    "prep":       "Stage 1 · data prep + catalogs",
    "rf":         "Stage 2 · receiver functions",
    "hk":         "Stage 3 · H-kappa stacking",
    "ccp":        "Stage 4 · CCP imaging",
    "ant":        "Stage 5 · ambient-noise cross-correlation",
    "dispersion": "Stage 6 · dispersion measurement",
    "tomo":       "Stage 7 · per-station dispersion curves",
    "dsurftomo":  "Stage 7-alt · DSurfTomo 3-D ANT inversion",
    "inversion":  "Stage 8 · joint RF+SWD inversion",
    "synthesis":  "Stage 9 · synthesis & figures",
}

# Terminal status glyphs (fall back cleanly on ASCII-only terminals via the label).
_ICON = {
    "pending": "·", "running": "▶", "done": "✔", "failed": "✗", "skipped": "–",
}


def _now_ts() -> float:
    return time.time()


def _iso(ts: float | None) -> str | None:
    return None if ts is None else datetime.fromtimestamp(ts).strftime("%Y-%m-%d %H:%M:%S")


def _fmt_dur(seconds: float | None) -> str:
    """Human duration: ``1h04m``, ``4m12s``, ``9.3s``, ``—`` if unknown."""
    if seconds is None:
        return "—"
    s = float(seconds)
    if s < 60:
        return f"{s:.1f}s"
    if s < 3600:
        return f"{int(s // 60)}m{int(s % 60):02d}s"
    return f"{int(s // 3600)}h{int((s % 3600) // 60):02d}m"


class ProgressTracker:
    """Reads/writes the shared ``logs/progress.json`` and renders the standing.

    One tracker instance owns one status file. Use :meth:`for_config` to get the
    tracker rooted at the project's ``logs/`` dir, then wrap each stage in
    :meth:`stage` (a context manager). The file is re-read before every mutation
    so concurrent single-stage runs in separate shells don't clobber each other.
    """

    def __init__(self, status_path: str | Path):
        self.path = Path(status_path)

    # -- construction ------------------------------------------------------
    @classmethod
    def for_config(cls, cfg: dict) -> "ProgressTracker":
        root = Path(cfg.get("_project_root", "."))
        return cls(root / "logs" / "progress.json")

    # -- persistence -------------------------------------------------------
    def _load(self) -> dict:
        try:
            with open(self.path) as f:
                data = json.load(f)
            if isinstance(data, dict) and isinstance(data.get("stages"), dict):
                return data
        except (FileNotFoundError, json.JSONDecodeError, OSError):
            pass
        return {"stages": {}}

    def _save(self, data: dict) -> None:
        data["updated"] = _iso(_now_ts())
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            tmp = self.path.with_suffix(".json.tmp")
            with open(tmp, "w") as f:
                json.dump(data, f, indent=2, sort_keys=False)
            tmp.replace(self.path)                      # atomic on POSIX
            self._write_markdown(data)
        except OSError as e:                            # never kill a run over status I/O
            LOG.debug(f"could not persist progress ({e!r})")

    def _update_stage(self, key: str, **fields) -> dict:
        data = self._load()
        rec = data["stages"].get(key, {})
        rec.setdefault("label", STAGE_META.get(key, key))
        rec.update(fields)
        data["stages"][key] = rec
        self._save(data)
        return rec

    # -- the stage lifecycle ----------------------------------------------
    @contextmanager
    def stage(self, key: str, position: int | None = None, total: int | None = None):
        """Time and record one stage; mark it running → done/failed.

        ``position``/``total`` (e.g. 3 of 7) are used only for the banner the
        orchestrator prints; single-stage runs may omit them.
        """
        label = STAGE_META.get(key, key)
        where = f" [{position}/{total}]" if position and total else ""
        start = _now_ts()
        self._update_stage(key, status="running", start=_iso(start), start_ts=start,
                           end=None, end_ts=None, duration_s=None, error=None, note=None)
        LOG.info(f"┏━ {_ICON['running']} {label}{where} — running")
        try:
            yield self
        except BaseException as e:  # noqa: BLE001 — record then re-raise unchanged
            dur = _now_ts() - start
            self._update_stage(key, status="failed", end=_iso(_now_ts()),
                               end_ts=_now_ts(), duration_s=dur,
                               error=f"{type(e).__name__}: {e}")
            LOG.error(f"┗━ {_ICON['failed']} {label} — FAILED after {_fmt_dur(dur)}: {e}")
            raise
        else:
            dur = _now_ts() - start
            self._update_stage(key, status="done", end=_iso(_now_ts()),
                               end_ts=_now_ts(), duration_s=dur)
            LOG.info(f"┗━ {_ICON['done']} {label} — done in {_fmt_dur(dur)}")

    def note(self, key: str, message: str) -> None:
        """Attach/append a short human note to a stage (e.g. an output count)."""
        self._update_stage(key, note=str(message))

    def mark_skipped(self, key: str, reason: str = "") -> None:
        self._update_stage(key, status="skipped", note=reason or "skipped",
                           start=None, end=None, duration_s=None)

    def is_done(self, key: str) -> bool:
        return self._load()["stages"].get(key, {}).get("status") == "done"

    # -- rendering ---------------------------------------------------------
    def render(self, selected: list[str] | None = None) -> str:
        """A monospace table of the current standing, newest state on disk."""
        data = self._load()
        keys = [k for k in STAGE_META if (selected is None or k in selected)]
        rows = []
        for i, k in enumerate(keys, 1):
            rec = data["stages"].get(k, {})
            status = rec.get("status", "pending")
            icon = _ICON.get(status, "?")
            note = rec.get("note") or (rec.get("error") if status == "failed" else "") or ""
            rows.append((f"{i}", icon, status, STAGE_META[k],
                         _fmt_dur(rec.get("duration_s")), note))
        wlabel = max((len(r[3]) for r in rows), default=5)
        wstat = max((len(r[2]) for r in rows), default=7)
        out = [f"  Pipeline progress   (updated {data.get('updated', '—')})", ""]
        for idx, icon, status, label, dur, note in rows:
            line = (f"  {idx:>2}. {icon} {label:<{wlabel}}  "
                    f"{status:<{wstat}}  {dur:>7}")
            if note:
                line += f"  {note}"
            out.append(line)
        done = sum(1 for k in keys if data["stages"].get(k, {}).get("status") == "done")
        out += ["", f"  {done}/{len(keys)} stages complete."]
        return "\n".join(out)

    def _write_markdown(self, data: dict) -> None:
        """Mirror the JSON as a readable ``logs/progress.md`` (best-effort)."""
        lines = ["# Pipeline progress", "",
                 f"_Updated {data.get('updated', '—')}_", "",
                 "| # | Stage | Status | Duration | Note |",
                 "|---|-------|--------|----------|------|"]
        for i, k in enumerate(STAGE_META, 1):
            rec = data["stages"].get(k, {})
            status = rec.get("status", "pending")
            note = rec.get("note") or (rec.get("error") if status == "failed" else "") or ""
            lines.append(f"| {i} | {STAGE_META[k]} | {_ICON.get(status, '?')} {status} "
                         f"| {_fmt_dur(rec.get('duration_s'))} | {note} |")
        try:
            (self.path.parent / "progress.md").write_text("\n".join(lines) + "\n")
        except OSError:
            pass


def run_stage(cfg: dict, key: str, fn, position: int | None = None,
              total: int | None = None):
    """Run one stage's ``fn(cfg)`` under the shared tracker, recording its result.

    Used by both the orchestrator and the thin ``run_*.py`` entrypoints so a
    single-stage run updates the same ``logs/progress.json`` as a full run. The
    stage's return value (usually its output ``Path``) is captured as the note.
    """
    tracker = ProgressTracker.for_config(cfg)
    with tracker.stage(key, position=position, total=total):
        result = fn(cfg)
        try:
            if result is not None:
                tracker.note(key, f"-> {result}")
        except Exception:  # noqa: BLE001 — note is cosmetic
            pass
        return result


def print_status(cfg: dict, selected: list[str] | None = None) -> None:
    """Print the current standing to stdout (for ``--status``)."""
    print(ProgressTracker.for_config(cfg).render(selected))
