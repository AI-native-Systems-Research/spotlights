"""Path resolution and atomic file ops for the repo bench.

Two roots, separated by purpose:

    data/repo_bench/        # CACHED INPUT (cheap to keep, expensive to rebuild)
      raw/<window_id>/{prs.jsonl, manifest.json, diffs/}
      views/<window_id>/<view_id>/{prs.jsonl, manifest.json}

    runs/repo_bench/<run_id>/   # PER-INVOCATION OUTPUT
      snapshot.json
      bench_spec.{json,md}
      OBSERVABILITY_BENCH_SPEC.md
      run_report.{json,md}
      matching/match_report.json (when match runs)

`window_id` is `YYYY-MM-DD__YYYY-MM-DD` from the start/end dates. Two scrapes
of the same window write to the same directory; output is overwritten
atomically (temp-file + rename).

Override either root via env var for tests.
"""

from __future__ import annotations

import json
import os
import tempfile
from collections.abc import Iterable
from datetime import date, datetime
from pathlib import Path

from pydantic import BaseModel

DATA_ROOT_ENV = "DISCOVERY_BENCHMARK_DATA_ROOT"
DEFAULT_DATA_ROOT = Path("data/repo_bench")

RUNS_ROOT_ENV = "DISCOVERY_BENCHMARK_RUNS_ROOT"
DEFAULT_RUNS_ROOT = Path("runs/repo_bench")


def _to_date(d: date | datetime | str) -> date:
    if isinstance(d, datetime):
        return d.date()
    if isinstance(d, date):
        return d
    return date.fromisoformat(d)


def window_id_for(start: date | datetime | str, end: date | datetime | str) -> str:
    return f"{_to_date(start).isoformat()}__{_to_date(end).isoformat()}"


def data_root() -> Path:
    """Where cached input lives (raw scrape, derived views).

    Override via env var for tests. Stable across runs — one cache for
    the whole project.
    """
    override = os.environ.get(DATA_ROOT_ENV)
    return Path(override) if override else DEFAULT_DATA_ROOT


def runs_root() -> Path:
    """Where per-invocation output lives. Override via env var for tests.

    One subdirectory per benchmark run, named by `run_id`. Distinct from
    `data_root()` so a run dir can be deleted without touching the
    cached input.
    """
    override = os.environ.get(RUNS_ROOT_ENV)
    return Path(override) if override else DEFAULT_RUNS_ROOT


def raw_dir(window_id: str, *, root: Path | None = None) -> Path:
    return (root or data_root()) / "raw" / window_id


def run_dir(run_id: str, *, root: Path | None = None) -> Path:
    """Resolve a per-run output directory under `runs_root()`."""
    return (root or runs_root()) / run_id


def atomic_write_text(path: Path, content: str) -> None:
    """Write `content` to `path` atomically: temp file in same dir, then rename.

    The same-dir constraint is what makes the rename atomic on Windows and
    POSIX both.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_str = tempfile.mkstemp(
        prefix=path.name + ".tmp-", dir=str(path.parent)
    )
    tmp = Path(tmp_str)
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as fh:
            fh.write(content)
        os.replace(tmp, path)
    except Exception:
        tmp.unlink(missing_ok=True)
        raise


def write_jsonl(path: Path, rows: Iterable[BaseModel | dict]) -> int:
    """Write pydantic models or dicts as JSONL. Returns count written."""
    count = 0
    lines: list[str] = []
    for row in rows:
        if isinstance(row, BaseModel):
            lines.append(row.model_dump_json())
        else:
            lines.append(json.dumps(row, separators=(",", ":"), default=str))
        count += 1
    atomic_write_text(path, "\n".join(lines) + ("\n" if lines else ""))
    return count


def append_jsonl(path: Path, row: BaseModel | dict) -> None:
    """Append a single row to a JSONL file. Creates parent dir if needed.

    Each call flushes + fsyncs so a crash mid-scrape leaves the file with
    only complete lines. We deliberately do NOT make this atomic — the whole
    point is to survive partial work.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    if isinstance(row, BaseModel):
        line = row.model_dump_json()
    else:
        line = json.dumps(row, separators=(",", ":"), default=str)
    with path.open("a", encoding="utf-8", newline="\n") as fh:
        fh.write(line + "\n")
        fh.flush()
        os.fsync(fh.fileno())


def read_jsonl_lenient(path: Path) -> list[dict]:
    """Read JSONL, skipping the last line if it's a torn write.

    Used by resume — if a process died mid-write, the trailing line may be
    truncated. Drop it; the missing PR will be re-fetched.
    """
    if not path.exists():
        return []
    rows: list[dict] = []
    raw = path.read_text(encoding="utf-8")
    lines = raw.splitlines()
    for i, line in enumerate(lines):
        if not line.strip():
            continue
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            # Tolerate only the final line being torn; anything earlier is
            # corruption we want to surface.
            if i == len(lines) - 1:
                break
            raise
    return rows


def write_json(path: Path, obj: BaseModel | dict) -> None:
    if isinstance(obj, BaseModel):
        text = obj.model_dump_json(indent=2)
    else:
        text = json.dumps(obj, indent=2, default=str)
    atomic_write_text(path, text + "\n")
