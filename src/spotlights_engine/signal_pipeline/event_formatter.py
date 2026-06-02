"""Per-stage formatter for live signal-pipeline progress lines.

Wraps a sink (default `print`) and applies three transforms to the
one-line summaries `summarize_event` produces in `_subprocess_util.py`:

1. **Stage-id prefix**: `[01] +12.3s  Bash: ...` — multi-stage runs stay
   scannable when several stages share one terminal.
2. **Burst dedup**: identical tool calls inside a short window collapse
   to a single line with `(×N)` suffix (the agent often fires five
   `Glob` calls in <1 s).
3. **Terminal-event rewrite**: the trailing `+...s  RESULT subtype=...`
   line becomes `[01] DONE in 6:57, 36 turns, $3.10` (or `ERROR ...`).

The formatter is normally invoked from one subprocess reader thread at a
time (one formatter per stage; fan-out stages run one item at a time
today), but a `threading.Lock` guards the pending-burst state defensively
so future parallel fan-out doesn't race. The runner calls `flush()` after
each stage so the last buffered burst is emitted; fan-out callers can
also call `flush()` between items to prevent dedup from bleeding across
candidates.
"""

from __future__ import annotations

import re
import threading
import time
from dataclasses import dataclass
from typing import Callable


_BURST_WINDOW_S = 1.5
_MAX_LINE = 120

# `_summarize_event` formats elapsed as `+{:6.1f}s` (right-justified, 6 chars).
_ELAPSED_RE = re.compile(r"^\+\s*[\d.]+s\s+")
_RESULT_RE = re.compile(
    r"^\+\s*([\d.]+)s\s+RESULT subtype=(\S+) turns=(\S+)(?: cost=\$([\d.]+))?"
)


def _fmt_duration(seconds: float) -> str:
    seconds = max(0.0, float(seconds))
    if seconds < 60:
        return f"{seconds:.1f}s"
    m, s = divmod(int(round(seconds)), 60)
    return f"{m}:{s:02d}"


@dataclass
class _Pending:
    line: str
    sig: str
    count: int
    last_t: float


class StageEventFormatter:
    """Stateful sink for a single stage's stream-json summaries.

    Pass an instance as the `on_event` argument to `run_claude` /
    `run_extraction`. Call `flush()` once the stage's reader thread
    has joined so any pending burst gets emitted.
    """

    def __init__(
        self,
        stage_id: str,
        sink: Callable[[str], None] | None = None,
    ) -> None:
        self.stage_id = stage_id
        self.sink: Callable[[str], None] = (
            sink if sink is not None else lambda s: print(s, flush=True)
        )
        self._pending: _Pending | None = None
        self._lock = threading.Lock()

    def __call__(self, line: str) -> None:
        m = _RESULT_RE.match(line)
        if m:
            with self._lock:
                self._flush_pending_locked()
            elapsed_s, subtype, turns, cost = m.groups()
            verb = "DONE" if subtype == "success" else subtype.upper()
            tail = f"{verb} in {_fmt_duration(float(elapsed_s))}, {turns} turns"
            if cost:
                tail += f", ${cost}"
            self._emit(tail)
            return

        sig = _ELAPSED_RE.sub("", line, count=1)
        now = time.monotonic()
        with self._lock:
            p = self._pending
            if p is not None and p.sig == sig and (now - p.last_t) <= _BURST_WINDOW_S:
                p.count += 1
                p.last_t = now
                return
            self._flush_pending_locked()
            self._pending = _Pending(line=line, sig=sig, count=1, last_t=now)

    def flush(self) -> None:
        with self._lock:
            self._flush_pending_locked()

    def _flush_pending_locked(self) -> None:
        """Emit and clear pending burst. Caller must hold self._lock."""
        p = self._pending
        if p is None:
            return
        self._pending = None
        line = p.line if p.count == 1 else f"{p.line} (×{p.count})"
        # _emit may call into user-supplied sink which we don't want to hold
        # the lock during; but we already swapped _pending to None, so it's
        # safe to release. Keep simple: emit while holding the lock.
        self._emit(line)

    def _emit(self, body: str) -> None:
        line = f"[{self.stage_id}] {body}"
        if len(line) > _MAX_LINE:
            line = line[: _MAX_LINE - 1] + "…"
        try:
            self.sink(line)
        except Exception:  # noqa: BLE001 — UI errors must not abort the run
            pass


__all__ = ["StageEventFormatter"]
