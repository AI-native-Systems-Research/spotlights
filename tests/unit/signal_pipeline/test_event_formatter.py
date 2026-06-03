"""Tests for `signal_pipeline.event_formatter.StageEventFormatter`.

Covers stage-id prefix, burst dedup behavior, RESULT-line rewrite, and
the explicit `flush()` boundary used between fan-out items.
"""

from __future__ import annotations

import threading
import time

import pytest

from spotlights_engine.signal_pipeline.event_formatter import StageEventFormatter


pytestmark = pytest.mark.no_stub_stages


def _ev(elapsed: float, body: str) -> str:
    """Mimic the format `_subprocess_util.summarize_event` produces."""
    return f"+{elapsed:6.1f}s  {body}"


@pytest.fixture
def captured() -> list[str]:
    return []


@pytest.fixture
def fmt(captured: list[str]) -> StageEventFormatter:
    return StageEventFormatter(stage_id="02", sink=captured.append)


def test_prefixes_stage_id(fmt: StageEventFormatter, captured: list[str]) -> None:
    fmt(_ev(1.0, "Bash: List repo"))
    fmt.flush()
    assert captured == ["[02] +   1.0s  Bash: List repo"]


def test_burst_dedup_collapses_identical_lines(
    fmt: StageEventFormatter, captured: list[str]
) -> None:
    """Five identical Glob calls within the burst window collapse to (×5)."""
    for i in range(5):
        # Different elapsed timestamps but same body — formatter strips elapsed
        # before signature comparison so all 5 are siblings of one burst.
        fmt(_ev(1.0 + i * 0.1, "Glob: vllm/**/*.py"))
    fmt.flush()
    assert len(captured) == 1
    line = captured[0]
    assert "Glob: vllm/**/*.py" in line
    assert line.endswith("(×5)")


def test_burst_resets_when_signature_changes(
    fmt: StageEventFormatter, captured: list[str]
) -> None:
    """Different bodies don't dedup; each gets its own line."""
    fmt(_ev(1.0, "Bash: List vllm/"))
    fmt(_ev(1.1, "Bash: List vllm/v1/"))
    fmt(_ev(1.2, "Bash: List vllm/v1/engine/"))
    fmt.flush()
    assert len(captured) == 3


def test_burst_window_expires(
    fmt: StageEventFormatter,
    captured: list[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A repeat past the burst window emits fresh, doesn't dedup."""
    clock = [100.0]

    def fake_monotonic() -> float:
        return clock[0]

    monkeypatch.setattr(
        "spotlights_engine.signal_pipeline.event_formatter.time.monotonic",
        fake_monotonic,
    )

    fmt(_ev(1.0, "Glob: *.py"))
    clock[0] += 5.0  # well past _BURST_WINDOW_S = 1.5
    fmt(_ev(6.0, "Glob: *.py"))
    fmt.flush()
    assert len(captured) == 2


def test_result_line_rewrite_success(
    fmt: StageEventFormatter, captured: list[str]
) -> None:
    """Success RESULT becomes `[NN] DONE in MM:SS, N turns, $X.YYY`."""
    fmt("+ 411.5s  RESULT subtype=success turns=27 cost=$3.025")
    assert len(captured) == 1
    assert captured[0] == "[02] DONE in 6:52, 27 turns, $3.025"


def test_result_line_rewrite_with_cost_omitted(
    fmt: StageEventFormatter, captured: list[str]
) -> None:
    fmt("+  10.0s  RESULT subtype=success turns=4")
    assert captured == ["[02] DONE in 10.0s, 4 turns"]


def test_result_line_rewrite_error(
    fmt: StageEventFormatter, captured: list[str]
) -> None:
    """Non-success subtypes surface as upper-cased verb."""
    fmt("+ 27.0s  RESULT subtype=error_max_turns turns=60 cost=$1.398")
    assert captured == ["[02] ERROR_MAX_TURNS in 27.0s, 60 turns, $1.398"]


def test_result_flushes_pending_burst(
    fmt: StageEventFormatter, captured: list[str]
) -> None:
    """A pending burst is flushed before the RESULT line is emitted."""
    fmt(_ev(1.0, "Glob: *.py"))
    fmt(_ev(1.1, "Glob: *.py"))
    fmt("+  2.0s  RESULT subtype=success turns=3 cost=$0.10")
    assert len(captured) == 2
    assert captured[0].endswith("(×2)")
    assert "DONE" in captured[1]


def test_explicit_flush_emits_pending(
    fmt: StageEventFormatter, captured: list[str]
) -> None:
    """flush() between fan-out items prevents cross-item dedup bleed."""
    fmt(_ev(1.0, "Bash: read file"))
    assert captured == []  # still pending
    fmt.flush()
    assert len(captured) == 1


def test_flush_between_items_blocks_cross_item_dedup(
    fmt: StageEventFormatter, captured: list[str]
) -> None:
    """If runner flushes between items, identical first events of two items
    must NOT collapse into a single (×2) line.
    """
    # Item 1 emits, then runner flushes
    fmt(_ev(1.0, "Read: foo.py"))
    fmt.flush()
    # Item 2 emits the same body
    fmt(_ev(2.0, "Read: foo.py"))
    fmt.flush()
    assert len(captured) == 2
    assert "(×" not in captured[0]
    assert "(×" not in captured[1]


def test_long_lines_truncated(
    fmt: StageEventFormatter, captured: list[str]
) -> None:
    long = "x" * 200
    fmt(_ev(1.0, f"text: {long}"))
    fmt.flush()
    assert len(captured) == 1
    assert len(captured[0]) <= 120
    assert captured[0].endswith("…")


def test_sink_exception_doesnt_propagate(captured: list[str]) -> None:
    """A broken sink must not abort the run — exceptions are swallowed."""
    calls = []

    def boom(_line: str) -> None:
        calls.append(1)
        raise RuntimeError("UI is broken")

    fmt = StageEventFormatter("03", sink=boom)
    fmt(_ev(1.0, "Bash: anything"))
    fmt.flush()  # would emit, sink raises, no exception propagates
    assert calls  # sink WAS called


def test_lock_serializes_concurrent_calls(
    fmt: StageEventFormatter, captured: list[str]
) -> None:
    """Defense-in-depth: with the lock, concurrent calls don't corrupt
    the pending-burst state. Smoke test: 10 threads × 100 emits, no exception
    and no garbled lines.
    """

    def worker(i: int) -> None:
        for j in range(100):
            fmt(_ev(0.1 * j, f"Bash: worker-{i}"))

    threads = [threading.Thread(target=worker, args=(i,)) for i in range(10)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    fmt.flush()
    # 10 workers × 100 emits = 1000 logical events. Each "Bash: worker-N" body
    # was emitted 100× consecutively per worker, so they collapse into burst
    # groups. Exact count is timing-dependent; we just check the formatter
    # didn't crash and produced *some* output, no malformed lines.
    assert captured
    assert all(line.startswith("[02] ") for line in captured)
