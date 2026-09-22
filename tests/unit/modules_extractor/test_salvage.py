"""Unit tests for `claude_stage.salvage_structured_payload`.

Local models (e.g. `Qwen/Qwen3.8-27B`) intermittently type the structured
answer as assistant *text* — ``StructuredOutput({...})`` or bare JSON — instead
of invoking the StructuredOutput tool, then loop against the Stop hook until
`max_turns`. The terminal event then carries no `structured_output`, so the run
hard-fails even though the model produced the answer. `salvage_structured_payload`
reconstructs the payload from the text stream.

The event shapes and payload values below are lifted verbatim from a real
`Qwen/Qwen3.8-27B` run over the `hrl_ocr/models/fonts_recognition` shard, so the
fixture stays faithful to what the model actually emits (fields split across
separate turns; ``StructuredOutput(...)`` wrapping; interleaved refusals).
"""

from __future__ import annotations

import json

import pytest

from spotlights_engine.modules_extractor import claude_stage
from spotlights_engine.modules_extractor.claude_stage import (
    salvage_structured_payload,
)
from spotlights_engine.modules_extractor.errors import ExtractorAgentError
from spotlights_engine.modules_extractor.stage_schemas import AssignmentTree
from spotlights_engine.signal_pipeline._subprocess_util import StreamingResult

# ── Real payload fragments captured from the live run ─────────────────────

_REAL_ASSIGNMENTS = {
    "hrl_ocr/models/fonts_recognition": "MODULE",
    "hrl_ocr/models/fonts_recognition/basic_module": "PART",
}
_REAL_MODULE_DECISIONS = {
    "hrl_ocr/models/fonts_recognition": {
        "keep_reason": (
            "Provides a self-contained fonts recognition feature within the "
            "hrl_ocr models package"
        )
    },
}


def _assistant_text(text: str) -> dict:
    return {"type": "assistant", "message": {"content": [{"type": "text", "text": text}]}}


def _stream(*events: dict) -> bytes:
    return "".join(json.dumps(e) + "\n" for e in events).encode("utf-8")


_TERMINAL_MAX_TURNS = {
    "type": "result",
    "subtype": "error_max_turns",
    "is_error": True,
    "result": "Reached max turns",
}


# ── The core bug: fields split across turns, wrapped in StructuredOutput(...) ─


def test_salvage_merges_fields_split_across_turns() -> None:
    """The observed loop: module_decisions in one turn, assignments in another,
    each typed as ``StructuredOutput({...})`` text, refusals interleaved."""
    stream = _stream(
        {"type": "system", "subtype": "init"},
        _assistant_text(
            "StructuredOutput("
            + json.dumps({"module_decisions": _REAL_MODULE_DECISIONS})
            + ")"
        ),
        _assistant_text("I cannot comply."),
        _assistant_text(
            "StructuredOutput("
            + json.dumps({"assignments": _REAL_ASSIGNMENTS})
            + ")"
        ),
        _assistant_text("We are stuck in an infinite loop."),
        _TERMINAL_MAX_TURNS,
    )
    payload = salvage_structured_payload(stream, AssignmentTree)
    assert payload is not None
    tree = AssignmentTree.model_validate_json(payload)
    assert tree.assignments == _REAL_ASSIGNMENTS
    assert set(tree.module_decisions) == set(_REAL_MODULE_DECISIONS)


def test_salvage_from_single_text_event_with_both_fields() -> None:
    """The model sometimes types the whole payload (both fields) as one bare
    JSON text block — verbatim from the real capture."""
    both = {"assignments": _REAL_ASSIGNMENTS, "module_decisions": _REAL_MODULE_DECISIONS}
    stream = _stream(
        _assistant_text(json.dumps(both, indent=2)),
        _TERMINAL_MAX_TURNS,
    )
    payload = salvage_structured_payload(stream, AssignmentTree)
    assert payload is not None
    AssignmentTree.model_validate_json(payload)


def test_salvage_latest_value_wins_per_field() -> None:
    """When the model re-emits a field, the last value is kept (its final
    intent), not the first."""
    stale = {"hrl_ocr/models/fonts_recognition": "PART"}
    stream = _stream(
        _assistant_text(json.dumps({"assignments": stale})),
        _assistant_text(json.dumps({"module_decisions": _REAL_MODULE_DECISIONS})),
        _assistant_text(json.dumps({"assignments": _REAL_ASSIGNMENTS})),
        _TERMINAL_MAX_TURNS,
    )
    payload = salvage_structured_payload(stream, AssignmentTree)
    assert payload is not None
    assert json.loads(payload)["assignments"] == _REAL_ASSIGNMENTS


# ── Negatives: salvage must refuse rather than fabricate ──────────────────


def test_partial_only_one_required_field_is_unsalvageable() -> None:
    """Only assignments ever emitted — module_decisions missing -> None, so the
    caller fails loudly instead of shipping an incomplete tree."""
    stream = _stream(
        _assistant_text(json.dumps({"assignments": _REAL_ASSIGNMENTS})),
        _assistant_text("I cannot comply."),
        _TERMINAL_MAX_TURNS,
    )
    assert salvage_structured_payload(stream, AssignmentTree) is None


def test_no_json_in_text_is_unsalvageable() -> None:
    stream = _stream(
        _assistant_text("We are stuck in an infinite loop. I cannot comply."),
        _TERMINAL_MAX_TURNS,
    )
    assert salvage_structured_payload(stream, AssignmentTree) is None


def test_example_json_without_required_keys_is_ignored() -> None:
    """Stray JSON in reasoning text that lacks the required keys must not be
    mistaken for the payload."""
    stream = _stream(
        _assistant_text('Here is the schema shape: {"foo": 1, "bar": [2, 3]}'),
        _TERMINAL_MAX_TURNS,
    )
    assert salvage_structured_payload(stream, AssignmentTree) is None


def test_only_terminal_and_tool_events_no_assistant_text() -> None:
    """A stream with no assistant text (only system/result) yields nothing."""
    stream = _stream({"type": "system", "subtype": "init"}, _TERMINAL_MAX_TURNS)
    assert salvage_structured_payload(stream, AssignmentTree) is None


def test_empty_and_garbage_streams() -> None:
    assert salvage_structured_payload(b"", AssignmentTree) is None
    assert salvage_structured_payload(b"not json\n[1,2]\n", AssignmentTree) is None


def test_nested_braces_and_strings_do_not_break_extraction() -> None:
    """Braces inside JSON string values must not confuse the brace matcher."""
    tricky = {
        "assignments": _REAL_ASSIGNMENTS,
        "module_decisions": {
            "hrl_ocr/models/fonts_recognition": {
                "keep_reason": "handles {curly} and \"quoted\" text; nested {a:{b}}"
            }
        },
    }
    stream = _stream(_assistant_text(json.dumps(tricky)), _TERMINAL_MAX_TURNS)
    payload = salvage_structured_payload(stream, AssignmentTree)
    assert payload is not None
    tree = AssignmentTree.model_validate_json(payload)
    assert "curly" in next(iter(tree.module_decisions.values())).keep_reason


# ── Caller integration: run_structured_claude_stage salvage gate ──────────
#
# No subprocess: run_streaming_claude is stubbed to return a fabricated loop
# stream. These lock the policy that salvage recovers a non-API terminal loop
# but never masks an API-transport failure (which must stay retryable), and is
# off by default.


def _loop_stream() -> bytes:
    return _stream(
        {"type": "system", "subtype": "init"},
        _assistant_text(
            "StructuredOutput("
            + json.dumps({"module_decisions": _REAL_MODULE_DECISIONS})
            + ")"
        ),
        _assistant_text("I cannot comply."),
        _assistant_text(
            "StructuredOutput(" + json.dumps({"assignments": _REAL_ASSIGNMENTS}) + ")"
        ),
        _TERMINAL_MAX_TURNS,
    )


def _stub_stream(monkeypatch, stdout: bytes, *, returncode: int = 0) -> None:
    monkeypatch.setattr(claude_stage, "resolve_and_check", lambda **kw: ["/bin/claude"])
    monkeypatch.setattr(claude_stage, "build_schema_text", lambda t: "{}")

    def fake_stream(**kwargs):
        return StreamingResult(
            stdout=stdout, stderr=b"", returncode=returncode, duration_s=1.0
        )

    monkeypatch.setattr(claude_stage, "run_streaming_claude", fake_stream)


def _run(tmp_path, **overrides):
    kwargs = dict(
        output_type=AssignmentTree,
        repo_path=tmp_path,
        prompt="p",
        stage_name="assign:fonts",
        attempt_dir=None,
        max_turns=8,
        timeout_s=5,
    )
    kwargs.update(overrides)
    return claude_stage.run_structured_claude_stage(**kwargs)


def test_caller_salvages_non_api_terminal_loop(tmp_path, monkeypatch) -> None:
    _stub_stream(monkeypatch, _loop_stream())
    result = _run(tmp_path, salvage=True)
    assert result.parsed is not None
    assert result.parsed.assignments == _REAL_ASSIGNMENTS


def test_caller_does_not_salvage_when_disabled(tmp_path, monkeypatch) -> None:
    _stub_stream(monkeypatch, _loop_stream())
    with pytest.raises(ExtractorAgentError):
        _run(tmp_path, salvage=False)


def test_caller_never_salvages_api_terminal(tmp_path, monkeypatch) -> None:
    """A rate-limit terminal must stay a retryable ExtractorAgentError even
    though the stream also carries a salvageable payload."""
    api_terminal = {
        "type": "result",
        "subtype": "success",
        "is_error": True,
        "result": "API Error: Request rejected (429) · Rate limit exceeded",
    }
    stream = _stream(
        _assistant_text(
            "StructuredOutput("
            + json.dumps(
                {
                    "assignments": _REAL_ASSIGNMENTS,
                    "module_decisions": _REAL_MODULE_DECISIONS,
                }
            )
            + ")"
        ),
        api_terminal,
    )
    _stub_stream(monkeypatch, stream)
    with pytest.raises(ExtractorAgentError) as excinfo:
        _run(tmp_path, salvage=True)
    assert excinfo.value.context.get("api_failure") is not None
