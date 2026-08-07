"""Candidate-mode step 3: per-candidate surveys, ids, tagging, attribution."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from spotlights_engine.candidate_deep_research import (
    research_candidates,
    research_candidates_with_telemetry,
)
from spotlights_engine.candidate_deep_research.api import (
    _candidate_options,
    _candidate_segment,
)
from spotlights_engine.module_deep_research.codex_exec import CodexExecOptions
from spotlights_engine.schemas.finding import Finding
from tests.unit.candidate_deep_research._fakes import (
    SEGMENT,
    FakeRunner,
    agent_payload,
    make_candidate,
    make_request,
    usage,
)

# per-candidate loop ----------------------------------------------------------


def test_one_survey_is_rendered_per_candidate() -> None:
    runner = FakeRunner(agent_payload("Flash attention tiling"))

    research_candidates(make_request(3), runner=runner, segment=SEGMENT)

    assert len(runner.prompts) == 3
    prompted = {
        f"cand-{SEGMENT}-{n:04d}"
        for n in (1, 2, 3)
        if any(f"cand-{SEGMENT}-{n:04d}" in p for p in runner.prompts)
    }
    assert len(prompted) == 3


def test_no_candidates_yields_empty_output_and_no_runner_calls() -> None:
    runner = FakeRunner(agent_payload("X"))

    output = research_candidates(make_request(0), runner=runner, segment=SEGMENT)

    assert output.findings == []
    assert output.issues == []
    assert runner.prompts == []


def test_missing_module_returns_unrecoverable_issue() -> None:
    runner = FakeRunner(agent_payload("X"))

    output = research_candidates(
        make_request(2, module_qualified_name="missing.module"),
        runner=runner,
        segment=SEGMENT,
    )

    assert output.findings == []
    assert len(output.issues) == 1
    assert output.issues[0].recoverable is False
    assert output.issues[0].step == "module_deep_research"
    assert runner.prompts == []


# finding ids and candidate tagging ------------------------------------------


def test_finding_ids_carry_the_per_candidate_segment() -> None:
    runner = FakeRunner(agent_payload("A", "B"))

    output = research_candidates(make_request(2), runner=runner, segment=SEGMENT)

    assert sorted(f.finding_id for f in output.findings) == [
        f"find-{SEGMENT}-0001-0001",
        f"find-{SEGMENT}-0001-0002",
        f"find-{SEGMENT}-0002-0001",
        f"find-{SEGMENT}-0002-0002",
    ]


def test_every_finding_is_tagged_with_its_candidate() -> None:
    runner = FakeRunner(agent_payload("A"))

    output = research_candidates(make_request(2), runner=runner, segment=SEGMENT)

    by_candidate = {f.candidate_id: f.finding_id for f in output.findings}
    assert by_candidate == {
        f"cand-{SEGMENT}-0001": f"find-{SEGMENT}-0001-0001",
        f"cand-{SEGMENT}-0002": f"find-{SEGMENT}-0002-0001",
    }


def test_search_queries_are_tagged_with_their_candidate() -> None:
    runner = FakeRunner(agent_payload("A", queries=("paged attention",)))

    output = research_candidates(make_request(2), runner=runner, segment=SEGMENT)

    assert len(output.search_queries) == 2
    assert {q.candidate_id for q in output.search_queries} == {
        f"cand-{SEGMENT}-0001",
        f"cand-{SEGMENT}-0002",
    }


def test_max_findings_per_candidate_caps_each_survey_independently() -> None:
    runner = FakeRunner(agent_payload("A", "B", "C", "D"))

    output = research_candidates(
        make_request(2, max_findings_per_candidate=2), runner=runner, segment=SEGMENT
    )

    assert len(output.findings) == 4  # 2 per candidate, not 2 in total
    per_candidate: dict[str | None, list[Finding]] = {}
    for f in output.findings:
        per_candidate.setdefault(f.candidate_id, []).append(f)
    assert {k: len(v) for k, v in per_candidate.items()} == {
        f"cand-{SEGMENT}-0001": 2,
        f"cand-{SEGMENT}-0002": 2,
    }


def test_dedup_is_per_candidate_so_the_same_url_survives_twice() -> None:
    # `merge_outcomes` allocates its `seen` set per call, so one call per
    # candidate means a shared source is kept once *per candidate*.
    runner = FakeRunner(agent_payload("A"))

    output = research_candidates(make_request(2), runner=runner, segment=SEGMENT)

    assert [f.url for f in output.findings] == [
        "https://example.com/1",
        "https://example.com/1",
    ]


# cross-attribution -----------------------------------------------------------


def test_distinct_per_candidate_payloads_do_not_bleed_across_candidates() -> None:
    def payload_for(prompt: str) -> str:
        if f"cand-{SEGMENT}-0001" in prompt:
            return agent_payload("first-candidate-finding")
        return agent_payload("second-candidate-finding")

    runner = FakeRunner(payload_for=payload_for)

    output = research_candidates(make_request(2), runner=runner, segment=SEGMENT)

    titles = {f.candidate_id: f.title for f in output.findings}
    assert titles == {
        f"cand-{SEGMENT}-0001": "first-candidate-finding",
        f"cand-{SEGMENT}-0002": "second-candidate-finding",
    }


def test_a_runner_failure_is_recoverable_and_scoped_to_its_candidate() -> None:
    def payload_for(prompt: str) -> str | None:
        if f"cand-{SEGMENT}-0001" in prompt:
            return None  # nonzero exit with no payload -> recoverable issue
        return agent_payload("healthy")

    runner = _FailFirstRunner(payload_for)

    output = research_candidates(make_request(2), runner=runner, segment=SEGMENT)

    assert [f.candidate_id for f in output.findings] == [f"cand-{SEGMENT}-0002"]
    assert output.issues
    assert all(i.recoverable for i in output.issues)
    assert any("exited with code 2" in i.message for i in output.issues)


class _FailFirstRunner(FakeRunner):
    """Exits nonzero (no payload) for the first candidate, succeeds otherwise."""

    def __init__(self, payload_for) -> None:
        super().__init__(payload_for=payload_for, stderr="network down")

    def run(self, prompt: str, *, check: bool = True):
        result = super().run(prompt, check=check)
        if result.final_message is None:
            return result.model_copy(update={"returncode": 2})
        return result


def test_a_post_merge_failure_degrades_only_its_candidate() -> None:
    # >9999 findings from one runner makes `_renumber_findings` mint
    # `find-<seg>-10000`, which `Finding.finding_id`'s pattern rejects. That
    # raise happens *after* the runner fan-out; it must not escape the worker
    # and fail the whole module (D7).
    overflowing = json.dumps(
        {
            "findings": [
                {
                    "finding_id": "find-0001",
                    "title": f"t{i}",
                    "url": f"https://example.com/{i}",
                    "source_type": "paper",
                    "technique_summary": "s",
                }
                for i in range(10_001)
            ],
            "issues": [],
            "search_queries": [],
        }
    )

    def payload_for(prompt: str) -> str:
        if f"cand-{SEGMENT}-0001" in prompt:
            return overflowing
        return agent_payload("healthy")

    output = research_candidates(
        make_request(2, max_findings_per_candidate=20_000),
        runner=FakeRunner(payload_for=payload_for),
        segment=SEGMENT,
    )

    assert [f.candidate_id for f in output.findings] == [f"cand-{SEGMENT}-0002"]
    assert len(output.issues) == 1
    assert output.issues[0].recoverable is True
    assert f"candidate_deep_research failed for cand-{SEGMENT}-0001" in output.issues[0].message


def test_a_duplicate_candidate_id_is_surveyed_once() -> None:
    # Two entries sharing an id derive the same finding-id segment, so the
    # module would carry two findings with the *same* `finding_id`.
    runner = FakeRunner(agent_payload("A"))
    request = make_request(1)
    duplicated = request.model_copy(update={"candidates": [make_candidate(0), make_candidate(0)]})

    output = research_candidates(duplicated, runner=runner, segment=SEGMENT)

    assert len(runner.prompts) == 1
    assert [f.finding_id for f in output.findings] == [f"find-{SEGMENT}-0001-0001"]
    assert len(output.issues) == 1
    assert output.issues[0].recoverable is True
    assert "duplicate candidate id" in output.issues[0].message


def test_candidate_id_from_another_module_is_reported_not_raised() -> None:
    request = make_request(1)
    request = request.model_copy(update={"candidates": [make_candidate(0, segment="other_module")]})

    output = research_candidates(request, runner=FakeRunner(agent_payload("A")), segment=SEGMENT)

    assert output.findings == []
    assert len(output.issues) == 1
    assert output.issues[0].recoverable is True
    assert "candidate_deep_research failed for cand-other_module-0001" in (output.issues[0].message)


# usage / duration attribution ------------------------------------------------


def test_usages_are_attributed_per_candidate_and_runner() -> None:
    codex = FakeRunner(agent_payload("A"), name="codex", usage=usage())
    claude = FakeRunner(agent_payload("B"), name="claude", usage=usage(1, 2))

    result = research_candidates_with_telemetry(
        make_request(2), runners=[codex, claude], segment=SEGMENT
    )

    assert sorted((u.candidate_id, u.cli) for u in result.usages) == [
        (f"cand-{SEGMENT}-0001", "claude"),
        (f"cand-{SEGMENT}-0001", "codex"),
        (f"cand-{SEGMENT}-0002", "claude"),
        (f"cand-{SEGMENT}-0002", "codex"),
    ]
    assert all(u.duration_s is not None and u.duration_s >= 0 for u in result.usages)


def test_runners_without_usage_contribute_no_usage_records() -> None:
    result = research_candidates_with_telemetry(
        make_request(2), runner=FakeRunner(agent_payload("A")), segment=SEGMENT
    )

    assert result.usages == []
    assert len(result.output.findings) == 2


# helpers ---------------------------------------------------------------------


def test_candidate_segment_appends_the_candidate_counter() -> None:
    assert _candidate_segment(make_candidate(11, segment="a-b"), segment="a-b") == "a-b-0012"


def test_candidate_segment_rejects_a_foreign_candidate() -> None:
    with pytest.raises(ValueError, match="expected"):
        _candidate_segment(make_candidate(0, segment="other"), segment=SEGMENT)


def test_candidate_options_give_each_candidate_a_private_last_message_file() -> None:
    base = CodexExecOptions(cwd=Path("/repo"), output_last_message=Path("/shared.md"))

    first = _candidate_options(
        base,
        repo_path=Path("/repo"),
        candidate=make_candidate(0),
        last_message_dir=Path("/artifacts/lm"),
    )
    second = _candidate_options(
        base,
        repo_path=Path("/repo"),
        candidate=make_candidate(1),
        last_message_dir=Path("/artifacts/lm"),
    )

    assert first is not None and second is not None
    assert first.output_last_message == Path(f"/artifacts/lm/cand-{SEGMENT}-0001.md")
    assert second.output_last_message == Path(f"/artifacts/lm/cand-{SEGMENT}-0002.md")
    # The caller's options object is untouched.
    assert base.output_last_message == Path("/shared.md")


def test_candidate_options_build_defaults_when_no_options_are_passed() -> None:
    options = _candidate_options(
        None,
        repo_path=Path("/repo"),
        candidate=make_candidate(0),
        last_message_dir=Path("/artifacts/lm"),
    )

    assert options is not None
    assert options.cwd == Path("/repo")
    assert options.output_last_message == Path(f"/artifacts/lm/cand-{SEGMENT}-0001.md")


def test_candidate_options_pass_through_when_there_is_nothing_to_isolate() -> None:
    # No directory and no caller-pinned file: `build_command` mints a private
    # temp file per invocation, so there is nothing to make candidate-local.
    base = CodexExecOptions(cwd=Path("/repo"))

    assert (
        _candidate_options(
            base,
            repo_path=Path("/repo"),
            candidate=make_candidate(0),
            last_message_dir=None,
        )
        is base
    )


def test_candidate_options_isolate_a_caller_pinned_last_message_file() -> None:
    # A standalone caller that pins one `output_last_message` and no directory
    # would otherwise have every concurrent candidate write/read the same file.
    base = CodexExecOptions(cwd=Path("/repo"), output_last_message=Path("/tmp/shared.md"))

    first = _candidate_options(
        base, repo_path=Path("/repo"), candidate=make_candidate(0), last_message_dir=None
    )
    second = _candidate_options(
        base, repo_path=Path("/repo"), candidate=make_candidate(1), last_message_dir=None
    )

    assert first is not None and second is not None
    assert first.output_last_message != second.output_last_message
    assert first.output_last_message == Path(f"/tmp/shared.cand-{SEGMENT}-0001.md")
    assert base.output_last_message == Path("/tmp/shared.md")
