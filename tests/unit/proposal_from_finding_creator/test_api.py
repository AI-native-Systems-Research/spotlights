"""End-to-end (with fake runner) tests for `create_proposals_with_telemetry`."""

from __future__ import annotations

from pathlib import Path

import pytest

from spotlights_engine.proposal_from_finding_creator import (
    ProposalFromFindingConfig,
    create_proposals,
    create_proposals_with_telemetry,
)
from spotlights_engine.proposal_from_finding_creator.claude_exec import PairRunResult
from spotlights_engine.utils.schema_compat import proposals_from
from tests.unit.proposal_from_finding_creator._fakes import (
    fake_runner_factory,
    make_input,
    make_proposal_payload,
)


@pytest.fixture
def repo(tmp_path: Path) -> Path:
    r = tmp_path / "repo"
    r.mkdir()
    return r


@pytest.fixture
def artifacts(tmp_path: Path) -> Path:
    a = tmp_path / "artifacts"
    a.mkdir()
    return a


def test_happy_path_one_proposal_per_pair(repo: Path, artifacts: Path) -> None:
    inp = make_input(n_candidates=1, n_findings=1)
    runner = fake_runner_factory(
        payloads={
            "cand-v1_kv_offload-0001__find-v1_kv_offload-0001": make_proposal_payload(finding_id="find-v1_kv_offload-0001")
        }
    )
    cfg = ProposalFromFindingConfig(repo_path=repo, artifacts_dir=artifacts)
    result = create_proposals_with_telemetry(inp, config=cfg, runner=runner)

    assert len(result.output.candidates.candidates) == 1
    c = result.output.candidates.candidates[0]
    proposals = proposals_from(c, "research_finding")
    assert len(proposals) == 1
    p = proposals[0]
    assert p.finding_ref_id == "find-v1_kv_offload-0001"
    assert p.id == "prop-v1_kv_offload-0001"
    assert p.description == "A detailed plan"
    assert p.rationale == "Because of Y"
    assert p.author == "proposal_from_finding_creator"
    assert result.output.issues == []
    assert "cand-v1_kv_offload-0001__find-v1_kv_offload-0001" in result.per_pair_durations_s


def test_zero_proposals_keeps_candidate_advanced(repo: Path, artifacts: Path) -> None:
    inp = make_input(n_candidates=1, n_findings=1)
    runner = fake_runner_factory(payloads={"cand-v1_kv_offload-0001__find-v1_kv_offload-0001": []})
    cfg = ProposalFromFindingConfig(repo_path=repo, artifacts_dir=artifacts)
    out = create_proposals(inp, config=cfg, runner=runner)

    c = out.candidates.candidates[0]
    assert proposals_from(c, "research_finding") == []
    assert out.issues == []


def test_multi_candidate_multi_finding_each_pair_runs(
    repo: Path, artifacts: Path
) -> None:
    inp = make_input(n_candidates=2, n_findings=2)
    runner = fake_runner_factory(
        payloads={
            "cand-v1_kv_offload-0001__find-v1_kv_offload-0001": make_proposal_payload(finding_id="find-v1_kv_offload-0001"),
            "cand-v1_kv_offload-0001__find-v1_kv_offload-0002": make_proposal_payload(finding_id="find-v1_kv_offload-0002"),
            "cand-v1_kv_offload-0002__find-v1_kv_offload-0001": make_proposal_payload(finding_id="find-v1_kv_offload-0001"),
            "cand-v1_kv_offload-0002__find-v1_kv_offload-0002": make_proposal_payload(finding_id="find-v1_kv_offload-0002"),
        }
    )
    cfg = ProposalFromFindingConfig(repo_path=repo, artifacts_dir=artifacts)
    out = create_proposals(inp, config=cfg, runner=runner)

    for c in out.candidates.candidates:
        assert len(proposals_from(c, "research_finding")) == 2
    assert out.issues == []


def test_runner_failure_becomes_recoverable_issue(
    repo: Path, artifacts: Path
) -> None:
    inp = make_input(n_candidates=1, n_findings=1)
    runner = fake_runner_factory(
        errors={"cand-v1_kv_offload-0001__find-v1_kv_offload-0001": "claude exit=137: oom"}
    )
    cfg = ProposalFromFindingConfig(repo_path=repo, artifacts_dir=artifacts)
    out = create_proposals(inp, config=cfg, runner=runner)

    c = out.candidates.candidates[0]
    assert proposals_from(c, "research_finding") == []
    assert any(
        iss.recoverable
        and "cand-v1_kv_offload-0001" in iss.message
        and "find-v1_kv_offload-0001" in iss.message
        for iss in out.issues
    )


def test_runner_exception_becomes_recoverable_issue_and_siblings_continue(
    repo: Path, artifacts: Path
) -> None:
    inp = make_input(n_candidates=1, n_findings=2)

    def _runner(
        *, pair_key, prompt, schema_text, repo_path, max_turns, wallclock_s,
        claude_model=None,
    ) -> PairRunResult:
        if pair_key == "cand-v1_kv_offload-0001__find-v1_kv_offload-0001":
            raise RuntimeError("transient runner crash")
        return PairRunResult(
            pair_key=pair_key,
            duration_s=0.001,
            structured_output=make_proposal_payload(finding_id="find-v1_kv_offload-0002"),
        )

    cfg = ProposalFromFindingConfig(repo_path=repo, artifacts_dir=artifacts)
    result = create_proposals_with_telemetry(inp, config=cfg, runner=_runner)

    c = result.output.candidates.candidates[0]
    assert [
        p.finding_ref_id for p in proposals_from(c, "research_finding")
    ] == ["find-v1_kv_offload-0002"]
    assert any(
        iss.recoverable
        and "RuntimeError: transient runner crash" in iss.message
        and "cand-v1_kv_offload-0001" in iss.message
        and "find-v1_kv_offload-0001" in iss.message
        for iss in result.output.issues
    )
    assert "cand-v1_kv_offload-0001__find-v1_kv_offload-0001" in result.per_pair_durations_s
    assert "cand-v1_kv_offload-0001__find-v1_kv_offload-0002" in result.per_pair_durations_s


def test_zero_findings_returns_advanced_candidates_with_no_issues(
    repo: Path, artifacts: Path
) -> None:
    inp = make_input(n_candidates=2, n_findings=0)
    runner = fake_runner_factory()
    cfg = ProposalFromFindingConfig(repo_path=repo, artifacts_dir=artifacts)
    out = create_proposals(inp, config=cfg, runner=runner)

    for c in out.candidates.candidates:
        assert proposals_from(c, "research_finding") == []
    assert out.issues == []


def test_zero_candidates_returns_empty_output(repo: Path, artifacts: Path) -> None:
    inp = make_input(n_candidates=0, n_findings=2)
    runner = fake_runner_factory()
    cfg = ProposalFromFindingConfig(repo_path=repo, artifacts_dir=artifacts)
    out = create_proposals(inp, config=cfg, runner=runner)

    assert out.candidates.candidates == []
    assert out.issues == []


def test_per_pair_debug_files_written_on_success(
    repo: Path, artifacts: Path
) -> None:
    inp = make_input(n_candidates=1, n_findings=1)
    runner = fake_runner_factory(
        payloads={
            "cand-v1_kv_offload-0001__find-v1_kv_offload-0001": make_proposal_payload(finding_id="find-v1_kv_offload-0001")
        }
    )
    cfg = ProposalFromFindingConfig(repo_path=repo, artifacts_dir=artifacts)
    create_proposals(inp, config=cfg, runner=runner)

    dbg_dir = artifacts / "proposal_from_finding_creator.last_messages"
    assert (dbg_dir / "cand-v1_kv_offload-0001__find-v1_kv_offload-0001.json").exists()
