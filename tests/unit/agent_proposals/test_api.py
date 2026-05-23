"""End-to-end (with fake runners) tests for `create_agent_proposals_with_telemetry`."""

from __future__ import annotations

from pathlib import Path

import pytest

from spotlights_engine.agent_proposals import (
    AgentProposalsConfig,
    create_agent_proposals,
    create_agent_proposals_with_telemetry,
)
from tests.unit.agent_proposals._fakes import (
    fake_claude_runner_factory,
    fake_codex_runner_factory,
    make_empty_proposal_payload,
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


def test_happy_path_one_proposal_per_agent(repo: Path, artifacts: Path) -> None:
    inp = make_input(n_candidates=1)
    claude_runner = fake_claude_runner_factory(
        payloads={"cand-0001": make_proposal_payload(agent_name="claude")}
    )
    codex_runner = fake_codex_runner_factory(
        payloads={"cand-0001": make_proposal_payload(agent_name="codex")}
    )
    cfg = AgentProposalsConfig(repo_path=repo, artifacts_dir=artifacts)
    result = create_agent_proposals_with_telemetry(
        inp,
        config=cfg,
        claude_runner=claude_runner,
        codex_runner=codex_runner,
    )

    cands = result.output.candidates.candidates
    assert len(cands) == 1
    c = cands[0]
    assert c.state == "AGENT_PROPOSALS_CREATED"
    assert {p.agent_name for p in c.agent_proposals} == {"claude", "codex"}
    assert result.output.issues == []
    assert "cand-0001" in result.per_candidate_durations_s
    assert "claude" in result.per_candidate_durations_s["cand-0001"]
    assert "codex" in result.per_candidate_durations_s["cand-0001"]


def test_zero_proposals_keeps_candidate_advanced(repo: Path, artifacts: Path) -> None:
    inp = make_input(n_candidates=1)
    claude_runner = fake_claude_runner_factory(
        payloads={"cand-0001": make_empty_proposal_payload()}
    )
    codex_runner = fake_codex_runner_factory(
        payloads={"cand-0001": make_empty_proposal_payload()}
    )
    cfg = AgentProposalsConfig(repo_path=repo, artifacts_dir=artifacts)
    out = create_agent_proposals(
        inp,
        config=cfg,
        claude_runner=claude_runner,
        codex_runner=codex_runner,
    )

    c = out.candidates.candidates[0]
    assert c.state == "AGENT_PROPOSALS_CREATED"
    assert c.agent_proposals == []
    assert out.issues == []


def test_runner_failure_becomes_recoverable_issue(
    repo: Path, artifacts: Path
) -> None:
    inp = make_input(n_candidates=1)
    claude_runner = fake_claude_runner_factory(
        errors={"cand-0001": "claude exit=137: oom"}
    )
    codex_runner = fake_codex_runner_factory(
        payloads={"cand-0001": make_proposal_payload(agent_name="codex")}
    )
    cfg = AgentProposalsConfig(repo_path=repo, artifacts_dir=artifacts)
    out = create_agent_proposals(
        inp,
        config=cfg,
        claude_runner=claude_runner,
        codex_runner=codex_runner,
    )

    c = out.candidates.candidates[0]
    assert c.state == "AGENT_PROPOSALS_CREATED"
    # Codex still ran successfully even though Claude failed.
    assert [p.agent_name for p in c.agent_proposals] == ["codex"]
    assert any(
        iss.recoverable
        and "cand-0001" in iss.message
        and "claude" in iss.message
        for iss in out.issues
    )


def test_runner_exception_does_not_stop_siblings(
    repo: Path, artifacts: Path
) -> None:
    inp = make_input(n_candidates=2)

    def _claude_runner(
        *,
        candidate_id: str,
        prompt,
        schema_text,
        repo_path,
        max_turns,
        wallclock_s,
    ):
        if candidate_id == "cand-0001":
            raise RuntimeError("transient runner crash")
        from spotlights_engine.agent_proposals.claude_exec import (
            CandidateAgentRunResult,
        )

        return CandidateAgentRunResult(
            candidate_id=candidate_id,
            duration_s=0.001,
            structured_output=make_proposal_payload(agent_name="claude"),
        )

    codex_runner = fake_codex_runner_factory(
        payloads={
            "cand-0001": make_proposal_payload(agent_name="codex"),
            "cand-0002": make_proposal_payload(agent_name="codex"),
        }
    )
    cfg = AgentProposalsConfig(repo_path=repo, artifacts_dir=artifacts)
    result = create_agent_proposals_with_telemetry(
        inp,
        config=cfg,
        claude_runner=_claude_runner,
        codex_runner=codex_runner,
    )

    cands = result.output.candidates.candidates
    assert len(cands) == 2
    by_id = {c.id: c for c in cands}
    # cand-0001: only codex's proposal — claude crashed
    assert [p.agent_name for p in by_id["cand-0001"].agent_proposals] == ["codex"]
    # cand-0002: both proposals
    assert {p.agent_name for p in by_id["cand-0002"].agent_proposals} == {
        "claude",
        "codex",
    }
    assert any(
        iss.recoverable
        and "RuntimeError: transient runner crash" in iss.message
        and "cand-0001" in iss.message
        and "claude" in iss.message
        for iss in result.output.issues
    )


def test_zero_candidates_returns_empty_output(repo: Path, artifacts: Path) -> None:
    inp = make_input(n_candidates=0)
    claude_runner = fake_claude_runner_factory()
    codex_runner = fake_codex_runner_factory()
    cfg = AgentProposalsConfig(repo_path=repo, artifacts_dir=artifacts)
    out = create_agent_proposals(
        inp,
        config=cfg,
        claude_runner=claude_runner,
        codex_runner=codex_runner,
    )

    assert out.candidates.candidates == []
    assert out.issues == []


def test_debug_first_n_candidates_truncates(repo: Path, artifacts: Path) -> None:
    inp = make_input(n_candidates=3)
    claude_runner = fake_claude_runner_factory(
        payloads={
            "cand-0001": make_proposal_payload(agent_name="claude"),
            "cand-0002": make_proposal_payload(agent_name="claude"),
            "cand-0003": make_proposal_payload(agent_name="claude"),
        }
    )
    codex_runner = fake_codex_runner_factory(
        payloads={
            "cand-0001": make_proposal_payload(agent_name="codex"),
            "cand-0002": make_proposal_payload(agent_name="codex"),
            "cand-0003": make_proposal_payload(agent_name="codex"),
        }
    )
    cfg = AgentProposalsConfig(
        repo_path=repo, artifacts_dir=artifacts, debug_first_n_candidates=1
    )
    result = create_agent_proposals_with_telemetry(
        inp,
        config=cfg,
        claude_runner=claude_runner,
        codex_runner=codex_runner,
    )

    cands = result.output.candidates.candidates
    assert len(cands) == 3
    assert all(c.state == "AGENT_PROPOSALS_CREATED" for c in cands)
    by_id = {c.id: c for c in cands}
    # First candidate ran agents
    assert {p.agent_name for p in by_id["cand-0001"].agent_proposals} == {
        "claude",
        "codex",
    }
    # Remaining candidates have empty agent_proposals (state still advanced)
    assert by_id["cand-0002"].agent_proposals == []
    assert by_id["cand-0003"].agent_proposals == []


def test_per_candidate_debug_files_written_on_success(
    repo: Path, artifacts: Path
) -> None:
    inp = make_input(n_candidates=1)
    claude_runner = fake_claude_runner_factory(
        payloads={"cand-0001": make_proposal_payload(agent_name="claude")}
    )
    codex_runner = fake_codex_runner_factory(
        payloads={"cand-0001": make_proposal_payload(agent_name="codex")}
    )
    cfg = AgentProposalsConfig(repo_path=repo, artifacts_dir=artifacts)
    create_agent_proposals(
        inp,
        config=cfg,
        claude_runner=claude_runner,
        codex_runner=codex_runner,
    )

    dbg_dir = artifacts / "agent_proposals.last_messages"
    assert (dbg_dir / "cand-0001.claude.json").exists()
    assert (dbg_dir / "cand-0001.codex.json").exists()
