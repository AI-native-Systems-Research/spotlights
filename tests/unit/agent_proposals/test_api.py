"""End-to-end (with fake runners) tests for `create_agent_proposals_with_telemetry`."""

from __future__ import annotations

from pathlib import Path

import pytest

from spotlights_engine.agent_proposals import (
    AgentProposalsConfig,
    create_agent_proposals,
    create_agent_proposals_with_telemetry,
)
from spotlights_engine.utils.schema_compat import proposals_from
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
        payloads={"cand-v1_kv_offload-0001": make_proposal_payload(agent_name="claude")}
    )
    codex_runner = fake_codex_runner_factory(
        payloads={"cand-v1_kv_offload-0001": make_proposal_payload(agent_name="codex")}
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
    agent_props = proposals_from(c, "agent_knowledge")
    assert {p.author for p in agent_props} == {"claude", "codex"}
    assert all(p.id and p.id.startswith("prop-") for p in agent_props)
    assert result.output.issues == []
    assert "cand-v1_kv_offload-0001" in result.per_candidate_durations_s
    assert "claude" in result.per_candidate_durations_s["cand-v1_kv_offload-0001"]
    assert "codex" in result.per_candidate_durations_s["cand-v1_kv_offload-0001"]


def test_zero_proposals_keeps_candidate_advanced(repo: Path, artifacts: Path) -> None:
    inp = make_input(n_candidates=1)
    claude_runner = fake_claude_runner_factory(
        payloads={"cand-v1_kv_offload-0001": make_empty_proposal_payload()}
    )
    codex_runner = fake_codex_runner_factory(
        payloads={"cand-v1_kv_offload-0001": make_empty_proposal_payload()}
    )
    cfg = AgentProposalsConfig(repo_path=repo, artifacts_dir=artifacts)
    out = create_agent_proposals(
        inp,
        config=cfg,
        claude_runner=claude_runner,
        codex_runner=codex_runner,
    )

    c = out.candidates.candidates[0]
    assert proposals_from(c, "agent_knowledge") == []
    assert out.issues == []


def test_runner_failure_becomes_recoverable_issue(
    repo: Path, artifacts: Path
) -> None:
    inp = make_input(n_candidates=1)
    claude_runner = fake_claude_runner_factory(
        errors={"cand-v1_kv_offload-0001": "claude exit=137: oom"}
    )
    codex_runner = fake_codex_runner_factory(
        payloads={"cand-v1_kv_offload-0001": make_proposal_payload(agent_name="codex")}
    )
    cfg = AgentProposalsConfig(repo_path=repo, artifacts_dir=artifacts)
    out = create_agent_proposals(
        inp,
        config=cfg,
        claude_runner=claude_runner,
        codex_runner=codex_runner,
    )

    c = out.candidates.candidates[0]
    # Codex still ran successfully even though Claude failed.
    assert [p.author for p in proposals_from(c, "agent_knowledge")] == ["codex"]
    assert any(
        iss.recoverable
        and "cand-v1_kv_offload-0001" in iss.message
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
        claude_model=None,
    ):
        if candidate_id == "cand-v1_kv_offload-0001":
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
            "cand-v1_kv_offload-0001": make_proposal_payload(agent_name="codex"),
            "cand-v1_kv_offload-0002": make_proposal_payload(agent_name="codex"),
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
    # cand-v1_kv_offload-0001: only codex's proposal — claude crashed
    assert [
        p.author for p in proposals_from(by_id["cand-v1_kv_offload-0001"], "agent_knowledge")
    ] == ["codex"]
    # cand-v1_kv_offload-0002: both proposals
    assert {
        p.author for p in proposals_from(by_id["cand-v1_kv_offload-0002"], "agent_knowledge")
    } == {
        "claude",
        "codex",
    }
    assert any(
        iss.recoverable
        and "RuntimeError: transient runner crash" in iss.message
        and "cand-v1_kv_offload-0001" in iss.message
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
            "cand-v1_kv_offload-0001": make_proposal_payload(agent_name="claude"),
            "cand-v1_kv_offload-0002": make_proposal_payload(agent_name="claude"),
            "cand-v1_kv_offload-0003": make_proposal_payload(agent_name="claude"),
        }
    )
    codex_runner = fake_codex_runner_factory(
        payloads={
            "cand-v1_kv_offload-0001": make_proposal_payload(agent_name="codex"),
            "cand-v1_kv_offload-0002": make_proposal_payload(agent_name="codex"),
            "cand-v1_kv_offload-0003": make_proposal_payload(agent_name="codex"),
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
    by_id = {c.id: c for c in cands}
    # First candidate ran agents
    assert {
        p.author for p in proposals_from(by_id["cand-v1_kv_offload-0001"], "agent_knowledge")
    } == {
        "claude",
        "codex",
    }
    # Remaining candidates have no agent-knowledge proposals (not run)
    assert proposals_from(by_id["cand-v1_kv_offload-0002"], "agent_knowledge") == []
    assert proposals_from(by_id["cand-v1_kv_offload-0003"], "agent_knowledge") == []


def test_per_candidate_debug_files_written_on_success(
    repo: Path, artifacts: Path
) -> None:
    inp = make_input(n_candidates=1)
    claude_runner = fake_claude_runner_factory(
        payloads={"cand-v1_kv_offload-0001": make_proposal_payload(agent_name="claude")}
    )
    codex_runner = fake_codex_runner_factory(
        payloads={"cand-v1_kv_offload-0001": make_proposal_payload(agent_name="codex")}
    )
    cfg = AgentProposalsConfig(repo_path=repo, artifacts_dir=artifacts)
    create_agent_proposals(
        inp,
        config=cfg,
        claude_runner=claude_runner,
        codex_runner=codex_runner,
    )

    dbg_dir = artifacts / "agent_proposals.last_messages"
    assert (dbg_dir / "cand-v1_kv_offload-0001.claude.json").exists()
    assert (dbg_dir / "cand-v1_kv_offload-0001.codex.json").exists()
