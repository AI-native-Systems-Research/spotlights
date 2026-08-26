"""Shared fakes/builders for agent_proposals unit tests."""

from __future__ import annotations

from pathlib import Path

from spotlights_engine.agent_proposals.claude_exec import CandidateAgentRunResult
from spotlights_engine.schemas.candidate import Candidate, Candidates
from spotlights_engine.schemas.common import SpotlightContext
from spotlights_engine.schemas.pipeline import AgentProposalsInput
from spotlights_engine.schemas.project import (
    Module,
    ProjectTree,
    Repository,
)
from spotlights_engine.utils.schema_compat import make_location


def make_candidate(idx: int = 0) -> Candidate:
    n = idx + 1
    return Candidate(
        id=f"cand-v1_kv_offload-{n:04d}",
        origin="code_agent",
        locations=[
            make_location(
                file="src/v1/kv_offload/core.py",
                line_start=1,
                line_end=10,
                symbol=f"hot_{n}",
                kind="function",
            )
        ],
        description="x",
        current_approach="x",
        evolve_rationale="x",
        estimated_impact="medium",
        estimated_impact_explanation="x",
        proposals=[],
    )


def make_candidates(n: int = 1, qn: str = "v1/kv_offload") -> Candidates:
    return Candidates(
        module_qualified_name=qn,
        candidates=[make_candidate(i) for i in range(n)],
    )


def make_project_tree() -> ProjectTree:
    leaf = Module(
        name="kv_offload",
        path="src/v1/kv_offload",
        submodules=[],
    )
    parent = Module(
        name="v1",
        path="src/v1",
        submodules=[leaf],
    )
    return ProjectTree(
        repository=Repository(name="repo", summary="x", source_root="src"),
        modules=[parent],
    )


def make_input(n_candidates: int = 1, qn: str = "v1/kv_offload") -> AgentProposalsInput:
    return AgentProposalsInput(
        project_tree=make_project_tree(),
        candidates=make_candidates(n_candidates, qn=qn),
        context=SpotlightContext(objective="reduce latency"),
    )


def make_proposal_payload(*, agent_name: str) -> dict:
    return {
        "proposals": [
            {
                "title": "Use technique X",
                "detailed_description": "A detailed plan",
                "agent_name": agent_name,
                "novelty_rationale": "Not covered by research findings",
            }
        ]
    }


def make_empty_proposal_payload() -> dict:
    return {"proposals": []}


def fake_claude_runner_factory(
    *,
    payloads: dict[str, dict | list | None] | None = None,
    errors: dict[str, str] | None = None,
    duration_s: float = 0.001,
):
    payloads = payloads or {}
    errors = errors or {}

    def _runner(
        *,
        candidate_id: str,
        prompt: str,
        schema_text: str,
        repo_path: Path,
        max_turns: int,
        wallclock_s: int,
        claude_model: str | None = None,
    ) -> CandidateAgentRunResult:
        if candidate_id in errors:
            return CandidateAgentRunResult(
                candidate_id=candidate_id,
                duration_s=duration_s,
                structured_output=None,
                error=errors[candidate_id],
                stdout=b"",
                stderr=errors[candidate_id].encode("utf-8"),
            )
        return CandidateAgentRunResult(
            candidate_id=candidate_id,
            duration_s=duration_s,
            structured_output=payloads.get(candidate_id),
        )

    return _runner


def fake_codex_runner_factory(
    *,
    payloads: dict[str, dict | list | None] | None = None,
    errors: dict[str, str] | None = None,
    duration_s: float = 0.001,
):
    payloads = payloads or {}
    errors = errors or {}

    def _runner(
        *,
        candidate_id: str,
        prompt: str,
        schema_text: str,
        repo_path: Path,
        wallclock_s: int,
        last_message_path: Path,
        schema_path: Path,
        codex_model: str | None,
        codex_reasoning_effort: str | None,
    ) -> CandidateAgentRunResult:
        if candidate_id in errors:
            return CandidateAgentRunResult(
                candidate_id=candidate_id,
                duration_s=duration_s,
                structured_output=None,
                error=errors[candidate_id],
                stdout=b"",
                stderr=errors[candidate_id].encode("utf-8"),
            )
        return CandidateAgentRunResult(
            candidate_id=candidate_id,
            duration_s=duration_s,
            structured_output=payloads.get(candidate_id),
        )

    return _runner


__all__ = [
    "fake_claude_runner_factory",
    "fake_codex_runner_factory",
    "make_candidate",
    "make_candidates",
    "make_empty_proposal_payload",
    "make_input",
    "make_project_tree",
    "make_proposal_payload",
]
