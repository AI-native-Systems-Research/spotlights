"""Shared fakes/builders for proposal_from_finding_creator unit tests."""

from __future__ import annotations

from pathlib import Path

from spotlights_engine.proposal_from_finding_creator.claude_exec import (
    PairRunResult,
)
from spotlights_engine.schemas.legacy.candidate import Candidate, Candidates
from spotlights_engine.schemas.legacy.common import SpotlightContext
from spotlights_engine.schemas.legacy.finding import Finding
from spotlights_engine.schemas.legacy.pipeline import ProposalFromFindingCreatorInput


def make_candidate(idx: int = 0) -> Candidate:
    n = idx + 1
    return Candidate(
        id=f"cand-{n:04d}",
        file="src/v1/kv_offload/core.py",
        line_start=1,
        line_end=10,
        symbol=f"hot_{n}",
        kind="function",
        description="x",
        current_approach="x",
        evolve_rationale="x",
        estimated_impact="medium",
        estimated_impact_explanation="x",
    )


def make_candidates(n: int = 1, qn: str = "v1.kv_offload") -> Candidates:
    return Candidates(
        module_qualified_name=qn,
        candidates=[make_candidate(i) for i in range(n)],
    )


def make_finding(idx: int = 0) -> Finding:
    n = idx + 1
    return Finding(
        finding_id=f"find-{n:04d}",
        title="t",
        url="https://example.com",
        source_type="paper",
        technique_summary="s",
    )


def make_input(
    n_candidates: int = 1, n_findings: int = 1
) -> ProposalFromFindingCreatorInput:
    return ProposalFromFindingCreatorInput(
        candidates=make_candidates(n_candidates),
        findings=[make_finding(i) for i in range(n_findings)],
        context=SpotlightContext(objective="reduce latency"),
    )


def make_proposal_payload(
    *,
    finding_id: str,
    created_by: str = "proposal_from_finding_creator",
    title: str = "Use technique X",
    detailed_description: str = "A detailed plan",
    proposal_rationale: str = "Because of Y",
) -> list[dict]:
    return [
        {
            "title": title,
            "detailed_description": detailed_description,
            "finding_id": finding_id,
            "proposal_rationale": proposal_rationale,
            "created_by": created_by,
        }
    ]


def fake_runner_factory(
    *,
    payloads: dict[str, list[dict] | None] | None = None,
    errors: dict[str, str] | None = None,
    duration_s: float = 0.001,
):
    """Build a runner that returns preset payloads/errors keyed by pair_key."""
    payloads = payloads or {}
    errors = errors or {}

    def _runner(
        *,
        pair_key: str,
        prompt: str,
        schema_text: str,
        repo_path: Path,
        max_turns: int,
        wallclock_s: int,
    ) -> PairRunResult:
        if pair_key in errors:
            return PairRunResult(
                pair_key=pair_key,
                duration_s=duration_s,
                structured_output=None,
                error=errors[pair_key],
                stdout=b"",
                stderr=errors[pair_key].encode("utf-8"),
            )
        return PairRunResult(
            pair_key=pair_key,
            duration_s=duration_s,
            structured_output=payloads.get(pair_key),
            error=None,
            stdout=b"",
            stderr=b"",
        )

    return _runner


__all__ = [
    "fake_runner_factory",
    "make_candidate",
    "make_candidates",
    "make_finding",
    "make_input",
    "make_proposal_payload",
]
