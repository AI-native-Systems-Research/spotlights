"""Shared fakes/builders for proposal_from_finding_creator unit tests."""

from __future__ import annotations

from pathlib import Path

from spotlights_engine.proposal_from_finding_creator.claude_exec import (
    PairRunResult,
)
from spotlights_engine.schemas.candidate import Candidate, Candidates
from spotlights_engine.schemas.common import SpotlightContext
from spotlights_engine.schemas.finding import Finding
from spotlights_engine.schemas.pipeline import ProposalFromFindingCreatorInput
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
    )


def make_candidates(n: int = 1, qn: str = "v1/kv_offload") -> Candidates:
    return Candidates(
        module_qualified_name=qn,
        candidates=[make_candidate(i) for i in range(n)],
    )


def make_finding(idx: int = 0, *, candidate_id: str | None = None) -> Finding:
    n = idx + 1
    return Finding(
        finding_id=f"find-v1_kv_offload-{n:04d}",
        candidate_id=candidate_id,
        title="t",
        url="https://example.com",
        source_type="paper",
        technique_summary="s",
    )


def make_input(
    n_candidates: int = 1, n_findings: int = 1
) -> ProposalFromFindingCreatorInput:
    """Build `n_candidates`, each owning `n_findings` findings (decision D6).

    Every finding is stamped with its owning candidate's id and finding ids are
    globally sequential, so the pair set is `Σ|F_c| = n_candidates * n_findings`
    — never the old `|C|×|F|` cross-product. Candidate `i` (0-based) owns
    findings `i*n_findings .. i*n_findings + n_findings - 1`.
    """
    cands = make_candidates(n_candidates)
    findings: list[Finding] = []
    idx = 0
    for c in cands.candidates:
        for _ in range(n_findings):
            findings.append(make_finding(idx, candidate_id=c.id))
            idx += 1
    return ProposalFromFindingCreatorInput(
        candidates=cands,
        findings=findings,
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
