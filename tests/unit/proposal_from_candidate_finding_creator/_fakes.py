"""Shared fakes/builders for proposal_from_candidate_finding_creator tests."""

from __future__ import annotations

from pathlib import Path

from spotlights_engine.proposal_from_finding_creator.claude_exec import PairRunResult
from spotlights_engine.schemas.candidate import Candidates
from spotlights_engine.schemas.common import SpotlightContext
from spotlights_engine.schemas.finding import Finding
from spotlights_engine.schemas.pipeline import ProposalFromFindingCreatorInput
from tests.unit.proposal_from_finding_creator._fakes import make_candidate

SEGMENT = "v1_kv_offload"
CREATED_BY = "proposal_from_candidate_finding_creator"


def candidate_id(idx: int) -> str:
    return f"cand-{SEGMENT}-{idx + 1:04d}"


def make_finding(
    candidate_idx: int, finding_idx: int = 0, *, candidate_id_override: str | None = None
) -> Finding:
    """Build a candidate-mode finding: `find-<segment>-<cand>-NNNN` + tag."""
    cand = (
        candidate_id_override if candidate_id_override is not None else candidate_id(candidate_idx)
    )
    return Finding(
        finding_id=f"find-{SEGMENT}-{candidate_idx + 1:04d}-{finding_idx + 1:04d}",
        title="t",
        url=f"https://example.com/{candidate_idx}-{finding_idx}",
        source_type="paper",
        technique_summary="s",
        candidate_id=cand,
    )


def make_input(
    *, n_candidates: int = 1, findings: list[Finding] | None = None
) -> ProposalFromFindingCreatorInput:
    """Candidate-mode step-4 input: findings are candidate-tagged by default."""
    if findings is None:
        findings = [make_finding(i) for i in range(n_candidates)]
    return ProposalFromFindingCreatorInput(
        candidates=Candidates(
            module_qualified_name="v1/kv_offload",
            candidates=[make_candidate(i) for i in range(n_candidates)],
        ),
        findings=findings,
        context=SpotlightContext(objective="reduce latency"),
    )


def make_proposal_payload(
    *,
    finding_id: str,
    created_by: str = CREATED_BY,
    title: str = "Use technique X",
    detailed_description: str = "A detailed plan",
    proposal_rationale: str = "Because of Y",
    mechanism: str = "Swap the loop for a pool",
    required_changes: str = "Rewrite core.py hot_1",
    expected_effect: str = "2x on the hot path",
    evaluation_metric: str = "p99 latency",
) -> list[dict]:
    return [
        {
            "title": title,
            "detailed_description": detailed_description,
            "finding_id": finding_id,
            "proposal_rationale": proposal_rationale,
            "created_by": created_by,
            "mechanism": mechanism,
            "required_changes": required_changes,
            "expected_effect": expected_effect,
            "evaluation_metric": evaluation_metric,
        }
    ]


def fake_runner_factory(
    *,
    payloads: dict[str, list[dict] | None] | None = None,
    errors: dict[str, str] | None = None,
    duration_s: float = 0.001,
    invoked: list[str] | None = None,
    default_payload: bool = False,
):
    """Runner returning preset payloads/errors keyed by pair_key.

    With `default_payload=True` any unlisted pair gets a valid payload derived
    from its own key, which keeps grouping tests from having to enumerate pairs.
    """
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
        if invoked is not None:
            invoked.append(pair_key)
        if pair_key in errors:
            return PairRunResult(
                pair_key=pair_key,
                duration_s=duration_s,
                structured_output=None,
                error=errors[pair_key],
                stdout=b"",
                stderr=errors[pair_key].encode("utf-8"),
            )
        if pair_key in payloads:
            payload = payloads[pair_key]
        elif default_payload:
            payload = make_proposal_payload(finding_id=pair_key.split("__")[1])
        else:
            payload = None
        return PairRunResult(
            pair_key=pair_key,
            duration_s=duration_s,
            structured_output=payload,
            error=None,
            stdout=b"",
            stderr=b"",
        )

    return _runner


__all__ = [
    "CREATED_BY",
    "SEGMENT",
    "candidate_id",
    "fake_runner_factory",
    "make_finding",
    "make_input",
    "make_proposal_payload",
]
