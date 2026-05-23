"""Public entrypoint for step 4 (`proposal_from_finding_creator`).

Drives one Claude Code session per `(candidate, finding)` pair, bounded by
`max_parallel_pairs`. The architecture-shaped entrypoint is
`create_proposals`; `create_proposals_with_telemetry` returns the same output
plus per-pair durations for the manager's telemetry record.
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
from pathlib import Path
from typing import Protocol

from pydantic import BaseModel, ConfigDict, Field

from spotlights_engine.proposal_from_finding_creator.agent_schema import (
    build_per_pair_schema_text,
)
from spotlights_engine.proposal_from_finding_creator.claude_exec import (
    PairRunResult,
    ensure_claude_available,
    run_pair as default_run_pair,
)
from spotlights_engine.proposal_from_finding_creator.errors import (
    ProposalFromFindingSetupError,
    ProposalFromFindingValidationError,
)
from spotlights_engine.proposal_from_finding_creator.prompts import build_prompt
from spotlights_engine.proposal_from_finding_creator.validation import (
    parse_pair_payload,
)
from spotlights_engine.schemas.candidate import Candidate, Candidates
from spotlights_engine.schemas.common import StepIssue
from spotlights_engine.schemas.finding import Finding
from spotlights_engine.schemas.pipeline import (
    ProposalFromFindingCreatorInput,
    ProposalFromFindingCreatorOutput,
)
from spotlights_engine.schemas.proposals import DeepResearchProposal


_log = logging.getLogger(__name__)


class ProposalFromFindingConfig(BaseModel):
    """Runtime/infra knobs for step 4.

    Architectural fields stay on `ProposalFromFindingCreatorInput`; this
    class is for filesystem layout, parallelism, and Claude-runner knobs.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    repo_path: Path | None = None
    artifacts_dir: Path | None = None

    max_parallel_pairs: int = Field(default=5, ge=1)

    claude_max_turns: int = Field(default=30, ge=1)
    per_pair_wallclock_s: int = Field(default=600, ge=1)

    debug_first_n_pairs: int | None = Field(default=None, ge=1)

    created_by: str = Field(
        default="proposal_from_finding_creator", min_length=1
    )


class ProposalFromFindingCreatorResult(BaseModel):
    """Runtime-rich variant returned by `create_proposals_with_telemetry`."""

    model_config = ConfigDict(extra="forbid")

    output: ProposalFromFindingCreatorOutput
    per_pair_durations_s: dict[str, float] = Field(default_factory=dict)
    total_duration_s: float


class _PairRunner(Protocol):
    def __call__(
        self,
        *,
        pair_key: str,
        prompt: str,
        schema_text: str,
        repo_path: Path,
        max_turns: int,
        wallclock_s: int,
    ) -> PairRunResult: ...


def _issue(message: str, *, severity: str = "error", recoverable: bool = True) -> StepIssue:
    return StepIssue(
        step="proposal_from_finding_creator",
        severity=severity,  # type: ignore[arg-type]
        message=message,
        recoverable=recoverable,
    )


def _validate_setup(
    input: ProposalFromFindingCreatorInput,
    config: ProposalFromFindingConfig,
    *,
    will_invoke_claude: bool,
) -> None:
    if config.repo_path is None:
        raise ProposalFromFindingSetupError(
            "ProposalFromFindingConfig.repo_path is required at step entry"
        )
    if config.artifacts_dir is None:
        raise ProposalFromFindingSetupError(
            "ProposalFromFindingConfig.artifacts_dir is required at step entry"
        )

    repo = config.repo_path
    if not repo.exists() or not repo.is_dir():
        raise ProposalFromFindingSetupError(
            f"repo_path does not exist or is not a directory: {repo}",
            repo_path=str(repo),
        )

    # Contract sanity: candidates must already be in the input-friendly state.
    for c in input.candidates.candidates:
        if c.state != "DISCOVERED":
            raise ProposalFromFindingValidationError(
                f"candidate {c.id} is in state {c.state!r}; "
                "step 4 expects 'DISCOVERED' candidates",
                candidate_id=c.id,
                state=c.state,
            )

    if will_invoke_claude:
        ensure_claude_available()


def _build_pair_keys(
    candidates: list[Candidate], findings: list[Finding]
) -> list[tuple[Candidate, Finding, str]]:
    """Return all pairs in candidate-outer, finding-inner order."""
    pairs: list[tuple[Candidate, Finding, str]] = []
    for c in candidates:
        for f in findings:
            pairs.append((c, f, f"{c.id}__{f.finding_id}"))
    return pairs


def _last_messages_dir(config: ProposalFromFindingConfig) -> Path | None:
    if config.artifacts_dir is None:
        return None
    return config.artifacts_dir / "proposal_from_finding_creator.last_messages"


def _persist_pair_debug(
    *,
    config: ProposalFromFindingConfig,
    pair_key: str,
    run_result: PairRunResult,
) -> None:
    """Write per-pair debug files. On success: only the structured payload.
    On failure: the structured payload (if any), plus stdout/stderr tails."""
    out_dir = _last_messages_dir(config)
    if out_dir is None:
        return
    try:
        out_dir.mkdir(parents=True, exist_ok=True)
    except OSError:
        return

    if run_result.structured_output is not None:
        try:
            (out_dir / f"{pair_key}.json").write_text(
                json.dumps(run_result.structured_output, indent=2),
                encoding="utf-8",
            )
        except (OSError, TypeError, ValueError):
            pass

    if run_result.error is not None:
        if run_result.stdout:
            try:
                (out_dir / f"{pair_key}.stdout").write_bytes(run_result.stdout)
            except OSError:
                pass
        if run_result.stderr:
            try:
                (out_dir / f"{pair_key}.stderr").write_bytes(run_result.stderr)
            except OSError:
                pass


async def _run_one_pair(
    *,
    candidate: Candidate,
    finding: Finding,
    pair_key: str,
    input: ProposalFromFindingCreatorInput,
    config: ProposalFromFindingConfig,
    runner: _PairRunner,
    semaphore: asyncio.Semaphore,
) -> tuple[str, list[DeepResearchProposal], list[StepIssue], float]:
    schema_text = build_per_pair_schema_text(
        finding_id=finding.finding_id,
        created_by=config.created_by,
    )
    prompt = build_prompt(
        candidate=candidate,
        finding=finding,
        context=input.context,
        module_qualified_name=input.candidates.module_qualified_name,
        created_by=config.created_by,
    )

    async with semaphore:
        assert config.repo_path is not None  # guarded in _validate_setup
        run_start = time.monotonic()
        try:
            run_result = await asyncio.to_thread(
                runner,
                pair_key=pair_key,
                prompt=prompt,
                schema_text=schema_text,
                repo_path=config.repo_path,
                max_turns=config.claude_max_turns,
                wallclock_s=config.per_pair_wallclock_s,
            )
        except Exception as exc:
            duration = time.monotonic() - run_start
            return (
                pair_key,
                [],
                [
                    _issue(
                        f"agent failure (candidate_id={candidate.id}, "
                        f"finding_id={finding.finding_id}): "
                        f"{type(exc).__name__}: {exc}",
                        recoverable=True,
                    )
                ],
                duration,
            )

    _persist_pair_debug(config=config, pair_key=pair_key, run_result=run_result)

    issues: list[StepIssue] = []
    proposals: list[DeepResearchProposal] = []

    if run_result.error is not None:
        issues.append(
            _issue(
                f"agent failure (candidate_id={candidate.id}, "
                f"finding_id={finding.finding_id}): {run_result.error}",
                recoverable=True,
            )
        )
        return pair_key, proposals, issues, run_result.duration_s

    parsed = parse_pair_payload(
        run_result.structured_output,
        candidate_id=candidate.id,
        finding_id=finding.finding_id,
        created_by=config.created_by,
    )
    proposals.extend(parsed.proposals)
    for warn in parsed.warnings:
        issues.append(_issue(warn, severity="warning", recoverable=True))

    return pair_key, proposals, issues, run_result.duration_s


def _rebuild_candidate(
    candidate: Candidate, proposals: list[DeepResearchProposal]
) -> Candidate:
    return candidate.model_copy(
        update={
            "state": "FINDING_PROPOSALS_CREATED",
            "deep_research_proposals": proposals,
            # `agent_proposals` round-trips through this step unchanged; step 5
            # owns it. Preserving the input value keeps the rebuild verbatim.
            "agent_proposals": list(candidate.agent_proposals),
        }
    )


async def _run_async(
    input: ProposalFromFindingCreatorInput,
    *,
    config: ProposalFromFindingConfig,
    runner: _PairRunner,
) -> ProposalFromFindingCreatorResult:
    start = time.monotonic()

    pairs = _build_pair_keys(
        list(input.candidates.candidates), list(input.findings)
    )

    truncated = False
    if config.debug_first_n_pairs is not None:
        if len(pairs) > config.debug_first_n_pairs:
            truncated = True
        pairs = pairs[: config.debug_first_n_pairs]

    if truncated:
        _log.warning(
            "proposal_from_finding_creator: debug_first_n_pairs=%s active — "
            "running %s of %s pairs (DEBUG MODE, do not use for production)",
            config.debug_first_n_pairs,
            len(pairs),
            len(input.candidates.candidates) * len(input.findings),
        )

    semaphore = asyncio.Semaphore(config.max_parallel_pairs)
    tasks = [
        asyncio.create_task(
            _run_one_pair(
                candidate=c,
                finding=f,
                pair_key=key,
                input=input,
                config=config,
                runner=runner,
                semaphore=semaphore,
            )
        )
        for (c, f, key) in pairs
    ]

    pair_results: dict[str, tuple[list[DeepResearchProposal], list[StepIssue], float]] = {}
    if tasks:
        gathered = await asyncio.gather(*tasks)
        for pair_key, proposals, issues, duration in gathered:
            pair_results[pair_key] = (proposals, issues, duration)

    # Assemble per-candidate proposal lists in input-finding order, restricted
    # to pairs that were actually scheduled (debug truncation drops the tail).
    per_candidate_proposals: dict[str, list[DeepResearchProposal]] = {
        c.id: [] for c in input.candidates.candidates
    }
    aggregated_issues: list[StepIssue] = []
    per_pair_durations_s: dict[str, float] = {}

    for c, f, key in pairs:
        if key not in pair_results:
            continue
        proposals, issues, duration = pair_results[key]
        per_candidate_proposals[c.id].extend(proposals)
        aggregated_issues.extend(issues)
        per_pair_durations_s[key] = duration

    rebuilt = [
        _rebuild_candidate(c, per_candidate_proposals.get(c.id, []))
        for c in input.candidates.candidates
    ]

    output = ProposalFromFindingCreatorOutput(
        candidates=Candidates(
            module_qualified_name=input.candidates.module_qualified_name,
            candidates=rebuilt,
        ),
        issues=aggregated_issues,
    )
    total_duration_s = time.monotonic() - start
    return ProposalFromFindingCreatorResult(
        output=output,
        per_pair_durations_s=per_pair_durations_s,
        total_duration_s=total_duration_s,
    )


def create_proposals_with_telemetry(
    input: ProposalFromFindingCreatorInput,
    *,
    config: ProposalFromFindingConfig,
    runner: _PairRunner | None = None,
) -> ProposalFromFindingCreatorResult:
    """Runtime-rich entrypoint: returns per-pair durations alongside the output."""
    will_invoke_claude = bool(input.candidates.candidates) and bool(input.findings)

    if runner is None:
        # Don't probe for `claude` on PATH unless we'll actually shell out.
        _validate_setup(input, config, will_invoke_claude=will_invoke_claude)
        active_runner: _PairRunner = default_run_pair
    else:
        _validate_setup(input, config, will_invoke_claude=False)
        active_runner = runner

    return asyncio.run(_run_async(input, config=config, runner=active_runner))


def create_proposals(
    input: ProposalFromFindingCreatorInput,
    *,
    config: ProposalFromFindingConfig,
    runner: _PairRunner | None = None,
) -> ProposalFromFindingCreatorOutput:
    """Architecture-shaped entrypoint: returns the contract output directly."""
    return create_proposals_with_telemetry(
        input, config=config, runner=runner
    ).output


__all__ = [
    "ProposalFromFindingConfig",
    "ProposalFromFindingCreatorResult",
    "create_proposals",
    "create_proposals_with_telemetry",
]
