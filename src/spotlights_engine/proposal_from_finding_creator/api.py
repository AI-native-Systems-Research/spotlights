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
from typing import TYPE_CHECKING, Protocol

from pydantic import BaseModel, ConfigDict, Field

from spotlights_engine.costing.usage import CliUsage
from spotlights_engine.proposal_from_finding_creator.agent_schema import (
    build_per_pair_schema_text,
)
from spotlights_engine.proposal_from_finding_creator.claude_exec import (
    PairRunResult,
    ensure_claude_available,
)
from spotlights_engine.proposal_from_finding_creator.claude_exec import (
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
from spotlights_engine.schemas.proposal import Proposal
from spotlights_engine.schemas.proposals import DeepResearchProposal
from spotlights_engine.utils.id_helpers import slug_for
from spotlights_engine.utils.schema_compat import mint_proposal_ids

if TYPE_CHECKING:  # pragma: no cover
    # Imported for typing only: importing the manager package at runtime would
    # create a cycle (manager -> orchestrator -> this step package).
    from spotlights_engine.spotlights_manager.pipeline_state import CandidateStateMap


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
    # Next free `prop-` number after this step's allocation (decision D3). The
    # manager hands this to step 5 so its proposal ids never collide with step
    # 4's. Defaults to 1 for standalone callers that don't track id blocks.
    next_proposal_id: int = 1
    usages_by_pair: dict[str, CliUsage] = Field(default_factory=dict)


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
    candidate_states: CandidateStateMap | None,
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
    # The schema `Candidate` no longer carries `state` (decision D2); the manager
    # passes a pipeline-internal state map. When invoked standalone (no map), the
    # caller owns the state contract and the guard is skipped.
    if candidate_states is not None:
        for c in input.candidates.candidates:
            state = candidate_states.get(c.id)
            if state != "DISCOVERED":
                raise ProposalFromFindingValidationError(
                    f"candidate {c.id} is in state {state!r}; "
                    "step 4 expects 'DISCOVERED' candidates",
                    candidate_id=c.id,
                    state=state,
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
) -> tuple[
    str,
    list[DeepResearchProposal],
    list[StepIssue],
    float,
    CliUsage | None,
]:
    # Usage is optional: a runner can fail before a CLI process is spawned.
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
                None,
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
        return pair_key, proposals, issues, run_result.duration_s, (
            CliUsage(cli="claude", usage=run_result.usage)
            if run_result.usage is not None
            else None
        )

    parsed = parse_pair_payload(
        run_result.structured_output,
        candidate_id=candidate.id,
        finding_id=finding.finding_id,
        created_by=config.created_by,
    )
    proposals.extend(parsed.proposals)
    for warn in parsed.warnings:
        issues.append(_issue(warn, severity="warning", recoverable=True))

    _log.debug(
        "[%s] proposal_from_finding: pair %s done in %.1fs — %d proposals",
        input.candidates.module_qualified_name,
        pair_key,
        run_result.duration_s,
        len(proposals),
    )

    return pair_key, proposals, issues, run_result.duration_s, (
        CliUsage(cli="claude", usage=run_result.usage)
        if run_result.usage is not None
        else None
    )


def _convert_proposal(drp: DeepResearchProposal, prop_id: str) -> Proposal:
    """Map an agent-facing `DeepResearchProposal` to the unified `Proposal`.

    `finding_ref_id` is the (already-global, see manager) finding id; `source`
    pins this as a research-backed proposal.
    """
    return Proposal(
        id=prop_id,
        source="research_finding",
        finding_ref_id=drp.finding_id,
        author=drp.created_by,
        title=drp.title,
        description=drp.detailed_description,
        rationale=drp.proposal_rationale,
    )


def _rebuild_candidate(
    candidate: Candidate,
    new_proposals: list[DeepResearchProposal],
    proposal_ids: list[str],
) -> Candidate:
    """Append research-backed `Proposal`s to the candidate's unified list.

    `proposal_ids` are minted from the module's proposal block (decision D3) and
    must be the same length as `new_proposals`. Existing proposals (none yet at
    step 4 on the normal path) are preserved.
    """
    converted = [
        _convert_proposal(drp, pid)
        for drp, pid in zip(new_proposals, proposal_ids, strict=True)
    ]
    return candidate.model_copy(
        update={"proposals": list(candidate.proposals) + converted}
    )


async def _run_async(
    input: ProposalFromFindingCreatorInput,
    *,
    config: ProposalFromFindingConfig,
    runner: _PairRunner,
    proposal_id_start: int,
    segment: str,
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

    pair_results: dict[
        str, tuple[list[DeepResearchProposal], list[StepIssue], float, CliUsage | None]
    ] = {}
    if tasks:
        gathered = await asyncio.gather(*tasks)
        for pair_key, proposals, issues, duration, usage in gathered:
            pair_results[pair_key] = (proposals, issues, duration, usage)

    # Assemble per-candidate proposal lists in input-finding order, restricted
    # to pairs that were actually scheduled (debug truncation drops the tail).
    per_candidate_proposals: dict[str, list[DeepResearchProposal]] = {
        c.id: [] for c in input.candidates.candidates
    }
    aggregated_issues: list[StepIssue] = []
    per_pair_durations_s: dict[str, float] = {}
    usages_by_pair: dict[str, CliUsage] = {}

    for c, _f, key in pairs:
        if key not in pair_results:
            continue
        proposals, issues, duration, usage = pair_results[key]
        per_candidate_proposals[c.id].extend(proposals)
        aggregated_issues.extend(issues)
        per_pair_durations_s[key] = duration
        if usage is not None:
            usages_by_pair[key] = usage

    # Deterministic post-gather rebuild: mint `prop-<segment>-NNNN` ids from the
    # module session's proposal counter (decision D3) in candidate order, so
    # step 5 can continue past them. The slug segment makes ids globally unique;
    # the counter is a plain per-module-session sequence (no run-wide block to
    # overflow). Allocating here (not inside the parallel `_run_one_pair`) keeps
    # assignment order-stable.
    next_id = proposal_id_start
    rebuilt: list[Candidate] = []
    for c in input.candidates.candidates:
        new_proposals = per_candidate_proposals.get(c.id, [])
        ids = mint_proposal_ids(next_id, len(new_proposals), segment=segment)
        next_id += len(new_proposals)
        rebuilt.append(_rebuild_candidate(c, new_proposals, ids))

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
        next_proposal_id=next_id,
        usages_by_pair=usages_by_pair,
    )


def create_proposals_with_telemetry(
    input: ProposalFromFindingCreatorInput,
    *,
    config: ProposalFromFindingConfig,
    runner: _PairRunner | None = None,
    candidate_states: CandidateStateMap | None = None,
    proposal_id_start: int = 1,
    segment: str | None = None,
) -> ProposalFromFindingCreatorResult:
    """Runtime-rich entrypoint: returns per-pair durations alongside the output.

    `candidate_states` is the manager-provided pipeline-internal state map (D2);
    when omitted the `DISCOVERED` guard is skipped. `proposal_id_start` is the
    per-module-session proposal counter and `segment` is the module id segment
    (D3); minted ids are `prop-<segment>-NNNN`. `segment` defaults to the
    candidates' module slug for standalone callers.
    """
    will_invoke_claude = bool(input.candidates.candidates) and bool(input.findings)

    if runner is None:
        # Don't probe for `claude` on PATH unless we'll actually shell out.
        _validate_setup(
            input,
            config,
            will_invoke_claude=will_invoke_claude,
            candidate_states=candidate_states,
        )
        active_runner: _PairRunner = default_run_pair
    else:
        _validate_setup(
            input,
            config,
            will_invoke_claude=False,
            candidate_states=candidate_states,
        )
        active_runner = runner

    seg = segment if segment is not None else slug_for(
        input.candidates.module_qualified_name
    )

    return asyncio.run(
        _run_async(
            input,
            config=config,
            runner=active_runner,
            proposal_id_start=proposal_id_start,
            segment=seg,
        )
    )


def create_proposals(
    input: ProposalFromFindingCreatorInput,
    *,
    config: ProposalFromFindingConfig,
    runner: _PairRunner | None = None,
    candidate_states: CandidateStateMap | None = None,
    proposal_id_start: int = 1,
    segment: str | None = None,
) -> ProposalFromFindingCreatorOutput:
    """Architecture-shaped entrypoint: returns the contract output directly."""
    return create_proposals_with_telemetry(
        input,
        config=config,
        runner=runner,
        candidate_states=candidate_states,
        proposal_id_start=proposal_id_start,
        segment=segment,
    ).output


__all__ = [
    "ProposalFromFindingConfig",
    "ProposalFromFindingCreatorResult",
    "create_proposals",
    "create_proposals_with_telemetry",
]
