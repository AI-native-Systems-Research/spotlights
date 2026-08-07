"""Public entrypoint for step 4 in candidate mode.

Sibling of `proposal_from_finding_creator`: instead of the cartesian
`candidates × findings` fan-out, each candidate is paired only with the findings
whose `Finding.candidate_id` names it (`Σ|F_c|` sessions), and the emitted
proposals carry the four structured fields (`mechanism`, `required_changes`,
`expected_effect`, `evaluation_metric`).

The Claude runner, error types, and id minting are imported from the existing
package/utilities; the prompt/schema/validation-bound orchestration is copied so
it binds *this* package's `prompts`, `agent_schema`, and `validation` modules
(the originals bind theirs at import time). Issues are reported under the
existing `proposal_from_finding_creator` `PipelineStep`, whose literal union is
closed.
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
from spotlights_engine.proposal_from_candidate_finding_creator.agent_schema import (
    build_per_pair_schema_text,
)
from spotlights_engine.proposal_from_candidate_finding_creator.prompts import (
    build_prompt,
)
from spotlights_engine.proposal_from_candidate_finding_creator.validation import (
    parse_pair_payload,
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


class ProposalFromCandidateFindingConfig(BaseModel):
    """Runtime/infra knobs for step 4 in candidate mode.

    Same knobs as `ProposalFromFindingConfig`; only `created_by` differs, so
    proposals record which step-4 implementation authored them. `created_by`
    lands on `Proposal.author` and is only ever rendered or copied — no
    consumer compares it to a literal.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    repo_path: Path | None = None
    artifacts_dir: Path | None = None

    max_parallel_pairs: int = Field(default=5, ge=1)

    claude_max_turns: int = Field(default=30, ge=1)
    per_pair_wallclock_s: int = Field(default=600, ge=1)

    debug_first_n_pairs: int | None = Field(default=None, ge=1)

    created_by: str = Field(default="proposal_from_candidate_finding_creator", min_length=1)


class ProposalFromCandidateFindingCreatorResult(BaseModel):
    """Runtime-rich variant returned by `create_proposals_with_telemetry`.

    Same shape as `ProposalFromFindingCreatorResult` so the manager's shared
    step-4 tail (persist, checkpoint, usage writes) is mode-agnostic.
    """

    model_config = ConfigDict(extra="forbid")

    output: ProposalFromFindingCreatorOutput
    per_pair_durations_s: dict[str, float] = Field(default_factory=dict)
    total_duration_s: float
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
    config: ProposalFromCandidateFindingConfig,
    *,
    will_invoke_claude: bool,
    candidate_states: CandidateStateMap | None,
) -> None:
    if config.repo_path is None:
        raise ProposalFromFindingSetupError(
            "ProposalFromCandidateFindingConfig.repo_path is required at step entry"
        )
    if config.artifacts_dir is None:
        raise ProposalFromFindingSetupError(
            "ProposalFromCandidateFindingConfig.artifacts_dir is required at step entry"
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
) -> tuple[list[tuple[Candidate, Finding, str]], list[StepIssue]]:
    """Group findings by `candidate_id` and pair each candidate with its own.

    Unlike the module-mode cartesian builder this yields `Σ|F_c|` pairs. A
    finding whose `candidate_id` is missing or names a candidate outside the
    input is reported as a recoverable issue before being dropped, so a broken
    candidate-mode step 3 cannot silently erase all of step 4's work.

    Repeated candidate ids are collapsed to their first occurrence: the pair key
    is `<candidate_id>__<finding_id>`, so a duplicate id would emit the same key
    twice — running the pair twice and attaching two sets of proposals to the
    same code site.
    """
    unique_candidates: list[Candidate] = []
    issues: list[StepIssue] = []
    seen_candidate_ids: set[str] = set()
    for c in candidates:
        if c.id in seen_candidate_ids:
            issues.append(
                _issue(
                    f"duplicate candidate id {c.id} in step-4 input; pairing it once",
                    severity="warning",
                    recoverable=True,
                )
            )
            continue
        seen_candidate_ids.add(c.id)
        unique_candidates.append(c)

    by_candidate: dict[str, list[Finding]] = {c.id: [] for c in unique_candidates}
    for f in findings:
        if f.candidate_id is None:
            issues.append(
                _issue(
                    f"finding {f.finding_id} carries no candidate_id in candidate "
                    "mode; dropping it from step 4",
                    severity="warning",
                    recoverable=True,
                )
            )
            continue
        if f.candidate_id not in by_candidate:
            issues.append(
                _issue(
                    f"finding {f.finding_id} references candidate "
                    f"{f.candidate_id} which is not in this step's input; "
                    "dropping it from step 4",
                    severity="warning",
                    recoverable=True,
                )
            )
            continue
        by_candidate[f.candidate_id].append(f)

    pairs: list[tuple[Candidate, Finding, str]] = []
    for c in unique_candidates:
        for f in by_candidate[c.id]:
            pairs.append((c, f, f"{c.id}__{f.finding_id}"))
    return pairs, issues


def _last_messages_dir(
    config: ProposalFromCandidateFindingConfig,
) -> Path | None:
    if config.artifacts_dir is None:
        return None
    return config.artifacts_dir / "proposal_from_candidate_finding_creator.last_messages"


def _persist_pair_debug(
    *,
    config: ProposalFromCandidateFindingConfig,
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
    config: ProposalFromCandidateFindingConfig,
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
        return (
            pair_key,
            proposals,
            issues,
            run_result.duration_s,
            (
                CliUsage(cli="claude", usage=run_result.usage)
                if run_result.usage is not None
                else None
            ),
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
        "[%s] proposal_from_candidate_finding: pair %s done in %.1fs — %d proposals",
        input.candidates.module_qualified_name,
        pair_key,
        run_result.duration_s,
        len(proposals),
    )

    return (
        pair_key,
        proposals,
        issues,
        run_result.duration_s,
        (CliUsage(cli="claude", usage=run_result.usage) if run_result.usage is not None else None),
    )


def _convert_proposal(drp: DeepResearchProposal, prop_id: str) -> Proposal:
    """Map an agent-facing `DeepResearchProposal` to the unified `Proposal`.

    `finding_ref_id` is the (already-global, see manager) finding id; `source`
    pins this as a research-backed proposal. Unlike the module-mode converter
    this also carries the four structured fields the candidate-mode agent
    schema requires.
    """
    return Proposal(
        id=prop_id,
        source="research_finding",
        finding_ref_id=drp.finding_id,
        author=drp.created_by,
        title=drp.title,
        description=drp.detailed_description,
        rationale=drp.proposal_rationale,
        mechanism=drp.mechanism,
        required_changes=drp.required_changes,
        expected_effect=drp.expected_effect,
        evaluation_metric=drp.evaluation_metric,
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
        _convert_proposal(drp, pid) for drp, pid in zip(new_proposals, proposal_ids, strict=True)
    ]
    return candidate.model_copy(update={"proposals": list(candidate.proposals) + converted})


async def _run_async(
    input: ProposalFromFindingCreatorInput,
    *,
    config: ProposalFromCandidateFindingConfig,
    runner: _PairRunner,
    proposal_id_start: int,
    segment: str,
) -> ProposalFromCandidateFindingCreatorResult:
    start = time.monotonic()

    pairs, grouping_issues = _build_pair_keys(
        list(input.candidates.candidates), list(input.findings)
    )
    grouped_pair_count = len(pairs)

    truncated = False
    if config.debug_first_n_pairs is not None:
        if len(pairs) > config.debug_first_n_pairs:
            truncated = True
        pairs = pairs[: config.debug_first_n_pairs]

    if truncated:
        _log.warning(
            "proposal_from_candidate_finding_creator: debug_first_n_pairs=%s "
            "active — running %s of %s pairs (DEBUG MODE, do not use for "
            "production)",
            config.debug_first_n_pairs,
            len(pairs),
            grouped_pair_count,
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
    aggregated_issues: list[StepIssue] = list(grouping_issues)
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
    # step 5 can continue past them.
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
    return ProposalFromCandidateFindingCreatorResult(
        output=output,
        per_pair_durations_s=per_pair_durations_s,
        total_duration_s=total_duration_s,
        next_proposal_id=next_id,
        usages_by_pair=usages_by_pair,
    )


def create_proposals_with_telemetry(
    input: ProposalFromFindingCreatorInput,
    *,
    config: ProposalFromCandidateFindingConfig,
    runner: _PairRunner | None = None,
    candidate_states: CandidateStateMap | None = None,
    proposal_id_start: int = 1,
    segment: str | None = None,
) -> ProposalFromCandidateFindingCreatorResult:
    """Runtime-rich entrypoint: returns per-pair durations alongside the output.

    `candidate_states` is the manager-provided pipeline-internal state map (D2);
    when omitted the `DISCOVERED` guard is skipped. `proposal_id_start` is the
    per-module-session proposal counter and `segment` is the module id segment
    (D3); minted ids are `prop-<segment>-NNNN`. `segment` defaults to the
    candidates' module slug for standalone callers.
    """
    # Unlike module mode, a non-empty findings list is not enough: every finding
    # may belong to a candidate outside this input, leaving zero pairs.
    pairs, _ = _build_pair_keys(list(input.candidates.candidates), list(input.findings))
    will_invoke_claude = bool(pairs)

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

    seg = segment if segment is not None else slug_for(input.candidates.module_qualified_name)

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
    config: ProposalFromCandidateFindingConfig,
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
    "ProposalFromCandidateFindingConfig",
    "ProposalFromCandidateFindingCreatorResult",
    "create_proposals",
    "create_proposals_with_telemetry",
]
