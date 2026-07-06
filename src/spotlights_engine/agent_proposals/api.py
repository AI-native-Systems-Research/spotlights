"""Public entrypoint for step 5 (`agent_proposals`).

Drives one Claude session and one Codex session per candidate, sequentially
within a candidate (Codex needs Claude's output) and bounded by
`max_parallel_candidates` across candidates.

Known limitation (v1): step 5 cannot resume per-candidate within a single
module run. If the module crashes mid-step-5, the manager clears the entire
step-5 sidecar and reruns the whole candidate fan-out. See plan §9.
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
from pathlib import Path
from typing import TYPE_CHECKING, Protocol

from pydantic import BaseModel, ConfigDict, Field

from spotlights_engine.agent_proposals.agent_schema import (
    build_per_candidate_schema_text,
)
from spotlights_engine.agent_proposals.claude_exec import (
    CandidateAgentRunResult,
    ensure_claude_available,
)
from spotlights_engine.agent_proposals.claude_exec import (
    run_candidate_claude as default_run_claude,
)
from spotlights_engine.agent_proposals.codex_exec import (
    ensure_codex_available,
)
from spotlights_engine.agent_proposals.codex_exec import (
    run_candidate_codex as default_run_codex,
)
from spotlights_engine.agent_proposals.errors import (
    AgentProposalsSetupError,
    AgentProposalsValidationError,
)
from spotlights_engine.agent_proposals.prompts import (
    build_claude_prompt,
    build_codex_prompt,
)
from spotlights_engine.agent_proposals.validation import (
    parse_candidate_payload,
)
from spotlights_engine.costing.usage import CliUsage
from spotlights_engine.schemas.candidate import Candidate, Candidates
from spotlights_engine.schemas.common import StepIssue
from spotlights_engine.schemas.pipeline import (
    AgentProposalsInput,
    AgentProposalsOutput,
)
from spotlights_engine.schemas.proposal import Proposal
from spotlights_engine.schemas.proposals import AgentProposal
from spotlights_engine.utils.id_helpers import slug_for
from spotlights_engine.utils.schema_compat import mint_proposal_ids

if TYPE_CHECKING:  # pragma: no cover
    # Typing-only: importing the manager package at runtime would create a cycle
    # (manager -> orchestrator -> this step package).
    from spotlights_engine.spotlights_manager.pipeline_state import CandidateStateMap


_log = logging.getLogger(__name__)


_CLAUDE_PASS = "claude"
_CODEX_PASS = "codex"


class AgentProposalsConfig(BaseModel):
    """Runtime/infra knobs for step 5."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    repo_path: Path | None = None
    artifacts_dir: Path | None = None

    max_parallel_candidates: int = Field(default=5, ge=1)

    claude_max_turns: int = Field(default=30, ge=1)
    claude_wallclock_s: int = Field(default=600, ge=1)
    codex_wallclock_s: int = Field(default=600, ge=1)
    codex_model: str | None = None
    codex_reasoning_effort: str | None = None

    debug_first_n_candidates: int | None = Field(default=None, ge=1)

    claude_agent_name: str = Field(default="claude", min_length=1)
    codex_agent_name: str = Field(default="codex", min_length=1)


class AgentProposalsResult(BaseModel):
    """Runtime-rich variant returned by `create_agent_proposals_with_telemetry`."""

    model_config = ConfigDict(extra="forbid")

    output: AgentProposalsOutput
    per_candidate_durations_s: dict[str, dict[str, float]] = Field(
        default_factory=dict
    )
    total_duration_s: float
    usages_by_candidate: dict[str, list[CliUsage]] = Field(default_factory=dict)


class _ClaudeRunner(Protocol):
    def __call__(
        self,
        *,
        candidate_id: str,
        prompt: str,
        schema_text: str,
        repo_path: Path,
        max_turns: int,
        wallclock_s: int,
    ) -> CandidateAgentRunResult: ...


class _CodexRunner(Protocol):
    def __call__(
        self,
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
    ) -> CandidateAgentRunResult: ...


def _issue(
    message: str, *, severity: str = "error", recoverable: bool = True
) -> StepIssue:
    return StepIssue(
        step="agent_proposals",
        severity=severity,  # type: ignore[arg-type]
        message=message,
        recoverable=recoverable,
    )


def _validate_setup(
    input: AgentProposalsInput,
    config: AgentProposalsConfig,
    *,
    will_invoke_claude: bool,
    will_invoke_codex: bool,
    candidate_states: CandidateStateMap | None,
) -> None:
    if config.repo_path is None:
        raise AgentProposalsSetupError(
            "AgentProposalsConfig.repo_path is required at step entry"
        )
    if config.artifacts_dir is None:
        raise AgentProposalsSetupError(
            "AgentProposalsConfig.artifacts_dir is required at step entry"
        )

    repo = config.repo_path
    if not repo.exists() or not repo.is_dir():
        raise AgentProposalsSetupError(
            f"repo_path does not exist or is not a directory: {repo}",
            repo_path=str(repo),
        )

    # The schema `Candidate` no longer carries `state` (decision D2); the manager
    # passes a pipeline-internal state map. When invoked standalone (no map), the
    # caller owns the state contract and the guard is skipped.
    if candidate_states is not None:
        for c in input.candidates.candidates:
            state = candidate_states.get(c.id)
            if state != "FINDING_PROPOSALS_CREATED":
                raise AgentProposalsValidationError(
                    f"candidate {c.id} is in state {state!r}; "
                    "step 5 expects 'FINDING_PROPOSALS_CREATED' candidates",
                    candidate_id=c.id,
                    state=state,
                )

    if will_invoke_claude:
        ensure_claude_available()
    if will_invoke_codex:
        ensure_codex_available()


def _last_messages_dir(config: AgentProposalsConfig) -> Path | None:
    if config.artifacts_dir is None:
        return None
    return config.artifacts_dir / "agent_proposals.last_messages"


def _persist_candidate_debug(
    *,
    config: AgentProposalsConfig,
    candidate_id: str,
    pass_label: str,
    run_result: CandidateAgentRunResult,
) -> None:
    """Write per-candidate, per-pass debug files. On success: only the
    structured payload. On failure: stdout/stderr tails."""
    out_dir = _last_messages_dir(config)
    if out_dir is None:
        return
    try:
        out_dir.mkdir(parents=True, exist_ok=True)
    except OSError:
        return

    if run_result.structured_output is not None:
        try:
            (out_dir / f"{candidate_id}.{pass_label}.json").write_text(
                json.dumps(run_result.structured_output, indent=2),
                encoding="utf-8",
            )
        except (OSError, TypeError, ValueError):
            pass

    if run_result.error is not None:
        if run_result.stdout:
            try:
                (out_dir / f"{candidate_id}.{pass_label}.stdout").write_bytes(
                    run_result.stdout
                )
            except OSError:
                pass
        if run_result.stderr:
            try:
                (out_dir / f"{candidate_id}.{pass_label}.stderr").write_bytes(
                    run_result.stderr
                )
            except OSError:
                pass


async def _run_claude_pass(
    *,
    candidate: Candidate,
    input: AgentProposalsInput,
    config: AgentProposalsConfig,
    runner: _ClaudeRunner,
) -> tuple[AgentProposal | None, list[StepIssue], float, CliUsage | None]:
    schema_text = build_per_candidate_schema_text(
        agent_name=config.claude_agent_name
    )
    prompt = build_claude_prompt(
        candidate=candidate,
        project_tree=input.project_tree,
        module_qualified_name=input.candidates.module_qualified_name,
        context=input.context,
        agent_name=config.claude_agent_name,
    )

    assert config.repo_path is not None  # guarded in _validate_setup
    run_start = time.monotonic()
    try:
        run_result = await asyncio.to_thread(
            runner,
            candidate_id=candidate.id,
            prompt=prompt,
            schema_text=schema_text,
            repo_path=config.repo_path,
            max_turns=config.claude_max_turns,
            wallclock_s=config.claude_wallclock_s,
        )
    except Exception as exc:  # noqa: BLE001
        duration = time.monotonic() - run_start
        return (
            None,
            [
                _issue(
                    f"agent failure (candidate_id={candidate.id}, agent=claude): "
                    f"{type(exc).__name__}: {exc}",
                    recoverable=True,
                )
            ],
            duration,
            None,
        )

    _persist_candidate_debug(
        config=config,
        candidate_id=candidate.id,
        pass_label=_CLAUDE_PASS,
        run_result=run_result,
    )

    issues: list[StepIssue] = []
    if run_result.error is not None:
        issues.append(
            _issue(
                f"agent failure (candidate_id={candidate.id}, agent=claude): "
                f"{run_result.error}",
                recoverable=True,
            )
        )
        return None, issues, run_result.duration_s, (
            CliUsage(cli="claude", usage=run_result.usage)
            if run_result.usage is not None
            else None
        )

    parsed = parse_candidate_payload(
        run_result.structured_output,
        candidate_id=candidate.id,
        agent_name=config.claude_agent_name,
    )
    for warn in parsed.warnings:
        issues.append(_issue(warn, severity="warning", recoverable=True))

    proposal = parsed.proposals[0] if parsed.proposals else None
    return proposal, issues, run_result.duration_s, (
        CliUsage(cli="claude", usage=run_result.usage)
        if run_result.usage is not None
        else None
    )


async def _run_codex_pass(
    *,
    candidate: Candidate,
    input: AgentProposalsInput,
    config: AgentProposalsConfig,
    runner: _CodexRunner,
    claude_proposal: AgentProposal | None,
) -> tuple[AgentProposal | None, list[StepIssue], float, CliUsage | None]:
    schema_text = build_per_candidate_schema_text(
        agent_name=config.codex_agent_name
    )
    prompt = build_codex_prompt(
        candidate=candidate,
        project_tree=input.project_tree,
        module_qualified_name=input.candidates.module_qualified_name,
        context=input.context,
        agent_name=config.codex_agent_name,
        claude_proposal=claude_proposal,
    )

    assert config.repo_path is not None
    assert config.artifacts_dir is not None
    last_messages_dir = _last_messages_dir(config)
    assert last_messages_dir is not None
    last_message_path = last_messages_dir / f"{candidate.id}.codex.last_message.json"
    schema_path = last_messages_dir / f"{candidate.id}.codex.schema.json"

    run_start = time.monotonic()
    try:
        run_result = await asyncio.to_thread(
            runner,
            candidate_id=candidate.id,
            prompt=prompt,
            schema_text=schema_text,
            repo_path=config.repo_path,
            wallclock_s=config.codex_wallclock_s,
            last_message_path=last_message_path,
            schema_path=schema_path,
            codex_model=config.codex_model,
            codex_reasoning_effort=config.codex_reasoning_effort,
        )
    except Exception as exc:  # noqa: BLE001
        duration = time.monotonic() - run_start
        return (
            None,
            [
                _issue(
                    f"agent failure (candidate_id={candidate.id}, agent=codex): "
                    f"{type(exc).__name__}: {exc}",
                    recoverable=True,
                )
            ],
            duration,
            None,
        )

    _persist_candidate_debug(
        config=config,
        candidate_id=candidate.id,
        pass_label=_CODEX_PASS,
        run_result=run_result,
    )

    issues: list[StepIssue] = []
    if run_result.error is not None:
        issues.append(
            _issue(
                f"agent failure (candidate_id={candidate.id}, agent=codex): "
                f"{run_result.error}",
                recoverable=True,
            )
        )
        return None, issues, run_result.duration_s, (
            CliUsage(cli="codex", usage=run_result.usage)
            if run_result.usage is not None
            else None
        )

    parsed = parse_candidate_payload(
        run_result.structured_output,
        candidate_id=candidate.id,
        agent_name=config.codex_agent_name,
    )
    for warn in parsed.warnings:
        issues.append(_issue(warn, severity="warning", recoverable=True))

    proposal = parsed.proposals[0] if parsed.proposals else None
    return proposal, issues, run_result.duration_s, (
        CliUsage(cli="codex", usage=run_result.usage)
        if run_result.usage is not None
        else None
    )


async def _run_one_candidate(
    *,
    candidate: Candidate,
    input: AgentProposalsInput,
    config: AgentProposalsConfig,
    claude_runner: _ClaudeRunner,
    codex_runner: _CodexRunner,
    semaphore: asyncio.Semaphore,
) -> tuple[str, list[AgentProposal], list[StepIssue], dict[str, float], list[CliUsage]]:
    """Run Claude then Codex for one candidate. Returns the new proposal list,
    aggregated issues, and per-pass durations."""
    async with semaphore:
        claude_proposal, claude_issues, claude_duration, claude_usage = await _run_claude_pass(
            candidate=candidate,
            input=input,
            config=config,
            runner=claude_runner,
        )
        codex_proposal, codex_issues, codex_duration, codex_usage = await _run_codex_pass(
            candidate=candidate,
            input=input,
            config=config,
            runner=codex_runner,
            claude_proposal=claude_proposal,
        )

    proposals: list[AgentProposal] = []
    if claude_proposal is not None:
        proposals.append(claude_proposal)
    if codex_proposal is not None:
        proposals.append(codex_proposal)

    issues = list(claude_issues) + list(codex_issues)
    durations = {
        _CLAUDE_PASS: claude_duration,
        _CODEX_PASS: codex_duration,
    }
    usages = [u for u in (claude_usage, codex_usage) if u is not None]
    _log.debug(
        "[%s] agent_proposals: candidate %s done in %.1fs — %d proposals",
        input.candidates.module_qualified_name,
        candidate.id,
        claude_duration + codex_duration,
        len(proposals),
    )
    return candidate.id, proposals, issues, durations, usages


def _convert_proposal(ap: AgentProposal, prop_id: str) -> Proposal:
    """Map an agent-facing `AgentProposal` to the unified `Proposal`."""
    return Proposal(
        id=prop_id,
        source="agent_knowledge",
        author=ap.agent_name,
        title=ap.title,
        description=ap.detailed_description,
        rationale=ap.novelty_rationale,
    )


def _rebuild_candidate(
    candidate: Candidate,
    new_proposals: list[AgentProposal],
    proposal_ids: list[str],
) -> Candidate:
    """Append agent-knowledge `Proposal`s to the candidate's unified list.

    `proposal_ids` are minted from the module's proposal block (decision D3),
    continuing past step 4's allocation. Existing research-backed proposals are
    preserved.
    """
    converted = [
        _convert_proposal(ap, pid)
        for ap, pid in zip(new_proposals, proposal_ids, strict=True)
    ]
    return candidate.model_copy(
        update={"proposals": list(candidate.proposals) + converted}
    )


async def _run_async(
    input: AgentProposalsInput,
    *,
    config: AgentProposalsConfig,
    claude_runner: _ClaudeRunner,
    codex_runner: _CodexRunner,
    proposal_id_start: int,
    segment: str,
) -> AgentProposalsResult:
    start = time.monotonic()

    all_candidates = list(input.candidates.candidates)
    scheduled: list[Candidate] = list(all_candidates)

    truncated = False
    if config.debug_first_n_candidates is not None:
        if len(scheduled) > config.debug_first_n_candidates:
            truncated = True
        scheduled = scheduled[: config.debug_first_n_candidates]

    if truncated:
        _log.warning(
            "agent_proposals: debug_first_n_candidates=%s active — "
            "running %s of %s candidates (DEBUG MODE, do not use for production)",
            config.debug_first_n_candidates,
            len(scheduled),
            len(all_candidates),
        )

    semaphore = asyncio.Semaphore(config.max_parallel_candidates)
    tasks = [
        asyncio.create_task(
            _run_one_candidate(
                candidate=c,
                input=input,
                config=config,
                claude_runner=claude_runner,
                codex_runner=codex_runner,
                semaphore=semaphore,
            )
        )
        for c in scheduled
    ]

    candidate_results: dict[
        str,
        tuple[list[AgentProposal], list[StepIssue], dict[str, float], list[CliUsage]],
    ] = {}
    if tasks:
        gathered = await asyncio.gather(*tasks)
        for cand_id, proposals, issues, durations, usages in gathered:
            candidate_results[cand_id] = (proposals, issues, durations, usages)

    aggregated_issues: list[StepIssue] = []
    per_candidate_durations_s: dict[str, dict[str, float]] = {}
    usages_by_candidate: dict[str, list[CliUsage]] = {}

    # Deterministic post-gather rebuild: mint `prop-<segment>-NNNN` ids from the
    # module session's proposal counter (decision D3), continuing past step 4's
    # allocation, in candidate order. The slug segment makes ids globally
    # unique; the counter is a plain per-module-session sequence (no run-wide
    # block to overflow). Allocating here (not inside the parallel
    # `_run_one_candidate`) keeps id assignment order-stable.
    new_by_id: dict[str, list[AgentProposal]] = {}
    for c in all_candidates:
        proposals = candidate_results[c.id][0] if c.id in candidate_results else []
        new_by_id[c.id] = proposals

    next_id = proposal_id_start
    rebuilt: list[Candidate] = []
    for c in all_candidates:
        if c.id in candidate_results:
            _, issues, durations, usages = candidate_results[c.id]
            aggregated_issues.extend(issues)
            per_candidate_durations_s[c.id] = durations
            if usages:
                usages_by_candidate[c.id] = usages
        new_proposals = new_by_id[c.id]
        ids = mint_proposal_ids(next_id, len(new_proposals), segment=segment)
        next_id += len(new_proposals)
        rebuilt.append(_rebuild_candidate(c, new_proposals, ids))

    output = AgentProposalsOutput(
        candidates=Candidates(
            module_qualified_name=input.candidates.module_qualified_name,
            candidates=rebuilt,
        ),
        issues=aggregated_issues,
    )
    total_duration_s = time.monotonic() - start
    return AgentProposalsResult(
        output=output,
        per_candidate_durations_s=per_candidate_durations_s,
        total_duration_s=total_duration_s,
        usages_by_candidate=usages_by_candidate,
    )


def create_agent_proposals_with_telemetry(
    input: AgentProposalsInput,
    *,
    config: AgentProposalsConfig,
    claude_runner: _ClaudeRunner | None = None,
    codex_runner: _CodexRunner | None = None,
    candidate_states: CandidateStateMap | None = None,
    proposal_id_start: int = 1,
    segment: str | None = None,
) -> AgentProposalsResult:
    """Runtime-rich entrypoint: per-candidate / per-agent durations alongside output.

    `candidate_states` is the manager-provided pipeline-internal state map (D2);
    when omitted the `FINDING_PROPOSALS_CREATED` guard is skipped.
    `proposal_id_start` is the per-module-session proposal counter advanced past
    step 4's allocation, and `segment` is the module id segment (D3); minted ids
    are `prop-<segment>-NNNN`. `segment` defaults to the candidates' module slug
    for standalone callers.
    """
    will_invoke = bool(input.candidates.candidates)

    if claude_runner is None:
        will_invoke_claude = will_invoke
        active_claude: _ClaudeRunner = default_run_claude
    else:
        will_invoke_claude = False
        active_claude = claude_runner

    if codex_runner is None:
        will_invoke_codex = will_invoke
        active_codex: _CodexRunner = default_run_codex
    else:
        will_invoke_codex = False
        active_codex = codex_runner

    _validate_setup(
        input,
        config,
        will_invoke_claude=will_invoke_claude,
        will_invoke_codex=will_invoke_codex,
        candidate_states=candidate_states,
    )

    seg = segment if segment is not None else slug_for(
        input.candidates.module_qualified_name
    )

    return asyncio.run(
        _run_async(
            input,
            config=config,
            claude_runner=active_claude,
            codex_runner=active_codex,
            proposal_id_start=proposal_id_start,
            segment=seg,
        )
    )


def create_agent_proposals(
    input: AgentProposalsInput,
    *,
    config: AgentProposalsConfig,
    claude_runner: _ClaudeRunner | None = None,
    codex_runner: _CodexRunner | None = None,
    candidate_states: CandidateStateMap | None = None,
    proposal_id_start: int = 1,
    segment: str | None = None,
) -> AgentProposalsOutput:
    """Architecture-shaped entrypoint: returns the contract output directly."""
    return create_agent_proposals_with_telemetry(
        input,
        config=config,
        claude_runner=claude_runner,
        codex_runner=codex_runner,
        candidate_states=candidate_states,
        proposal_id_start=proposal_id_start,
        segment=segment,
    ).output


__all__ = [
    "AgentProposalsConfig",
    "AgentProposalsResult",
    "create_agent_proposals",
    "create_agent_proposals_with_telemetry",
]
