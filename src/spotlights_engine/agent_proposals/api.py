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
import inspect
import json
import logging
import time
from pathlib import Path
from typing import Protocol

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
from spotlights_engine.schemas.candidate import Candidate, Candidates
from spotlights_engine.schemas.common import StepIssue
from spotlights_engine.schemas.pipeline import (
    AgentProposalsInput,
    AgentProposalsOutput,
)
from spotlights_engine.schemas.proposals import AgentProposal

_log = logging.getLogger(__name__)


def _accepts_keyword(fn: object, name: str) -> bool:
    try:
        sig = inspect.signature(fn)
    except (TypeError, ValueError):
        return True
    return name in sig.parameters or any(
        p.kind == inspect.Parameter.VAR_KEYWORD for p in sig.parameters.values()
    )


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
    codex_profile: str | None = None
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
        codex_profile: str | None,
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

    for c in input.candidates.candidates:
        if c.state != "FINDING_PROPOSALS_CREATED":
            raise AgentProposalsValidationError(
                f"candidate {c.id} is in state {c.state!r}; "
                "step 5 expects 'FINDING_PROPOSALS_CREATED' candidates",
                candidate_id=c.id,
                state=c.state,
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
) -> tuple[AgentProposal | None, list[StepIssue], float]:
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
        return None, issues, run_result.duration_s

    parsed = parse_candidate_payload(
        run_result.structured_output,
        candidate_id=candidate.id,
        agent_name=config.claude_agent_name,
    )
    for warn in parsed.warnings:
        issues.append(_issue(warn, severity="warning", recoverable=True))

    proposal = parsed.proposals[0] if parsed.proposals else None
    return proposal, issues, run_result.duration_s


async def _run_codex_pass(
    *,
    candidate: Candidate,
    input: AgentProposalsInput,
    config: AgentProposalsConfig,
    runner: _CodexRunner,
    claude_proposal: AgentProposal | None,
) -> tuple[AgentProposal | None, list[StepIssue], float]:
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
        runner_kwargs = dict(
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
        if _accepts_keyword(runner, "codex_profile"):
            runner_kwargs["codex_profile"] = config.codex_profile
        run_result = await asyncio.to_thread(runner, **runner_kwargs)
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
        return None, issues, run_result.duration_s

    parsed = parse_candidate_payload(
        run_result.structured_output,
        candidate_id=candidate.id,
        agent_name=config.codex_agent_name,
    )
    for warn in parsed.warnings:
        issues.append(_issue(warn, severity="warning", recoverable=True))

    proposal = parsed.proposals[0] if parsed.proposals else None
    return proposal, issues, run_result.duration_s


async def _run_one_candidate(
    *,
    candidate: Candidate,
    input: AgentProposalsInput,
    config: AgentProposalsConfig,
    claude_runner: _ClaudeRunner,
    codex_runner: _CodexRunner,
    semaphore: asyncio.Semaphore,
) -> tuple[str, list[AgentProposal], list[StepIssue], dict[str, float]]:
    """Run Claude then Codex for one candidate. Returns the new proposal list,
    aggregated issues, and per-pass durations."""
    async with semaphore:
        claude_proposal, claude_issues, claude_duration = await _run_claude_pass(
            candidate=candidate,
            input=input,
            config=config,
            runner=claude_runner,
        )
        codex_proposal, codex_issues, codex_duration = await _run_codex_pass(
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
    _log.debug(
        "[%s] agent_proposals: candidate %s done in %.1fs — %d proposals",
        input.candidates.module_qualified_name,
        candidate.id,
        claude_duration + codex_duration,
        len(proposals),
    )
    return candidate.id, proposals, issues, durations


def _rebuild_candidate(
    candidate: Candidate, proposals: list[AgentProposal]
) -> Candidate:
    return candidate.model_copy(
        update={
            "state": "AGENT_PROPOSALS_CREATED",
            "agent_proposals": proposals,
            "deep_research_proposals": list(candidate.deep_research_proposals),
        }
    )


async def _run_async(
    input: AgentProposalsInput,
    *,
    config: AgentProposalsConfig,
    claude_runner: _ClaudeRunner,
    codex_runner: _CodexRunner,
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

    candidate_results: dict[str, tuple[list[AgentProposal], list[StepIssue], dict[str, float]]] = {}
    if tasks:
        gathered = await asyncio.gather(*tasks)
        for cand_id, proposals, issues, durations in gathered:
            candidate_results[cand_id] = (proposals, issues, durations)

    aggregated_issues: list[StepIssue] = []
    per_candidate_durations_s: dict[str, dict[str, float]] = {}
    rebuilt: list[Candidate] = []

    for c in all_candidates:
        if c.id in candidate_results:
            proposals, issues, durations = candidate_results[c.id]
            aggregated_issues.extend(issues)
            per_candidate_durations_s[c.id] = durations
            rebuilt.append(_rebuild_candidate(c, proposals))
        else:
            # Candidate not scheduled (debug truncation): advance state with
            # empty agent_proposals.
            rebuilt.append(_rebuild_candidate(c, []))

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
    )


def create_agent_proposals_with_telemetry(
    input: AgentProposalsInput,
    *,
    config: AgentProposalsConfig,
    claude_runner: _ClaudeRunner | None = None,
    codex_runner: _CodexRunner | None = None,
) -> AgentProposalsResult:
    """Runtime-rich entrypoint: per-candidate / per-agent durations alongside output."""
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
    )

    return asyncio.run(
        _run_async(
            input,
            config=config,
            claude_runner=active_claude,
            codex_runner=active_codex,
        )
    )


def create_agent_proposals(
    input: AgentProposalsInput,
    *,
    config: AgentProposalsConfig,
    claude_runner: _ClaudeRunner | None = None,
    codex_runner: _CodexRunner | None = None,
) -> AgentProposalsOutput:
    """Architecture-shaped entrypoint: returns the contract output directly."""
    return create_agent_proposals_with_telemetry(
        input,
        config=config,
        claude_runner=claude_runner,
        codex_runner=codex_runner,
    ).output


__all__ = [
    "AgentProposalsConfig",
    "AgentProposalsResult",
    "create_agent_proposals",
    "create_agent_proposals_with_telemetry",
]
