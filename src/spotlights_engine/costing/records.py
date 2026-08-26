"""The durable per-invocation usage record.

One record per in-scope model call (Claude or Codex) for the four run steps,
written next to the step output it belongs to (`<module_dir>/<step>.usage/`). The
run manifest is a pure aggregation over these records, which is what makes
resume/crash recovery correct: records are keyed by
`(step, module_qualified_name, session_index, invocation_index, cli)` and a
step that re-runs clears its session's records before writing fresh ones.

`"one_shot_apply"` is the exception on both counts: it is a post-run step, and
it persists nothing here. It makes exactly one model call per candidate and
writes its own sibling `manifest.json` in the same all-or-nothing artifact
write as the patch, so there is no partial state for a durable intermediate
record to protect. It uses this model only to reach `compute_cost` and
`aggregate_models_used`.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from spotlights_engine.costing.usage import AgentUsage

UsageStep = Literal[
    "candidate_discovery",
    "module_deep_research",
    "proposal_from_finding_creator",
    "agent_proposals",
    # Post-run, and deliberately NOT part of any run's aggregation — see the
    # module docstring and `persistence.read_usage_records`.
    "one_shot_apply",
]

UsageCli = Literal["claude", "codex"]

PROVIDER_FOR_CLI: dict[str, str] = {
    "claude": "anthropic",
    "codex": "openai",
}


class UsageRecord(BaseModel):
    """Usage of one CLI invocation, durable on disk (see module docstring)."""

    model_config = ConfigDict(extra="forbid")

    step: UsageStep
    module_qualified_name: str = Field(min_length=1)
    session_index: int = Field(default=1, ge=1)
    invocation_index: int = Field(ge=0)
    # Human-debuggable id (e.g. "cand-x__find-y:claude"); not the primary key.
    invocation_id: str = ""
    provider: Literal["anthropic", "openai", "litellm"]
    cli: UsageCli
    # Resolved model id when the CLI reported one; None means unresolved (the
    # aggregation falls back to the CLI family and notes it).
    model: str | None = None
    role: str = Field(min_length=1)
    input: int = Field(default=0, ge=0)
    output: int = Field(default=0, ge=0)
    cache_read: int = Field(default=0, ge=0)
    cache_create: int = Field(default=0, ge=0)
    api_time_s: float | None = None
    # Audit only — never used for billing (CLI cost is Anthropic list price;
    # we bill through the IBM LiteLLM proxy's contracted rates).
    cli_reported_cost_usd: float | None = None

    @property
    def filename(self) -> str:
        """Stable on-disk name derived from the idempotency key."""
        return f"s{self.session_index}.i{self.invocation_index:04d}.{self.cli}.json"

    @property
    def total_tokens(self) -> int:
        return self.input + self.output + self.cache_read + self.cache_create

    @classmethod
    def from_usage(
        cls,
        usage: AgentUsage,
        *,
        step: UsageStep,
        module_qualified_name: str,
        session_index: int,
        invocation_index: int,
        invocation_id: str,
        cli: UsageCli,
        role: str,
        fallback_model: str | None = None,
        fallback_api_time_s: float | None = None,
    ) -> UsageRecord:
        return cls(
            step=step,
            module_qualified_name=module_qualified_name,
            session_index=session_index,
            invocation_index=invocation_index,
            invocation_id=invocation_id,
            provider=PROVIDER_FOR_CLI[cli],  # type: ignore[arg-type]
            cli=cli,
            model=usage.model or fallback_model,
            role=role,
            input=usage.input,
            output=usage.output,
            cache_read=usage.cache_read,
            cache_create=usage.cache_create,
            api_time_s=(
                usage.api_time_s if usage.api_time_s is not None else fallback_api_time_s
            ),
            cli_reported_cost_usd=usage.cli_reported_cost_usd,
        )


__all__ = ["PROVIDER_FOR_CLI", "UsageCli", "UsageRecord", "UsageStep"]
