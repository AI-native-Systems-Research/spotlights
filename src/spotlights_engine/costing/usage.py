"""Token-usage capture shape and tolerant per-CLI parsers.

The four token buckets are **disjoint by construction** so the rate-table
cost formula (`Σ bucket * rate`) never double-charges:

- Claude already reports them disjoint (`input_tokens` excludes cache
  reads/creates).
- Codex reports `cached_input_tokens` as a *subset* of `input_tokens`
  (OpenAI convention), so `codex_usage_from_stream` normalizes at capture
  time: `cache_read = cached_input_tokens`, `input -= cached_input_tokens`,
  `cache_create = 0` (Codex has no cache-write charge class).

Parsers are best-effort: a stream that carries no recognizable usage payload
yields `None` (the caller notes the degraded capture) rather than raising.
"""

from __future__ import annotations

import json

from pydantic import BaseModel, ConfigDict, Field


class AgentUsage(BaseModel):
    """Normalized usage extracted from one CLI invocation."""

    model_config = ConfigDict(extra="forbid")

    input: int = Field(default=0, ge=0)
    output: int = Field(default=0, ge=0)
    cache_read: int = Field(default=0, ge=0)
    cache_create: int = Field(default=0, ge=0)
    model: str | None = None
    api_time_s: float | None = None
    # Captured for audit only; billing goes through the contracted rate table.
    cli_reported_cost_usd: float | None = None


class CliUsage(BaseModel):
    """`AgentUsage` tagged with the CLI that produced it (fan-out steps)."""

    model_config = ConfigDict(extra="forbid")

    cli: str
    usage: AgentUsage


def _as_nonneg_int(value: object) -> int:
    try:
        n = int(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return 0
    return max(n, 0)


def _as_float(value: object) -> float | None:
    if value is None or isinstance(value, bool):
        return None
    try:
        return float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None


def _iter_json_lines(stdout: bytes | str):
    text = stdout.decode("utf-8", "replace") if isinstance(stdout, bytes) else stdout
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            obj = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(obj, dict):
            yield obj


def _claude_model_from_payload(payload: dict) -> str | None:
    """Resolve the model id from a Claude result payload when present.

    Recent CLIs report per-model usage under `modelUsage`; when several models
    appear (e.g. a haiku sub-agent), pick the one with the largest token
    volume — that is the model the run's spend belongs to.
    """
    model_usage = payload.get("modelUsage")
    if isinstance(model_usage, dict) and model_usage:
        def _volume(item: tuple[str, object]) -> int:
            _, v = item
            if not isinstance(v, dict):
                return 0
            return sum(
                _as_nonneg_int(v.get(k))
                for k in (
                    "inputTokens",
                    "outputTokens",
                    "input_tokens",
                    "output_tokens",
                    "cache_read_input_tokens",
                    "cacheReadInputTokens",
                )
            )

        return max(model_usage.items(), key=_volume)[0]
    model = payload.get("model")
    return model if isinstance(model, str) and model else None


def claude_usage_from_payload(payload: dict) -> AgentUsage | None:
    """Extract usage from a Claude result event / `--output-format json` payload.

    Works for both the terminal `type == "result"` stream-json event and the
    top-level payload of `--output-format json` (which has the same fields but
    no `type` requirement).
    """
    usage = payload.get("usage")
    if not isinstance(usage, dict):
        return None

    api_ms = _as_float(payload.get("duration_api_ms"))
    return AgentUsage(
        input=_as_nonneg_int(usage.get("input_tokens")),
        output=_as_nonneg_int(usage.get("output_tokens")),
        cache_read=_as_nonneg_int(usage.get("cache_read_input_tokens")),
        cache_create=_as_nonneg_int(usage.get("cache_creation_input_tokens")),
        model=_claude_model_from_payload(payload),
        api_time_s=api_ms / 1000.0 if api_ms is not None else None,
        cli_reported_cost_usd=_as_float(payload.get("total_cost_usd")),
    )


def claude_usage_from_stream(stdout: bytes | str) -> AgentUsage | None:
    """Extract usage from a Claude `--output-format stream-json` transcript.

    Totals live only on the terminal `type == "result"` event; a stream that
    never reached it (timeout, crash) yields `None`.
    """
    result_event: dict | None = None
    for obj in _iter_json_lines(stdout):
        if obj.get("type") == "result":
            result_event = obj
    if result_event is None:
        return None
    return claude_usage_from_payload(result_event)


def codex_usage_from_stream(stdout: bytes | str) -> AgentUsage | None:
    """Extract usage from a `codex exec --json` event stream.

    Codex usage payloads are cumulative; keep the **latest** complete payload
    rather than summing events (summing would multiply the total). Handles
    both the flat `{"usage": {...}}` event shape and the nested
    `{"msg": {"type": "token_count", "info": {"total_token_usage": {...}}}}`
    shape.
    """
    token_payload: dict | None = None
    model: str | None = None
    cost: float | None = None
    api_time_s: float | None = None

    for obj in _iter_json_lines(stdout):
        msg = obj.get("msg")
        if isinstance(msg, dict):
            info = msg.get("info")
            if isinstance(info, dict):
                total = info.get("total_token_usage")
                if isinstance(total, dict) and "input_tokens" in total:
                    token_payload = total
            if isinstance(msg.get("model"), str):
                model = msg["model"]
        usage = obj.get("usage")
        if isinstance(usage, dict) and "input_tokens" in usage:
            token_payload = usage
        if isinstance(obj.get("model"), str) and obj["model"]:
            model = obj["model"]
        found_cost = _as_float(obj.get("total_cost_usd"))
        if found_cost is None:
            found_cost = _as_float(obj.get("cost_usd"))
        if found_cost is not None:
            cost = found_cost
        for key, scale in (("duration_s", 1.0), ("duration_ms", 1000.0)):
            value = _as_float(obj.get(key))
            if value is not None:
                api_time_s = value / scale

    if token_payload is None:
        return None

    input_tokens = _as_nonneg_int(token_payload.get("input_tokens"))
    cached = _as_nonneg_int(token_payload.get("cached_input_tokens"))
    cached = min(cached, input_tokens)
    return AgentUsage(
        input=input_tokens - cached,
        output=_as_nonneg_int(token_payload.get("output_tokens")),
        cache_read=cached,
        cache_create=0,
        model=model,
        api_time_s=api_time_s,
        cli_reported_cost_usd=cost,
    )


def _add_optional(a: float | None, b: float | None) -> float | None:
    """Sum two optional numbers, staying `None` only when both are."""
    if a is None:
        return b
    if b is None:
        return a
    return a + b


def merge_agent_usage(
    earlier: AgentUsage | None, later: AgentUsage | None
) -> AgentUsage | None:
    """Add up the usage of two invocations that produced one result.

    Needed by the step-5 retry: an attempt that died of a rate limit has usually
    already paid for tokens -- a timed-out claude call carries the usage parsed
    out of its partial stream, and codex emits `token_count` events before it
    gives up -- so returning only the winning attempt's usage would bill the run
    for less than it spent. Cost under-reporting is the one error class this
    codebase cannot tolerate silently, since the figure is quoted to people.

    Counters add. `model` takes the first one named, on the assumption that
    retrying a call does not change the model under it. `api_time_s` and
    `cli_reported_cost_usd` add across whichever attempts reported them, and stay
    `None` only when nobody did -- distinguishing "no time spent" from "the CLI
    never said", which the rate table depends on.
    """
    if earlier is None:
        return later
    if later is None:
        return earlier
    return AgentUsage(
        input=earlier.input + later.input,
        output=earlier.output + later.output,
        cache_read=earlier.cache_read + later.cache_read,
        cache_create=earlier.cache_create + later.cache_create,
        model=earlier.model or later.model,
        api_time_s=_add_optional(earlier.api_time_s, later.api_time_s),
        cli_reported_cost_usd=_add_optional(
            earlier.cli_reported_cost_usd, later.cli_reported_cost_usd
        ),
    )


__all__ = [
    "AgentUsage",
    "CliUsage",
    "claude_usage_from_payload",
    "claude_usage_from_stream",
    "codex_usage_from_stream",
    "merge_agent_usage",
]
