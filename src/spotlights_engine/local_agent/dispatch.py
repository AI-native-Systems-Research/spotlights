"""Model-based dispatch: route agent invocations to the local pi/Qwen backend.

Every spawn seam in the engine (Claude stream-json, `codex exec`, `claude -p`)
checks `is_local_model(model)`. When it matches, the seam calls one of the
adapters here instead of spawning the hosted CLI. Each adapter runs `pi` against
the local Qwen vLLM (`pi_runner.run_pi`) and translates the pi output into the
exact result shape that seam's downstream parser already consumes — a
synthesized Claude stream-json byte stream, or an `AgentExecResult`, or a
schema-conforming `last_message.json`. Non-qwen models are untouched.
"""

from __future__ import annotations

import json
from pathlib import Path

from spotlights_engine.costing.usage import AgentUsage
from spotlights_engine.local_agent.base import AgentExecResult
from spotlights_engine.local_agent.pi_runner import (
    LocalAgentTimeout,
    PiRun,
    extract_json_object,
    is_local_model,
    run_pi,
)


def model_from_argv(argv: list[str]) -> str | None:
    """Return the value following `--model` in a CLI argv, if present."""
    for i, tok in enumerate(argv):
        if tok == "--model" and i + 1 < len(argv):
            return argv[i + 1]
    return None


def schema_from_argv(argv: list[str]) -> str | None:
    """Return the value following `--json-schema` in a CLI argv, if present."""
    for i, tok in enumerate(argv):
        if tok == "--json-schema" and i + 1 < len(argv):
            return argv[i + 1]
    return None


def structured_from_pi(pi: PiRun, schema_text: str | None) -> object | None:
    """Parse pi's final text into a JSON object when a schema was requested."""
    if not schema_text:
        return None
    candidate = extract_json_object(pi.final_text)
    if candidate is None:
        return None
    try:
        return json.loads(candidate)
    except json.JSONDecodeError:
        return None


def synth_stream_json(
    pi: PiRun, *, structured: object | None, cwd: Path | str | None
) -> bytes:
    """Render a `PiRun` as Claude-Code stream-json bytes.

    Emits `system/init`, one assistant text event, and a terminal `result`
    event carrying `structured_output` (when parsed) plus token usage — exactly
    the events `extract_result_event` / stage-telemetry parsers expect.
    """
    sid = pi.session_id or "pi-local"
    events: list[dict] = [
        {
            "type": "system",
            "subtype": "init",
            "session_id": sid,
            "cwd": str(cwd) if cwd is not None else "",
            "model": pi.model,
            "tools": [],
        },
        {
            "type": "assistant",
            "session_id": sid,
            "message": {
                "role": "assistant",
                "content": [{"type": "text", "text": pi.final_text}],
            },
        },
    ]
    result_event: dict = {
        "type": "result",
        "subtype": "success",
        "is_error": pi.returncode != 0,
        "session_id": sid,
        "result": pi.final_text,
        "duration_ms": int(pi.duration_s * 1000),
        "duration_api_ms": int(pi.duration_s * 1000),
        "num_turns": 1,
        "total_cost_usd": 0.0,
        "model": pi.model,
        "usage": {
            "input_tokens": pi.input_tokens,
            "output_tokens": pi.output_tokens,
            "cache_read_input_tokens": 0,
            "cache_creation_input_tokens": 0,
        },
    }
    if structured is not None:
        result_event["structured_output"] = structured
    events.append(result_event)
    return ("\n".join(json.dumps(ev) for ev in events) + "\n").encode("utf-8")


def run_streaming_local(
    *,
    argv: list[str],
    prompt: str,
    env: dict[str, str],
    cwd: Path | str | None,
    timeout_s: float,
):
    """Local counterpart to `run_streaming_claude`; returns a `StreamingResult`.

    Reads the model + `--json-schema` out of the Claude argv, runs pi, and
    synthesizes an equivalent stream-json byte stream so the caller's existing
    parser path is unchanged.
    """
    # Imported here to avoid a package import cycle (the streaming util imports
    # this module lazily from inside run_streaming_claude).
    from spotlights_engine.signal_pipeline._subprocess_util import (
        StreamingResult,
        StreamingTimeout,
    )

    model = model_from_argv(argv)
    schema_text = schema_from_argv(argv)
    try:
        pi = run_pi(
            prompt=prompt,
            cwd=cwd,
            env=env,
            timeout_s=timeout_s,
            model=model,
            schema_text=schema_text,
        )
    except LocalAgentTimeout as exc:
        raise StreamingTimeout(
            timeout_s=exc.timeout_s,
            duration_s=exc.duration_s,
            stdout=b"",
            stderr=exc.stderr.encode("utf-8", "replace"),
        ) from exc

    structured = structured_from_pi(pi, schema_text)
    stdout = synth_stream_json(pi, structured=structured, cwd=cwd)
    return StreamingResult(
        stdout=stdout,
        stderr=pi.stderr.encode("utf-8", "replace"),
        returncode=pi.returncode,
        duration_s=pi.duration_s,
    )


def run_agent_exec_local(
    *,
    prompt: str,
    cwd: Path | str | None,
    env: dict[str, str],
    timeout_s: float,
    model: str | None,
    command: list[str],
    schema_text: str | None = None,
    prefer_structured: bool = True,
    check: bool = True,
) -> AgentExecResult:
    """Local counterpart for the `AgentExecResult`-shaped seams (codex/claude).

    When `schema_text` is given and `prefer_structured` is True, the extracted
    JSON object becomes `final_message`; otherwise pi's final text is used.
    """
    pi = run_pi(
        prompt=prompt,
        cwd=cwd,
        env=env,
        timeout_s=timeout_s,
        model=model,
        schema_text=schema_text,
    )
    final = pi.final_text
    if schema_text and prefer_structured:
        candidate = extract_json_object(pi.final_text)
        if candidate is not None:
            final = candidate
    result = AgentExecResult(
        command=command,
        returncode=pi.returncode,
        stdout=pi.stdout,
        stderr=pi.stderr,
        final_message=final,
        usage=AgentUsage(
            input=pi.input_tokens, output=pi.output_tokens, model=pi.model
        ),
    )
    if check:
        result.raise_for_status()
    return result


def local_final_json(pi: PiRun, schema_text: str | None) -> str:
    """Return schema-conforming JSON text for `last_message.json`-style seams."""
    if schema_text:
        candidate = extract_json_object(pi.final_text)
        if candidate is not None:
            return candidate
    return pi.final_text


__all__ = [
    "is_local_model",
    "local_final_json",
    "model_from_argv",
    "run_agent_exec_local",
    "run_pi",
    "run_streaming_local",
    "schema_from_argv",
    "structured_from_pi",
    "synth_stream_json",
]
