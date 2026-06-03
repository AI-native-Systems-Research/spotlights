"""Claude Code subprocess runner for the modules extractor.

The agent is invoked non-interactively over the target repo: `cwd` is set to
`repo_path`, the prompt is sent on stdin, and the result is constrained by a
JSON schema derived from `ProjectTree`. The single terminal `result` event
from `stream-json` carries the validated object on `structured_output`,
which we parse back into a `ProjectTree`.

Permission mode is `plan`, which pre-approves the read/list tools (Bash,
Read, Glob, Grep) needed for a structural map and blocks mutations. Older
CLIs (≤ 2.1.87) auto-denied `ExitPlanMode` in non-interactive `-p` runs,
sending the agent into a retry loop until `--max-turns` was exhausted; CLI
2.1.140 fixes that, so plan mode is once again the right choice here.

Subprocess plumbing — argv resolution (Windows shim bypass), live streaming
via reader threads, deadline-based kill — lives in
`signal_pipeline._subprocess_util`. This file is the parts specific to the
ProjectTree extraction: schema, prompt delivery, and result parsing.
"""

from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

from pydantic import ValidationError

from spotlights_engine.modules_extractor.errors import (
    ExtractorAgentError,
    ExtractorSetupError,
    ExtractorValidationError,
)
from spotlights_engine.schemas.project import ProjectTree
from spotlights_engine.signal_pipeline._subprocess_util import (
    ClaudeResolutionError,
    StreamingTimeout,
    default_on_event,
    resolve_claude_argv0,
    run_streaming_claude,
)


_DROP_EXACT = frozenset(
    {
        "OPENAI_BASE_URL",
        "OPENAI_API_BASE",
        "ANTHROPIC_BASE_URL",
        "HTTP_PROXY",
        "HTTPS_PROXY",
        "ALL_PROXY",
        "VIRTUAL_ENV",
    }
)
_DROP_PREFIX = ("VSCODE_", "OPTQUEST_", "SPOTLIGHTS_")

# Linux MAX_ARG_STRLEN is 131_072 bytes per argument. `claude --json-schema`
# inlines the schema as one argv string; leave headroom for environment growth.
_MAX_SCHEMA_BYTES = 120_000


def _clean_env() -> dict[str, str]:
    env = os.environ.copy()
    for key in list(env):
        if key in _DROP_EXACT or key.startswith(_DROP_PREFIX):
            env.pop(key)
    return env


@dataclass
class ExtractionInvocation:
    """Side-channel telemetry from one Claude run."""

    session_id: str | None
    duration_s: float
    cost_usd: float | None
    input_tokens: int | None
    output_tokens: int | None


@dataclass
class ExtractionRunResult:
    project_tree: ProjectTree
    invocation: ExtractionInvocation
    raw_payload: str


def run_extraction(
    *,
    repo_path: Path,
    prompt: str,
    claude_bin: str = "claude",
    max_turns: int = 60,
    timeout_s: int = 1800,
    artifacts_dir: Path | None = None,
    on_event: Callable[[str], None] | None = default_on_event,
) -> ExtractionRunResult:
    """Invoke Claude Code over `repo_path` and return a validated `ProjectTree`.

    `artifacts_dir`, when provided, receives the rendered prompt, the JSON
    schema handed to the CLI, the raw stdout/stderr, and the final structured
    payload. The caller owns directory creation and uniqueness; this function
    writes into the directory as-is.

    `on_event` is called with a one-line summary of each stream-json event as
    it arrives. Default is `print(..., flush=True)`. Pass `None` to silence.
    """
    try:
        argv0 = resolve_claude_argv0(claude_bin)
    except ClaudeResolutionError as e:
        raise ExtractorSetupError(str(e), executable=claude_bin) from e

    if not repo_path.exists() or not repo_path.is_dir():
        raise ExtractorSetupError(
            f"repo_path does not exist or is not a directory: {repo_path}",
            repo_path=str(repo_path),
        )

    schema_text = json.dumps(ProjectTree.model_json_schema(), indent=2)
    schema_bytes = len(schema_text.encode("utf-8"))
    if schema_bytes >= _MAX_SCHEMA_BYTES:
        raise ExtractorSetupError(
            f"ProjectTree schema ({schema_bytes} bytes) exceeds claude "
            f"--json-schema argv ceiling ({_MAX_SCHEMA_BYTES} bytes)",
            schema_bytes=schema_bytes,
            limit=_MAX_SCHEMA_BYTES,
        )

    if artifacts_dir is not None:
        (artifacts_dir / "prompt.md").write_text(prompt, encoding="utf-8")
        (artifacts_dir / "schema.json").write_text(schema_text, encoding="utf-8")

    argv = argv0 + [
        "-p",
        "--output-format",
        "stream-json",
        "--verbose",
        "--json-schema",
        schema_text,
        "--permission-mode",
        "plan",
        "--max-turns",
        str(max_turns),
    ]

    try:
        result = run_streaming_claude(
            argv=argv,
            prompt=prompt,
            env=_clean_env(),
            cwd=repo_path,
            timeout_s=timeout_s,
            on_event=on_event,
        )
    except StreamingTimeout as exc:
        if artifacts_dir is not None:
            _write_streams(artifacts_dir, exc.stdout, exc.stderr)
        raise ExtractorAgentError(
            f"claude timed out after {exc.duration_s:.1f}s",
            timeout_s=timeout_s,
        ) from exc

    if artifacts_dir is not None:
        _write_streams(artifacts_dir, result.stdout, result.stderr)

    if result.returncode != 0:
        stderr_tail = result.stderr[-500:].decode("utf-8", "replace")
        stdout_tail = result.stdout[-500:].decode("utf-8", "replace")
        raise ExtractorAgentError(
            f"claude exit={result.returncode}",
            returncode=result.returncode,
            stderr_tail=stderr_tail,
            stdout_tail=stdout_tail,
        )

    result_event = _extract_result_event(result.stdout)
    if result_event is None:
        raise ExtractorAgentError(
            "claude stream-json had no terminal result event",
        )

    payload = _final_message_text(result_event)
    if not payload.strip():
        raise ExtractorAgentError("claude returned an empty final message")

    if artifacts_dir is not None:
        (artifacts_dir / "last_message.json").write_text(payload, encoding="utf-8")

    try:
        tree = ProjectTree.model_validate_json(payload)
    except ValidationError as exc:
        raise ExtractorValidationError(
            f"ProjectTree validation failed: {exc}",
            payload_preview=payload[:1000],
        ) from exc

    if artifacts_dir is not None:
        tree.to_json(artifacts_dir / "project_tree.json")

    usage = result_event.get("usage") or {}
    reported_duration = _duration_seconds(result_event)
    invocation = ExtractionInvocation(
        session_id=result_event.get("session_id"),
        duration_s=reported_duration if reported_duration is not None else result.duration_s,
        cost_usd=_as_float(result_event.get("total_cost_usd")),
        input_tokens=_as_int(usage.get("input_tokens")),
        output_tokens=_as_int(usage.get("output_tokens")),
    )

    return ExtractionRunResult(project_tree=tree, invocation=invocation, raw_payload=payload)


def _write_streams(artifacts_dir: Path, stdout: bytes, stderr: bytes) -> None:
    (artifacts_dir / "raw_stdout.log").write_bytes(stdout or b"")
    (artifacts_dir / "raw_stderr.log").write_bytes(stderr or b"")


def _extract_result_event(stdout: bytes) -> dict | None:
    last_event: dict | None = None
    for line in stdout.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            obj = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ExtractorAgentError(
                f"claude stream-json line not JSON: {exc}",
            ) from exc
        if not isinstance(obj, dict):
            raise ExtractorAgentError("claude stream-json line was not an object")
        last_event = obj
    if last_event is None or last_event.get("type") != "result":
        return None
    return last_event


def _final_message_text(result_event: dict) -> str:
    structured = result_event.get("structured_output")
    if isinstance(structured, (dict, list)):
        return json.dumps(structured)
    result = result_event.get("result")
    if isinstance(result, str) and result:
        return result
    message = result_event.get("message") or {}
    content = message.get("content")
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        chunks = [c.get("text", "") for c in content if isinstance(c, dict)]
        return "".join(chunks)
    return ""


def _duration_seconds(event: dict) -> float | None:
    for key in ("duration_s", "elapsed_s"):
        value = _as_float(event.get(key))
        if value is not None:
            return value
    for key in ("duration_ms", "elapsed_ms"):
        value = _as_float(event.get(key))
        if value is not None:
            return value / 1000.0
    return None


def _as_int(v: object) -> int | None:
    if v is None:
        return None
    try:
        return int(v)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None


def _as_float(v: object) -> float | None:
    if v is None:
        return None
    try:
        return float(v)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None


__all__ = [
    "ExtractionInvocation",
    "ExtractionRunResult",
    "run_extraction",
]
