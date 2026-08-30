"""Claude Code subprocess runner for the modules extractor (legacy path).

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
`signal_pipeline._subprocess_util`. The provider-neutral stream/usage/artifact
helpers are shared with the two-phase path in `claude_stage`. This file is the
parts specific to the single-shot ProjectTree extraction: schema, prompt
delivery, retry loop, and result parsing.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from pydantic import ValidationError

from spotlights_engine.modules_extractor.claude_stage import (
    MAX_SCHEMA_BYTES as _MAX_SCHEMA_BYTES,
)
from spotlights_engine.modules_extractor.claude_stage import (
    add_optional_float as _add_optional_float,
)
from spotlights_engine.modules_extractor.claude_stage import (
    add_optional_int as _add_optional_int,
)
from spotlights_engine.modules_extractor.claude_stage import as_float as _as_float
from spotlights_engine.modules_extractor.claude_stage import as_int as _as_int
from spotlights_engine.modules_extractor.claude_stage import clean_env as _clean_env
from spotlights_engine.modules_extractor.claude_stage import (
    duration_seconds as _duration_seconds,
)
from spotlights_engine.modules_extractor.claude_stage import (
    extract_result_event as _extract_result_event,
)
from spotlights_engine.modules_extractor.claude_stage import (
    final_message_text as _final_message_text,
)
from spotlights_engine.modules_extractor.claude_stage import notify as _notify
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

_VALIDATION_RETRIES = 1
_RETRY_PAYLOAD_MAX_CHARS = 80_000


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
    claude_model: str | None = None,
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
    if claude_model:
        argv += ["--model", claude_model]

    attempt_prompt = prompt
    attempts = _VALIDATION_RETRIES + 1
    total_duration_s = 0.0
    total_cost_usd: float | None = None
    total_input_tokens: int | None = None
    total_output_tokens: int | None = None
    for attempt in range(attempts):
        if artifacts_dir is not None and attempt:
            (artifacts_dir / f"prompt_attempt_{attempt}.md").write_text(
                attempt_prompt, encoding="utf-8"
            )

        try:
            result = run_streaming_claude(
                argv=argv,
                prompt=attempt_prompt,
                env=_clean_env(),
                cwd=repo_path,
                timeout_s=timeout_s,
                on_event=on_event,
            )
        except StreamingTimeout as exc:
            if artifacts_dir is not None:
                _write_streams(artifacts_dir, exc.stdout, exc.stderr, attempt=attempt)
            raise ExtractorAgentError(
                f"claude timed out after {exc.duration_s:.1f}s",
                timeout_s=timeout_s,
            ) from exc

        if artifacts_dir is not None:
            _write_streams(artifacts_dir, result.stdout, result.stderr, attempt=attempt)

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
        usage = result_event.get("usage") or {}
        reported_duration = _duration_seconds(result_event)
        total_duration_s += (
            reported_duration if reported_duration is not None else result.duration_s
        )
        total_cost_usd = _add_optional_float(
            total_cost_usd,
            _as_float(result_event.get("total_cost_usd")),
        )
        total_input_tokens = _add_optional_int(
            total_input_tokens,
            _as_int(usage.get("input_tokens")),
        )
        total_output_tokens = _add_optional_int(
            total_output_tokens,
            _as_int(usage.get("output_tokens")),
        )

        payload = _final_message_text(result_event)
        if not payload.strip():
            raise ExtractorAgentError("claude returned an empty final message")

        if artifacts_dir is not None:
            (artifacts_dir / "last_message.json").write_text(payload, encoding="utf-8")
            (artifacts_dir / f"last_message_attempt_{attempt}.json").write_text(
                payload, encoding="utf-8"
            )

        try:
            tree = ProjectTree.model_validate_json(payload)
        except ValidationError as exc:
            if artifacts_dir is not None:
                (artifacts_dir / f"validation_error_attempt_{attempt}.txt").write_text(
                    str(exc), encoding="utf-8"
                )
            if attempt < _VALIDATION_RETRIES:
                _notify(
                    on_event,
                    "extractor: ProjectTree validation failed; retrying once with "
                    "validation feedback",
                )
                attempt_prompt = _validation_retry_prompt(prompt, exc, payload)
                continue
            raise ExtractorValidationError(
                f"ProjectTree validation failed after {attempts} attempts: {exc}",
                attempts=attempts,
                payload_preview=payload[:1000],
            ) from exc

        if artifacts_dir is not None:
            tree.to_json(artifacts_dir / "project_tree.json")

        invocation = ExtractionInvocation(
            session_id=result_event.get("session_id"),
            duration_s=total_duration_s,
            cost_usd=total_cost_usd,
            input_tokens=total_input_tokens,
            output_tokens=total_output_tokens,
        )

        return ExtractionRunResult(
            project_tree=tree,
            invocation=invocation,
            raw_payload=payload,
        )

    raise AssertionError("unreachable: extraction attempts loop exhausted")


def _write_streams(
    artifacts_dir: Path,
    stdout: bytes,
    stderr: bytes,
    *,
    attempt: int | None = None,
) -> None:
    (artifacts_dir / "raw_stdout.log").write_bytes(stdout or b"")
    (artifacts_dir / "raw_stderr.log").write_bytes(stderr or b"")
    if attempt is not None:
        (artifacts_dir / f"raw_stdout_attempt_{attempt}.log").write_bytes(stdout or b"")
        (artifacts_dir / f"raw_stderr_attempt_{attempt}.log").write_bytes(stderr or b"")


def _validation_retry_prompt(prompt: str, exc: ValidationError, payload: str) -> str:
    payload_preview = payload
    if len(payload_preview) > _RETRY_PAYLOAD_MAX_CHARS:
        payload_preview = payload_preview[:_RETRY_PAYLOAD_MAX_CHARS] + "\n...<truncated>"
    return (
        f"{prompt}\n\n"
        "## Retry after local ProjectTree validation failure\n\n"
        "Your previous final JSON failed Spotlights' ProjectTree validators. "
        "Repair it and return ONLY one complete valid JSON object in the same schema.\n\n"
        "Validation error:\n"
        f"{exc}\n\n"
        "Common fixes:\n"
        "- Every module/submodule `name` must equal the normalized basename of its "
        "`path`.\n"
        "- A child submodule `path` must be nested under its parent and must point "
        "to the real filesystem unit represented by that child.\n"
        "- Do not split one directory into conceptual children that reuse the "
        "parent path. If the code has conceptual responsibilities inside one "
        "flat directory and no real nested paths for those children, keep the "
        "directory as one LEAF and capture the responsibilities in its "
        "description/main_files roles.\n"
        "- Do not invent path suffixes that do not exist.\n\n"
        "Previous invalid JSON:\n"
        f"{payload_preview}\n"
    )


__all__ = [
    "ExtractionInvocation",
    "ExtractionRunResult",
    "run_extraction",
]
