"""Common interface for local CLI-backed agents."""

from __future__ import annotations

import json
import subprocess
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal, Protocol

from spotlights_engine.cli_agents.structured import StructuredOutputSpec

OutputFormat = Literal["text", "json", "stream-json"]
RuntimeName = Literal["codex", "claude", "gemini"]


@dataclass(frozen=True)
class CliAgentRequest:
    """One prompt sent to a local CLI agent."""

    prompt: str
    cwd: Path | str | None = None
    model: str | None = None
    profile: str | None = None
    system_prompt: str | None = None
    output_format: OutputFormat = "json"
    timeout_seconds: int | None = None
    allow_write: bool = False
    allow_network: bool = False
    env: Mapping[str, str] = field(default_factory=dict)
    extra_args: Sequence[str] = ()
    structured_output: StructuredOutputSpec | None = None


@dataclass(frozen=True)
class CliAgentResult:
    """Normalized result returned by a local CLI agent subprocess."""

    runtime: RuntimeName
    command: list[str]
    returncode: int
    text: str
    stdout: str
    stderr: str
    parsed_json: Any | None = None
    structured_data: Any | None = None
    structured_error: str | None = None
    session_id: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def ok(self) -> bool:
        return self.returncode == 0

    def raise_for_status(self) -> CliAgentResult:
        if not self.ok:
            raise RuntimeError(
                f"{self.runtime} CLI failed with exit code {self.returncode}\n"
                f"COMMAND: {self.command!r}\nSTDOUT:\n{self.stdout}\nSTDERR:\n{self.stderr}"
            )
        return self

    def require_structured(self) -> Any:
        if self.structured_error is not None:
            raise ValueError(self.structured_error)
        if self.structured_data is None:
            raise ValueError("result does not contain structured data")
        return self.structured_data


class CliAgent(Protocol):
    """Protocol implemented by all local CLI agent adapters."""

    runtime: RuntimeName

    def build_command(self, request: CliAgentRequest) -> list[str]: ...

    def build_env(self, request: CliAgentRequest) -> dict[str, str]: ...

    def run(self, request: CliAgentRequest) -> CliAgentResult: ...


def effective_output_format(request: CliAgentRequest) -> OutputFormat:
    """Use machine-readable CLI output whenever structured data is requested."""
    if request.structured_output is not None:
        return "json"
    return request.output_format


def run_subprocess_agent(
    *,
    runtime: RuntimeName,
    command: list[str],
    env: Mapping[str, str],
    cwd: Path | str | None,
    timeout_seconds: int | None,
    stdin: str | None = None,
    structured_output: StructuredOutputSpec | None = None,
) -> CliAgentResult:
    """Run a CLI command and normalize text/JSON/session output."""
    proc = subprocess.run(
        command,
        input=stdin,
        text=True,
        capture_output=True,
        cwd=str(Path(cwd or ".").expanduser().resolve()),
        timeout=timeout_seconds,
        env=dict(env),
        check=False,
    )
    parsed = parse_jsonish(proc.stdout) or parse_jsonish(proc.stderr)
    text = extract_text(parsed) if parsed is not None else proc.stdout
    structured_data: Any | None = None
    structured_error: str | None = None
    if structured_output is not None:
        try:
            structured_data = structured_output.parse(text=text, parsed_json=parsed)
        except (TypeError, ValueError) as exc:
            structured_error = str(exc)
    return CliAgentResult(
        runtime=runtime,
        command=command,
        returncode=proc.returncode,
        text=text,
        stdout=proc.stdout,
        stderr=proc.stderr,
        parsed_json=parsed,
        structured_data=structured_data,
        structured_error=structured_error,
        session_id=extract_session_id(parsed),
    )


def parse_jsonish(text: str) -> Any | None:
    stripped = text.strip()
    if not stripped:
        return None
    try:
        return json.loads(stripped)
    except json.JSONDecodeError:
        pass
    events: list[Any] = []
    for line in stripped.splitlines():
        try:
            events.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return events or None


def extract_text(payload: Any) -> str:
    if isinstance(payload, dict):
        for key in ("result", "text", "message", "content", "answer"):
            value = payload.get(key)
            if isinstance(value, str):
                return value
        error = payload.get("error")
        if isinstance(error, dict) and isinstance(error.get("message"), str):
            return error["message"]
        return json.dumps(payload, ensure_ascii=False)
    if isinstance(payload, list):
        chunks: list[str] = []
        for event in payload:
            if isinstance(event, dict):
                text = extract_text(event)
                if text and text != json.dumps(event, ensure_ascii=False):
                    chunks.append(text)
        return "\n".join(chunks) if chunks else json.dumps(payload, ensure_ascii=False)
    return str(payload)


def extract_session_id(payload: Any) -> str | None:
    if isinstance(payload, dict):
        for key in ("session_id", "sessionId", "id"):
            value = payload.get(key)
            if isinstance(value, str) and value:
                return value
    if isinstance(payload, list):
        for event in reversed(payload):
            session_id = extract_session_id(event)
            if session_id:
                return session_id
    return None
