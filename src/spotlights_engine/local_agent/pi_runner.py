"""Run the `pi` coding agent against the local Qwen vLLM backend.

`pi` (pi-coding-agent) speaks OpenAI-format to the VELA-served Qwen model; its
`--mode json` output is a pi-native NDJSON, NOT Claude-Code stream-json. This
module runs pi non-interactively and normalizes its stream into a small,
transport-neutral `PiRun`. The dispatch adapters (`dispatch.py`) translate that
into whatever result shape a given engine seam already parses, so no downstream
parser has to learn about pi.

The model is served by vLLM on the VELA OpenShift cluster and reached locally
via `oc port-forward`; pi talks to it through its `vela-qwen` provider entry.
The port-forward + login bootstrap lives in the co-located `qwen.sh` (moved
here from `scripts/`); `ensure_backend` shells out to it when the endpoint is
down.
"""

from __future__ import annotations

import json
import shutil
import subprocess
import time
import urllib.request
from dataclasses import dataclass
from pathlib import Path

PI_PROVIDER = "vela-qwen"
PI_THINKING = "xhigh"
PI_DEFAULT_MODEL = "Qwen/Qwen3.8-27B"
_BACKEND_MODELS_URL = "http://localhost:18000/v1/models"
_QWEN_SH = Path(__file__).with_name("qwen.sh")


class LocalAgentError(RuntimeError):
    """pi could not be launched or the local backend is unreachable."""


class LocalAgentTimeout(RuntimeError):
    """pi exceeded the wall-clock deadline; partial streams are attached."""

    def __init__(self, *, timeout_s: float, duration_s: float, stdout: str, stderr: str) -> None:
        super().__init__(f"pi timed out after {duration_s:.1f}s")
        self.timeout_s = timeout_s
        self.duration_s = duration_s
        self.stdout = stdout
        self.stderr = stderr


def is_local_model(model: str | None) -> bool:
    """True when `model` should route to the local pi/Qwen backend."""
    return bool(model) and "qwen" in model.lower()


@dataclass
class PiRun:
    """Normalized outcome of one non-interactive pi invocation."""

    returncode: int
    final_text: str
    session_id: str | None
    input_tokens: int
    output_tokens: int
    duration_s: float
    stdout: str
    stderr: str
    model: str | None = None


def _pi_bin() -> str | None:
    return shutil.which("pi")


def _backend_up() -> bool:
    try:
        with urllib.request.urlopen(_BACKEND_MODELS_URL, timeout=3):
            return True
    except Exception:  # noqa: BLE001 - any failure means "not reachable"
        return False


def ensure_backend(env: dict[str, str], *, timeout_s: int = 120) -> None:
    """Ensure the vLLM port-forward is live (delegates to qwen.sh)."""
    if _backend_up():
        return
    if _QWEN_SH.exists():
        subprocess.run(
            ["/usr/bin/env", "zsh", str(_QWEN_SH), "ensure"],
            env=env,
            timeout=timeout_s,
            check=False,
        )
    if not _backend_up():
        raise LocalAgentError(
            "local Qwen backend unreachable at "
            f"{_BACKEND_MODELS_URL}; run `qwen.sh ensure` (oc login + port-forward)"
        )


def wrap_schema_prompt(prompt: str, schema_text: str) -> str:
    """Append a schema contract so pi emits one schema-conforming JSON object."""
    return (
        f"{prompt}\n\n"
        "## Required output format\n"
        "Return your FINAL answer as a SINGLE JSON object that validates against "
        "the JSON Schema below. Output ONLY that JSON object — no markdown code "
        "fences, no prose before or after it.\n\n"
        "JSON Schema:\n"
        f"{schema_text}\n"
    )


def extract_json_object(text: str) -> str | None:
    """Best-effort pull of one JSON object/array from a model's final text."""
    if not text:
        return None
    stripped = text.strip()
    if stripped.startswith("```"):
        # ```json\n...\n``` or ```\n...\n```
        body = stripped.split("\n", 1)[1] if "\n" in stripped else ""
        if body.endswith("```"):
            body = body[: -3]
        stripped = body.strip()
    # Locate the outermost {...} or [...] span and validate it parses.
    for open_ch, close_ch in (("{", "}"), ("[", "]")):
        start = stripped.find(open_ch)
        end = stripped.rfind(close_ch)
        if start != -1 and end != -1 and end > start:
            candidate = stripped[start : end + 1]
            try:
                json.loads(candidate)
            except json.JSONDecodeError:
                continue
            return candidate
    return None


def _parse_pi_stream(stdout: str) -> tuple[str, str | None, int, int]:
    """Return (final_assistant_text, session_id, input_tokens, output_tokens)."""
    session_id: str | None = None
    final_text = ""
    input_tokens = 0
    output_tokens = 0
    for line in stdout.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            ev = json.loads(line)
        except json.JSONDecodeError:
            continue
        if not isinstance(ev, dict):
            continue
        etype = ev.get("type")
        if etype == "session" and ev.get("id"):
            session_id = ev.get("id")
            continue
        if etype != "message_end":
            continue
        message = ev.get("message") or {}
        if message.get("role") != "assistant":
            continue
        content = message.get("content") or []
        chunks = [
            c.get("text", "")
            for c in content
            if isinstance(c, dict) and c.get("type") == "text"
        ]
        text = "".join(chunks).strip()
        if text:
            final_text = text  # last assistant turn wins
        usage = message.get("usage") or {}
        if isinstance(usage, dict):
            input_tokens = int(usage.get("input", input_tokens) or input_tokens)
            output_tokens = int(usage.get("output", output_tokens) or output_tokens)
    return final_text, session_id, input_tokens, output_tokens


def run_pi(
    *,
    prompt: str,
    cwd: Path | str | None,
    env: dict[str, str],
    timeout_s: float,
    model: str | None,
    schema_text: str | None = None,
    read_only: bool = True,
) -> PiRun:
    """Run `pi -p --mode json` and normalize its NDJSON into a `PiRun`."""
    pi = _pi_bin()
    if pi is None:
        raise LocalAgentError(
            "pi CLI not on PATH — install @earendil-works/pi-coding-agent"
        )
    ensure_backend(env)

    full_prompt = wrap_schema_prompt(prompt, schema_text) if schema_text else prompt
    cmd = [
        pi,
        "--provider",
        PI_PROVIDER,
        "--model",
        model or PI_DEFAULT_MODEL,
        "--thinking",
        PI_THINKING,
        "-p",
        "--mode",
        "json",
        "--no-session",
        # Skip global AGENTS.md/CLAUDE.md discovery: the user's ~/.pi caveman
        # AGENTS.md would otherwise leak prose into schema-constrained output.
        "--no-context-files",
    ]
    if read_only:
        cmd += ["--exclude-tools", "edit,write"]

    start = time.monotonic()
    try:
        completed = subprocess.run(
            cmd,
            input=full_prompt,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            cwd=str(Path(cwd).expanduser().resolve()) if cwd is not None else None,
            env=env,
            timeout=timeout_s,
            check=False,
        )
    except subprocess.TimeoutExpired as exc:
        duration = time.monotonic() - start
        raise LocalAgentTimeout(
            timeout_s=timeout_s,
            duration_s=duration,
            stdout=exc.stdout or "" if isinstance(exc.stdout, str) else "",
            stderr=exc.stderr or "" if isinstance(exc.stderr, str) else "",
        ) from exc

    duration = time.monotonic() - start
    final_text, session_id, input_tokens, output_tokens = _parse_pi_stream(
        completed.stdout
    )
    return PiRun(
        returncode=completed.returncode,
        final_text=final_text,
        session_id=session_id,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        duration_s=duration,
        stdout=completed.stdout,
        stderr=completed.stderr,
        model=model or PI_DEFAULT_MODEL,
    )


__all__ = [
    "LocalAgentError",
    "LocalAgentTimeout",
    "PiRun",
    "PI_DEFAULT_MODEL",
    "PI_PROVIDER",
    "ensure_backend",
    "extract_json_object",
    "is_local_model",
    "run_pi",
    "wrap_schema_prompt",
]
