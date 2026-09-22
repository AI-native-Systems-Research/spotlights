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
import logging
import shutil
import subprocess
import threading
import time
import urllib.request
from dataclasses import dataclass
from pathlib import Path

_log = logging.getLogger(__name__)

PI_PROVIDER = "vela-qwen"
PI_THINKING = "xhigh"
PI_DEFAULT_MODEL = "Qwen/Qwen3.8-27B"
_BACKEND_BASE = "http://localhost:18000/v1"
_BACKEND_MODELS_URL = f"{_BACKEND_BASE}/models"
_BACKEND_CHAT_URL = f"{_BACKEND_BASE}/chat/completions"
_QWEN_SH = Path(__file__).with_name("qwen.sh")

# How often the run_pi heartbeat logs liveness while the blocking subprocess
# runs. Qwen-over-proxy calls take minutes; without this a slow call and a
# hung call look identical (silent capture) until the wall-clock timeout.
_HEARTBEAT_INTERVAL_S = 30.0


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


@dataclass
class BackendHealth:
    """Result of an active liveness probe against the vLLM backend.

    Three failure modes are distinguished, because they need different fixes:

    - `reachable is False`: the `oc port-forward` tunnel is down or vLLM is not
      listening. Nothing generates; every `run_pi` call would hang until its
      wall-clock timeout. Fix: `qwen.sh ensure`.
    - `reachable is True, generating is False`: the models endpoint answers but
      a tiny completion did not return within the probe deadline. The backend
      is *blocked* (queue wedged, model deadlocked, or saturated) — the exact
      state a plain reachability check (`_backend_up`) cannot see and the one
      that made a run look silently hung.
    - both True: healthy; `latency_s` is the round-trip for a 1-token decode.
    """

    reachable: bool
    generating: bool
    latency_s: float | None
    detail: str

    @property
    def ok(self) -> bool:
        return self.reachable and self.generating


def probe_backend(
    model: str | None = None, *, timeout_s: float = 20.0
) -> BackendHealth:
    """Actively check whether the backend *generates*, not just accepts TCP.

    Sends a 1-token chat completion and measures round-trip latency. Used as an
    on-demand health check (`qwen.sh probe`) and as an optional preflight so a
    wedged-but-listening backend is caught before a multi-minute call, instead
    of after it silently times out.
    """
    if not _backend_up():
        return BackendHealth(
            reachable=False,
            generating=False,
            latency_s=None,
            detail=f"models endpoint unreachable at {_BACKEND_MODELS_URL}",
        )
    body = json.dumps(
        {
            "model": model or PI_DEFAULT_MODEL,
            "messages": [{"role": "user", "content": "ping"}],
            "max_tokens": 1,
            "temperature": 0,
            "stream": False,
        }
    ).encode("utf-8")
    req = urllib.request.Request(
        _BACKEND_CHAT_URL,
        data=body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    start = time.monotonic()
    try:
        with urllib.request.urlopen(req, timeout=timeout_s) as resp:
            payload = json.loads(resp.read().decode("utf-8", "replace"))
    except Exception as exc:  # noqa: BLE001 - reachable but not generating
        latency = time.monotonic() - start
        return BackendHealth(
            reachable=True,
            generating=False,
            latency_s=latency,
            detail=(
                f"reachable but no completion in {latency:.1f}s "
                f"(timeout={timeout_s:.0f}s): {type(exc).__name__}: {exc} "
                "— backend blocked/saturated"
            ),
        )
    latency = time.monotonic() - start
    if isinstance(payload, dict) and payload.get("choices"):
        return BackendHealth(
            reachable=True,
            generating=True,
            latency_s=latency,
            detail=f"generated in {latency:.2f}s",
        )
    return BackendHealth(
        reachable=True,
        generating=False,
        latency_s=latency,
        detail=f"completion returned no choices: {payload!r:.200}",
    )


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


def _heartbeat_loop(
    stop: threading.Event, start: float, model: str, timeout_s: float
) -> None:
    """Log liveness every `_HEARTBEAT_INTERVAL_S` while run_pi blocks.

    Cheap per beat: only a reachability check (`_backend_up`), never a
    generation probe — a probe would contend with the in-flight decode and
    could false-alarm under load. A dropped `oc port-forward` tunnel (the most
    common real "hang") makes the endpoint unreachable, so this catches it and
    warns that the call will now block until the wall-clock timeout, instead of
    the run sitting silent. For the deeper "listening but not generating" case,
    use `probe_backend` on demand (`qwen.sh probe`).
    """
    beat = 0
    while not stop.wait(_HEARTBEAT_INTERVAL_S):
        beat += 1
        elapsed = time.monotonic() - start
        if _backend_up():
            _log.info(
                "pi[%s] alive: %.0fs/%.0fs elapsed; backend reachable",
                model,
                elapsed,
                timeout_s,
            )
        else:
            _log.warning(
                "pi[%s] %.0fs/%.0fs elapsed; BACKEND UNREACHABLE at %s "
                "— call will block until timeout (oc port-forward dropped?)",
                model,
                elapsed,
                timeout_s,
                _BACKEND_MODELS_URL,
            )


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
    stop = threading.Event()
    heartbeat = threading.Thread(
        target=_heartbeat_loop,
        args=(stop, start, model or PI_DEFAULT_MODEL, timeout_s),
        daemon=True,
    )
    heartbeat.start()
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
        stop.set()
        duration = time.monotonic() - start
        raise LocalAgentTimeout(
            timeout_s=timeout_s,
            duration_s=duration,
            stdout=exc.stdout or "" if isinstance(exc.stdout, str) else "",
            stderr=exc.stderr or "" if isinstance(exc.stderr, str) else "",
        ) from exc
    finally:
        stop.set()

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
    "BackendHealth",
    "LocalAgentError",
    "LocalAgentTimeout",
    "PiRun",
    "PI_DEFAULT_MODEL",
    "PI_PROVIDER",
    "ensure_backend",
    "extract_json_object",
    "is_local_model",
    "probe_backend",
    "run_pi",
    "wrap_schema_prompt",
]
