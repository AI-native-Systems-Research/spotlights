"""Ops probe: does the hosted web-search tool actually reach a backend?

`WebSearch` (Anthropic-hosted) and `codex --search` (OpenAI-hosted) are
*server-side* tools. In this deployment both CLIs are pointed at a LiteLLM
gateway rather than the vendor endpoints, and a generic OpenAI-compatible proxy
does not necessarily forward hosted-tool calls. If it doesn't, the tool is
silently unavailable and the agent answers from parametric memory (the
hallucination path). This script proves — against the *same* live env the
engine uses — whether each runner's web search actually works.

opencode is different in kind: pointed at the same LiteLLM gateway (a plain
OpenAI-compatible provider), it has *no* vendor-hosted web_search to forward.
Instead it grounds with a **client-side** `webfetch`/`websearch` tool — the
model picks a URL/query and opencode itself makes the network call, then feeds
the result back. So the opencode probe's evidence of a working search is a
`webfetch`/`websearch` tool call that actually *completes* (reaches the
network) plus the same PyPI grounding, rather than a server-side tool block.

antigravity is different again: it is probed through the **Python SDK**
(`google-antigravity`), not a CLI. The Antigravity `agy` CLI is OAuth-only and
cannot be routed through a proxy, but the SDK's `GeminiAPIEndpoint(base_url=…,
api_key=…)` accepts a custom base URL + key, so we point it at the *same*
LiteLLM gateway. Credentials come from `~/.claude/settings.json` (the
`env.ANTHROPIC_BASE_URL` + `env.ANTHROPIC_AUTH_TOKEN` the gateway already uses)
or, when that file is not available, the same process environment. The model
defaults to `gcp/gemini-3.1-pro-preview`. The SDK exposes a
`BuiltinTools.SEARCH_WEB` tool; a WORKS verdict requires that tool to actually
be dispatched (surfaced as a `search_web` tool-call in the response stream),
plus the same PyPI grounding.

The LiteLLM gateway serves the model in the Gemini wire format, and some
LiteLLM versions emit Gemini streaming chunks as Python bytes-repr strings
inside SSE frames (`data: b'data: {...}'`), which the SDK's local harness
rejects mid-stream. The engine's Antigravity runner
(spotlights_engine.module_deep_research.antigravity_exec) fixes this with a
local normalizer proxy that unwraps those frames back to standard Gemini SSE;
by default this probe routes the SDK through the *same* normalizer, so the
verdict reflects the exact path the runner uses. In normalizer mode the real
bearer token is injected upstream by the proxy and the SDK only sees a
placeholder key (current harness builds crash on non-Gemini key formats and
custom endpoint headers). Pass --no-antigravity-sse-normalizer to probe the
gateway directly: then the token is sent as both the SDK API key and an
explicit bearer header, and a bytes-repr gateway stream surfaces as ERROR with
the gateway's own message rather than a false WORKS.

This is a manual/ops probe, NOT a unit test: it needs the real proxy plus
credentials and makes live network calls. Run it before trusting the Claude,
Codex, opencode or antigravity runners' web findings.

Two layers of evidence back a WORKS verdict:
  1. Invocation — the JSON event stream must show a hosted web-search tool call
     actually completing (for codex, a `web_search` item with a non-empty
     query the model chose), not merely the tool being offered/registered.
  2. Grounding — the answer must cite the *live* latest version, fetched
     independently from PyPI. A post-cutoff version can only appear if the
     search returned fresh data, so this proves the tool returned usable
     results rather than the model hallucinating. A proven invocation whose
     answer omits the ground truth is reported SUSPECT (disable with
     --no-ground-truth). Grounding applies to the codex, opencode and
     antigravity probes.

Usage:
    uv run --no-sync python scripts/verify_hosted_search.py
    uv run --no-sync python scripts/verify_hosted_search.py --only claude
    uv run --no-sync python scripts/verify_hosted_search.py --project uv
    uv run --no-sync python scripts/verify_hosted_search.py --only codex --no-ground-truth
    uv run --no-sync python scripts/verify_hosted_search.py --only opencode
    uv run --no-sync python scripts/verify_hosted_search.py --only antigravity
    uv run --no-sync python scripts/verify_hosted_search.py --only antigravity \
        --antigravity-model gemini-2.5-pro

Exit code is nonzero unless every probed runner is WORKS (i.e. SUSPECT, INERT,
or ERROR all fail), so this can gate a deployment check.
"""

from __future__ import annotations

import argparse
import asyncio
import contextlib
import json
import os
import re
import subprocess
import sys
import tempfile
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass
from pathlib import Path

# Where the LiteLLM gateway credentials live. The Antigravity SDK probe reads
# the gateway base_url + token straight from here (the same env the Claude CLI
# uses), so it hits the identical live gateway as the other runners.
DEFAULT_CLAUDE_SETTINGS = Path.home() / ".claude" / "settings.json"

# Default Antigravity model. This is a LiteLLM model-group name (provider/model
# form) served by the gateway, not a vendor model id.
DEFAULT_ANTIGRAVITY_MODEL = "gcp/gemini-3.1-pro-preview"

# The one correct answer requires a live search (fact newer than training
# cutoff). The sentinel gives the model a legal move when it *cannot* search,
# so a missing tool surfaces as INERT rather than a confident hallucination.
SENTINEL = "SEARCH_UNAVAILABLE"

_PROMPT_TEMPLATE = (
    "What is the latest released version of the project '{project}', and cite "
    "the exact source URL you searched to find it? Answer in one line. "
    "If you cannot run a web search, reply with exactly {sentinel} and nothing "
    "else — do not guess from memory."
)

# Substrings that, in the raw JSON event stream / transcript, evidence an
# actual hosted-tool invocation (not merely a plausible fresh-looking answer).
_CLAUDE_TOOL_MARKERS = ("web_search", "server_tool_use", '"type":"web_search')

# opencode grounds via client-side network tools rather than a hosted
# web_search; these are the tool names whose *completed* invocation proves it
# actually reached the network (webfetch = model-chosen URL fetch, websearch =
# plugin-provided search). See _opencode_search_tool.
_OPENCODE_SEARCH_TOOLS = ("webfetch", "websearch", "web_search", "web_fetch")

# The Antigravity SDK's hosted web-search builtin. A genuine invocation shows up
# as a `search_web` tool-call in the response stream (BuiltinTools.SEARCH_WEB ==
# "search_web"). We also treat the string appearing in a mid-turn gateway error
# payload as evidence the tool was dispatched (the LiteLLM gateway can emit the
# Gemini functionCall for search_web and then break the SSE stream before the
# SDK can surface it as a clean tool-call chunk).
_ANTIGRAVITY_SEARCH_TOOL = "search_web"
_ANTIGRAVITY_STREAM_PARSE_MARKERS = (
    "iterateresponsestream",
    "error unmarshalling data",
    "invalid character 'b' looking for beginning of value",
    "b'data:",
)

Verdict = str  # "WORKS" | "SUSPECT" | "INERT" | "ERROR"


@dataclass
class ProbeResult:
    runner: str
    verdict: Verdict
    detail: str


def pypi_latest_version(project: str, *, timeout: float = 15.0) -> str | None:
    """Fetch the true latest version of a PyPI project, independent of any LLM.

    Returns the ``info.version`` string (e.g. "0.11.26"), or None if the
    project isn't on PyPI or the fetch fails. Used to *ground* a runner's
    answer: a post-cutoff version can only appear if a live search actually
    returned fresh data, so matching it is proof the tool works — not merely
    that it was invoked.
    """
    url = f"https://pypi.org/pypi/{urllib.parse.quote(project)}/json"
    try:
        with urllib.request.urlopen(url, timeout=timeout) as resp:
            data = json.load(resp)
    except (urllib.error.URLError, TimeoutError, json.JSONDecodeError, ValueError):
        return None
    version = data.get("info", {}).get("version")
    return version if isinstance(version, str) and version.strip() else None


def _codex_search_query(stdout: str) -> str | None:
    """Return the query of a genuinely-invoked web_search, else None.

    Codex emits one JSON event per line. A real hosted search surfaces as a
    ``web_search`` item that reaches ``item.completed`` carrying a non-empty
    ``query`` the model chose (e.g. "PyPI uv latest version"). Merely
    registering/enabling the tool never produces such an item, so this is far
    stronger evidence than substring-matching "search" in the raw transcript —
    that string also appears in tool-schema/config events when the tool is
    inert.
    """
    for line in stdout.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            event = json.loads(line)
        except (json.JSONDecodeError, ValueError):
            continue
        if not isinstance(event, dict) or event.get("type") != "item.completed":
            continue
        item = event.get("item")
        if not isinstance(item, dict) or item.get("type") != "web_search":
            continue
        query = item.get("query")
        if isinstance(query, str) and query.strip():
            return query.strip()
    return None


def _prompt(project: str) -> str:
    return _PROMPT_TEMPLATE.format(project=project, sentinel=SENTINEL)


def probe_claude(project: str, *, claude_bin: str = "claude") -> ProbeResult:
    """Invoke `claude -p` with WebSearch allowed and inspect the event stream.

    Uses streaming/verbose JSON so tool-use blocks are visible; compact
    `--output-format json` may expose only the final result + metadata, not the
    `server_tool_use` / `web_search` blocks we need to see.
    """
    prompt = _prompt(project)
    cmd = [
        claude_bin,
        "-p",
        "--output-format",
        "stream-json",
        "--verbose",
        "--permission-mode",
        "acceptEdits",
        "--max-turns",
        "6",
        "--tools",
        "WebSearch",
        "--allowedTools",
        "WebSearch",
    ]
    try:
        completed = subprocess.run(
            cmd,
            input=prompt,
            capture_output=True,
            text=True,
            timeout=240,
            check=False,
        )
    except FileNotFoundError:
        return ProbeResult("claude", "ERROR", f"{claude_bin} not found on PATH")
    except subprocess.TimeoutExpired:
        return ProbeResult("claude", "ERROR", "claude probe timed out after 240s")

    transcript = completed.stdout + "\n" + completed.stderr
    low = transcript.lower()

    if completed.returncode != 0 and not transcript.strip():
        return ProbeResult(
            "claude", "ERROR", f"claude exited {completed.returncode} with no output"
        )

    tool_used = any(m in low for m in _CLAUDE_TOOL_MARKERS)
    sentinel_seen = SENTINEL in transcript

    if tool_used and not sentinel_seen:
        return ProbeResult(
            "claude", "WORKS", "web_search / server_tool_use block present in stream"
        )
    if sentinel_seen:
        return ProbeResult(
            "claude",
            "INERT",
            f"model emitted {SENTINEL} — could not run a web search",
        )
    if "web_search" in low and ("error" in low or "not available" in low):
        return ProbeResult("claude", "ERROR", "gateway rejected the hosted tool")
    # No tool block, no sentinel: model likely answered from memory. Do NOT
    # classify WORKS on a plausible answer alone — require explicit tool evidence.
    return ProbeResult(
        "claude",
        "INERT",
        "no hosted-tool block and no sentinel — answer likely from memory",
    )


def probe_codex(
    project: str,
    *,
    codex_bin: str = "codex",
    profile: str = "litellm",
    expected_version: str | None = None,
) -> ProbeResult:
    """Invoke `codex exec --search` (through the litellm profile) and inspect
    its JSON event stream for a web-search / tool event, else the sentinel.

    When ``expected_version`` (independent PyPI ground truth) is supplied, WORKS
    additionally requires that exact version to appear in the model's answer —
    proof the search returned *live data the model used*, not just that the tool
    was invoked. A proven invocation whose answer omits the ground truth is
    reported as SUSPECT (the gateway may have accepted the call but returned
    nothing, leaving the model to hallucinate)."""
    prompt = _prompt(project)
    with tempfile.TemporaryDirectory(prefix="verify-hosted-search-") as tmp:
        last_message = Path(tmp) / "codex_last.md"
        cmd = [
            codex_bin,
            "--profile",
            profile,
            "--ask-for-approval",
            "never",
            "--search",
            "exec",
            "--sandbox",
            "read-only",
            "--skip-git-repo-check",
            "--json",
            "--output-last-message",
            str(last_message),
            "-",
        ]
        try:
            completed = subprocess.run(
                cmd,
                input=prompt,
                capture_output=True,
                text=True,
                timeout=240,
                check=False,
            )
        except FileNotFoundError:
            return ProbeResult("codex", "ERROR", f"{codex_bin} not found on PATH")
        except subprocess.TimeoutExpired:
            return ProbeResult("codex", "ERROR", "codex probe timed out after 240s")

        final = ""
        if last_message.exists():
            final = last_message.read_text(encoding="utf-8", errors="replace")

    transcript = completed.stdout + "\n" + completed.stderr + "\n" + final
    low = transcript.lower()

    if completed.returncode != 0 and not transcript.strip():
        return ProbeResult("codex", "ERROR", f"codex exited {completed.returncode} with no output")

    # Parse the JSON event stream for a completed web_search carrying a
    # non-empty query — proof the tool actually ran, not merely that it was
    # offered. The sentinel is checked against the whole transcript so a
    # "cannot search" reply in the final message is never masked.
    search_query = _codex_search_query(completed.stdout)
    sentinel_seen = SENTINEL in transcript

    if search_query is not None and not sentinel_seen:
        if expected_version is None:
            return ProbeResult(
                "codex", "WORKS", f"web_search completed with query {search_query!r}"
            )
        if expected_version in (final or transcript):
            return ProbeResult(
                "codex",
                "WORKS",
                f"web_search ran ({search_query!r}) and answer cites live "
                f"version {expected_version}",
            )
        return ProbeResult(
            "codex",
            "SUSPECT",
            f"web_search ran ({search_query!r}) but answer omits ground-truth "
            f"version {expected_version} — search may have returned nothing",
        )
    if sentinel_seen:
        return ProbeResult(
            "codex", "INERT", f"model emitted {SENTINEL} — could not run a web search"
        )
    if ("search" in low) and ("error" in low or "not supported" in low or "unavailable" in low):
        return ProbeResult("codex", "ERROR", "gateway rejected the hosted --search tool")
    return ProbeResult(
        "codex",
        "INERT",
        "no web-search event and no sentinel — answer likely from memory",
    )


def _opencode_search_tool(stdout: str) -> str | None:
    """Return the name of a genuinely-completed opencode network tool, else None.

    opencode's `--format json` emits one JSON event per line. A tool invocation
    surfaces as a ``tool_use`` event whose ``part.state.status`` transitions to
    ``completed``; the tool name is ``part.tool``. We accept only the network
    tools in ``_OPENCODE_SEARCH_TOOLS`` (webfetch/websearch/…) and only in the
    ``completed`` state — a tool that ``error``ed (e.g. a malformed URL the
    model retried) or merely started is not proof the search reached the
    network. This mirrors ``_codex_search_query``: evidence of a real
    invocation, not of the tool merely being registered.
    """
    for line in stdout.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            event = json.loads(line)
        except (json.JSONDecodeError, ValueError):
            continue
        if not isinstance(event, dict) or event.get("type") != "tool_use":
            continue
        part = event.get("part")
        if not isinstance(part, dict):
            continue
        tool = part.get("tool")
        state = part.get("state")
        status = state.get("status") if isinstance(state, dict) else None
        if tool in _OPENCODE_SEARCH_TOOLS and status == "completed":
            return tool
    return None


def probe_opencode(
    project: str,
    *,
    opencode_bin: str = "opencode",
    model: str = "litellm/gemini-2.5-pro",
    expected_version: str | None = None,
) -> ProbeResult:
    """Invoke `opencode run --format json` and inspect its event stream for a
    completed client-side network tool (webfetch/websearch), else the sentinel.

    Unlike claude/codex, opencode has no vendor-hosted web_search to forward
    through the gateway; it grounds by having the model drive its own
    ``webfetch``/``websearch`` tool. WORKS therefore means such a tool actually
    *completed* (reached the network). When ``expected_version`` (independent
    PyPI ground truth) is supplied, WORKS additionally requires that exact
    version in the answer — proof the fetch returned live data the model used;
    a completed fetch whose answer omits it is SUSPECT (the model may have
    fetched the wrong URL or ignored the result)."""
    prompt = _prompt(project)
    cmd = [
        opencode_bin,
        "run",
        "--format",
        "json",
        "-m",
        model,
        prompt,
    ]
    try:
        completed = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=240,
            check=False,
        )
    except FileNotFoundError:
        return ProbeResult("opencode", "ERROR", f"{opencode_bin} not found on PATH")
    except subprocess.TimeoutExpired:
        return ProbeResult("opencode", "ERROR", "opencode probe timed out after 240s")

    transcript = completed.stdout + "\n" + completed.stderr
    low = transcript.lower()

    if completed.returncode != 0 and not transcript.strip():
        return ProbeResult(
            "opencode", "ERROR", f"opencode exited {completed.returncode} with no output"
        )

    search_tool = _opencode_search_tool(completed.stdout)
    sentinel_seen = SENTINEL in transcript

    if search_tool is not None and not sentinel_seen:
        if expected_version is None:
            return ProbeResult(
                "opencode", "WORKS", f"{search_tool} tool completed (reached network)"
            )
        if expected_version in transcript:
            return ProbeResult(
                "opencode",
                "WORKS",
                f"{search_tool} tool completed and answer cites live version {expected_version}",
            )
        return ProbeResult(
            "opencode",
            "SUSPECT",
            f"{search_tool} tool completed but answer omits ground-truth "
            f"version {expected_version} — fetch may have missed live data",
        )
    if sentinel_seen:
        return ProbeResult(
            "opencode", "INERT", f"model emitted {SENTINEL} — could not run a web search"
        )
    if ("webfetch" in low or "websearch" in low) and (
        "error" in low or "unavailable" in low or "not available" in low
    ):
        return ProbeResult("opencode", "ERROR", "opencode network tool failed")
    return ProbeResult(
        "opencode",
        "INERT",
        "no completed network-tool event and no sentinel — answer likely from memory",
    )


def _one_line(text: str, *, limit: int = 240) -> str:
    """Collapse whitespace and truncate — gateway errors are large multi-line
    JSON blobs, unusable in the one-line verdict table."""
    collapsed = re.sub(r"\s+", " ", text).strip()
    return collapsed if len(collapsed) <= limit else collapsed[: limit - 1] + "…"


def _antigravity_error_mentions_search_call(text: str) -> bool:
    """Return True when a raw gateway/SDK error appears to contain a search call.

    A bare ``search_web`` substring can appear in diagnostics, so require either
    a tool/function-call marker or a quoted tool name from a serialized payload.
    """
    low = text.lower()
    if _ANTIGRAVITY_SEARCH_TOOL not in low:
        return False
    if any(marker in low for marker in ("functioncall", "function_call", "toolcall")):
        return True
    return any(
        marker in low
        for marker in (
            f'"{_ANTIGRAVITY_SEARCH_TOOL}"',
            f"'{_ANTIGRAVITY_SEARCH_TOOL}'",
            f'\\"{_ANTIGRAVITY_SEARCH_TOOL}\\"',
        )
    )


def _antigravity_error_detail(text: str, *, search_seen: bool = False) -> str:
    """Turn Antigravity SDK/gateway failures into compact, actionable detail."""
    low = text.lower()
    if "publisher model" in low and "was not found" in low:
        return (
            "model rejected by upstream Vertex/LiteLLM (not found or no project "
            f"access); choose a served model/region: {_one_line(text, limit=170)}"
        )
    if all(marker in low for marker in _ANTIGRAVITY_STREAM_PARSE_MARKERS[:3]) or (
        "iterateresponsestream" in low and "b'data:" in low
    ):
        if search_seen:
            prefix = "search_web dispatched, but Antigravity SDK/Gemini SSE parsing failed"
        else:
            prefix = (
                "Antigravity SDK/Gemini SSE parsing failed before a "
                "search_web tool-call surfaced"
            )
        return (
            f"{prefix} (gateway stream frame is not valid JSON for the SDK): "
            f"{_one_line(text, limit=160)}"
        )
    if search_seen:
        return f"search_web dispatched but the turn did not complete: {_one_line(text)}"
    return _one_line(text)


def load_gateway_creds(
    settings_path: Path,
) -> tuple[str | None, str | None, str | None]:
    """Read the LiteLLM gateway base_url + token from a Claude settings.json.

    Returns ``(base_url, api_key, error)``. On success ``error`` is None; on any
    problem the first two are None and ``error`` is a human-readable reason.
    The settings file wins over process env when present; env fallback keeps
    the probe usable in CI or stripped-down ops shells that inject the gateway
    values directly. The gateway is shared across runners, so the Claude CLI's
    ``env.ANTHROPIC_BASE_URL`` / ``env.ANTHROPIC_AUTH_TOKEN`` are exactly the
    creds the Antigravity SDK needs to reach the same endpoint.
    """
    path = Path(settings_path).expanduser()
    file_error: str | None = None
    settings_env: dict[str, object] = {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        file_error = f"settings file not found: {path}"
    except (OSError, json.JSONDecodeError, ValueError) as exc:
        file_error = f"could not read/parse {path}: {exc}"
    else:
        env = data.get("env") if isinstance(data, dict) else None
        settings_env = env if isinstance(env, dict) else {}

    base_url = settings_env.get("ANTHROPIC_BASE_URL") or os.environ.get("ANTHROPIC_BASE_URL")
    api_key = (
        settings_env.get("ANTHROPIC_AUTH_TOKEN")
        or settings_env.get("ANTHROPIC_API_KEY")
        or os.environ.get("ANTHROPIC_AUTH_TOKEN")
        or os.environ.get("ANTHROPIC_API_KEY")
    )
    if not base_url:
        detail = "no ANTHROPIC_BASE_URL in settings or environment"
        if file_error:
            detail = f"{file_error}; {detail}"
        return None, None, detail
    if not api_key:
        detail = "no ANTHROPIC_AUTH_TOKEN/ANTHROPIC_API_KEY in settings or environment"
        if file_error:
            detail = f"{file_error}; {detail}"
        return None, None, detail
    if not isinstance(base_url, str) or not isinstance(api_key, str):
        return None, None, "gateway credentials must be strings"
    return base_url.rstrip("/"), api_key, None


def _gateway_api_key(api_key: str) -> str:
    """Return the raw token value, accepting either raw or ``Bearer ...`` input."""
    token = api_key.strip()
    if token.lower().startswith("bearer "):
        token = token[7:].strip()
    return token


def _gateway_http_headers(api_key: str) -> dict[str, str]:
    """Headers for LiteLLM/OpenAI-compatible auth through GeminiAPIEndpoint."""
    return {"Authorization": f"Bearer {_gateway_api_key(api_key)}"}


def probe_antigravity(
    project: str,
    *,
    model: str = DEFAULT_ANTIGRAVITY_MODEL,
    settings_path: Path = DEFAULT_CLAUDE_SETTINGS,
    expected_version: str | None = None,
    timeout_seconds: int = 240,
    use_sse_normalizer: bool = True,
) -> ProbeResult:
    """Drive the Antigravity Python SDK against the LiteLLM gateway and inspect
    the response stream for a dispatched `search_web` tool-call, else the
    sentinel.

    Unlike the CLI probes this shells nothing out: it configures a
    ``GeminiAPIEndpoint(base_url=…, api_key=…)`` pointed at the gateway (creds
    from ``settings_path``), enables only ``BuiltinTools.SEARCH_WEB``, and reads
    back the streamed chunks. WORKS means the model actually dispatched
    ``search_web``; when ``expected_version`` (independent PyPI ground truth) is
    supplied, WORKS additionally requires that exact version in the answer, else
    SUSPECT.

    With ``use_sse_normalizer`` (the default), the SDK is routed through the
    engine runner's local SSE normalizer proxy
    (``spotlights_engine.module_deep_research.antigravity_exec``), which unwraps
    LiteLLM's Python-bytes-repr SSE frames back to standard Gemini SSE — the
    same path the Antigravity runner uses, so the verdict reflects what the
    engine actually experiences. The proxy injects the real bearer token
    upstream and the SDK only sees a placeholder key. Without it, the probe
    talks to the gateway directly: a dispatched search whose stream then fails
    to complete is reported ERROR (the tool fired but no grounded answer
    landed), and a hard gateway rejection (404/auth) is ERROR carrying the
    gateway's own message."""
    try:
        from google.antigravity import (
            Agent,
            BuiltinTools,
            CapabilitiesConfig,
            GeminiAPIEndpoint,
            LocalAgentConfig,
            ModelTarget,
            ModelType,
        )
        from google.antigravity.types import Text, ToolCall
    except ImportError as exc:
        return ProbeResult(
            "antigravity",
            "ERROR",
            f"google-antigravity not importable ({exc}); run `pip install google-antigravity`",
        )

    sse_normalizer = None
    placeholder_api_key = None
    force_model_targets = None
    if use_sse_normalizer:
        try:
            from spotlights_engine.module_deep_research.antigravity_exec import (
                _PROXY_PLACEHOLDER_API_KEY as placeholder_api_key,
                _force_agent_model_targets as force_model_targets,
                _sse_bytes_repr_normalizer as sse_normalizer,
            )
        except ImportError as exc:
            return ProbeResult(
                "antigravity",
                "ERROR",
                f"spotlights_engine SSE normalizer not importable ({exc}); run "
                "inside the repo env or pass --no-antigravity-sse-normalizer",
            )

    base_url, api_key, cred_err = load_gateway_creds(settings_path)
    if cred_err:
        return ProbeResult("antigravity", "ERROR", cred_err)

    prompt = _prompt(project)

    def _is_search(name: object) -> bool:
        return _ANTIGRAVITY_SEARCH_TOOL in str(name).lower()

    async def _drive() -> tuple[str, list[str], str | None]:
        gateway_api_key = _gateway_api_key(api_key)
        with contextlib.ExitStack() as stack:
            if sse_normalizer is not None:
                # Same setup as AntigravityExecClient's proxy path: the SDK
                # talks to the local normalizer with a placeholder key, and the
                # proxy injects the real bearer auth upstream.
                local_base_url = stack.enter_context(
                    sse_normalizer(
                        base_url,
                        timeout_seconds=float(timeout_seconds),
                        upstream_headers=_gateway_http_headers(api_key),
                    )
                )
                sdk_api_key = placeholder_api_key
                endpoint = GeminiAPIEndpoint(
                    base_url=local_base_url,
                    api_key=sdk_api_key,
                )
            else:
                sdk_api_key = gateway_api_key
                endpoint = GeminiAPIEndpoint(
                    base_url=base_url,
                    http_headers=_gateway_http_headers(api_key),
                    api_key=sdk_api_key,
                )
            target = ModelTarget(name=model, types=[ModelType.TEXT], endpoint=endpoint)
            config = LocalAgentConfig(
                system_instructions=(
                    "Use the web search tool to look up live facts before you "
                    "answer. Do not answer from memory."
                ),
                capabilities=CapabilitiesConfig(enabled_tools=[BuiltinTools.SEARCH_WEB]),
                model=target,
                # Also set at top level so the SDK's auto-added default image model
                # slot validates. The probe never enables or uses image generation,
                # but the full config is validated up-front.
                api_key=sdk_api_key,
            )
            text_parts: list[str] = []
            search_queries: list[str] = []
            stream_error: str | None = None
            agent_session = Agent(config)
            if force_model_targets is not None:
                # Strip SDK-added default (image) model targets so the proxy-only
                # text model is the sole upstream, as the engine runner does.
                force_model_targets(agent_session, [target])
            async with agent_session as agent:
                response = await agent.chat(prompt)
                try:
                    async for chunk in response.chunks:
                        if isinstance(chunk, Text):
                            text_parts.append(chunk.text)
                        elif isinstance(chunk, ToolCall) and _is_search(chunk.name):
                            q = ""
                            if isinstance(chunk.args, dict):
                                q = str(chunk.args.get("query") or "").strip()
                            search_queries.append(q)
                except Exception as exc:  # noqa: BLE001 — mid-turn gateway/stream failure
                    # Cancellation (timeout) is BaseException, not Exception, so it
                    # still propagates; only real stream errors are captured here.
                    stream_error = str(exc)
        return "".join(text_parts), search_queries, stream_error

    try:
        text, search_queries, stream_error = asyncio.run(
            asyncio.wait_for(_drive(), timeout=timeout_seconds)
        )
    except TimeoutError:
        return ProbeResult(
            "antigravity", "ERROR", f"antigravity probe timed out after {timeout_seconds}s"
        )
    except Exception as exc:  # noqa: BLE001 — connect/validate failure before any chunk
        return ProbeResult("antigravity", "ERROR", _antigravity_error_detail(str(exc)))

    transcript = text + "\n" + (stream_error or "")
    sentinel_seen = SENTINEL in transcript
    # The tool counts as dispatched if we saw a clean search_web tool-call OR the
    # gateway surfaced the search_web functionCall inside a broken-stream error.
    search_seen = bool(search_queries) or (
        stream_error is not None and _antigravity_error_mentions_search_call(stream_error)
    )
    query_note = next((q for q in search_queries if q), None)

    if search_seen and not sentinel_seen:
        if stream_error is not None:
            # The model chose to search, but the turn did not finish cleanly
            # (often a broken Gemini SSE stream from the gateway). Not WORKS.
            return ProbeResult(
                "antigravity",
                "ERROR",
                _antigravity_error_detail(stream_error, search_seen=True),
            )
        if expected_version is None:
            detail = "search_web tool-call dispatched"
            if query_note:
                detail += f" (query {query_note!r})"
            return ProbeResult("antigravity", "WORKS", detail)
        if expected_version in transcript:
            return ProbeResult(
                "antigravity",
                "WORKS",
                f"search_web dispatched and answer cites live version {expected_version}",
            )
        return ProbeResult(
            "antigravity",
            "SUSPECT",
            f"search_web dispatched but answer omits ground-truth version "
            f"{expected_version} — search may have returned nothing",
        )
    if sentinel_seen:
        return ProbeResult(
            "antigravity", "INERT", f"model emitted {SENTINEL} — could not run a web search"
        )
    if stream_error is not None:
        return ProbeResult("antigravity", "ERROR", _antigravity_error_detail(stream_error))
    return ProbeResult(
        "antigravity",
        "INERT",
        "no search_web tool-call and no sentinel — answer likely from memory",
    )


def _print_verdict_table(results: list[ProbeResult]) -> None:
    width = max((len(r.runner) for r in results), default=6)
    print("\n=== hosted web-search verdicts ===")
    for r in results:
        print(f"  {r.runner.ljust(width)}  {r.verdict:<7}  {r.detail}")
    print()


def _build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--only",
        choices=["claude", "codex", "opencode", "antigravity"],
        default=None,
        help="Probe only one runner (default: all).",
    )
    p.add_argument(
        "--project",
        default="uv",
        help="Fast-moving project whose latest version requires a live search.",
    )
    p.add_argument(
        "--codex-profile",
        default="litellm",
        help="Codex profile to run through (default: litellm).",
    )
    p.add_argument(
        "--opencode-model",
        default="litellm/gemini-2.5-pro",
        help="opencode model (provider/model) to run through (default: litellm/gemini-2.5-pro).",
    )
    p.add_argument(
        "--antigravity-model",
        default=DEFAULT_ANTIGRAVITY_MODEL,
        help="Antigravity SDK model (LiteLLM model-group name) to run through "
        f"(default: {DEFAULT_ANTIGRAVITY_MODEL}).",
    )
    p.add_argument(
        "--claude-settings",
        type=Path,
        default=DEFAULT_CLAUDE_SETTINGS,
        help="Claude settings.json to read the LiteLLM gateway base_url + token "
        f"from for the antigravity probe (default: {DEFAULT_CLAUDE_SETTINGS}).",
    )
    p.add_argument(
        "--no-ground-truth",
        action="store_true",
        help="Skip the PyPI ground-truth grounding check (accept a proven "
        "tool invocation as WORKS even if the answer omits the live version).",
    )
    p.add_argument(
        "--no-antigravity-sse-normalizer",
        dest="antigravity_sse_normalizer",
        action="store_false",
        help="Probe the gateway directly instead of through the engine "
        "runner's local SSE normalizer proxy (LiteLLM bytes-repr streams "
        "will then surface as ERROR).",
    )
    p.set_defaults(antigravity_sse_normalizer=True)
    return p


def main(argv: list[str] | None = None) -> int:
    args = _build_argparser().parse_args(argv)

    base_url = os.environ.get("ANTHROPIC_BASE_URL", "(unset — vendor endpoint)")
    print(f"ANTHROPIC_BASE_URL = {base_url}")
    print(f"probing project    = {args.project!r}")

    expected_version: str | None = None
    if not args.no_ground_truth:
        expected_version = pypi_latest_version(args.project)
        if expected_version is None:
            print(
                f"ground truth       = (PyPI lookup failed for "
                f"{args.project!r}; grounding disabled)\n"
            )
        else:
            print(f"ground truth       = {args.project} {expected_version} (PyPI)\n")
    else:
        print()

    results: list[ProbeResult] = []
    if args.only in (None, "claude"):
        print("[claude] probing WebSearch …")
        results.append(probe_claude(args.project))
    if args.only in (None, "codex"):
        print("[codex] probing --search …")
        results.append(
            probe_codex(
                args.project,
                profile=args.codex_profile,
                expected_version=expected_version,
            )
        )
    if args.only in (None, "opencode"):
        print("[opencode] probing webfetch/websearch …")
        results.append(
            probe_opencode(
                args.project,
                model=args.opencode_model,
                expected_version=expected_version,
            )
        )
    if args.only in (None, "antigravity"):
        via = (
            "local SSE normalizer proxy"
            if args.antigravity_sse_normalizer
            else "gateway direct"
        )
        print(
            f"[antigravity] probing SDK search_web via {args.antigravity_model!r} "
            f"({via}) …"
        )
        results.append(
            probe_antigravity(
                args.project,
                model=args.antigravity_model,
                settings_path=args.claude_settings,
                expected_version=expected_version,
                use_sse_normalizer=args.antigravity_sse_normalizer,
            )
        )

    _print_verdict_table(results)

    # Machine-readable line for CI capture.
    print("verdicts_json=" + json.dumps({r.runner: r.verdict for r in results}))

    bad = [r for r in results if r.verdict != "WORKS"]
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
