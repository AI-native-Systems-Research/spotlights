"""Ops probe: does the hosted web-search tool actually reach a backend?

`WebSearch` (Anthropic-hosted) and `codex --search` (OpenAI-hosted) are
*server-side* tools. In this deployment both CLIs are pointed at a LiteLLM
gateway rather than the vendor endpoints, and a generic OpenAI-compatible proxy
does not necessarily forward hosted-tool calls. If it doesn't, the tool is
silently unavailable and the agent answers from parametric memory (the
hallucination path). This script proves — against the *same* live env the
engine uses — whether each runner's web search actually works.

This is a manual/ops probe, NOT a unit test: it needs the real proxy plus
credentials and makes live network calls. Run it before trusting the Claude or
Codex runners' web findings.

Usage:
    uv run --no-sync python scripts/verify_hosted_search.py
    uv run --no-sync python scripts/verify_hosted_search.py --only claude
    uv run --no-sync python scripts/verify_hosted_search.py --project uv

Exit code is nonzero if either probed runner is INERT or ERROR, so this can
gate a deployment check.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path

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
_CLAUDE_TOOL_MARKERS = ("web_search", "server_tool_use", "\"type\":\"web_search")
_CODEX_TOOL_MARKERS = ("web_search", "web.search", "\"search\"", "browser")

Verdict = str  # "WORKS" | "INERT" | "ERROR"


@dataclass
class ProbeResult:
    runner: str
    verdict: Verdict
    detail: str


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


def probe_codex(project: str, *, codex_bin: str = "codex", profile: str = "litellm") -> ProbeResult:
    """Invoke `codex exec --search` (through the litellm profile) and inspect
    its JSON event stream for a web-search / tool event, else the sentinel."""
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
        return ProbeResult(
            "codex", "ERROR", f"codex exited {completed.returncode} with no output"
        )

    tool_used = any(m in low for m in _CODEX_TOOL_MARKERS)
    sentinel_seen = SENTINEL in transcript

    if tool_used and not sentinel_seen:
        return ProbeResult("codex", "WORKS", "web-search tool event present in stream")
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


def _print_verdict_table(results: list[ProbeResult]) -> None:
    width = max((len(r.runner) for r in results), default=6)
    print("\n=== hosted web-search verdicts ===")
    for r in results:
        print(f"  {r.runner.ljust(width)}  {r.verdict:<6}  {r.detail}")
    print()


def _build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--only",
        choices=["claude", "codex"],
        default=None,
        help="Probe only one runner (default: both).",
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
    return p


def main(argv: list[str] | None = None) -> int:
    args = _build_argparser().parse_args(argv)

    base_url = os.environ.get("ANTHROPIC_BASE_URL", "(unset — vendor endpoint)")
    print(f"ANTHROPIC_BASE_URL = {base_url}")
    print(f"probing project    = {args.project!r}\n")

    results: list[ProbeResult] = []
    if args.only in (None, "claude"):
        print("[claude] probing WebSearch …")
        results.append(probe_claude(args.project))
    if args.only in (None, "codex"):
        print("[codex] probing --search …")
        results.append(probe_codex(args.project, profile=args.codex_profile))

    _print_verdict_table(results)

    # Machine-readable line for CI capture.
    print("verdicts_json=" + json.dumps({r.runner: r.verdict for r in results}))

    bad = [r for r in results if r.verdict != "WORKS"]
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
