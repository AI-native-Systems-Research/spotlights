"""Resolve the Claude / Codex CLI binary, dodging the Windows `.CMD` shim.

Three different answers to this problem used to live in the tree:
`resolve_claude_argv0` (strict — refuses the `.CMD`/`.bat` shim because it
buffers child stdout and deadlocks streaming), `resolve_cli_executable`
(POSIX passthrough, `shutil.which` on Windows), and bare `shutil.which(...)`
sites that silently accepted the shim. This module is the single policy:

- `resolve_claude_argv0` keeps the strict behaviour and is exported for the
  streaming path.
- `resolve_cli` is the unified resolver used by every session; it refuses the
  shim for both `claude` and `codex` so the blocking path can no longer
  silently regress onto a buffering shim either.
"""

from __future__ import annotations

import os
import shutil
import sys
from pathlib import Path


class CLIResolutionError(RuntimeError):
    """A required CLI couldn't be resolved to a non-shim binary."""


# Back-compat alias: the streaming helper historically raised this name.
ClaudeResolutionError = CLIResolutionError


def _override_env_var(cli: str) -> str | None:
    if cli == "claude":
        return "SPOTLIGHTS_CLAUDE_BIN"
    if cli == "codex":
        return "SPOTLIGHTS_CODEX_BIN"
    return None


def resolve_cli(cli: str, bin_name: str | None = None) -> str:
    """Resolve ``cli`` ("claude"/"codex") to a concrete, non-shim binary path.

    On POSIX: `shutil.which`. On Windows: prefer the npm-installed `.exe`
    (for claude) and refuse a `.CMD`/`.bat`/`.ps1` shim, which buffers child
    stdout and breaks streaming / deadlocks `subprocess.run(capture_output)`.

    A per-CLI override env var (`SPOTLIGHTS_CLAUDE_BIN`/`SPOTLIGHTS_CODEX_BIN`)
    short-circuits the search when set.
    """
    name = bin_name or cli

    override_var = _override_env_var(cli)
    if override_var:
        override = os.environ.get(override_var)
        if override:
            if not Path(override).exists():
                raise CLIResolutionError(f"{override_var}={override!r} does not exist")
            return override

    if sys.platform == "win32":
        if cli == "claude":
            appdata = os.environ.get("APPDATA")
            if appdata:
                base = (
                    Path(appdata)
                    / "npm"
                    / "node_modules"
                    / "@anthropic-ai"
                    / "claude-code"
                )
                for candidate in (base / "claude.exe", base / "bin" / "claude.exe"):
                    if candidate.exists():
                        return str(candidate)
        resolved = shutil.which(name)
        if resolved is None:
            raise CLIResolutionError(f"required CLI not on PATH: {name}")
        if resolved.lower().endswith((".cmd", ".bat", ".ps1")):
            raise CLIResolutionError(
                f"resolved {name} to a cmd shim ({resolved}); the shim buffers "
                f"child stdout and breaks live streaming. Install a real .exe "
                f"or set {override_var} to a real binary"
            )
        return resolved

    resolved = shutil.which(name)
    if resolved is None:
        raise CLIResolutionError(f"required CLI not on PATH: {name}")
    return resolved


def resolve_claude_argv0(claude_bin: str = "claude") -> list[str]:
    """argv prefix that invokes `claude` without going through the cmd shim.

    Kept for the streaming callers that expect a list prefix. Delegates to
    :func:`resolve_cli`.
    """
    return [resolve_cli("claude", claude_bin)]


__all__ = [
    "CLIResolutionError",
    "ClaudeResolutionError",
    "resolve_claude_argv0",
    "resolve_cli",
]
