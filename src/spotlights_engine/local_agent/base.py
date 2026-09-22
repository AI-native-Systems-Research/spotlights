"""Base agent-execution API shared by every CLI runner in the engine.

This is the canonical home for the agent-runner contract: the captured-result
model (`AgentExecResult`), the minimal runner `Protocol` (`ModuleResearchRunner`),
and the Windows shim resolver (`resolve_cli_executable`). Codex, Claude, the
local pi/Qwen backend, and the test fakes all speak this shape so the
orchestration layer can treat them interchangeably.

Historically this lived in `module_deep_research.agent_exec`; it moved here when
the local-model dispatch was added, so all agent transports (hosted + local)
sit under one package.
"""

from __future__ import annotations

import os
import shutil
from typing import Protocol

from pydantic import BaseModel, ConfigDict

from spotlights_engine.costing.usage import AgentUsage

WINDOWS_SUBPROCESS_NEEDS_SHIM_RESOLUTION = os.name == "nt"


class AgentExecResult(BaseModel):
    """Captured result from one non-interactive research-agent invocation."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    command: list[str]
    returncode: int
    stdout: str
    stderr: str
    final_message: str | None = None
    # Token usage parsed from the CLI output while the raw stream is still
    # available; None when the runner emitted no parseable usage (e.g. a
    # timeout before any usage event was written).
    usage: AgentUsage | None = None

    @property
    def ok(self) -> bool:
        return self.returncode == 0

    def raise_for_status(self) -> None:
        if self.returncode != 0:
            raise RuntimeError(
                f"agent failed with exit code {self.returncode}\n"
                f"COMMAND: {' '.join(self.command)}\n"
                f"STDOUT:\n{self.stdout}\nSTDERR:\n{self.stderr}"
            )


class ModuleResearchRunner(Protocol):
    """Minimal interface shared by Codex, Claude, pi/Qwen, and test fakes."""

    name: str

    def run(self, prompt: str, *, check: bool = True) -> AgentExecResult: ...


def resolve_cli_executable(executable: str) -> str:
    """Resolve CLI shims on Windows while preserving POSIX command behavior.

    npm-installed CLIs are commonly exposed as `.cmd` shims on Windows. Passing a
    bare command name to `subprocess.run(..., shell=False)` can fail with
    `FileNotFoundError` there, even when the command works in `cmd.exe` or
    PowerShell. `shutil.which` applies Windows `PATHEXT` lookup and returns the
    concrete shim path. POSIX callers keep the configured command unchanged so
    command shapes remain stable in logs and tests.
    """
    if not WINDOWS_SUBPROCESS_NEEDS_SHIM_RESOLUTION:
        return executable
    return shutil.which(executable) or executable


__all__ = ["AgentExecResult", "ModuleResearchRunner", "resolve_cli_executable"]
