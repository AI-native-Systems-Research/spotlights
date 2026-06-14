"""Shared subprocess result and runner protocol for module deep research."""

from __future__ import annotations

from typing import Protocol

from pydantic import BaseModel, ConfigDict


class AgentExecResult(BaseModel):
    """Captured result from one non-interactive research-agent invocation."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    command: list[str]
    returncode: int
    stdout: str
    stderr: str
    final_message: str | None = None

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
    """Minimal interface shared by Codex, Claude, Gemini, and test fakes."""

    name: str

    def run(self, prompt: str, *, check: bool = True) -> AgentExecResult: ...


__all__ = ["AgentExecResult", "ModuleResearchRunner"]
