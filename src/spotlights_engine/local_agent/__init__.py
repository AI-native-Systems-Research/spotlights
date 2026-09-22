"""Local-model agent transport + the shared agent-execution base API.

- `base`: the canonical agent-runner contract (`AgentExecResult`,
  `ModuleResearchRunner`, `resolve_cli_executable`) shared by every CLI runner.
- `pi_runner`: run the `pi` coding agent against the local Qwen vLLM backend.
- `dispatch`: `is_local_model` + adapters that translate a pi run into each
  seam's native result shape.

Co-located `qwen.sh` bootstraps the vLLM port-forward / OpenShift login.
"""

from __future__ import annotations

from spotlights_engine.local_agent.base import (
    AgentExecResult,
    ModuleResearchRunner,
    resolve_cli_executable,
)
from spotlights_engine.local_agent.dispatch import (
    is_local_model,
    local_final_json,
    run_agent_exec_local,
    run_streaming_local,
)
from spotlights_engine.local_agent.pi_runner import (
    LocalAgentError,
    LocalAgentTimeout,
    PiRun,
    run_pi,
)

__all__ = [
    "AgentExecResult",
    "LocalAgentError",
    "LocalAgentTimeout",
    "ModuleResearchRunner",
    "PiRun",
    "is_local_model",
    "local_final_json",
    "resolve_cli_executable",
    "run_agent_exec_local",
    "run_pi",
    "run_streaming_local",
]
