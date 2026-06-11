"""Default CLI-agent runner for expanded module deep research."""

from __future__ import annotations

from pathlib import Path
from typing import Protocol

from pydantic import BaseModel, Field

from spotlights_engine.cli_agents import CliAgentRequest, StructuredOutputSpec
from spotlights_engine.cli_agents.claude_cli import ClaudeCliAgent
from spotlights_engine.cli_agents.codex_cli import CodexCliAgent
from spotlights_engine.cli_agents.config import ClaudeCliConfig, CodexCliConfig, GeminiCliConfig
from spotlights_engine.cli_agents.gemini_cli import GeminiCliAgent
from spotlights_engine.module_deep_research.expanded.models import (
    AgentSearchOutput,
    ExpandedResearchConfig,
    PaperEvidence,
    ResearchPaper,
    ResearchTask,
)


class AgentTaskRunner(Protocol):
    """Executes one expanded research task and returns normalized paper output."""

    def run_task(
        self, *, task: ResearchTask, prompt: str, repo_path: Path
    ) -> AgentSearchOutput: ...


class AgentPaperPayload(BaseModel):
    """Schema requested from CLI agents for one module-scoped paper search."""

    papers: list[ResearchPaper] = Field(
        default_factory=list,
        description="Research papers relevant to the target module and objective.",
    )
    notes: list[str] = Field(
        default_factory=list,
        description="Short caveats, search gaps, or verification notes.",
    )


class CliAgentTaskRunner:
    """Run expanded research tasks through Codex, Claude, and Gemini CLI adapters."""

    def __init__(self, config: ExpandedResearchConfig) -> None:
        self.config = config
        self._agents = {
            "codex": CodexCliAgent(CodexCliConfig.ibm_litellm(profile=config.codex_profile)),
            "claude": ClaudeCliAgent(ClaudeCliConfig.ibm_litellm(model=config.claude_model)),
            "gemini": GeminiCliAgent(GeminiCliConfig.ibm_litellm(model=config.gemini_model)),
        }
        self._spec = StructuredOutputSpec.from_pydantic(
            AgentPaperPayload,
            name="module_paper_search",
            description="Papers must be specific to the Spotlight target module.",
        )

    def run_task(self, *, task: ResearchTask, prompt: str, repo_path: Path) -> AgentSearchOutput:
        """Execute one task and normalize success or recoverable failure."""
        agent = self._agents[task.agent_name]
        result = agent.run(
            CliAgentRequest(
                prompt=prompt,
                cwd=repo_path,
                allow_network=True,
                allow_write=False,
                timeout_seconds=self.config.timeout_seconds,
                structured_output=self._spec,
            )
        )
        if not result.ok:
            message = (
                f"{task.agent_name} exited with code {result.returncode}: {result.stderr[-500:]}"
            )
            return AgentSearchOutput(
                task_id=task.task_id,
                agent_name=task.agent_name,
                prompt_variant_id=task.prompt_variant.variant_id,
                phase=task.phase,
                papers=[],
                notes=[],
                error=message,
            )
        if result.structured_error is not None:
            return AgentSearchOutput(
                task_id=task.task_id,
                agent_name=task.agent_name,
                prompt_variant_id=task.prompt_variant.variant_id,
                phase=task.phase,
                papers=[],
                notes=[],
                error=result.structured_error,
            )
        payload = result.require_structured()
        if isinstance(payload, AgentPaperPayload):
            papers = payload.papers
            notes = payload.notes
        else:
            coerced = AgentPaperPayload.model_validate(payload)
            papers = coerced.papers
            notes = coerced.notes
        return AgentSearchOutput(
            task_id=task.task_id,
            agent_name=task.agent_name,
            prompt_variant_id=task.prompt_variant.variant_id,
            phase=task.phase,
            papers=[_attach_missing_evidence(paper, task) for paper in papers],
            notes=notes,
        )


def _attach_missing_evidence(paper: ResearchPaper, task: ResearchTask) -> ResearchPaper:
    if paper.evidence:
        return paper
    return paper.model_copy(
        update={
            "evidence": [
                PaperEvidence(
                    agent_name=task.agent_name,
                    prompt_variant_id=task.prompt_variant.variant_id,
                    quote_or_note=paper.why_relevant,
                    source_url=paper.source_url or paper.pdf_url,
                )
            ]
        }
    )
