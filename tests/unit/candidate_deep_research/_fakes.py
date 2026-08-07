"""Shared fakes/builders for candidate_deep_research unit tests."""

from __future__ import annotations

import json
from pathlib import Path

from spotlights_engine.costing.usage import AgentUsage
from spotlights_engine.module_deep_research.agent_exec import AgentExecResult
from spotlights_engine.schemas.candidate import Candidate, Candidates
from spotlights_engine.schemas.common import SpotlightContext
from spotlights_engine.schemas.pipeline import CandidateDeepResearchInput
from spotlights_engine.schemas.project import Module, ProjectTree, Repository
from spotlights_engine.utils.schema_compat import make_location

MODULE_QN = "inference/attention"
SEGMENT = "inference_attention"


def tree() -> ProjectTree:
    return ProjectTree(
        repository=Repository(name="demo", summary="Demo repository.", source_root="src"),
        modules=[
            Module(
                name="inference",
                path="src/inference",
                submodules=[
                    Module(
                        name="attention",
                        path="src/inference/attention",
                        description="Attention implementation.",
                    )
                ],
            )
        ],
    )


def make_candidate(idx: int = 0, *, segment: str = SEGMENT) -> Candidate:
    n = idx + 1
    return Candidate(
        id=f"cand-{segment}-{n:04d}",
        origin="code_agent",
        locations=[
            make_location(
                file="src/inference/attention/core.py",
                line_start=1,
                line_end=10,
                symbol=f"hot_{n}",
                kind="function",
            )
        ],
        description=f"description {n}",
        current_approach=f"current approach {n}",
        evolve_rationale=f"evolve rationale {n}",
        estimated_impact="medium",
        estimated_impact_explanation=f"impact {n}",
    )


def make_candidates(n: int = 1) -> Candidates:
    return Candidates(
        module_qualified_name=MODULE_QN,
        candidates=[make_candidate(i) for i in range(n)],
    )


def make_request(
    n_candidates: int = 1,
    *,
    module_qualified_name: str = MODULE_QN,
    max_findings_per_candidate: int = 5,
    repo_path: Path = Path("/tmp/example-repo"),
) -> CandidateDeepResearchInput:
    return CandidateDeepResearchInput(
        project_tree=tree(),
        module_qualified_name=module_qualified_name,
        context=SpotlightContext(objective="reduce latency"),
        repo_path=repo_path,
        candidates=[make_candidate(i) for i in range(n_candidates)],
        max_findings_per_candidate=max_findings_per_candidate,
    )


def agent_payload(*titles: str, queries: tuple[str, ...] = ()) -> str:
    """Build one runner's raw JSON payload with local `find-NNNN` ids."""
    return json.dumps(
        {
            "findings": [
                {
                    "finding_id": f"find-{i:04d}",
                    "title": title,
                    "url": f"https://example.com/{i}",
                    "source_type": "paper",
                    "technique_summary": f"summary for {title}",
                }
                for i, title in enumerate(titles, start=1)
            ],
            "issues": [],
            "search_queries": [{"query": q, "tool": "web"} for q in queries],
        }
    )


class FakeRunner:
    """A `ModuleResearchRunner` that returns a per-prompt payload.

    `payload_for` receives the rendered prompt so a test can hand a different
    survey result to each candidate (candidate ids appear in the prompt).
    """

    def __init__(
        self,
        payload: str | None = None,
        *,
        name: str = "codex",
        returncode: int = 0,
        stderr: str = "",
        usage: AgentUsage | None = None,
        payload_for=None,
        raises: Exception | None = None,
    ) -> None:
        self.payload = payload
        self.name = name
        self.returncode = returncode
        self.stderr = stderr
        self.usage = usage
        self.payload_for = payload_for
        self.raises = raises
        self.prompts: list[str] = []

    def run(self, prompt: str, *, check: bool = True) -> AgentExecResult:
        self.prompts.append(prompt)
        if self.raises is not None:
            raise self.raises
        final = self.payload_for(prompt) if self.payload_for is not None else self.payload
        return AgentExecResult(
            command=[self.name, "exec"],
            returncode=self.returncode,
            stdout="",
            stderr=self.stderr,
            final_message=final,
            usage=self.usage,
        )


def usage(input_tokens: int = 10, output_tokens: int = 20) -> AgentUsage:
    return AgentUsage(input=input_tokens, output=output_tokens)


__all__ = [
    "MODULE_QN",
    "SEGMENT",
    "FakeRunner",
    "agent_payload",
    "make_candidate",
    "make_candidates",
    "make_request",
    "tree",
    "usage",
]
