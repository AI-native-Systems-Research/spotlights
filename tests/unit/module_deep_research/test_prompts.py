"""Unit tests for module deep-research per-candidate prompt assembly."""

from __future__ import annotations

from pathlib import Path

from spotlights_engine.module_deep_research.prompts import (
    render_candidate_deep_research_prompt,
)
from spotlights_engine.schemas.candidate import (
    Candidate,
    CodeLocation,
    CodeSpan,
)
from spotlights_engine.schemas.common import SpotlightContext
from spotlights_engine.schemas.pipeline import ModuleDeepResearchInput
from spotlights_engine.schemas.project import File, Module, ProjectTree, Repository


def _make_candidate(cid: str = "cand-kv_offload-0001") -> Candidate:
    return Candidate(
        id=cid,
        module_qualified_name="kv_offload",
        origin="code_agent",
        locations=[
            CodeLocation(
                file="cpu/manager.py",
                spans=[
                    CodeSpan(
                        line_start=10,
                        line_end=42,
                        symbol="score_multi_vector",
                        kind="function",
                    )
                ],
            )
        ],
        description="Scores multi-vector embeddings sequentially.",
        current_approach="Python loop over vectors.",
        evolve_rationale="Hot path in decode; batching would cut latency.",
        estimated_impact="high",
        estimated_impact_explanation="Dominates the decode profile.",
    )


def _tree() -> ProjectTree:
    return ProjectTree(
        repository=Repository(
            name="vllm",
            summary="Inference engine.",
            external_dependencies=["torch", "cuda"],
        ),
        modules=[
            Module(
                name="kv_offload",
                path="vllm/v1/kv_offload",
                description="CPU KV offload manager.",
                depends_on=["scheduler"],
                main_files=[File(path="cpu/manager.py", role="CPU offload manager")],
            )
        ],
    )


def _request(**overrides: object) -> ModuleDeepResearchInput:
    kwargs: dict = {
        "project_tree": _tree(),
        "module_qualified_name": "kv_offload",
        "context": SpotlightContext(
            objective="reduce decode latency for long-context serving",
            workload_hints=["single-node 8xH100"],
            validation_plan=["compare tokens/sec on existing benchmark"],
        ),
        "repo_path": Path("/tmp/example-repo"),
        "max_findings_per_candidate": 3,
        "candidates": [_make_candidate()],
    }
    kwargs.update(overrides)
    return ModuleDeepResearchInput(**kwargs)


def test_prompt_is_built_from_repository_module_context_and_candidate() -> None:
    request = _request()
    repo_path = request.repo_path

    prompt = render_candidate_deep_research_prompt(
        request, request.project_tree.modules[0], request.candidates[0]
    )

    # Repository + module grounding context.
    assert "vllm" in prompt
    assert "Inference engine." in prompt
    assert "kv_offload" in prompt
    assert "vllm/v1/kv_offload" in prompt
    assert "cpu/manager.py: CPU offload manager" in prompt
    # Caller context.
    assert "reduce decode latency for long-context serving" in prompt
    assert "single-node 8xH100" in prompt
    assert "compare tokens/sec on existing benchmark" in prompt
    # Per-candidate cap + schema.
    assert "at most 3 findings" in prompt
    assert "ModuleDeepResearchOutput JSON schema" in prompt
    assert "Empty findings are valid" in prompt
    assert "Workflow:" in prompt
    assert "Source quality" in prompt
    assert str(repo_path) in prompt
    assert "Repository working directory" in prompt
    # Search-transparency rules + the schema dump now includes search_queries.
    assert "Search transparency" in prompt
    assert "Record every web/literature search query you issued" in prompt
    assert "search_queries" in prompt


def test_prompt_focuses_on_the_candidate_code_site() -> None:
    request = _request()

    prompt = render_candidate_deep_research_prompt(
        request, request.project_tree.modules[0], request.candidates[0]
    )

    # The candidate is the subject of the survey (decision D1/D2).
    assert "Target candidate (the subject of this survey):" in prompt
    assert "cand-kv_offload-0001" in prompt
    assert "score_multi_vector" in prompt
    assert "cpu/manager.py" in prompt
    assert "10-42" in prompt
    assert "Python loop over vectors." in prompt
    assert "Hot path in decode; batching would cut latency." in prompt
    # Explicit candidate relevance gate.
    assert "Candidate relevance gate" in prompt
    # No leftover module-wide "hot spots" framing.
    assert "Identified hot spots" not in prompt
