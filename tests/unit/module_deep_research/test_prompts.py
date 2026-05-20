"""Unit tests for module deep-research prompt assembly."""

from __future__ import annotations

from spotlights_engine.module_deep_research.prompts import render_module_deep_research_prompt
from spotlights_engine.schemas.deep_research import ModuleDeepResearchInput, SpotlightContext
from spotlights_engine.schemas.modules import File, Module, ProjectTree, Repository


def test_prompt_is_built_from_repository_module_and_context() -> None:
    tree = ProjectTree(
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
    request = ModuleDeepResearchInput(
        project_tree=tree,
        module_qualified_name="kv_offload",
        context=SpotlightContext(
            objective="reduce decode latency for long-context serving",
            workload_hints=["single-node 8xH100"],
        ),
        max_findings_per_module=3,
    )

    prompt = render_module_deep_research_prompt(request, tree.modules[0])

    assert "vllm" in prompt
    assert "Inference engine." in prompt
    assert "kv_offload" in prompt
    assert "vllm/v1/kv_offload" in prompt
    assert "cpu/manager.py: CPU offload manager" in prompt
    assert "reduce decode latency for long-context serving" in prompt
    assert "single-node 8xH100" in prompt
    assert "at most 3 findings" in prompt
    assert "ModuleDeepResearchOutput JSON schema" in prompt
    assert "Empty findings are valid" in prompt
