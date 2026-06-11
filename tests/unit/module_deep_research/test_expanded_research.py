"""Tests for expanded multi-agent module deep research."""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

from spotlights_engine.module_deep_research.expanded import orchestrator as orchestrator_module
from spotlights_engine.module_deep_research.expanded.dedup import dedup_keys, merge_papers
from spotlights_engine.module_deep_research.expanded.identifiers import extract_arxiv_id
from spotlights_engine.module_deep_research.expanded.models import (
    AgentSearchOutput,
    ExpandedResearchConfig,
    ModuleResearchPacket,
    PaperEvidence,
    ResearchPaper,
    ResearchTask,
)
from spotlights_engine.module_deep_research.expanded.orchestrator import (
    ExpandedModuleResearchOrchestrator,
)
from spotlights_engine.schemas.common import SpotlightContext
from spotlights_engine.schemas.finding import Finding
from spotlights_engine.schemas.pipeline import ModuleDeepResearchInput, ModuleDeepResearchOutput
from spotlights_engine.schemas.project import File, Module, ProjectTree, Repository


class FakeExpandedRunner:
    def __init__(self) -> None:
        self.tasks: list[ResearchTask] = []
        self.prompts: list[str] = []

    def run_task(self, *, task: ResearchTask, prompt: str, repo_path: Path) -> AgentSearchOutput:
        self.tasks.append(task)
        self.prompts.append(prompt)
        paper = ResearchPaper(
            title=f"{task.agent_name} {task.prompt_variant.family} paper",
            authors=["A. Researcher"],
            year=2026,
            arxiv_id="2606.12345" if task.agent_name in {"codex", "gemini"} else None,
            source_url=f"https://example.com/{task.agent_name}/{task.prompt_variant.family}",
            why_relevant="It improves the module's long-context path.",
            transferable_idea="Use an implementation-ready scheduling technique for this module.",
            evidence=[
                PaperEvidence(
                    agent_name=task.agent_name,
                    prompt_variant_id=task.prompt_variant.variant_id,
                    quote_or_note="Section 3 reports the implementation technique.",
                    source_url=f"https://example.com/{task.agent_name}",
                )
            ],
            relevance_score=0.9,
        )
        return AgentSearchOutput(
            task_id=task.task_id,
            agent_name=task.agent_name,
            prompt_variant_id=task.prompt_variant.variant_id,
            phase=task.phase,
            papers=[paper],
        )


class OutOfOrderExpandedRunner(FakeExpandedRunner):
    def run_task(self, *, task: ResearchTask, prompt: str, repo_path: Path) -> AgentSearchOutput:
        if task.prompt_variant.family == "focused_module_lit_review":
            time.sleep(0.02)
        return super().run_task(task=task, prompt=prompt, repo_path=repo_path)


class BatchRecordingOrchestrator(ExpandedModuleResearchOrchestrator):
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.task_batches: list[list[ResearchTask]] = []

    def _run_tasks(
        self, packet: ModuleResearchPacket, tasks: list[ResearchTask]
    ) -> list[AgentSearchOutput]:
        self.task_batches.append(list(tasks))
        return super()._run_tasks(packet, tasks)


def _module() -> Module:
    return Module(
        name="attention",
        path="src/inference/attention",
        description="Attention kernel orchestration.",
        main_files=[File(path="src/inference/attention/core.py", role="attention core")],
    )


def _request(tmp_path: Path) -> ModuleDeepResearchInput:
    return ModuleDeepResearchInput(
        project_tree=ProjectTree(
            repository=Repository(name="demo", summary="Demo inference repository."),
            modules=[Module(name="inference", path="src/inference", submodules=[_module()])],
        ),
        module_qualified_name="inference.attention",
        context=SpotlightContext(
            objective="reduce long-context decode latency",
            workload_hints=["batch size 1-8"],
            validation_plan=["tokens/sec benchmark"],
        ),
        repo_path=tmp_path,
        max_findings_per_module=3,
    )


def _baseline() -> ModuleDeepResearchOutput:
    return ModuleDeepResearchOutput(
        findings=[
            Finding(
                finding_id="find-0001",
                title="Vanilla Codex baseline paper",
                url="https://example.com/baseline",
                source_type="paper",
                technique_summary="Baseline technique.",
                supporting_evidence="Baseline evidence.",
            )
        ]
    )


def test_dedup_prefers_arxiv_key_over_title() -> None:
    paper = ResearchPaper(
        title="Paged Attention for Long Context",
        arxiv_id="2606.12345v2",
        source_url="https://arxiv.org/abs/2606.12345",
        why_relevant="Relevant to attention.",
        transferable_idea="Use paged attention allocation.",
    )

    assert "arxiv:2606.12345" in dedup_keys(paper)


def test_malformed_arxiv_ids_are_not_promoted_to_dedup_keys() -> None:
    paper = ResearchPaper(
        title="Malformed ID Candidate",
        arxiv_id="9054.2026",
        source_url="https://arxiv.org/abs/2026.13376",
        why_relevant="Relevant.",
        transferable_idea="Transferable.",
    )

    assert paper.arxiv_id is None
    assert extract_arxiv_id("https://arxiv.org/abs/9054.2026") is None
    assert extract_arxiv_id("https://arxiv.org/abs/2401.10774v2") == "2401.10774"
    assert all(not key.startswith("arxiv:") for key in dedup_keys(paper))


def test_merge_papers_cross_checks_agents_on_same_identifier() -> None:
    base = ResearchPaper(
        title="Shared Paper",
        arxiv_id="2606.12345",
        why_relevant="Relevant.",
        transferable_idea="Transferable.",
    )

    merged = merge_papers({"codex": [base], "gemini": [base]})

    assert len(merged) == 1
    assert merged[0].seen_by_agents == ["codex", "gemini"]
    assert merged[0].verification_status == "cross_checked"


def test_expanded_orchestrator_returns_contract_and_writes_sidecars(tmp_path: Path) -> None:
    runner = FakeExpandedRunner()
    config = ExpandedResearchConfig(
        artifacts_dir=tmp_path / "expanded",
        include_agents=("codex", "gemini"),
        prompt_generations=1,
        max_parallel_searches=2,
        max_papers_per_task=1,
    )

    orchestrator = BatchRecordingOrchestrator(
        request=_request(tmp_path),
        module=_module(),
        baseline_output=_baseline(),
        config=config,
        runner=runner,
    )
    output = orchestrator.run()

    assert output.findings
    assert output.findings[0].finding_id == "find-0001"
    assert all(finding.source_type == "paper" for finding in output.findings)
    assert len(runner.tasks) == 12  # 2 agents * (5 Step-1 map variants + 1 strategy pass).
    assert len(orchestrator.task_batches) == 1
    assert {
        task.phase for task in orchestrator.task_batches[0]
    } == {"map", "strategy_expansion"}
    assert [task.phase for task in orchestrator.task_batches[0][:4]] == [
        "map",
        "strategy_expansion",
        "map",
        "strategy_expansion",
    ]
    assert [task.agent_name for task in orchestrator.task_batches[0][:4]] == [
        "codex",
        "codex",
        "gemini",
        "gemini",
    ]
    assert {task.phase for task in runner.tasks} == {"map", "strategy_expansion"}
    assert sum(task.phase == "strategy_expansion" for task in runner.tasks) == 2
    assert "bfs_dfs_search_strategy" in {task.prompt_variant.family for task in runner.tasks}
    assert all(task.prompt_variant.family != "citation_chaser" for task in runner.tasks)
    assert all("Attention kernel orchestration" in prompt for prompt in runner.prompts)
    assert (tmp_path / "expanded" / "coverage_ui.json").exists()
    assert (tmp_path / "expanded" / "before_after.json").exists()
    assert (tmp_path / "expanded" / "prompt_evolution.json").exists()
    assert (tmp_path / "expanded" / "knowledge" / "archive" / "records.jsonl").exists()
    wiki_index = tmp_path / "expanded" / "knowledge" / "wiki" / "index.md"
    assert wiki_index.exists()
    assert "Expanded deep research summary: inference.attention" in wiki_index.read_text()


def test_expanded_orchestrator_can_disable_wiki_archive(tmp_path: Path) -> None:
    config = ExpandedResearchConfig(
        artifacts_dir=tmp_path / "expanded",
        include_agents=("codex",),
        max_papers_per_task=1,
        archive_to_wiki=False,
    )

    ExpandedModuleResearchOrchestrator(
        request=_request(tmp_path),
        module=_module(),
        baseline_output=_baseline(),
        config=config,
        runner=FakeExpandedRunner(),
    ).run()

    assert not (tmp_path / "expanded" / "knowledge").exists()


def test_expanded_orchestrator_can_disable_bfs_dfs_strategy_pass(tmp_path: Path) -> None:
    runner = FakeExpandedRunner()
    config = ExpandedResearchConfig(
        artifacts_dir=tmp_path / "expanded",
        include_agents=("codex", "gemini"),
        max_papers_per_task=1,
        include_bfs_dfs_strategy_pass=False,
    )

    ExpandedModuleResearchOrchestrator(
        request=_request(tmp_path),
        module=_module(),
        baseline_output=_baseline(),
        config=config,
        runner=runner,
    ).run()

    assert len(runner.tasks) == 10
    assert {task.phase for task in runner.tasks} == {"map"}
    assert "bfs_dfs_search_strategy" not in {
        task.prompt_variant.family for task in runner.tasks
    }


def test_parallel_bfs_dfs_strategy_uses_one_shared_worker_pool(
    tmp_path: Path,
    monkeypatch: Any,
) -> None:
    max_workers_seen: list[int | None] = []
    original_executor = orchestrator_module.concurrent.futures.ThreadPoolExecutor

    class RecordingThreadPoolExecutor:
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            max_workers = kwargs.get("max_workers")
            if max_workers is None and args:
                max_workers = args[0]
            max_workers_seen.append(max_workers)
            self._executor = original_executor(*args, **kwargs)

        def __enter__(self) -> Any:
            self._executor.__enter__()
            return self

        def __exit__(self, *args: Any) -> bool | None:
            return self._executor.__exit__(*args)

        def submit(self, *args: Any, **kwargs: Any) -> Any:
            return self._executor.submit(*args, **kwargs)

    monkeypatch.setattr(
        orchestrator_module.concurrent.futures,
        "ThreadPoolExecutor",
        RecordingThreadPoolExecutor,
    )
    runner = FakeExpandedRunner()
    config = ExpandedResearchConfig(
        artifacts_dir=tmp_path / "expanded",
        include_agents=("codex", "claude", "gemini"),
        prompt_generations=1,
        max_parallel_searches=5,
        max_papers_per_task=1,
        include_bfs_dfs_strategy_pass=True,
        parallel_strategy_pass=True,
    )

    orchestrator = BatchRecordingOrchestrator(
        request=_request(tmp_path),
        module=_module(),
        baseline_output=_baseline(),
        config=config,
        runner=runner,
    )
    orchestrator.run()

    assert max_workers_seen == [5]
    assert [len(batch) for batch in orchestrator.task_batches] == [18]
    assert [
        (task.agent_name, task.phase)
        for task in orchestrator.task_batches[0][:6]
    ] == [
        ("codex", "map"),
        ("codex", "strategy_expansion"),
        ("claude", "map"),
        ("claude", "strategy_expansion"),
        ("gemini", "map"),
        ("gemini", "strategy_expansion"),
    ]
    assert sum(task.phase == "strategy_expansion" for task in orchestrator.task_batches[0]) == 3
    assert sum(task.phase == "map" for task in orchestrator.task_batches[0]) == 15


def test_expanded_orchestrator_can_run_bfs_dfs_after_prompt_evolution(
    tmp_path: Path,
) -> None:
    runner = FakeExpandedRunner()
    config = ExpandedResearchConfig(
        artifacts_dir=tmp_path / "expanded",
        include_agents=("codex", "gemini"),
        max_papers_per_task=1,
        parallel_strategy_pass=False,
    )

    orchestrator = BatchRecordingOrchestrator(
        request=_request(tmp_path),
        module=_module(),
        baseline_output=_baseline(),
        config=config,
        runner=runner,
    )
    orchestrator.run()

    assert [len(batch) for batch in orchestrator.task_batches] == [10, 2]
    assert {task.phase for task in orchestrator.task_batches[0]} == {"map"}
    assert {task.phase for task in orchestrator.task_batches[1]} == {
        "strategy_expansion"
    }


def test_expanded_orchestrator_accepts_custom_knowledge_root(tmp_path: Path) -> None:
    config = ExpandedResearchConfig(
        artifacts_dir=tmp_path / "expanded",
        knowledge_root=tmp_path / "knowledge-root",
        include_agents=("codex",),
        max_papers_per_task=1,
    )

    ExpandedModuleResearchOrchestrator(
        request=_request(tmp_path),
        module=_module(),
        baseline_output=_baseline(),
        config=config,
        runner=FakeExpandedRunner(),
    ).run()

    record_pages = list((tmp_path / "knowledge-root" / "wiki" / "records").glob("*.md"))
    assert record_pages
    assert any("Expanded deep-search summary" in page.read_text() for page in record_pages)


def test_expanded_orchestrator_writes_deterministic_agent_output_order(tmp_path: Path) -> None:
    config = ExpandedResearchConfig(
        artifacts_dir=tmp_path / "expanded",
        include_agents=("codex",),
        prompt_generations=1,
        max_parallel_searches=2,
        max_papers_per_task=1,
    )

    ExpandedModuleResearchOrchestrator(
        request=_request(tmp_path),
        module=_module(),
        baseline_output=_baseline(),
        config=config,
        runner=OutOfOrderExpandedRunner(),
    ).run()

    payload = json.loads((tmp_path / "expanded" / "agent_outputs.json").read_text())

    assert [item["prompt_variant_id"] for item in payload] == [
        "g0.codex.focused_module_lit_review",
        "g0.codex.bfs_dfs_search_strategy",
        "g0.codex.implementation_transfer",
        "g0.codex.adversarial_missing_work",
        "g0.codex.benchmark_oriented",
        "g0.codex.skeptical_verifier",
    ]


def test_expanded_orchestrator_can_opt_into_citation_expansion(tmp_path: Path) -> None:
    runner = FakeExpandedRunner()
    config = ExpandedResearchConfig(
        artifacts_dir=tmp_path / "expanded",
        include_agents=("codex", "gemini"),
        prompt_generations=1,
        max_parallel_searches=2,
        max_papers_per_task=1,
        include_citation_chaser_prompts=True,
        enable_citation_expansion=True,
    )

    ExpandedModuleResearchOrchestrator(
        request=_request(tmp_path),
        module=_module(),
        baseline_output=_baseline(),
        config=config,
        runner=runner,
    ).run()

    assert len(runner.tasks) == 16  # 2 agents * 6 map variants + 2 strategy + 2 expansion.
    assert "citation_chaser" in {task.prompt_variant.family for task in runner.tasks}
    assert "strategy_expansion" in {task.phase for task in runner.tasks}
    assert "citation_expansion" in {task.phase for task in runner.tasks}
