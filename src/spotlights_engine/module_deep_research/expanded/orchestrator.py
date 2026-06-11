"""High-level orchestration for expanded module deep research."""

from __future__ import annotations

import concurrent.futures
from collections import defaultdict
from dataclasses import dataclass

from spotlights_engine.module_deep_research.expanded.agent_runner import (
    AgentTaskRunner,
    CliAgentTaskRunner,
)
from spotlights_engine.module_deep_research.expanded.artifacts import write_expanded_artifacts
from spotlights_engine.module_deep_research.expanded.citation_expansion import (
    citation_expansion_variants,
)
from spotlights_engine.module_deep_research.expanded.compare import (
    build_coverage_report,
    compare_findings,
)
from spotlights_engine.module_deep_research.expanded.dedup import merge_papers
from spotlights_engine.module_deep_research.expanded.models import (
    AgentSearchOutput,
    ExpandedResearchConfig,
    ExpandedResearchReport,
    MergedPaper,
    ModuleResearchPacket,
    PaperEvidence,
    PromptVariant,
    ResearchPaper,
    ResearchTask,
    SearchPhase,
)
from spotlights_engine.module_deep_research.expanded.prompt_variants import (
    bfs_dfs_strategy_variants,
    build_prompt_evolution_report,
    evolve_variants,
    initial_prompt_variants,
    render_search_prompt,
    score_outputs,
)
from spotlights_engine.module_deep_research.expanded.reduce import reduce_to_findings
from spotlights_engine.module_deep_research.expanded.wiki_archive import (
    archive_expanded_research_to_wiki,
)
from spotlights_engine.schemas.common import StepIssue
from spotlights_engine.schemas.finding import Finding
from spotlights_engine.schemas.pipeline import ModuleDeepResearchInput, ModuleDeepResearchOutput
from spotlights_engine.schemas.project import File, Module


@dataclass(frozen=True)
class _SearchResult:
    """Internal search state produced before final Spotlight reduction."""

    variants: list[PromptVariant]
    outputs: list[AgentSearchOutput]
    merged_papers: list[MergedPaper]


class ExpandedModuleResearchOrchestrator:
    """Run the expanded research pipeline for one already-resolved module.

    `run()` is intentionally algorithm-shaped for review:

    1. prepare the module packet;
    2. search with agent/prompt variants;
    3. merge and reduce papers into the public output contract;
    4. write audit sidecars for comparison and UI coverage.

    Lower-level subprocess, prompt, dedup, and artifact mechanics stay behind
    focused helpers so the control flow remains readable.
    """

    def __init__(
        self,
        *,
        request: ModuleDeepResearchInput,
        module: Module,
        baseline_output: ModuleDeepResearchOutput,
        config: ExpandedResearchConfig,
        runner: AgentTaskRunner | None = None,
    ) -> None:
        self.request = request
        self.module = module
        self.baseline_output = baseline_output
        self.config = config
        self.runner = runner or CliAgentTaskRunner(config)

    def run(self) -> ModuleDeepResearchOutput:
        """Execute expanded deep search while preserving the public output contract."""
        packet = self._prepare_module_packet()
        search = self._run_expanded_search(packet)
        output = self._reduce_to_spotlight_contract(search)
        report = self._build_review_report(packet=packet, search=search, output=output)
        self._write_review_artifacts(report=report, output=output)
        issue_count = len(output.issues)
        output = self._archive_to_wiki(report=report, output=output)
        if len(output.issues) != issue_count:
            self._write_review_artifacts(report=report, output=output)
        return output

    def _prepare_module_packet(self) -> ModuleResearchPacket:
        """Collect the stable module context shared by every agent prompt."""
        repository = self.request.project_tree.repository
        return ModuleResearchPacket(
            module_qualified_name=self.request.module_qualified_name,
            repository_name=repository.name,
            repository_summary=repository.summary,
            external_dependencies=list(repository.external_dependencies),
            module_name=self.module.name,
            module_path=self.module.path,
            module_description=self.module.description,
            main_files=[_format_file(file) for file in self.module.main_files],
            depends_on=list(self.module.depends_on),
            objective=self.request.context.objective,
            workload_hints=list(self.request.context.workload_hints),
            validation_plan=list(self.request.context.validation_plan),
        )

    def _run_expanded_search(self, packet: ModuleResearchPacket) -> _SearchResult:
        """Run Step-1 map/reduce search, plus opt-in later citation expansion."""
        variants, outputs = self._run_prompt_evolution(packet)
        search = _SearchResult(
            variants=variants,
            outputs=outputs,
            merged_papers=self._merge(outputs),
        )

        if self.config.enable_citation_expansion:
            search = self._with_citation_expansion(packet=packet, search=search)

        return search

    def _reduce_to_spotlight_contract(self, search: _SearchResult) -> ModuleDeepResearchOutput:
        """Convert merged papers into the stable `ModuleDeepResearchOutput` contract."""
        findings = reduce_to_findings(
            merged_papers=search.merged_papers,
            max_findings=self.request.max_findings_per_module,
        )
        return self._choose_public_output(findings, search.outputs)

    def _write_review_artifacts(
        self,
        *,
        report: ExpandedResearchReport,
        output: ModuleDeepResearchOutput,
    ) -> None:
        """Write sidecars used for Venn, before/after, and prompt-evolution review."""
        write_expanded_artifacts(
            artifacts_dir=self.config.artifacts_dir,
            report=report,
            output=output,
        )

    def _archive_to_wiki(
        self,
        *,
        report: ExpandedResearchReport,
        output: ModuleDeepResearchOutput,
    ) -> ModuleDeepResearchOutput:
        """Archive a human-readable generated wiki without changing the return shape."""
        if not self.config.archive_to_wiki:
            return output
        try:
            archive_expanded_research_to_wiki(
                report=report,
                output=output,
                artifacts_dir=self.config.artifacts_dir,
                knowledge_root=self.config.knowledge_root,
            )
        except Exception as exc:  # pragma: no cover - defensive optional archive boundary
            return output.model_copy(
                update={
                    "issues": [
                        *output.issues,
                        _issue(f"expanded research wiki archive failed: {exc}"),
                    ]
                }
            )
        return output

    def _with_citation_expansion(
        self, *, packet: ModuleResearchPacket, search: _SearchResult
    ) -> _SearchResult:
        """Extend Step 1 results with the gated Step 3 citation/reference phase."""
        expansion_variants, expansion_outputs = self._run_citation_expansion(
            packet=packet,
            seed_papers=search.merged_papers,
        )
        all_variants = [*search.variants, *expansion_variants]
        all_outputs = [*search.outputs, *expansion_outputs]
        return _SearchResult(
            variants=all_variants,
            outputs=all_outputs,
            merged_papers=self._merge(all_outputs),
        )

    def _run_citation_expansion(
        self, *, packet: ModuleResearchPacket, seed_papers: list[MergedPaper]
    ) -> tuple[list[PromptVariant], list[AgentSearchOutput]]:
        """Run the opt-in Step-3 citation/reference expansion phase."""
        variants = citation_expansion_variants(
            agents=self.config.include_agents,
            seed_papers=seed_papers,
            generation=self.config.prompt_generations,
        )
        outputs = self._run_tasks(
            packet,
            self._tasks_for_variants(
                variants,
                generation=self.config.prompt_generations,
                phase="citation_expansion",
            ),
        )
        return variants, outputs

    def _build_review_report(
        self,
        *,
        packet: ModuleResearchPacket,
        search: _SearchResult,
        output: ModuleDeepResearchOutput,
    ) -> ExpandedResearchReport:
        """Assemble the review/UI sidecar without changing public output shape."""
        return ExpandedResearchReport(
            packet=packet,
            baseline=self._baseline_as_agent_output(),
            agent_outputs=search.outputs,
            prompt_evolution=build_prompt_evolution_report(
                variants=search.variants,
                scores=score_outputs(search.outputs),
                top_per_agent=self.config.top_prompts_per_agent,
            ),
            merged_papers=search.merged_papers,
            comparison=compare_findings(
                baseline_findings=self.baseline_output.findings,
                expanded_findings=output.findings,
            ),
            coverage=build_coverage_report(
                module_qualified_name=self.request.module_qualified_name,
                merged_papers=search.merged_papers,
            ),
        )

    def _run_prompt_evolution(
        self, packet: ModuleResearchPacket
    ) -> tuple[list[PromptVariant], list[AgentSearchOutput]]:
        """Explore standard prompts and optionally union them with BFS/DFS coverage."""
        variants = initial_prompt_variants(
            self.config.include_agents,
            include_citation_chaser=self.config.include_citation_chaser_prompts,
        )
        all_variants = list(variants)
        all_outputs: list[AgentSearchOutput] = []
        parallel_strategy_variants = (
            self._strategy_variants(generation=0)
            if self.config.include_bfs_dfs_strategy_pass
            and self.config.parallel_strategy_pass
            else []
        )
        all_variants.extend(parallel_strategy_variants)

        for generation in range(self.config.prompt_generations):
            tasks = self._tasks_for_variants(variants, generation=generation)
            if generation == 0 and parallel_strategy_variants:
                tasks = _interleave_task_lanes(
                    tasks,
                    self._strategy_tasks(
                        parallel_strategy_variants,
                        generation=generation,
                    ),
                )
            generation_outputs = self._run_tasks(packet, tasks)
            all_outputs.extend(generation_outputs)
            standard_outputs = [
                output
                for output in generation_outputs
                if output.phase != "strategy_expansion"
            ]

            if generation + 1 >= self.config.prompt_generations:
                break
            scores = score_outputs(standard_outputs)
            variants = evolve_variants(
                previous_variants=variants,
                outputs=standard_outputs,
                scores=scores,
                top_per_agent=self.config.top_prompts_per_agent,
                next_generation=generation + 1,
            )
            all_variants.extend(variants)

        if (
            self.config.include_bfs_dfs_strategy_pass
            and not self.config.parallel_strategy_pass
        ):
            strategy_generation = self.config.prompt_generations
            strategy_variants = self._strategy_variants(generation=strategy_generation)
            all_variants.extend(strategy_variants)
            all_outputs.extend(
                self._run_tasks(
                    packet,
                    self._strategy_tasks(
                        strategy_variants,
                        generation=strategy_generation,
                    ),
                )
            )

        return all_variants, all_outputs

    def _strategy_variants(self, *, generation: int) -> list[PromptVariant]:
        """Build the additive BFS/DFS variants for a specific execution point."""
        return bfs_dfs_strategy_variants(
            self.config.include_agents,
            generation=generation,
        )

    def _strategy_tasks(
        self, variants: list[PromptVariant], *, generation: int
    ) -> list[ResearchTask]:
        """Map BFS/DFS variants to the strategy phase without changing their family."""
        return self._tasks_for_variants(
            variants,
            generation=generation,
            phase="strategy_expansion",
        )

    def _tasks_for_variants(
        self,
        variants: list[PromptVariant],
        *,
        generation: int,
        phase: SearchPhase | None = None,
    ) -> list[ResearchTask]:
        return [
            ResearchTask(
                task_id=f"{self.request.module_qualified_name}.{variant.variant_id}",
                phase=phase or ("map" if generation == 0 else "prompt_evolution"),
                agent_name=variant.agent_name,
                prompt_variant=variant,
                module_qualified_name=self.request.module_qualified_name,
                max_papers=self.config.max_papers_per_task,
            )
            for variant in variants
        ]

    def _run_tasks(
        self, packet: ModuleResearchPacket, tasks: list[ResearchTask]
    ) -> list[AgentSearchOutput]:
        if not tasks:
            return []
        max_workers = min(self.config.max_parallel_searches, len(tasks))
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {
                executor.submit(self._run_one_task, packet=packet, task=task): index
                for index, task in enumerate(tasks)
            }
            ordered: list[AgentSearchOutput | None] = [None] * len(tasks)
            for future in concurrent.futures.as_completed(futures):
                ordered[futures[future]] = future.result()
            return [output for output in ordered if output is not None]

    def _run_one_task(
        self, *, packet: ModuleResearchPacket, task: ResearchTask
    ) -> AgentSearchOutput:
        prompt = render_search_prompt(
            packet=packet,
            variant=task.prompt_variant,
            max_papers=task.max_papers,
            require_pdf=self.config.require_open_access_pdf,
        )
        try:
            return self.runner.run_task(task=task, prompt=prompt, repo_path=self.request.repo_path)
        except Exception as exc:  # pragma: no cover - defensive boundary around external CLIs
            return AgentSearchOutput(
                task_id=task.task_id,
                agent_name=task.agent_name,
                prompt_variant_id=task.prompt_variant.variant_id,
                phase=task.phase,
                papers=[],
                notes=[],
                error=f"expanded research task failed: {exc}",
            )

    def _merge(self, outputs: list[AgentSearchOutput]) -> list[MergedPaper]:
        papers_by_agent: dict[str, list[ResearchPaper]] = defaultdict(list)
        for output in outputs:
            papers_by_agent[output.agent_name].extend(output.papers)
        return merge_papers(papers_by_agent)

    def _choose_public_output(
        self, expanded_findings: list[Finding], outputs: list[AgentSearchOutput]
    ) -> ModuleDeepResearchOutput:
        issues = list(self.baseline_output.issues)
        issues.extend(self._issues_from_outputs(outputs))
        if expanded_findings:
            return ModuleDeepResearchOutput(findings=expanded_findings, issues=issues)
        if self.baseline_output.findings:
            issues.append(
                _issue("expanded research produced no findings; returning vanilla Codex baseline")
            )
            return ModuleDeepResearchOutput(findings=self.baseline_output.findings, issues=issues)
        return ModuleDeepResearchOutput(findings=[], issues=issues)

    def _baseline_as_agent_output(self) -> AgentSearchOutput | None:
        if not self.baseline_output.findings:
            return None
        return AgentSearchOutput(
            task_id=f"{self.request.module_qualified_name}.vanilla_codex",
            agent_name="vanilla_codex",
            prompt_variant_id="g0.vanilla_codex.vanilla",
            phase="baseline",
            papers=[_finding_as_paper(finding) for finding in self.baseline_output.findings],
            notes=["Converted from vanilla Codex-only ModuleDeepResearchOutput."],
        )

    def _issues_from_outputs(self, outputs: list[AgentSearchOutput]) -> list[StepIssue]:
        return [_issue(output.error) for output in outputs if output.error]


def _format_file(file: File) -> str:
    return f"{file.path}: {file.role}"


def _finding_as_paper(finding: Finding) -> ResearchPaper:
    return ResearchPaper(
        title=finding.title,
        source_url=finding.url,
        why_relevant=finding.technique_summary,
        transferable_idea=finding.technique_summary,
        evidence=[
            PaperEvidence(
                agent_name="vanilla_codex",
                prompt_variant_id="g0.vanilla_codex.vanilla",
                quote_or_note=finding.supporting_evidence or finding.technique_summary,
                source_url=finding.url,
            )
        ],
        relevance_score=0.5,
    )


def _issue(message: str) -> StepIssue:
    return StepIssue(
        step="module_deep_research",
        severity="warning",
        message=message,
        recoverable=True,
    )


def _interleave_task_lanes(
    primary: list[ResearchTask], secondary: list[ResearchTask]
) -> list[ResearchTask]:
    """Fairly submit two task lanes to one executor and one concurrency cap."""
    primary = _round_robin_by_agent(primary)
    secondary = _round_robin_by_agent(secondary)
    interleaved: list[ResearchTask] = []
    for index in range(max(len(primary), len(secondary))):
        if index < len(primary):
            interleaved.append(primary[index])
        if index < len(secondary):
            interleaved.append(secondary[index])
    return interleaved


def _round_robin_by_agent(tasks: list[ResearchTask]) -> list[ResearchTask]:
    """Keep deterministic order while avoiding one agent monopolizing early workers."""
    by_agent: dict[str, list[ResearchTask]] = {}
    for task in tasks:
        by_agent.setdefault(task.agent_name, []).append(task)

    ordered: list[ResearchTask] = []
    buckets = list(by_agent.values())
    for index in range(max((len(bucket) for bucket in buckets), default=0)):
        for bucket in buckets:
            if index < len(bucket):
                ordered.append(bucket[index])
    return ordered
