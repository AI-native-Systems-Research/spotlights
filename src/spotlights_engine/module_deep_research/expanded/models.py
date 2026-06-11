"""Internal models for expanded module deep research.

The public Spotlight contract remains `ModuleDeepResearchInput ->
ModuleDeepResearchOutput`.  These models describe the richer sidecar data used
for audit, prompt evolution, deduplication, and UI coverage views.
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

from spotlights_engine.module_deep_research.expanded.identifiers import normalize_arxiv_id

AgentName = Literal["vanilla_codex", "codex", "claude", "gemini"]
PromptFamily = Literal[
    "vanilla",
    "focused_module_lit_review",
    "implementation_transfer",
    "adversarial_missing_work",
    "citation_chaser",
    "benchmark_oriented",
    "skeptical_verifier",
    "bfs_dfs_search_strategy",
    "evolved",
]
SearchPhase = Literal[
    "baseline",
    "map",
    "prompt_evolution",
    "strategy_expansion",
    "citation_expansion",
    "reduce",
]
VerificationStatus = Literal["unverified", "single_source", "cross_checked", "rejected"]


class ExpandedResearchConfig(BaseModel):
    """Runtime knobs for the expanded multi-agent deep-research implementation."""

    model_config = ConfigDict(extra="forbid", arbitrary_types_allowed=True)

    enabled: bool = Field(
        default=True,
        description=(
            "Run expanded multi-agent research instead of returning only the vanilla baseline."
        ),
    )
    run_vanilla_baseline: bool = Field(
        default=False,
        description=(
            "Also run the legacy Codex-only module_deep_research step before expanded "
            "search so before/after reports include a live vanilla baseline. Disabled "
            "by default so normal expanded runs execute only the new multi-agent path."
        ),
    )
    artifacts_dir: Path | None = Field(
        default=None,
        description="Directory where expanded sidecar artifacts are written for this module.",
    )
    archive_to_wiki: bool = Field(
        default=True,
        description=(
            "Archive expanded research into the generated module-knowledge wiki when "
            "artifacts_dir or knowledge_root is available. This writes human-readable "
            "wiki pages without changing ModuleDeepResearchOutput."
        ),
    )
    knowledge_root: Path | None = Field(
        default=None,
        description=(
            "Optional module-knowledge root. Defaults to artifacts_dir / 'knowledge' "
            "when archive_to_wiki is enabled."
        ),
    )
    max_parallel_searches: int = Field(
        default=5,
        ge=1,
        le=5,
        description="Hard cap for concurrent agent/search tasks across map and expansion phases.",
    )
    prompt_generations: int = Field(
        default=1,
        ge=1,
        le=4,
        description="Number of prompt-evolution generations to run per module.",
    )
    top_prompts_per_agent: int = Field(
        default=2,
        ge=1,
        le=5,
        description="Number of highest-scoring prompt variants retained per agent for evolution.",
    )
    include_bfs_dfs_strategy_pass: bool = Field(
        default=True,
        description=(
            "Run an additive BFS/DFS search-strategy lane. Its papers are merged and "
            "deduplicated with the standard lane so the strategy can add coverage without "
            "replacing papers found by normal evolution."
        ),
    )
    parallel_strategy_pass: bool = Field(
        default=True,
        description=(
            "When the BFS/DFS strategy lane is enabled, run it in the first shared task "
            "batch beside the standard map prompts while still respecting "
            "max_parallel_searches. Set false to run the strategy lane after standard "
            "prompt evolution for easier A/B isolation."
        ),
    )
    include_citation_chaser_prompts: bool = Field(
        default=False,
        description=(
            "Include citation/reference-chaser prompt personalities in the initial "
            "multi-agent map search. Disabled by default for the Step 1 workflow."
        ),
    )
    enable_citation_expansion: bool = Field(
        default=False,
        description=(
            "Run the later seed-paper citation/reference expansion phase. This is "
            "Step 3 behavior and is intentionally disabled for the current Step 1 "
            "deep-search preparation workflow."
        ),
    )
    max_papers_per_task: int = Field(
        default=6,
        ge=1,
        le=20,
        description="Maximum paper records requested from each agent task.",
    )
    include_agents: tuple[Literal["codex", "claude", "gemini"], ...] = Field(
        default=("codex", "claude", "gemini"),
        description=(
            "Agent CLIs participating in expanded search; vanilla Codex "
            "is tracked separately as the control."
        ),
    )
    codex_profile: str = Field(
        default="gpt55",
        description=("Codex CLI profile for expanded Codex, typically Azure GPT-5.5 via LiteLLM."),
    )
    claude_model: str = Field(
        default="claude-opus-4-7",
        description="Claude Code model ID used through the Anthropic-compatible route.",
    )
    gemini_model: str = Field(
        default="gcp/gemini-3.1-pro-preview",
        description="Gemini CLI model ID used through the Gemini-compatible LiteLLM route.",
    )
    timeout_seconds: int = Field(
        default=900,
        ge=30,
        description="Wall-clock timeout for one CLI-agent research task.",
    )
    require_open_access_pdf: bool = Field(
        default=False,
        description="Require a direct open-access PDF URL for every paper.",
    )

    @field_validator("include_agents")
    @classmethod
    def _at_least_one_agent(
        cls, value: tuple[Literal["codex", "claude", "gemini"], ...]
    ) -> tuple[Literal["codex", "claude", "gemini"], ...]:
        if not value:
            raise ValueError("include_agents must contain at least one agent")
        return tuple(dict.fromkeys(value))


class ModuleResearchPacket(BaseModel):
    """Module-scoped input packet used by prompt builders and sidecar artifacts."""

    model_config = ConfigDict(extra="forbid")

    module_qualified_name: str = Field(description="Dot-qualified Spotlight target module name.")
    repository_name: str = Field(description="Repository name from the extracted project tree.")
    repository_summary: str = Field(
        description="Repository summary from the extracted project tree."
    )
    external_dependencies: list[str] = Field(
        default_factory=list,
        description="Repository-level external dependencies relevant to research prompts.",
    )
    module_name: str = Field(description="Target module short name.")
    module_path: str = Field(description="Repository-relative target module path.")
    module_description: str = Field(description="Target module description emitted by extraction.")
    main_files: list[str] = Field(
        default_factory=list,
        description="Repository-relative main files with their module roles.",
    )
    depends_on: list[str] = Field(
        default_factory=list,
        description="Other module names this module depends on.",
    )
    objective: str = Field(description="Caller objective from SpotlightContext.")
    workload_hints: list[str] = Field(
        default_factory=list,
        description="Caller workload hints used to judge module relevance.",
    )
    validation_plan: list[str] = Field(
        default_factory=list,
        description="Caller validation plan used to prefer testable findings.",
    )


class PromptVariant(BaseModel):
    """One prompt strategy for one agent and generation."""

    model_config = ConfigDict(extra="forbid")

    variant_id: str = Field(description="Stable identifier for this prompt variant.")
    agent_name: AgentName = Field(description="Agent this variant is intended for.")
    family: PromptFamily = Field(description="Prompt family or role strategy.")
    generation: int = Field(ge=0, description="Prompt evolution generation number.")
    title: str = Field(description="Human-readable prompt variant label.")
    instructions: str = Field(description="Prompt-specific instructions appended to the base task.")
    parent_variant_id: str | None = Field(
        default=None,
        description="Previous-generation variant that produced this evolved prompt, if any.",
    )


class ResearchTask(BaseModel):
    """A single module-scoped search task executed by one CLI agent."""

    model_config = ConfigDict(extra="forbid")

    task_id: str = Field(description="Stable task identifier used for artifacts and logs.")
    phase: SearchPhase = Field(description="Pipeline phase that created this task.")
    agent_name: AgentName = Field(description="Agent expected to execute the task.")
    prompt_variant: PromptVariant = Field(description="Prompt variant used for this task.")
    module_qualified_name: str = Field(description="Target module qualified name.")
    max_papers: int = Field(ge=1, description="Maximum number of papers requested.")


class PaperEvidence(BaseModel):
    """Agent-provided evidence for why a paper belongs in this module search."""

    model_config = ConfigDict(extra="forbid")

    agent_name: AgentName = Field(description="Agent that supplied this evidence.")
    prompt_variant_id: str = Field(description="Prompt variant that supplied this evidence.")
    quote_or_note: str = Field(
        description="Short quote, source pointer, or paraphrased evidence note."
    )
    source_url: str | None = Field(default=None, description="URL supporting this evidence item.")


class ResearchPaper(BaseModel):
    """Structured paper record emitted by an agent or produced by reduction."""

    model_config = ConfigDict(extra="forbid")

    title: str = Field(description="Exact paper title as reported by the source.")
    authors: list[str] = Field(default_factory=list, description="Paper authors, if known.")
    year: int | None = Field(
        default=None, ge=1900, le=2100, description="Publication year, if known."
    )
    date: str | None = Field(default=None, description="Publication/submission date, ISO if known.")
    abstract: str | None = Field(default=None, description="Short abstract or source summary.")
    doi: str | None = Field(default=None, description="DOI without URL prefix, if known.")
    arxiv_id: str | None = Field(
        default=None, description="arXiv identifier without version, if known."
    )
    openreview_id: str | None = Field(default=None, description="OpenReview identifier, if known.")
    source_url: str | None = Field(default=None, description="Canonical source URL for the paper.")
    pdf_url: str | None = Field(default=None, description="Direct PDF URL, if available.")
    venue: str | None = Field(
        default=None, description="Venue, workshop, journal, or preprint host."
    )
    why_relevant: str = Field(description="Why this paper matters for the target module.")
    transferable_idea: str = Field(
        description="Concrete method or design idea the module could adopt."
    )
    evidence: list[PaperEvidence] = Field(
        default_factory=list,
        description="Evidence snippets and source pointers supporting relevance.",
    )
    relevance_score: float = Field(
        default=0.5,
        ge=0.0,
        le=1.0,
        description="Agent-estimated relevance to module and objective.",
    )

    @field_validator("arxiv_id")
    @classmethod
    def _normalize_arxiv_id(cls, value: str | None) -> str | None:
        return normalize_arxiv_id(value)


class AgentSearchOutput(BaseModel):
    """Structured output from one agent task."""

    model_config = ConfigDict(extra="forbid")

    task_id: str = Field(description="ResearchTask identifier.")
    agent_name: AgentName = Field(description="Agent that executed the task.")
    prompt_variant_id: str = Field(description="PromptVariant identifier.")
    phase: SearchPhase = Field(description="Phase that produced this output.")
    papers: list[ResearchPaper] = Field(
        default_factory=list,
        description="Papers found by the agent for this task.",
    )
    notes: list[str] = Field(default_factory=list, description="Agent notes or caveats.")
    error: str | None = Field(default=None, description="Recoverable task error, if any.")


class PromptVariantScore(BaseModel):
    """Deterministic score assigned to one prompt variant after a generation."""

    model_config = ConfigDict(extra="forbid")

    variant_id: str = Field(description="Prompt variant being scored.")
    agent_name: AgentName = Field(description="Agent that used the variant.")
    generation: int = Field(description="Prompt generation that produced the score.")
    papers_found: int = Field(ge=0, description="Raw count of papers found.")
    unique_papers_found: int = Field(ge=0, description="Deduped count contributed by the variant.")
    average_relevance: float = Field(ge=0.0, le=1.0, description="Mean relevance score.")
    score: float = Field(ge=0.0, description="Composite prompt quality score.")


class PromptEvolutionReport(BaseModel):
    """Prompt evolution trace for audit and review."""

    model_config = ConfigDict(extra="forbid")

    variants: list[PromptVariant] = Field(description="All prompt variants that were planned.")
    scores: list[PromptVariantScore] = Field(description="Scores observed for prompt variants.")
    selected_variant_ids: dict[str, list[str]] = Field(
        default_factory=dict,
        description="Best variant IDs retained per agent name.",
    )


class MergedPaper(BaseModel):
    """Canonical deduplicated paper record."""

    model_config = ConfigDict(extra="forbid")

    canonical_id: str = Field(description="Stable deduplication identifier.")
    paper: ResearchPaper = Field(description="Best representative paper record.")
    seen_by_agents: list[AgentName] = Field(description="Agents that found this paper.")
    seen_by_prompt_variants: list[str] = Field(description="Prompt variants that found this paper.")
    duplicate_keys: list[str] = Field(
        description="All deduplication keys associated with the group."
    )
    duplicate_count: int = Field(ge=1, description="Number of records merged into this paper.")
    verification_status: VerificationStatus = Field(description="Source verification state.")


class BeforeAfterComparison(BaseModel):
    """Comparison between vanilla Codex-only output and expanded output."""

    model_config = ConfigDict(extra="forbid")

    baseline_finding_titles: list[str] = Field(
        description="Finding titles from vanilla Codex-only."
    )
    expanded_finding_titles: list[str] = Field(description="Finding titles after expansion.")
    added_titles: list[str] = Field(description="Titles present only after expansion.")
    baseline_only_titles: list[str] = Field(description="Titles present only in the baseline.")
    overlap_titles: list[str] = Field(
        description="Titles present in both baseline and expanded output."
    )


class CoverageReport(BaseModel):
    """UI-ready coverage sets for Venn/matrix views."""

    model_config = ConfigDict(extra="forbid")

    module_qualified_name: str = Field(description="Target module qualified name.")
    agents: list[AgentName] = Field(description="Agents included in the coverage report.")
    sets: dict[str, list[str]] = Field(description="Canonical paper IDs found by each agent.")
    only_by: dict[str, list[str]] = Field(description="Canonical paper IDs unique to each agent.")
    all_agents: list[str] = Field(description="Canonical paper IDs found by every expanded agent.")


class ExpandedResearchReport(BaseModel):
    """Complete sidecar report for one module expanded deep-search run."""

    model_config = ConfigDict(extra="forbid")

    packet: ModuleResearchPacket = Field(description="Module-scoped research input packet.")
    baseline: AgentSearchOutput | None = Field(
        default=None,
        description="Vanilla Codex-only baseline converted to paper-like records when available.",
    )
    agent_outputs: list[AgentSearchOutput] = Field(description="All expanded agent outputs.")
    prompt_evolution: PromptEvolutionReport = Field(description="Prompt evolution trace.")
    merged_papers: list[MergedPaper] = Field(description="Merged and deduplicated papers.")
    comparison: BeforeAfterComparison = Field(description="Baseline-vs-expanded comparison.")
    coverage: CoverageReport = Field(description="UI-ready agent coverage data.")
