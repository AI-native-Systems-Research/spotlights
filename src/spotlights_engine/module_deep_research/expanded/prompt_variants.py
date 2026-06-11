"""Prompt planning and evolution for expanded module deep research."""

from __future__ import annotations

from collections import defaultdict
from datetime import date

from spotlights_engine.module_deep_research.expanded.dedup import dedup_keys
from spotlights_engine.module_deep_research.expanded.models import (
    AgentName,
    AgentSearchOutput,
    ModuleResearchPacket,
    PromptEvolutionReport,
    PromptVariant,
    PromptVariantScore,
)

_BASE_VARIANTS: tuple[tuple[str, str, str], ...] = (
    (
        "focused_module_lit_review",
        "Focused module literature review",
        (
            "Find papers with concrete methods that map directly to this "
            "module's owned responsibilities."
        ),
    ),
    (
        "implementation_transfer",
        "Implementation transfer scout",
        (
            "Prefer papers whose algorithms, data structures, or systems "
            "tricks can be translated into code changes."
        ),
    ),
    (
        "adversarial_missing_work",
        "Adversarial missing-work finder",
        (
            "Look for important papers that a normal web search would miss; "
            "challenge obvious keywords and search adjacent terminology."
        ),
    ),
    (
        "citation_chaser",
        "Citation and reference chaser",
        (
            "Use seed-like and discovered papers to identify their most "
            "relevant references and citing papers."
        ),
    ),
    (
        "benchmark_oriented",
        "Benchmark-oriented researcher",
        (
            "Prioritize papers with empirical claims, benchmarks, ablations, "
            "or reproducible implementation details."
        ),
    ),
    (
        "skeptical_verifier",
        "Skeptical verifier",
        (
            "Prefer fewer papers with stronger source evidence; reject "
            "topical but non-transferable work."
        ),
    ),
)


def initial_prompt_variants(
    agents: tuple[str, ...], *, include_citation_chaser: bool = False
) -> list[PromptVariant]:
    """Return first-generation prompt variants for each configured agent."""
    variants: list[PromptVariant] = []
    for agent in agents:
        for family, title, instructions in _BASE_VARIANTS:
            if family == "citation_chaser" and not include_citation_chaser:
                continue
            variants.append(
                PromptVariant(
                    variant_id=f"g0.{agent}.{family}",
                    agent_name=agent,  # type: ignore[arg-type]
                    family=family,  # type: ignore[arg-type]
                    generation=0,
                    title=title,
                    instructions=_agent_tuned_instructions(agent, instructions),
                )
            )
    return variants


def bfs_dfs_strategy_variants(
    agents: tuple[str, ...],
    *,
    generation: int,
) -> list[PromptVariant]:
    """Return additive BFS/DFS prompt variants for Step-1 coverage expansion."""
    return [
        PromptVariant(
            variant_id=f"g{generation}.{agent}.bfs_dfs_search_strategy",
            agent_name=agent,  # type: ignore[arg-type]
            family="bfs_dfs_search_strategy",
            generation=generation,
            title="BFS/DFS coverage strategist",
            instructions=_agent_tuned_instructions(
                agent,
                (
                    "Use an explicit graph-search mental model. First run a BFS-style "
                    "broad frontier over neighboring terminology, communities, and "
                    "implementation approaches. Then run DFS-style deep dives on the "
                    "most module-relevant branches until you reach concrete papers "
                    "with implementation-ready mechanisms. Return a balanced set: at "
                    "least one canonical anchor, one adjacent non-obvious paper, one "
                    "systems/serving paper, and one kernel/verification or scheduler "
                    "paper when available. Avoid citation/reference expansion; this "
                    "is an additive Step-1 map-search strategy."
                ),
            ),
        )
        for agent in agents
    ]


def render_search_prompt(
    *, packet: ModuleResearchPacket, variant: PromptVariant, max_papers: int, require_pdf: bool
) -> str:
    """Render one agent prompt for one module-scoped paper search task."""
    pdf_rule = (
        "Every paper must include a direct open-access PDF URL."
        if require_pdf
        else "Prefer direct open-access PDF URLs when available."
    )
    return f"""You are the {variant.title} for the Spotlight module_deep_research step.

Target module:
- Qualified name: {packet.module_qualified_name}
- Name: {packet.module_name}
- Path: {packet.module_path}
- Description: {packet.module_description or "(none)"}
- Depends on: {", ".join(packet.depends_on) or "(none)"}
- Main files: {"; ".join(packet.main_files) or "(none)"}

Repository:
- Name: {packet.repository_name}
- Summary: {packet.repository_summary}
- External dependencies: {", ".join(packet.external_dependencies) or "(none)"}

Caller context:
- Current date: {date.today().isoformat()}
- Objective: {packet.objective}
- Workload hints: {"; ".join(packet.workload_hints) or "(none)"}
- Validation plan: {"; ".join(packet.validation_plan) or "(none)"}

Prompt strategy:
{variant.instructions}

Task:
Find up to {max_papers} research papers that could improve this specific module.
{pdf_rule}

Selection rules:
- Keep only papers with a concrete transferable method, algorithm, data
  structure, or systems technique.
- The relevance must be to the target module, not just to the repository's
  broad domain.
- Prefer primary sources: arXiv, OpenReview, proceedings, publisher pages,
  official project pages, or implementation-rich papers.
- Include DOI, arXiv ID, OpenReview ID, source URL, and PDF URL when known.
- Do not synthesize identifiers. For arXiv IDs, use only verified IDs in
  `YYMM.NNNNN` form with month `01`-`12`; otherwise leave `arxiv_id` null.
- Explain the transferable idea in implementation-facing language.
- Include short evidence notes with source pointers; do not invent titles,
  URLs, authors, dates, or benchmark claims.
""".strip()


def score_outputs(outputs: list[AgentSearchOutput]) -> list[PromptVariantScore]:
    """Score prompt variants using deterministic quality and uniqueness signals."""
    all_keys: dict[str, int] = defaultdict(int)
    keys_by_variant: dict[str, set[str]] = defaultdict(set)
    papers_by_variant: dict[str, list[float]] = defaultdict(list)
    metadata: dict[str, tuple[AgentName, int]] = {}

    for output in outputs:
        for paper in output.papers:
            keys = dedup_keys(paper)
            keys_by_variant[output.prompt_variant_id].update(keys)
            papers_by_variant[output.prompt_variant_id].append(paper.relevance_score)
            for key in keys:
                all_keys[key] += 1
        generation = _generation_from_variant_id(output.prompt_variant_id)
        metadata[output.prompt_variant_id] = (output.agent_name, generation)

    scores: list[PromptVariantScore] = []
    for variant_id, keys in keys_by_variant.items():
        unique = sum(1 for key in keys if all_keys[key] == 1)
        relevance_scores = papers_by_variant[variant_id]
        avg_relevance = sum(relevance_scores) / len(relevance_scores) if relevance_scores else 0.0
        papers_found = len(relevance_scores)
        score = unique * 2.0 + papers_found + avg_relevance
        agent_name, generation = metadata[variant_id]
        scores.append(
            PromptVariantScore(
                variant_id=variant_id,
                agent_name=agent_name,
                generation=generation,
                papers_found=papers_found,
                unique_papers_found=unique,
                average_relevance=avg_relevance,
                score=score,
            )
        )
    return sorted(scores, key=lambda item: (item.agent_name, -item.score, item.variant_id))


def select_best_variants(
    scores: list[PromptVariantScore], *, top_per_agent: int
) -> dict[str, list[str]]:
    """Select top prompt variants per agent."""
    selected: dict[str, list[str]] = defaultdict(list)
    for score in sorted(scores, key=lambda item: (-item.score, item.variant_id)):
        bucket = selected[score.agent_name]
        if len(bucket) < top_per_agent:
            bucket.append(score.variant_id)
    return dict(selected)


def evolve_variants(
    *,
    previous_variants: list[PromptVariant],
    outputs: list[AgentSearchOutput],
    scores: list[PromptVariantScore],
    top_per_agent: int,
    next_generation: int,
) -> list[PromptVariant]:
    """Create next-generation prompt variants from the strongest prior variants."""
    selected = select_best_variants(scores, top_per_agent=top_per_agent)
    variants_by_id = {variant.variant_id: variant for variant in previous_variants}
    titles_by_variant = _top_titles_by_variant(outputs)
    evolved: list[PromptVariant] = []
    for agent_name, variant_ids in selected.items():
        for rank, variant_id in enumerate(variant_ids, start=1):
            parent = variants_by_id.get(variant_id)
            if parent is None:
                continue
            examples = (
                "; ".join(titles_by_variant.get(variant_id, [])[:3]) or "no strong papers yet"
            )
            evolved.append(
                PromptVariant(
                    variant_id=f"g{next_generation}.{agent_name}.evolved{rank}",
                    agent_name=agent_name,  # type: ignore[arg-type]
                    family="evolved",
                    generation=next_generation,
                    title=f"Evolved {parent.title}",
                    instructions=(
                        f"Improve the previous strategy `{parent.title}`. It found: {examples}. "
                        "Search for adjacent papers that either improve on those methods, "
                        "contradict them, or provide a more implementation-ready version "
                        "for this module. Avoid duplicates."
                    ),
                    parent_variant_id=variant_id,
                )
            )
    return evolved


def build_prompt_evolution_report(
    *, variants: list[PromptVariant], scores: list[PromptVariantScore], top_per_agent: int
) -> PromptEvolutionReport:
    """Assemble a serializable prompt evolution report."""
    return PromptEvolutionReport(
        variants=variants,
        scores=scores,
        selected_variant_ids=select_best_variants(scores, top_per_agent=top_per_agent),
    )


def _agent_tuned_instructions(agent: str, base: str) -> str:
    if agent == "claude":
        return f"{base} Be skeptical and prefer source-backed reasoning over breadth."
    if agent == "gemini":
        return f"{base} Use broad web coverage and URL/source context aggressively."
    if agent == "codex":
        return f"{base} Ground claims in the target files and implementation transfer path."
    return base


def _generation_from_variant_id(variant_id: str) -> int:
    prefix = variant_id.split(".", 1)[0]
    if prefix.startswith("g") and prefix[1:].isdigit():
        return int(prefix[1:])
    return 0


def _top_titles_by_variant(outputs: list[AgentSearchOutput]) -> dict[str, list[str]]:
    titles: dict[str, list[str]] = defaultdict(list)
    for output in outputs:
        for paper in sorted(output.papers, key=lambda item: item.relevance_score, reverse=True):
            titles[output.prompt_variant_id].append(paper.title)
    return dict(titles)
