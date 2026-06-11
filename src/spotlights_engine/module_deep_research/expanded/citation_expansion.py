"""Citation/reference expansion planning for expanded module deep research."""

from __future__ import annotations

from spotlights_engine.module_deep_research.expanded.models import MergedPaper, PromptVariant


def citation_expansion_variants(
    *, agents: tuple[str, ...], seed_papers: list[MergedPaper], generation: int
) -> list[PromptVariant]:
    """Build one citation/reference expansion prompt per agent from merged seeds."""
    if not seed_papers:
        return []
    seeds = "; ".join(paper.paper.title for paper in seed_papers[:5])
    variants: list[PromptVariant] = []
    for agent in agents:
        variants.append(
            PromptVariant(
                variant_id=f"g{generation}.{agent}.citation_expansion",
                agent_name=agent,  # type: ignore[arg-type]
                family="citation_chaser",
                generation=generation,
                title="Citation/reference expansion",
                instructions=(
                    "Use these seed papers as anchors: "
                    f"{seeds}. Inspect their references and citing papers. "
                    "Return only additional papers that improve module coverage, "
                    "fill a gap in the current set, or provide stronger evidence."
                ),
            )
        )
    return variants
