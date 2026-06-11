# Expanded Module Deep Research

This package implements the Step-1 expanded deep-search path while preserving
Spotlight's public contract:

```text
ModuleDeepResearchInput -> ModuleDeepResearchOutput
```

The review entrypoint is:

```text
src/spotlights_engine/module_deep_research/api.py
    research_module(...)
        -> ExpandedModuleResearchOrchestrator(...).run()

Optional comparison mode:

```text
ExpandedResearchConfig(run_vanilla_baseline=True)
    research_module(...)
        -> vanilla Codex baseline
        -> ExpandedModuleResearchOrchestrator(...).run()
```

## Algorithm shape

The orchestrator is intentionally written as a small functional pipeline:

```python
def run() -> ModuleDeepResearchOutput:
    packet = prepare_module_packet(input, module)
    search = run_expanded_search(packet)
    output = reduce_to_spotlight_contract(search)
    write_review_artifacts(packet, search, output)
    return output
```

Each line maps to one private method in:

```text
src/spotlights_engine/module_deep_research/expanded/orchestrator.py
```

The lower-level mechanics stay isolated:

```text
prompt_variants.py   # prompt families, prompt evolution, prompt scoring
agent_runner.py      # Claude/Codex/Gemini task execution through cli_agents
dedup.py             # paper merge keys and cross-agent canonicalization
identifiers.py       # DOI/arXiv extraction and malformed-ID rejection
reduce.py            # merged papers -> ModuleDeepResearchOutput findings
compare.py           # vanilla-vs-expanded and Venn coverage sets
artifacts.py         # JSON sidecar writer
```

## Step-1 search flow

```python
def run_expanded_search(packet):
    variants = initial_prompt_variants(agents=[codex, claude, gemini])
    strategy = (
        bfs_dfs_strategy_variants()
        if config.include_bfs_dfs_strategy_pass and config.parallel_strategy_pass
        else []
    )
    outputs = []

    for generation in prompt_generations:
        tasks = map_variants_to_agent_tasks(variants)
        if generation == 0:
            tasks = fair_interleave(tasks, map_strategy_to_agent_tasks(strategy))
        outputs += run_tasks_in_parallel(tasks, max_parallel_searches=5)

        standard_outputs = exclude_strategy_outputs(outputs)
        variants = evolve_best_prompts(standard_outputs)  # skipped after last generation

    if config.include_bfs_dfs_strategy_pass and not config.parallel_strategy_pass:
        outputs += run_bfs_dfs_strategy_pass(packet)

    search = merge_and_dedup(outputs)

    if config.enable_citation_expansion:
        search = run_step3_citation_expansion(search)

    return search
```

Default behavior is Step 1 only:

```python
include_citation_chaser_prompts = False
enable_citation_expansion = False
include_bfs_dfs_strategy_pass = True
parallel_strategy_pass = True
run_vanilla_baseline = False
max_parallel_searches = 5
archive_to_wiki = True
```

Citation/reference expansion is implemented behind the flag because it belongs
to the later Step-3 workflow, not the current Step-1 preparation run.
BFS/DFS is an additive Step-1 strategy lane: it is merged with standard prompt
evolution instead of replacing it, so the strategy can add papers without
dropping papers found by the normal search. By default the standard map pass
and BFS/DFS strategy pass share the first task batch and the same
`max_parallel_searches <= 5` worker cap; set `parallel_strategy_pass=False` to
run the strategy lane after prompt evolution for cleaner A/B timing.
The legacy Codex-only baseline remains available for before/after review, but
it is opt-in through `run_vanilla_baseline=True`; normal expanded runs execute
only the new multi-agent path.

## Agent roles and models

The expanded path uses three local CLI adapters:

```text
Codex  -> Codex CLI profile `gpt55`
Claude -> Claude Code model `claude-opus-4-7`
Gemini -> Gemini CLI model `gcp/gemini-3.1-pro-preview`
```

Those defaults live in:

```text
src/spotlights_engine/module_deep_research/expanded/models.py
src/spotlights_engine/module_deep_research/expanded/agent_runner.py
src/spotlights_engine/cli_agents/config.py
```

LiteLLM is treated as configurable proxy metadata, not as a fourth agent.

## Review artifacts

When `ExpandedResearchConfig.artifacts_dir` is set, each module writes:

```text
expanded_research_report.json          # full audit packet
module_deep_research.expanded.json     # public output contract
prompt_evolution.json                  # variants, scores, selected prompts
merged_papers.json                     # canonical deduplicated papers
before_after.json                      # vanilla Codex vs expanded output
coverage_ui.json                       # Venn/matrix-ready agent coverage
agent_outputs.json                     # raw normalized agent task outputs
```

These sidecars are review/UI data only. The public pipeline still returns
`ModuleDeepResearchOutput`.

## Wiki archive

Expanded research can also archive a human-readable generated wiki using the
existing `module_knowledge` archive/wiki system. This is default-on when an
artifact directory is available:

```python
ExpandedResearchConfig(
    artifacts_dir=Path(".../expanded_research"),
    archive_to_wiki=True,       # default
    knowledge_root=None,        # default: artifacts_dir / "knowledge"
)
```

The archive writes:

```text
knowledge/
  archive/
    records.jsonl               # source of truth for archived records
  wiki/
    index.md                    # generated human-readable index
    records/*.md                # one page per summary/paper record
```

You can redirect the archive to a shared project knowledge root:

```python
ExpandedResearchConfig(
    artifacts_dir=Path(".../expanded_research"),
    archive_to_wiki=True,
    knowledge_root=Path(".spotlights/knowledge"),
)
```

The wiki archive is intentionally a side effect only. It does not change:

```text
ModuleDeepResearchInput -> ModuleDeepResearchOutput
```

Archived records are generated from the same review report:

```python
expanded report -> KnowledgeRecord(summary)
expanded report -> KnowledgeRecord(paper) for each MergedPaper
KnowledgeBase.archive_many(records)
KnowledgeBase.render_wiki()
```

## Result structure

### Public return value

The only value returned to the rest of Spotlight is still:

```python
ModuleDeepResearchOutput(
    findings=[
        Finding(
            finding_id="find-0001",
            title="Paper or source title",
            url="https://...",
            source_type="paper",
            technique_summary="Implementation-facing transfer idea.",
            supporting_evidence="Short source-backed evidence note.",
        ),
    ],
    issues=[
        StepIssue(
            step="module_deep_research",
            severity="warning",
            message="Recoverable issue, if any.",
            recoverable=True,
        ),
    ],
)
```

Expanded research therefore does not leak internal agent-specific structures
into downstream Spotlight steps. The reducer turns merged papers into ordinary
`Finding` records.

### Full review sidecar

`expanded_research_report.json` is the complete audit packet:

```python
ExpandedResearchReport(
    packet=ModuleResearchPacket(...),        # module description, files, objective
    baseline=AgentSearchOutput(...),         # vanilla Codex converted to paper-like records
    agent_outputs=[AgentSearchOutput(...)],  # raw Codex/Claude/Gemini task outputs
    prompt_evolution=PromptEvolutionReport(...),
    merged_papers=[MergedPaper(...)],
    comparison=BeforeAfterComparison(...),
    coverage=CoverageReport(...),
)
```

### Agent output shape

Each Codex / Claude / Gemini task normalizes to:

```python
AgentSearchOutput(
    task_id="module.g0.codex.implementation_transfer",
    agent_name="codex",  # codex | claude | gemini
    prompt_variant_id="g0.codex.implementation_transfer",
    phase="map",  # map | prompt_evolution | citation_expansion
    papers=[
        ResearchPaper(
            title="Exact paper title",
            authors=["..."],
            year=2026,
            doi="10....",
            arxiv_id="2606.12345",  # normalized; malformed IDs become None
            openreview_id=None,
            source_url="https://...",
            pdf_url="https://...",
            venue="...",
            why_relevant="Why this paper matters for this module.",
            transferable_idea="Concrete code/design idea to try.",
            evidence=[
                PaperEvidence(
                    agent_name="codex",
                    prompt_variant_id="g0.codex.implementation_transfer",
                    quote_or_note="Short evidence note.",
                    source_url="https://...",
                ),
            ],
            relevance_score=0.9,
        ),
    ],
    notes=["Search caveats or gaps."],
    error=None,
)
```

Valid phases are:

```text
baseline | map | prompt_evolution | strategy_expansion | citation_expansion | reduce
```

### Merged paper shape

After map/reduce deduplication, papers become:

```python
MergedPaper(
    canonical_id="arxiv:2606.12345",  # strongest valid key, else title key
    paper=ResearchPaper(...),
    seen_by_agents=["codex", "gemini"],
    seen_by_prompt_variants=[
        "g0.codex.implementation_transfer",
        "g0.gemini.benchmark_oriented",
    ],
    duplicate_keys=["arxiv:2606.12345", "doi:10...."],
    duplicate_count=2,
    verification_status="cross_checked",  # single_source | cross_checked
)
```

### Prompt evolution shape

`prompt_evolution.json` contains:

```python
PromptEvolutionReport(
    variants=[PromptVariant(...)],
    scores=[
        PromptVariantScore(
            variant_id="g0.codex.adversarial_missing_work",
            agent_name="codex",
            generation=0,
            papers_found=4,
            unique_papers_found=3,
            average_relevance=0.92,
            score=10.92,
        ),
    ],
    selected_variant_ids={
        "codex": ["g0.codex.adversarial_missing_work"],
        "claude": ["g1.claude.evolved1"],
        "gemini": ["g1.gemini.evolved1"],
    },
)
```

### Venn / before-after UI shape

`coverage_ui.json` is ready for Venn or matrix rendering:

```python
CoverageReport(
    module_qualified_name="spec_decode",
    agents=["codex", "claude", "gemini"],
    sets={
        "codex": ["arxiv:..."],
        "claude": ["arxiv:..."],
        "gemini": ["arxiv:..."],
    },
    only_by={
        "codex": ["arxiv:..."],
        "claude": ["arxiv:..."],
        "gemini": ["arxiv:..."],
    },
    all_agents=["arxiv:..."],
)
```

`before_after.json` compares vanilla Codex against the expanded result:

```python
BeforeAfterComparison(
    baseline_finding_titles=["..."],
    expanded_finding_titles=["..."],
    added_titles=["Found only by expanded multi-agent search"],
    baseline_only_titles=["Found only by vanilla Codex"],
    overlap_titles=["Found by both"],
)
```

### Wiki archive record shape

The generated wiki uses existing module-knowledge records, not a bespoke format:

```python
KnowledgeRecord(
    record_id="expanded_research:spec_decode:summary",
    source_type="experiment",  # summary page
    title="Expanded deep research summary: spec_decode",
    text="Markdown evidence summary...",
    source=SourceRef(
        source_id="expanded_research:spec_decode",
        title="Expanded deep research for spec_decode",
        trust_tier="internal",
    ),
    provenance=Provenance(
        locator="expanded_module_deep_research:spec_decode:summary",
        extractor="expanded_module_deep_research",
        artifact_path="../expanded_research_report.json",
    ),
    tags=["spec_decode", "expanded-deep-research", "summary"],
    metadata={...},
)
```

Paper pages use `source_type="paper"` and include the canonical paper ID,
agents that found it, prompt variants, evidence notes, and links back to the
expanded research JSON artifact.

## Identifier guardrail

Agent outputs are not trusted blindly. `ResearchPaper.arxiv_id` is normalized
through `identifiers.normalize_arxiv_id`.

Only modern arXiv IDs in verified `YYMM.NNNNN` form with month `01` through
`12` survive. Malformed values become `None`, so they cannot become canonical
deduplication keys.
