# Module Knowledge

`module_knowledge` is the public Spotlights memory layer. It is intentionally
small: an Archive API, a Retrieve API, and a generated Markdown wiki. It does
not import the old phased Bundle B layout.

## Responsibilities

- Archive evidence-bearing records from existing Spotlights modules.
- Retrieve local records with transparent dependency-free lexical ranking.
- Render a generated wiki for humans and agents.
- Verify generated wiki pages against the JSONL source hash.

## Source of truth

`archive/records.jsonl` is the source of truth. Wiki pages are derived views and
carry `generated_from` plus `source_hash` metadata so stale pages are detectable.

## Initial feeders

- `module_deep_research` findings become `finding` records.
- `modules_extractor` `ProjectTree` modules become `module_map` records.
- `candidate_discovery` `Candidates` become `candidate` records.
- `proposal_from_finding_creator` and `agent_proposals` outputs attached to candidates become `proposal` records.
- `SpotlightsResult` / `ModuleRun` adapters collect findings, candidates, and proposals from completed manager runs.

Future feeders can archive issues, PRs, experiment outcomes, and user/agent notes
through the same `KnowledgeBase.archive(...)` or `archive_many(...)` path.

## Public API sketch

```python
from spotlights_engine.module_knowledge import KnowledgeBase, records_from_project_tree

kb = KnowledgeBase.open(".spotlights/knowledge")
kb.archive_many(records_from_project_tree(project_tree, artifact_path="artifacts/ProjectTree.json"))

results = kb.retrieve("kv cache scheduler", top_k=10)
kb.render_wiki()
kb.render_retrieval_wiki("kv cache scheduler", results, query_id="kv-cache-scheduler")
kb.verify_wiki(strict=True)
```

## V1 non-goals

- No phase folders.
- No telemetry feeder.
- No embeddings/vector DB.
- No live GitHub or web discovery agent.
- No IBM-specific runtime configuration.
