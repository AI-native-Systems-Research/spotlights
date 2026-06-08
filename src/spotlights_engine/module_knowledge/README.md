# `module_knowledge`

`module_knowledge` is the lightweight Spotlights memory layer. It provides only
three public responsibilities:

1. **Archive** evidence-bearing records into local JSONL.
2. **Retrieve** ranked local records for a query.
3. **Render / verify** a generated Markdown wiki for humans and agents.

There are no phase folders, no live web/GitHub discovery, no telemetry feeder,
and no vector database in v1. The source of truth is always JSONL; the wiki is a
derived view.

## Storage layout

```text
<knowledge-root>/
  archive/
    records.jsonl        # source of truth
  wiki/
    index.md             # generated
    records/*.md         # generated record pages
    queries/*.md         # generated retrieval pages
```

Generated wiki pages include:

- `generated_from` metadata pointing back to `archive/records.jsonl`
- `source_hash` metadata so stale pages are detectable
- local artifact links when `provenance.artifact_path` exists on disk

## Basic usage

```python
from spotlights_engine.module_knowledge import (
    KnowledgeBase,
    KnowledgeRecord,
    Provenance,
    SourceRef,
)

kb = KnowledgeBase.open(".spotlights/knowledge")

record = KnowledgeRecord(
    record_id="paper:paged-kv",
    source_type="paper",
    title="Paged KV cache allocation",
    text="Paged allocation reduces KV cache fragmentation during decode.",
    source=SourceRef(
        source_id="src:paged-kv",
        title="Paged KV cache allocation",
        url="https://example.com/paged-kv",
        trust_tier="credible",
    ),
    provenance=Provenance(
        locator="paper:abstract",
        extractor="module_deep_research",
    ),
    tags=["kv-cache", "decode"],
)

kb.archive(record)

results = kb.retrieve("kv cache decode fragmentation", top_k=10)
kb.render_wiki()
kb.render_retrieval_wiki("kv cache decode fragmentation", results, query_id="kv-cache")
report = kb.verify_wiki(strict=True)
assert report.ok, report.issues
```

## Archive API

```python
kb.archive(record)
kb.archive_many(records)
```

Archive writes are upserts by `record_id`. The archive layer writes a
deterministic `provenance.content_hash` from the record content so stored records
are stable and replayable.

## Retrieve API

```python
from spotlights_engine.module_knowledge import RetrieveRequest

results = kb.retrieve(
    RetrieveRequest(
        query="kv cache scheduler",
        top_k=10,
        source_types=["paper", "module_map"],
        tags=["kv-cache"],
    )
)
```

Retrieval is dependency-free lexical ranking in v1. It ranks transparent matches
over title, tags, source type, source title, provenance locator, and text. A
future embedding or vector backend can sit behind the same API without changing
callers.

## Wiki API

```python
kb.render_wiki()
kb.render_retrieval_wiki("kv cache scheduler", results, query_id="kv-cache-scheduler")
kb.verify_wiki(strict=True)
```

`verify_wiki(strict=True)` checks generated metadata, source hashes, local links,
and that every archived record has a generated record page.


## CLI usage

The top-level `spotlights-engine` command exposes deterministic knowledge
commands. These commands do not run agents.

```bash
# Archive one KnowledgeRecord JSON object, or a JSON list of records.
spotlights-engine knowledge --root .spotlights/knowledge archive \
  --record-json record.json \
  --render-wiki

# Retrieve local memory.
spotlights-engine knowledge --root .spotlights/knowledge retrieve \
  "kv cache scheduler" \
  --source-type paper \
  --tag kv-cache

# Retrieve as JSON and optionally render a generated query page.
spotlights-engine knowledge --root .spotlights/knowledge retrieve \
  "kv cache scheduler" \
  --json \
  --render-query-page \
  --query-id kv-cache-scheduler

# Render / verify the generated wiki.
spotlights-engine knowledge --root .spotlights/knowledge wiki render
spotlights-engine knowledge --root .spotlights/knowledge wiki verify
```

For stdin archival use `--record-json -`.

## Adapters from existing Spotlights modules

### From `module_deep_research`

```python
from spotlights_engine.module_knowledge import records_from_module_deep_research

records = records_from_module_deep_research(
    module_deep_research_output,
    module_qualified_name="engine/cache",
)
kb.archive_many(records)
```

Each deep-research `Finding` becomes a `finding` knowledge record preserving:

- source URL and source type
- technique summary and supporting evidence
- target module name
- `module_deep_research:<module>:<finding_id>` provenance locator

### From `modules_extractor`

```python
from spotlights_engine.module_knowledge import records_from_project_tree

records = records_from_project_tree(
    project_tree,
    artifact_path="artifacts/ProjectTree.json",
)
kb.archive_many(records)
```

Each `ProjectTree` module becomes a `module_map` record with module path,
description, dependencies, main files, and optional artifact link.


### From `candidate_discovery` and proposal steps

```python
from spotlights_engine.module_knowledge import (
    records_from_candidates,
    records_from_module_run,
    records_from_spotlights_result,
)

# Archive one Candidates object, including attached deep-research and agent proposals.
kb.archive_many(records_from_candidates(candidates))

# Archive one manager module run: findings + candidates + proposals.
kb.archive_many(records_from_module_run(module_run))

# Archive a full Spotlights manager result across all module runs.
kb.archive_many(records_from_spotlights_result(spotlights_result))
```

Candidate records use `source_type="candidate"` and preserve file, line range,
symbol, kind, state, estimated impact, and anomaly references. Proposal records use
`source_type="proposal"` and preserve whether they came from
`proposal_from_finding_creator` or `agent_proposals`, plus candidate/finding/agent
metadata.

## Design rules

- Keep JSONL as source of truth.
- Keep the wiki generated and verifiable.
- Keep archive/retrieve/wiki independent from agent runtimes.
- Keep organization-specific runtime settings out of this module.
- Add new feeders through adapters that produce `KnowledgeRecord` objects.
