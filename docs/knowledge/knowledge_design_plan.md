# Knowledge Wiki — Design Plan

A design plan for a shared, LLM-curated wiki that participates in the Spotlights pipeline. The wiki is inspired by Karpathy's [LLM wiki](https://gist.github.com/karpathy/442a6bf555914893e9891c11519de94f): raw sources stay immutable, the wiki is a markdown layer of concept and entity pages maintained by an LLM, and a small set of well-defined workflows (ingest, query, lint) keep it alive.

This plan **builds on the existing [module_knowledge](../../src/spotlights_engine/module_knowledge/) package** rather than replacing it. The existing code is a solid records layer (deterministic JSONL archive + 1:1 markdown projection + lexical retrieval). What's missing is the synthesis layer: concept-keyed pages that an LLM merges across many records and that the pipeline reads back as warm-start priors.

The design is **two layers, two subtrees**:

- **Records layer** (existing) — `module_knowledge.records`, writing `archive/records.jsonl` and `wiki/records/*.md`. Deterministic, replayable, the audit log.
- **Concepts layer** (new) — `module_knowledge.concepts`, writing `wiki/concepts/{techniques,entities}/*.md`. LLM-curated, accumulating, the warm-start cache.

The plan covers:

1. **Package restructure** — moving the existing code under `module_knowledge/records/` and adding `module_knowledge/concepts/` as a sibling.
2. **How records and concepts relate** — the lifecycle of a record, when ingest fires, how concept pages cite records, and what happens when records change.
3. **Operating modes** — `concepts-only`, `records-only`, `records-and-concepts`. The user-visible knob that decides which layers run.
4. **The concepts wiki itself** — on-disk shape, page schema, public API, invariants. Independent of any pipeline stage. Lint is described here with explicit "when to implement" markers.
5. **Wiring concepts into `module_deep_research`** — the ingest and query workflows, plus the orchestrator's wave-based ingest.

---

## 1. Package restructure

### 1.1 Why split

`module_knowledge` today *is* the records layer, but the name implies it covers all of Spotlights' memory. Adding a synthesis layer alongside it without renaming would leave the asymmetry: one half named after the package, the other half a sibling of "everything else." Splitting into two subpackages — `records/` and `concepts/` — fixes the naming, gives each layer its own files and tests, and lets the `module_knowledge` package as a whole accurately describe both.

### 1.2 Target layout

```text
src/spotlights_engine/module_knowledge/
  __init__.py                 # re-exports public surface from both subpackages
  layout.py                   # KnowledgeRoot — owns the shared root path + index
  index.py                    # cross-layer wiki/index.md generator
  records/                    # everything that exists today
    __init__.py
    schemas.py                # KnowledgeRecord, SourceRef, Provenance, ...
    archive.py                # KnowledgeBase
    store.py                  # read_records, write_records, sha256_file
    retrieve.py               # retrieve_records (lexical) — reused by concepts
    layout.py                 # RecordsLayout — JSONL + wiki/records subtree
    wiki.py                   # WikiRenderer (1:1 record → page)
    adapters.py               # records_from_module_run, ...
  concepts/                   # new subpackage
    __init__.py
    schemas.py                # ConceptPage, IngestRequest, IngestReport, LintReport
    layout.py                 # ConceptsLayout — wiki/concepts subtree
    ingest.py                 # LLM-driven ingest pass
    query.py                  # query over concept pages (reuses records.retrieve)
    wiki.py                   # render concept index, verify (sources/links/orphans)
    prompts.py                # ingest prompt templates
    lint.py                   # deferred — see 4.6
```

### 1.3 What changes for callers

The package-level `__init__.py` re-exports everything from both subpackages, so existing imports keep working without a flag day:

```python
# src/spotlights_engine/module_knowledge/__init__.py
from spotlights_engine.module_knowledge.records import (
    KnowledgeBase,
    KnowledgeRecord,
    RecordsLayout,
    retrieve_records,
    WikiRenderer,
    verify_wiki,
    records_from_module_run,
    # ... all existing names
)
from spotlights_engine.module_knowledge.concepts import (
    Wiki,
    ConceptPage,
    IngestRequest,
    IngestReport,
)
```

New code prefers explicit imports (`from spotlights_engine.module_knowledge.records import ...` or `... .concepts import ...`).

### 1.4 Disk layout after the split

Both layers live under one shared knowledge root: **`<artifacts-dir>/.spotlights/knowledge/`**, one per `--artifacts-dir`. Records and concepts are sibling subtrees under that root.

```text
<artifacts-dir>/.spotlights/knowledge/
  archive/
    records.jsonl                       # existing, source of truth for records
  wiki/
    index.md                            # cross-cutting landing page (records + concepts)
    records/                            # existing — deterministic 1:1 from JSONL
      finding-abc123.md
      candidate-xyz.md
      module_map-vllm.v1.kv_offload.md
      ...
    concepts/                           # new — LLM-synthesized
      techniques/                       # global — shared across subject systems
        paged-attention.md
        layer-wise-kv-offload.md
      entities/                         # subject-system-namespaced
        vllm/
          v1.kv_offload.md
          v1.scheduler.md
        sglang/
          scheduler.md
    queries/                            # existing — generated retrieval pages
      kv-cache-scheduler.md
    log/
      ingest-2026-06-09T14-12-03Z.json  # one ingest pass per file
  wiki_state.json                       # wave-level resume marker (see 5.4)
```

The technique-vs-entity split is the v1 cross-subject reuse model: technique pages are universal knowledge ("paged attention as a concept") and accumulate across runs against any subject system, while entity pages are anchored to a specific code region in a specific subject system. See §3.8.

`wiki/index.md` is regenerated after every records-side render and every concepts-side ingest. It's deterministic — a directory walk over both subtrees with counts and links.

### 1.5 Recommended commit order

1. **Pure directory move.** `module_knowledge/*.py` → `module_knowledge/records/*.py`. `__init__.py` re-exports preserve every existing import path. No behavior change, no new code, easy to review, easy to revert. Tests pass on the same fixtures.
2. **Optional rename.** `KnowledgeLayout` → `RecordsLayout`, plus introduce `KnowledgeRoot` as the shared parent. Skip if the existing names still read cleanly after the move.
3. **Add `concepts/` as a fresh subpackage**, with its own schemas, layout, ingest, query, and wiki rendering. Pure addition; records side is untouched.
4. **Pipeline wiring** (section 3) lands after concepts is unit-tested in isolation.

The first commit is load-bearing. If it merges cleanly, everything after is additive and the records layer stays frozen — which is the right state for stable infrastructure.

---

## 2. How records and concepts relate

The two layers are not independent stores that happen to live under the same root. They sit in a strict upstream/downstream relationship: **records is the audit log of evidence; concepts is LLM-synthesized commentary anchored to that log**. Every claim on a concept page must trace back to one or more records. Records have no knowledge of concepts at all — the relationship is one-way.

This section describes that relationship in four parts: the lifecycle of a single record, when ingest fires, how citation works, and what happens when records change.

### 2.1 The lifecycle of a record

A record is born when a pipeline stage produces evidence. It does not enter the concepts layer until ingest folds it in. The full path:

```
1. Pipeline stage produces structured output
   (Finding, Candidate, Proposal, ProjectTree module)
       │
       ▼
2. records-side adapter converts it to KnowledgeRecord
   (records_from_module_deep_research, records_from_candidates, ...)
       │
       ▼
3. KnowledgeBase.archive_many(...) appends to archive/records.jsonl
   (deterministic, no LLM, immediate)
       │
       ▼
4. WikiRenderer.render() projects each record to wiki/records/<record_id>.md
   (deterministic, no LLM, can be regenerated from JSONL at any time)
       │
       ▼
   ─── records side complete ───
       │
       ▼ (only at a wave boundary; see 2.2)
5. Orchestrator collects record_ids produced this wave
       │
       ▼
6. Wiki.ingest(IngestRequest(record_ids=[...])) reads those records from JSONL,
   retrieves relevant existing concept pages, runs an LLM merge pass
       │
       ▼
7. LLM emits a structured edit plan: (slug, action, body) tuples for 1..N pages
       │
       ▼
8. concepts/wiki.py applies edits atomically to wiki/concepts/{techniques,entities}/*.md
   Each page's frontmatter sources: list now contains the record_ids it cites
```

Steps 1–4 happen during a module's pipeline (today's behavior, unchanged). Steps 5–8 happen *between* modules, at wave boundaries. A record always exists in the records layer before it is ever cited by a concept page.

### 2.2 When ingest fires

Ingest is the **only** moment when a record crosses from the records layer into the concepts layer. It fires:

- **Between waves of module processing.** Records produced by all successful modules in a wave are batched into a single `IngestRequest`. With `--max-parallel 1` this means after every module; with `--max-parallel N` this means every N modules.
- **Never inside a module's pipeline.** The deep-research stage *reads* concept pages but never *writes* them.
- **Never on records that didn't survive the wave.** If a module fails, its records (if any were written before the failure) are excluded from the ingest request. This is enforced in the orchestrator (see 5.5): only successful module runs contribute `record_ids`.
- **Never on records that already have full citations.** Ingest is incremental — each pass only sees the new wave's record IDs, not the entire archive. Concept pages from prior waves are loaded by retrieval, not by re-reading their full source records.

This is what makes the system scale. The records archive grows linearly with pipeline output. The ingest cost per wave is bounded by the wave's record count + the retrieval depth on the concepts side, not by the total size of either store.

### 2.3 How concept pages cite records

A concept page's `sources` frontmatter list is a list of `record_id`s. The records layer owns those IDs; concepts only references them. This gives three concrete properties:

- **Every claim is traceable.** Open `concepts/techniques/paged-attention.md`, read the `sources:` block, look up each `record_id` in `archive/records.jsonl` (or open `wiki/records/<record_id>.md`). You see the exact finding, candidate, or paper a claim came from.
- **Citations survive ingest churn.** When a concept page is updated by a later wave, its `sources` list grows or is rewritten by the LLM, but the underlying records don't move. A page can cite records from runs months apart.
- **`verify()` enforces referential integrity.** The concepts-side verifier reads each page's `sources` list and checks that every `record_id` exists in `archive/records.jsonl`. A page that cites a missing record is a hard error — either the JSONL was tampered with or the page was hand-written incorrectly.

The reverse relationship — "which concept pages cite this record?" — is **not** stored. It can be derived by scanning all concept pages, but the records layer doesn't carry back-references. This keeps records side-effect-free and ensures the records layer remains regeneratable from raw inputs alone.

### 2.4 What happens when records change

Records are append-mostly. The records-layer API supports upsert by `record_id` (the existing `KnowledgeBase.archive(...)` is idempotent), so a record can be re-archived with updated content. This raises an obvious question: what about the concept pages that already cite it?

The answer is **records are mutable in shape, immutable in identity, and concept pages don't auto-update on records changes**:

- **Mutable in shape.** The body of a record can be updated by re-archiving with the same `record_id`. The records layer recomputes its content hash and the wiki/records page regenerates.
- **Immutable in identity.** A `record_id` once cited never goes away. Concept-page citations remain valid.
- **No auto-propagation.** Updating a record does *not* trigger re-ingest of every concept page that cites it. Concept pages may now cite a stale view of the evidence — that's a known tradeoff, and it's where lint (4.6) eventually earns its keep. For v1, the assumption is that records are mostly write-once: a finding is what it was at the time of the run, a project-tree module is what it was at extraction time. Re-archiving the same `record_id` should be rare.
- **Records deletion is not supported in v1.** Removing a record while concept pages cite it would create dangling citations. If a record needs to be retracted, mark it via `metadata` rather than deleting it; future lint can flag retracted records on concept pages.

### 2.5 What records are eligible for ingest

Not every record produced by the pipeline is a useful ingest input. The v1 policy is conservative:

| Source type | Ingested in v1? | Notes |
|---|---|---|
| `finding` | **Yes** | Primary input. Findings are the deep-research stage's output and have the richest synthesis content. |
| `module_map` | **Yes** | Used to seed entity pages with module descriptions and dependencies. |
| `candidate` | Deferred | Becomes a section on entity pages (`## Techniques considered here`) in a follow-up. The records exist; ingest just doesn't read them yet. |
| `proposal` | Deferred | Same as candidates — entity-page material, not technique-page material. |
| `paper`, `blog`, `docs`, `issue`, `pr`, `talk` | If present | Currently produced indirectly via finding metadata. If a future feeder writes these as standalone records, ingest can consume them. |
| `experiment`, `note` | No | Out of scope for v1. |

The eligible-types list lives in `concepts/ingest.py` as a constant, not in the records layer. Records doesn't know which types concepts consumes.

### 2.6 What this means for failure modes

Because the relationship is one-way and ingest is the only crossing point, failure modes are clean:

- **Concepts ingest crashes mid-wave** → records are intact, concept-side files use atomic rename so the previous version is preserved. Re-running the wave (or the next wave) re-ingests cleanly.
- **A concept page is hand-edited** → the records archive is unaffected. The page's `content_hash` mismatches and `verify()` flags it; the human-edited content is preserved unless ingest deliberately rewrites the page.
- **The concepts subtree is deleted** → records is unaffected. Re-running the pipeline (or running a hypothetical "rebuild concepts from records" tool) regenerates concept pages from the JSONL. The pipeline degrades gracefully — concept queries return empty until concepts is repopulated.
- **The records archive is corrupted** → concepts pages may cite missing record IDs. `verify()` flags this. Concepts is not designed to survive records corruption; the records layer is the load-bearing one.

This asymmetry is deliberate: the records layer is infrastructure-grade (append-mostly, deterministic, regeneratable), the concepts layer is best-effort synthesis (LLM-driven, lossy, regeneratable from records but expensive to do so).

---

## 3. Operating modes

The two layers can run together or independently. Which layers run is a user-visible config choice — the `mode` field on the knowledge config — not a hidden architectural assumption. There are three supported modes plus a kill-switch:

| Mode | Records JSONL? | Concepts pages? | Citations | What `query()` returns to deep research |
|---|---|---|---|---|
| **`concepts-only`** | No | Yes (LLM-synthesized) | Light references in `sources:` (`kind` + `id` + optional `title`/`url`). Best-effort resolution against the run's outputs and the web. | Concept pages (synthesis) |
| **`records-only`** | Yes | No | N/A — records is the only knowledge artifact. `wiki/records/<id>.md` is the deterministic 1:1 projection (today's behavior). | Records (raw evidence) |
| **`records-and-concepts`** | Yes | Yes | Concepts `sources:` list `record_id`s that resolve through `archive/records.jsonl` for full audit. **Default.** | Concept pages (synthesis) |
| `off` | No | No | Knowledge subsystem disabled entirely. Pipeline runs as if `module_knowledge` did not exist. | Nothing (no warm-start) |

The default is `records-and-concepts`. The other modes are equally first-class and fully supported — the architecture does not assume records is always present.

### 3.1 Why three modes

A single on/off toggle conflates two independent decisions:

1. **Do you want a structured audit log (records)?** Useful for downstream tooling, validation, replay, debugging. Cheap on disk, deterministic, no LLM.
2. **Do you want LLM-synthesized priors (concepts)?** Useful for compounding knowledge across runs and warm-starting deep research. Costs LLM calls per ingest, accumulates across runs.

These are orthogonal. Some users want one without the other. The mode field exposes that orthogonality directly instead of hiding it behind an opinion about defaults.

### 3.2 `concepts-only` — light references

`concepts-only` is the lightest-weight mode. No JSONL is written. Concept pages are self-sufficient: each page's `sources:` block contains direct references to whatever evidence backs the synthesis.

Frontmatter shape in this mode:

```yaml
sources:
  - kind: finding
    id: "vllm/v1.kv_offload/run-2026-06-09-a/finding-04"
    title: "Layer-wise offload reduces TTFT 1.4×"
    url: "spotlights-out/modules/v1.kv_offload.md#finding-04"
  - kind: paper
    id: "arxiv:2403.12345"
    title: "LMCache: long-context KV reuse"
    url: "https://arxiv.org/abs/2403.12345"
  - kind: issue
    id: "vllm#8234"
    url: "https://github.com/vllm-project/vllm/issues/8234"
```

Each entry is a *pointer*, not the evidence itself. The body of a finding lives in the run's output directory; the body of a paper lives on arxiv; the body of an issue lives on GitHub. The page never duplicates them.

What `verify()` does in this mode:

- Every `sources` entry is structurally valid (`kind`, `id` present).
- For `kind: finding` and similar local references, the path component (after `:` or `#`) exists on disk under the run output directory if available.
- For URLs, `verify()` does not fetch by default (would couple verification to the network). A `verify(check_urls=True)` opt-in does best-effort HEAD checks.
- Every `[[link]]` between concept pages resolves.

What you give up vs. `records-and-concepts`:

- **No audit replay.** A finding's exact text at the time of capture is not preserved unless the run's output directory is preserved alongside the wiki.
- **No structured programmatic surface.** Downstream tools cannot iterate over `KnowledgeRecord` objects; they have to parse markdown.
- **Weaker `verify()`.** External references (papers, issues) are checked for syntactic validity, not for resolution.

What you gain:

- One layer to reason about. No JSONL, no records-side adapters running, no records-side wiki rendering.
- Fewer moving parts when prototyping or running a single experiment.
- This is the design that would have shipped first if `module_knowledge` had not already existed. It remains the simplest viable wiki on its own.

### 3.3 `records-only` — structured audit log, no synthesis

`records-only` is today's behavior: the records-side adapters write `archive/records.jsonl`, `WikiRenderer` projects each record to a deterministic markdown page, and no LLM ingest runs. There is no concepts subtree at all.

This mode is for users who want:

- **Structured pipeline output for downstream tooling** — validation discovery, dashboards, cross-run comparisons read JSONL with a typed schema.
- **A baseline run with the LLM synthesis layer disabled** — useful for debugging "is concepts making things better or worse?" by toggling.
- **Maximum determinism** — every byte of the output is reproducible from pipeline inputs alone.

The records layer's own citations (per-record `provenance` blocks with locator, content_hash, and source URL) carry the audit information. There is no separate `sources:` story because there is no synthesis layer to cite *from*.

### 3.4 `records-and-concepts` — default, full audit

This is the design described in §2. Both layers run. Concept pages cite records by `record_id`, the records archive is the upstream of truth, `verify()` enforces referential integrity across the two stores.

Frontmatter shape in this mode:

```yaml
sources:
  - kind: finding
    record_id: "finding:vllm:v1.kv_offload:run-2026-06-09-a:f-04"
  - kind: paper
    record_id: "paper:arxiv:2403.12345"
```

The references are even lighter than in `concepts-only` — just a `kind` + `record_id` — because the actual evidence (title, url, body, provenance) is one JSONL lookup away. Page rendering can dereference `record_id`s on demand to show titles inline; the storage stays minimal.

This mode is the default because:

- Disk cost of JSONL is negligible relative to the LLM cost of running concepts at all.
- Audit, replay, and recovery come essentially for free once concepts is on.
- Downstream tooling (validation, future archive systems) gets the structured surface it needs.

### 3.5 Mode selection in code

The mode is a config field on the manager:

```python
class KnowledgeConfig(BaseModel):
    mode: Literal["concepts-only", "records-only", "records-and-concepts", "off"] = (
        "records-and-concepts"
    )
    # ...other concepts/records-specific settings
```

CLI surface:

```bash
spotlights-engine --knowledge-mode concepts-only ...
spotlights-engine --knowledge-mode records-only ...
spotlights-engine --knowledge-mode off ...        # equivalent to today's --no-wiki idea
```

The orchestrator's wave loop branches on `mode`:

- `off` — no `Wiki.open(...)`, no records-side archive calls, no ingest, no concepts query. The pipeline behaves as if the knowledge subsystem did not exist.
- `records-only` — adapters run as today, `WikiRenderer.render()` runs after each module, no `Wiki.open(...)`, no ingest.
- `concepts-only` — `Wiki.open(...)` runs, ingest runs at wave boundaries, query runs before deep research. Records-side adapters are bypassed; ingest receives source bodies inline (see 5.3).
- `records-and-concepts` — both halves run. The orchestrator collects `record_id`s from successful modules' checkpoint state (see 5.4) and passes them to ingest.

### 3.6 What changes between modes

The page body is identical across `concepts-only` and `records-and-concepts`. The same LLM ingest prompt produces the same synthesis. Only three things differ:

1. **`sources:` shape** — light references vs. `record_id` references, as shown above.
2. **`IngestRequest` shape** — see 5.3. Carries inline source bodies in `concepts-only`, carries `record_id`s in `records-and-concepts`.
3. **`verify()` strictness** — referential integrity check is enforced only when records is present.

This is deliberate: switching modes should not change what the wiki *says*, only how its claims are anchored. A user moving from `concepts-only` to `records-and-concepts` mid-project gets a stricter verifier and a richer audit trail without re-synthesizing pages. (A small migration tool — `concepts upgrade-sources` — can rewrite light references into `record_id` references once records is populated.)

### 3.7 Cross-objective reuse

The wiki is shared across runs of different objectives against the same subject system. A run with `--objective "reduce TTFT"` and a later run with `--objective "reduce P99 TPOT"` both read and update the same concept pages. This is the compounding behavior the design exists for; restricting pages to a single objective would defeat the point.

Two consequences for the page schema:

- **Entity pages carry an `objectives_seen` frontmatter list.** Each ingest pass that touches an entity page appends the current run's objective if not already present. This lets readers (human or agent) see which objectives have already considered this region.
- **The `## Observations across runs` section on entity pages records the objective per observation.** Same-region findings under different objectives often surface different bottlenecks (e.g. a region might be TTFT-bound under one workload and TPOT-bound under another); the section preserves both.

The ingest prompt is told to treat the wiki as cumulative across objectives — never delete prior observations because the new run has a different goal.

### 3.8 Cross-subject-system reuse — global techniques, namespaced entities

Technique pages and entity pages have different reuse semantics across subject systems (vLLM, SGLang, etc.):

- **Technique pages are global.** A `paged-attention.md` page is universal knowledge — the technique exists independent of any one inference engine. All runs against any subject system read and update the same technique pages.
- **Entity pages are subject-system-namespaced.** `entities/vllm/v1.kv_offload.md` describes a specific code region in a specific repo. A different subject system gets its own subtree.

This split lives on disk:

```text
wiki/concepts/
  techniques/                # global
    paged-attention.md
    speculative-decoding.md
  entities/
    vllm/                    # subject-system-namespaced
      v1.kv_offload.md
    sglang/
      scheduler.md
```

Two design rules fall out of this:

1. **Technique pages cite entity pages from any subject system.** A technique page's `## Where it has been considered in subject systems` section can list `[[entities/vllm/v1.kv_offload]]` and `[[entities/sglang/scheduler]]` side by side. The cross-link is what makes a technique page "the index" for "given this technique, where could it apply" — the inverse-discovery direction the README mentions.
2. **Technique pages carry a `subject_systems_seen` frontmatter field**, auto-maintained by ingest, listing the subject systems that have contributed to the synthesis. Not used for filtering — purely transparency.

What this gives you in practice:

- Run 1 against vLLM populates `paged-attention.md` and `entities/vllm/v1.kv_offload.md`.
- Run 2 against SGLang queries the wiki, retrieves `paged-attention.md` (warm-started from vLLM's prior synthesis), proceeds with deep research informed by it, and on ingest creates `entities/sglang/scheduler.md` plus updates the technique page to add SGLang to `subject_systems_seen`.
- A user asking "where else has paged-attention been considered?" reads one technique page and follows links into both `entities/vllm/...` and `entities/sglang/...`.

What's deliberately not in v1:

- No per-subject-system filtering of technique pages. `Wiki.query(...)` returns the same technique pages regardless of which subject system you're running against. If techniques diverge sharply across subject systems, the LLM should record that on the page (a `## Subject-system-specific caveats` subsection) rather than fork the page.
- No "promotion" path for moving a subject-system-specific finding into a global technique page. Ingest decides on the spot whether a finding belongs on a technique or entity page.

`subject_system` itself is derived from a config field (default: repo path basename), and the plan recommends pinning it explicitly when running against a renamed/moved repo so entity pages don't bifurcate silently.

### 3.9 What this means for the rest of the plan

The records-vs-concepts relationship described in §2 applies specifically to `records-and-concepts`. The description in §4 of the concepts subpackage and in §5 of the pipeline wiring is mode-aware where needed, and otherwise mode-agnostic. The implementation order in §6 builds the layers in a sequence that supports all three modes — concepts standalone is achievable from the first concepts-side commit, records-only is already today's behavior, and the default mode is the union.

---

## 4. The concepts layer

Everything in this section describes the new `module_knowledge.concepts` subpackage. The records layer is untouched by this design.

### 4.1 Goals and non-goals

**Goals**

- A markdown layer that humans and agents can read directly.
- LLM-maintained: pages are written and updated by an LLM, never by deterministic transforms over a single input.
- Concept-keyed (techniques, subject-system entities), not run-keyed.
- Cross-linked: every claim points back to a raw source; pages link to related pages.
- Verifiable: every page carries provenance metadata so stale pages are detectable.
- Reuses the records layer's lexical retrieval primitive — no new ranking system.

**Non-goals (v1)**

- No vector database, no embeddings. Lexical retrieval is sufficient at the page granularity concepts produces.
- No live web/GitHub crawler beyond what the deep-research stage already retrieves.
- No multi-tenant wiki. One knowledge root (`<artifacts-dir>/.spotlights/knowledge/`) per `--artifacts-dir`. Multi-repo reuse is supported via the technique-vs-entity split (§3.8), not via parallel wikis.
- No cross-objective merging logic beyond what falls out of LLM ingest. The LLM is the merge layer.

### 4.2 Page schema

Every concept page is markdown with a YAML frontmatter block. The frontmatter is the contract; the body is the LLM's domain. The body is identical across operating modes; only the `sources:` block differs.

**Technique page (global, shared across subject systems)** — `records-and-concepts` mode:

```markdown
---
schema_version: concepts.v1
kind: technique
slug: layer-wise-kv-offload                # filename without .md
title: "Layer-wise KV offload"
as_of_run: 2026-06-09T14-12-03Z            # last ingest that touched this page
subject_systems_seen: [vllm, sglang]       # auto-maintained by ingest (techniques only)
sources:                                   # record-id references
  - record_id: "finding:vllm:v1.kv_offload:f-04"
    kind: finding
  - record_id: "paper:arxiv:2403.12345"
    kind: paper
links:                                     # outbound page links
  techniques: [async-h2d-copy, paged-attention]
  entities: ["vllm/v1.kv_offload", "sglang/scheduler"]
content_hash: "sha256:..."
---

## Synthesis
...

## Where it has been considered in subject systems
- [[entities/vllm/v1.kv_offload]] — proposed in run 2026-06-09-a, candidate cand-3 (open)
- [[entities/sglang/scheduler]] — surfaced in run 2026-07-12-b

## Known caveats
- ...
```

**Entity page (subject-system-namespaced)** — `records-and-concepts` mode:

```markdown
---
schema_version: concepts.v1
kind: entity
slug: vllm/v1.kv_offload
title: "vLLM v1.kv_offload — KV cache offload region"
subject_system: vllm                       # entities only
as_of_run: 2026-07-12-b
objectives_seen:                           # auto-maintained by ingest (entities only)
  - "reduce median TTFT"
  - "reduce P99 TPOT under bursty decode"
sources:
  - record_id: "finding:vllm:v1.kv_offload:f-04"
    kind: finding
links:
  techniques: [layer-wise-kv-offload, async-h2d-copy]
content_hash: "sha256:..."
---

## What this region does
...

## Observations across runs
- Run 2026-06-09-a (objective: reduce TTFT, workload: long-context multi-turn): ...
- Run 2026-07-12-b (objective: reduce P99 TPOT, workload: bursty decode): ...

## Techniques considered here
- [[layer-wise-kv-offload]] — proposed (Run 1, cand-3); de-emphasized in Run 2

## Open questions
- ...
```

**`concepts-only` mode** — `sources:` carries light references (`kind` + `id` + optional `title`/`url`):

```yaml
sources:
  - kind: finding
    id: "vllm/v1.kv_offload/run-2026-06-09-a/finding-04"
    title: "Layer-wise offload reduces TTFT 1.4×"
    url: "spotlights-out/modules/v1.kv_offload.md#finding-04"
  - kind: paper
    id: "arxiv:2403.12345"
    title: "LMCache: long-context KV reuse"
    url: "https://arxiv.org/abs/2403.12345"
```

Body conventions (enforced by the ingest prompt, not by code):

- Every claim in `## Synthesis` points to at least one entry in `sources:`.
- Cross-page links use `[[slug]]` syntax. `verify()` resolves them.
- Sections are stable per kind:
  - **Technique pages:** `## Synthesis`, `## Sources`, `## Where it has been considered in subject systems`, `## Known caveats`.
  - **Entity pages:** `## What this region does`, `## Observations across runs`, `## Techniques considered here`, `## Open questions`.

The LLM may add subsections inside these but should not rename the top-level ones.

`verify()` is mode-aware: in `records-and-concepts` it checks that every `record_id` in `sources` exists in `archive/records.jsonl`; in `concepts-only` it checks that each `sources` entry is structurally well-formed and that local `id`/`url` paths exist on disk where applicable.

#### Slug discipline — how concepts are merged

Concept pages are concept-keyed, not run-keyed. When two findings from different runs both describe "speculative decoding," they update the *same* `techniques/speculative-decoding.md` page. The mechanism is naming discipline, enforced by the ingest LLM:

- **Slugs are derived from canonical concept names**, not from finding IDs or run IDs. The ingest prompt instructs the LLM to choose a kebab-case slug that names the underlying concept (`speculative-decoding`, not `vllm-spec-decode-finding-04`).
- **Before creating a new page, the LLM must check for an existing one.** The prompt receives the list of all current technique-page slugs and entity-page slugs. The LLM is told: if any existing slug names the same concept (even under a synonym — "spec decoding" ↔ "speculative decoding"), update that page rather than creating a new one.
- **Aliases are recorded on the page, not as separate slugs.** If a finding refers to "ZeroQuant" but the existing page is `kv-quantization-int8.md`, the LLM updates the existing page and adds "ZeroQuant" to a `## Synthesis` paragraph or an aliases hint, not a new file.
- **Splits and merges are explicit and rare.** If two pages have drifted to describe the same concept and need to merge, that's a lint operation (4.6), not an ingest operation. Ingest prefers idempotent updates over restructuring.

The ingest prompt enforces this; `verify()` does not (slug semantics aren't checkable mechanically). The cost of getting this wrong is duplicated synthesis across slug variants — exactly the kind of decay the lint pass exists to repair.

Entity-page slugs follow a different discipline: they are mechanical, derived from `<subject_system>/<module-qualified-name>`. There is no LLM judgment about entity-page naming — the path is unambiguous.

### 4.3 Public API

```python
from spotlights_engine.module_knowledge.concepts import (
    Wiki, ConceptPage, IngestRequest, IngestReport, QueryResult,
)

class Wiki:
    @classmethod
    def open(cls, root: Path) -> "Wiki":
        """Open or create the concepts wiki at `<root>/wiki/concepts/`. Idempotent."""

    # ---------- query ----------
    def query(self, q: str, *, top_k: int = 5,
              kinds: list[str] | None = None) -> list[QueryResult]:
        """Lexical ranking over concept pages.
        Returns full page bodies + provenance for the LLM to read.
        Reuses module_knowledge.records.retrieve scoring primitives."""

    # ---------- ingest ----------
    async def ingest(self, request: IngestRequest) -> IngestReport:
        """Run an LLM ingest pass (claude-driven) that folds the given sources
        into concept pages. Updates 1..N existing pages and/or creates new ones.
        Atomic per page; updates wiki_state.json after successful commit."""

    # ---------- resume support (see 5.4) ----------
    def read_resume_marker(self, run_id: str) -> int:
        """Return last_ingested_wave_idx for run_id, or -1 if no marker matches."""

    def bump_resume_marker(
        self, run_id: str, wave_idx: int, ingested_qns: list[str]
    ) -> None:
        """Atomically update wiki_state.json after a successful ingest."""

    def filter_uningested(self, qns: list[str], run_id: str) -> list[str]:
        """Drop QNs that already appear in wiki_state.ingested_module_qns_in_current_run."""

    # ---------- lint (deferred; see 4.6) ----------
    async def lint(self) -> LintReport: ...

    # ---------- utilities ----------
    def render_index(self) -> None:
        """Regenerate the cross-cutting wiki/index.md.
        Deterministic walk; no LLM."""

    def verify(self) -> VerifyReport:
        """Check content_hash matches body and every [[link]] resolves.
        In records-and-concepts mode, also checks that every record_id in
        sources exists in archive/records.jsonl."""
```

Data shapes — `IngestRequest` is a discriminated union over operating mode:

```python
class RecordsBackedIngestRequest(BaseModel):
    """Ingest input in records-and-concepts mode."""
    mode: Literal["records-and-concepts"] = "records-and-concepts"
    record_ids: list[str]            # resolved against archive/records.jsonl
    subject_system: str
    objective: str | None
    run_id: str

class InlineIngestRequest(BaseModel):
    """Ingest input in concepts-only mode."""
    mode: Literal["concepts-only"] = "concepts-only"
    sources: list[InlineSource]      # carried inline; no JSONL involvement
    subject_system: str
    objective: str | None
    run_id: str

IngestRequest = RecordsBackedIngestRequest | InlineIngestRequest

class InlineSource(BaseModel):
    kind: Literal["finding", "paper", "issue", "pr", "blog", "docs", "note"]
    id: str                          # stable string the LLM can cite
    title: str
    body: str                        # the actual evidence — only sent inline once,
                                     # not stored on every concept page
    url: str | None = None
    module_qn: str | None = None     # if anchored to a subject-system module

class IngestReport(BaseModel):
    pages_updated: list[str]         # slugs
    pages_created: list[str]
    llm_calls: int
    cost_usd: float
    duration_s: float

class QueryResult(BaseModel):
    page: ConceptPage                # full page including frontmatter
    score: float
    matched_terms: list[str]
```

`RecordsBackedIngestRequest` takes `record_ids` rather than embedding source bodies — the ingest pass loads the record contents from the records archive. This keeps the request small and lets ingest see the full records (including provenance) rather than a flattened summary.

`InlineIngestRequest` carries source bodies inline because there is no JSONL to resolve against. The LLM still emits page `sources:` blocks as light references (just `kind` + `id` + `title` + `url`); the inline `body` is consumed during ingest and not stored on the resulting concept pages.

### 4.4 Invariants and concurrency

- **Source of truth for synthesis:** the markdown files in `wiki/concepts/`. Frontmatter `content_hash` lets `verify()` detect drift.
- **Source of truth for evidence:** `archive/records.jsonl`. Concept pages cite records by `record_id`; the records layer is unchanged.
- **Ingest is serial.** Only one `ingest()` runs at a time. The orchestrator provides the synchronization point (between waves; see §5). Inside `ingest()`, page writes use a per-file lock — write to `*.tmp`, fsync, rename — so a crash mid-write leaves the previous version intact.
- **Query is concurrent-safe.** Pure-read; no locking needed.
- **Page deletes do not happen during ingest.** Pages can become stale, but ingest never removes them. Lint is the only writer that deletes (and only with explicit configuration).

### 4.5 What the records layer already gives us

The new code reuses, not duplicates:

- **Lexical retrieval primitive.** `concepts/query.py` calls into `records.retrieve._score_record` with `ConceptPage`-shaped inputs (title + frontmatter tags + body). No new ranker.
- **Hash + atomic write helpers.** `records/store.py` already has `sha256_file` and the JSONL write pattern. Concepts uses the same `*.tmp` + rename idiom.
- **Verification skeleton.** `records.wiki.verify_wiki` already checks `<!-- generated_from -->` and local link resolution. `concepts/wiki.py:verify` follows the same pattern, with extra checks for sources resolution and orphan pages.
- **Adapters.** Records adapters keep populating the JSONL during a run. The ingest request just passes the `record_id`s for the records produced this wave; nothing about adapters changes.

### 4.6 Lint — when and why

Lint is a periodic LLM-driven pass that does **not** run as part of normal pipeline execution. Implement later, when the wiki is large enough that decay becomes visible. Concrete triggers for prioritizing it:

- **Trigger A: contradictions across pages become measurable.** Two technique pages disagree on whether a technique helps a workload archetype, or two entity pages cite different bottleneck root causes for the same module. When users start hitting these manually, lint earns its keep.
- **Trigger B: the wiki has > ~50 pages.** Below that threshold, the LLM context comfortably holds enough of the wiki that contradictions get caught at ingest time. Above it, the ingest prompt only sees retrieved pages — contradictions on unretrieved pages will accumulate.
- **Trigger C: orphan pages appear.** Pages with no inbound `[[links]]` from any other page. Often a sign that an entity page was created speculatively and never wired in, or that an old technique page lost all its references after a re-synthesis.

Lint scope when it ships:

- **Contradiction check.** Sample N pairs of pages that share at least one tag, ask the LLM to identify direct contradictions, write findings to `wiki/log/lint-<ts>.md` for human review. Does *not* auto-edit pages — humans decide.
- **Stale-claim detection.** Find claims in entity pages that reference a candidate or proposal the most recent run no longer surfaces. Mark with `<!-- stale: as-of <run> -->` rather than removing.
- **Orphan + dead-link sweep.** Deterministic, no LLM. Lists orphan pages and unresolved `[[links]]`. Cheap to run; can ship before the LLM lint passes.

Out of scope for lint v1: auto-merging pages, auto-deleting orphans, rewriting page bodies. All destructive operations remain manual.

The deterministic orphan + dead-link sweep is the cheapest piece and could ship alongside `verify()`. Treat it as a pre-lint stepping stone.

---

## 5. Ingest and Query for `module_deep_research`

This section describes how the concepts layer is woven into the deep-research stage at [src/spotlights_engine/module_deep_research/](../../src/spotlights_engine/module_deep_research/) and how the orchestrator coordinates ingests between modules. The records layer continues to be populated by adapters as it is today.

### 5.1 Where it plugs in

The deep-research stage is the most expensive LLM stage in the pipeline (one or more web/literature sweeps per module). It is also the stage whose findings are the most reusable across modules within a run and across runs on the same subject system. Two integration points:

- **Query (read-side):** at the start of `research_module(...)`, before the agent prompt is built, the concepts wiki is queried with terms derived from the module qualified name, the objective, and the workload hints. Retrieved concept pages are inlined into the agent prompt as a "Prior synthesis" section.
- **Ingest (write-side):** after `research_module(...)` returns its findings and the records adapter writes them to `archive/records.jsonl` (or, in `concepts-only` mode, the findings are kept in memory), the orchestrator schedules an ingest pass that folds those sources into the concepts subtree. Ingest does not run inside the per-module pipeline — it runs at wave boundaries (see 5.5) so the wiki state is well-defined when the next wave starts.

The deep-research module itself does not call ingest. It exposes its findings; the records adapter persists them; the orchestrator triggers the ingest. This keeps the ingest cost out of every module's critical path and makes wiki integration something the orchestrator owns end-to-end.

### 5.2 Query workflow

Query is mode-aware. There is one query call per module, but the store it hits depends on the operating mode:

| Mode | Queried store | What's returned |
|---|---|---|
| `records-and-concepts` | concepts | Concept pages (synthesis) |
| `concepts-only` | concepts | Concept pages (synthesis) |
| `records-only` | records | Records (raw evidence) |
| `off` | — | No query call is made |

Concepts and records are **never both queried in the same run**. Mixing them in a single prompt would force the agent to reconcile two different content shapes (synthesized priors vs. raw evidence) on every call; the design avoids that complexity. In `records-and-concepts`, records remains available for audit and verify(), but warm-starting deep research is exclusively the concepts layer's job.

```python
# Inside research_module(...) before the agent invocation:

if mode == "off":
    prior = []
elif mode == "records-only":
    prior = records_kb.retrieve(
        query=f"{module.qualified_name} {objective} {' '.join(workload_hints)}",
        top_k=8,
    )
else:  # concepts-only or records-and-concepts
    prior = wiki.query(
        q=f"{module.qualified_name} {objective} {' '.join(workload_hints)}",
        kinds=["technique", "entity"],
        top_k=8,
    )

prompt = build_prompt(
    module=module,
    objective=objective,
    workload_hints=workload_hints,
    prior=prior,             # shape varies by mode
    prior_kind=mode,         # so the prompt template knows what it got
)
```

Prompt contract for the deep-research agent (additions):

- A new prior-knowledge block appears before the existing instruction set. Its header is **"Prior synthesis from the wiki"** when querying concepts, **"Prior raw evidence from the records archive"** when querying records.
- For each retrieved concept page, the agent is shown the page body verbatim (frontmatter stripped except for `slug`, `as_of_run`, and where applicable `subject_systems_seen` / `objectives_seen`).
- For each retrieved record, the agent is shown title, source URL, and body — same shape as today's `wiki/records/<id>.md` projection.
- The agent is instructed to **treat prior knowledge as a starting point, not as ground truth**: cite it when reusing, but verify any claim it acts on, and explicitly note when the wiki is wrong or incomplete.

When the queried store is empty (cold start in any mode), `prior` is `[]` and the prompt has an empty prior block. The agent prompt should degrade gracefully — same instructions, just less context.

### 5.3 Ingest workflow

The ingest pass takes a batch of input (one wave's worth, see 5.5) and produces a structured set of page updates. The input shape depends on the operating mode:

- **`records-and-concepts`:** a `RecordsBackedIngestRequest` carrying `record_ids` collected from each successful module's checkpoint state (see 5.4). The ingest pass loads record bodies from `archive/records.jsonl`.
- **`concepts-only`:** an `InlineIngestRequest` carrying `list[InlineSource]` directly. There is no JSONL; the orchestrator passes findings inline from each module's deep-research output.

Both shapes are defined in the public API at §4.3. `IngestReport` is identical across modes:

```python
class IngestReport(BaseModel):
    pages_updated: list[str]         # slugs
    pages_created: list[str]
    llm_calls: int
    cost_usd: float
    duration_s: float
```

The ingest prompt is the heart of the design and deserves careful iteration. Its responsibilities:

1. **Load the new sources.** In `records-and-concepts`, look up record bodies in `archive/records.jsonl` by id. In `concepts-only`, read them from the inline `sources` list in the request.
2. **Read the current concept-wiki state** for pages that might be relevant — retrieved by the same lexical query the agent uses, plus any pages whose entity matches a source's `module_qn`.
3. **Decide which pages to touch.** A typical source updates 2–5 pages: one technique page (deepening synthesis or adding a citation), one entity page (recording the observation against this module), and possibly a related-technique page for cross-links.
4. **Produce a structured edit plan**, not free-text. The LLM emits a JSON object listing (page slug, action: create|update, body) tuples. The orchestrator applies the edits with file locks and atomic renames.
5. **Update frontmatter automatically** — `as_of_run`, `content_hash`, `links`, `sources` are computed, not LLM-written. The `sources:` shape follows the operating mode (light references in `concepts-only`, `record_id` references in `records-and-concepts`).

A key rule: the ingest LLM never sees more than the retrieved concept pages plus the new sources. It does not see the whole wiki. This is what makes ingest scale — the work per wave is bounded by retrieval depth, not by total wiki size. The cost of *not* seeing everything is what lint exists to catch (4.6).

### 5.4 Tracking archived record IDs in module checkpoint state

Wave-based ingest needs an answer to one specific question: "for the modules that just finished this wave, which `record_id`s did they archive?" The answer must be:

- **Scoped to this run.** A module re-archived in a later run shouldn't drag its prior records into the current ingest.
- **Scoped to this module.** Ingest should only see records produced by modules in the wave, not records from siblings still running in earlier waves.
- **Available without scanning the JSONL.** Walking `archive/records.jsonl` to filter by `provenance.locator` is fragile (locator format changes are quiet breakages) and does not scope to the current run.

The cleanest place to store this is the **per-module checkpoint state** that the orchestrator already maintains. Each module already owns its own subtree under `<artifacts>/spotlights_manager/modules/<slug>/` with a `status.json` checkpoint and step-output sidecars. Adding a small sidecar — say `archived_records.json` — that lists the `record_id`s archived by that module's records-side adapters this run is the right shape.

Concretely:

1. **A new sidecar path** on `ModulePaths` ([src/spotlights_engine/spotlights_manager/persistence.py:139](../../src/spotlights_engine/spotlights_manager/persistence.py#L139)):

   ```python
   @property
   def archived_records_path(self) -> Path:
       return self.dir / "archived_records.json"
   ```

2. **A new field** on `LoadedModuleState` populated by `read_module_state(...)` from that file when present:

   ```python
   @dataclass
   class LoadedModuleState:
       ...
       archived_record_ids: list[str] = field(default_factory=list)
   ```

3. **Records-side adapters return `record_id`s on archive.** `KnowledgeBase.archive_many(...)` already returns an `ArchiveResult`; extend it (or add a sibling method) so the caller gets back the ids it just wrote. The orchestrator appends those ids to the module's `archived_records.json` after each step that archives.

4. **Reset semantics on retry.** If a module is rerun (e.g. on resume after a failure), `archived_records.json` is overwritten, not appended. This keeps the file scoped to the current run and avoids double-counting records from a half-finished prior attempt.

5. **`status: COMPLETED` is the gate.** Only modules whose final checkpoint reports success contribute their `archived_record_ids` to the wave's ingest. A module that failed mid-pipeline may have written some records to the JSONL, but those records are not seen by ingest — they will be re-produced and ingested on the next run.

#### Wave-level resume marker

Module-level checkpointing is necessary but not sufficient. Ingest runs *between* modules at wave boundaries, so a crash mid-run can leave the system in an awkward state: wave 3's modules all completed, but the ingest pass for wave 3 either crashed mid-write or never started.

The wiki keeps a small sidecar at `<knowledge-root>/wiki_state.json` to make this recoverable:

```json
{
  "schema_version": "wiki_state.v1",
  "run_id": "2026-06-09-a",
  "last_ingested_wave_idx": 2,
  "ingested_module_qns_in_current_run": [
    "v1.kv_offload", "v1.scheduler", ...
  ],
  "updated_at": "2026-06-09T14:12:03Z"
}
```

Resume semantics:

- **On run start**, the orchestrator reads `wiki_state.json`. If `run_id` matches the resuming run, it skips waves whose index is `<= last_ingested_wave_idx`. Modules in those waves are not re-ingested even though their checkpoint state may still claim they're "to be ingested."
- **On run start with a different `run_id`**, `wiki_state.json` is treated as stale and is overwritten with a fresh marker. Records from prior runs were already folded into concept pages on those runs; they don't get re-folded.
- **Atomicity per wave.** `Wiki.ingest()` updates `wiki_state.json` *after* all page edits for the wave commit successfully (atomic rename, same as the page writes). A crash mid-ingest leaves `last_ingested_wave_idx` at the previous wave's value, and on resume that wave's ingest re-runs against the same set of modules. Re-running ingest on the same record IDs is acceptable because ingest is mostly idempotent — the LLM may produce slightly different page bodies on retry, but `content_hash` will reflect the final state.
- **`ingested_module_qns_in_current_run`** is the dedupe key for "did this module already contribute to ingest in this run?" The orchestrator filters successful modules through this set before building the next wave's `IngestRequest`. This protects against a half-completed wave whose `last_ingested_wave_idx` didn't bump but whose modules' `archived_records.json` exists from before the crash.

`wiki_state.json` is not the source of truth — `archive/records.jsonl` and the concept pages are. It's a coordination file that survives a crash. Deleting it forces ingest to redo all waves on the next run, which is wasteful but not incorrect.

### 5.5 Orchestrator integration — wave-based ingest

This is the wiring change in [orchestrator.py](../../src/spotlights_engine/spotlights_manager/orchestrator.py). The current code creates one task per module and `gather`s them all under a single semaphore. The wave-based version chunks `ordered_qns` and dispatches an ingest request between chunks; the request shape depends on the operating mode (see §3):

```python
# At src/spotlights_engine/spotlights_manager/orchestrator.py:1340 (sketch)

sem = asyncio.Semaphore(config.max_parallel_sessions)
manifest_lock = asyncio.Lock()
cancel_event = (
    None if input.continue_on_module_failure else asyncio.Event()
)

mode = config.knowledge.mode
wiki = (
    Wiki.open(paths.knowledge_root)
    if mode in ("concepts-only", "records-and-concepts")
    else None
)
wave_size = config.max_parallel_sessions

# Resume support: skip waves already ingested in this run.
resume_after = wiki.read_resume_marker(run_id) if wiki else -1   # -1 = no resume

results: list[ModuleCheckpoint | BaseException] = []
for wave_idx, wave in enumerate(_chunks(ordered_qns, wave_size)):
    if cancel_event is not None and cancel_event.is_set():
        break

    tasks = [asyncio.create_task(_wrapped(qn)) for qn in wave]
    wave_results = await asyncio.gather(*tasks, return_exceptions=True)
    results.extend(wave_results)

    if wiki is None or wave_idx <= resume_after:
        continue   # ingest already happened in a prior crashed attempt

    successful = [
        qn for qn, r in zip(wave, wave_results)
        if not isinstance(r, BaseException) and r.status != "FAILED"
    ]
    successful = wiki.filter_uningested(successful, run_id)   # dedupe vs prior crash
    if not successful:
        continue

    if mode == "records-and-concepts":
        # Read record IDs from each successful module's checkpoint state.
        record_ids: list[str] = []
        for qn in successful:
            state = P.read_module_state(paths.for_module(qn))
            record_ids.extend(state.archived_record_ids)
        if record_ids:
            await wiki.ingest(RecordsBackedIngestRequest(
                record_ids=record_ids,
                subject_system=config.subject_system_name,
                objective=input.objective,
                run_id=run_id,
            ))
    elif mode == "concepts-only":
        # Read findings inline from each successful module's deep-research output.
        sources: list[InlineSource] = []
        for qn in successful:
            state = P.read_module_state(paths.for_module(qn))
            sources.extend(_findings_to_inline_sources(state.deep_research, qn))
        if sources:
            await wiki.ingest(InlineIngestRequest(
                sources=sources,
                subject_system=config.subject_system_name,
                objective=input.objective,
                run_id=run_id,
            ))

    # Bump the resume marker only after a successful ingest commit.
    wiki.bump_resume_marker(run_id, wave_idx, successful)
```

Properties of this design:

- **`max_parallel_sessions = 1` ⇒ one module per wave ⇒ ingest after every module.** The Karpathy-pure pattern.
- **`max_parallel_sessions = N` ⇒ waves of N ⇒ ingest fires ⌈M/N⌉ times.** Modules in the same wave do not see each other; modules in wave k+1 see everything waves 1..k produced.
- **Failed modules are excluded from ingest** so a crashed run does not poison the wiki. Their data is picked up on the next run after a successful retry.
- **Cancellation is honored at wave boundaries** — a fail-fast run stops cleanly between waves.
- **In `records-and-concepts`, records are written by adapters first**, ingest runs second. The records layer is always the upstream of truth; concepts is downstream synthesis.
- **In `concepts-only`, findings are read directly from the per-module deep-research sidecar** (`state.deep_research`) and turned into `InlineSource` objects on the fly. No JSONL is involved.
- **No JSONL re-scan, no high-water mark, in either mode.** Each module's checkpoint state is the single source of truth for "what did this module just produce" — same pattern as the existing `state.candidates`, `state.deep_research`, etc. sidecars.
- **Resume-safe.** The `wiki_state.json` marker (see 5.4) lets the orchestrator skip waves whose ingest committed before a crash. Re-running the same waves is acceptable but wasteful; the marker avoids it.

### 5.6 Disable / opt-out

Disabling is now expressed through the operating mode (see §3), not a separate flag:

- `--knowledge-mode off` — no `Wiki.open(...)`, no records-side adapters, no ingest, no concepts query. Pipeline behaves as if the knowledge subsystem did not exist. The orchestrator's outer loop becomes byte-equivalent to today's single `gather`.
- `--knowledge-mode records-only` — keeps the structured audit log, drops the LLM synthesis layer. The agent prompt has no `prior_synthesis` block. No ingest passes run between waves.
- `--knowledge-mode concepts-only` — runs concepts standalone with light references (see §3.2). No JSONL is written. The records-side adapters and `WikiRenderer` are bypassed.

The mode field is the single user-facing knob. There is no separate `--no-wiki`, `--no-records`, or `wiki_enabled: bool` toggle — those would conflict with the mode field's semantics.

### 5.7 Cost model

For a run over M modules with `max_parallel_sessions = N`, the concepts layer adds:

- **Per module:** one wiki query (cheap, lexical, no LLM) + a longer agent prompt (more input tokens). Input-token cost only.
- **Per wave:** one ingest pass (1–2 LLM calls; updates ~5–10 pages). Output-token cost.
- **Total concepts cost:** O(M) input-token overhead on the agent + O(M/N) ingest passes.

**Ingest is driven by `claude`** in v1 — same CLI the rest of the engine uses for orchestration-heavy LLM stages. Routing through a different model (e.g. `codex`, a cheaper Haiku tier, or a LiteLLM-proxied provider) is a follow-up; v1 keeps the model choice unified to avoid an extra config knob during early prompt iteration.

The expected payoff is a reduction in deep-research output tokens per module, because the agent reuses prior synthesis instead of re-deriving it. The break-even point is empirical and should be measured on the vLLM canonical run.

### 5.8 Open questions for v1

- **Retrieval relevance for cold-start runs.** With < 10 concept pages, lexical retrieval is noisy. Consider a "skip query when concepts has < K pages" guard for the first few modules of a run.
- **Findings vs. proposals as ingest input.** Records already cover findings, candidates, and proposals. v1 ingest folds findings only — they're the upstream signal. Candidates and proposals can become entity-page sections in a follow-up; the records will already be in the JSONL when that work starts.
- **Run identity in `as_of_run`.** The orchestrator already has a stable run id for artifacts. Reuse it; do not invent a new one for the wiki.
- **Ingest LLM output validation.** The LLM emits a structured edit plan (page slug, action, body), but the body is markdown that must conform to the page schema (frontmatter contract, stable section names, valid `[[link]]` syntax). v1 retries on schema-validation failure; if retries exhaust, the wave's ingest is skipped and a warning is logged. Worth measuring how often this fires before deciding whether to invest in stricter structured-output guards.
- **`verify()` cost at scale.** With > 100 concept pages and a large records archive, full verification on every run becomes non-trivial. v1 runs `verify()` only on explicit CLI invocation, not as part of pipeline execution. Worth revisiting once the wiki has accumulated enough state to be worth defending against drift.

---

## 6. Implementation order

Suggested order of work, smallest viable steps first.

**Phase 1 — Restructure (no behavior change)**

1. Move `module_knowledge/*.py` → `module_knowledge/records/*.py`. Top-level `__init__.py` re-exports preserve all import paths. Tests pass on existing fixtures.
2. (Optional) Rename `KnowledgeLayout` → `RecordsLayout`; introduce `KnowledgeRoot` if it improves readability.

**Phase 2 — Concepts subpackage (additive, no pipeline change)**

3. `concepts/schemas.py` — `ConceptPage`, `IngestRequest`, `IngestReport`, `LintReport`. Pydantic, `extra="forbid"`, mirroring records-side conventions.
4. `concepts/layout.py` + `concepts/wiki.py` — directory shape, atomic page writes, `verify()`.
5. `concepts/query.py` — wraps `records.retrieve` scoring with `ConceptPage` inputs. Unit tests with hand-written pages.
6. `concepts/ingest.py` + `concepts/prompts.py` — the LLM ingest pass. The single biggest implementation effort. Iterate on prompts against canned record sets before wiring to the orchestrator.

**Phase 3 — Pipeline wiring**

7. Module checkpoint extension (section 5.4) — `archived_records.json` sidecar, `LoadedModuleState.archived_record_ids`, adapter return-value plumbing. Only required for `records-and-concepts` mode.
8. Orchestrator wiring (section 5.5) — wave-based ingest, mode dispatch, run-id threading.
9. `module_deep_research` query integration (section 5.2) — prompt change, prior-synthesis block, agent instructions.
10. Cross-cutting `wiki/index.md` generator that lists records + concepts together.

**Phase 4 — Lint (deferred)**

11. Deterministic orphan + dead-link sweep — first lint piece, ships in `concepts/wiki.py` next to `verify()`.
12. LLM lint passes — only after the concepts wiki passes the triggers in section 4.6.

Phases 1–3 are the v1 critical path. Phase 4 is deferred and depends on real-world wiki growth.
