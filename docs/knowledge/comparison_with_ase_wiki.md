# Spotlights `knowledge_design_plan` vs. agentic-strategy-evolution wiki

A side-by-side comparison of two LLM-curated knowledge systems built by adjacent projects:

- **Spotlights** — design plan in [knowledge_design_plan.md](knowledge_design_plan.md). Wiki-in-the-loop layered over the existing `module_knowledge` records archive.
- **ASE (agentic-strategy-evolution)** — implemented in [`~/.nous/wiki/`](https://github.com/...) via four `/post-campaign`, `/index-wiki`, `/suggest-next`, `/visualize-*` slash-command skills.

## TL;DR

| | Spotlights | ASE |
|---|---|---|
| What it is | Wiki-in-the-loop synthesizer | Post-campaign extractor + cross-campaign registry |
| When it runs | Inside the live pipeline (wave boundaries) | After a campaign completes (manual trigger) |
| Storage | Markdown pages with YAML frontmatter | JSON files, one per concept-category |
| Synthesis style | LLM merges N findings into M concept pages | LLM extracts campaign output into typed JSON |
| Cross-run shape | Concept pages grow across runs | Per-campaign JSONs immutable; `registry.json` routes between them |
| Primary consumer | The next module's deep-research agent | The next campaign's planning agent (`/suggest-next`) |

## Side-by-side

| Dimension | Spotlights (planned) | ASE (built) |
|---|---|---|
| **When the wiki updates** | Inside the live pipeline at every wave boundary | After a run completes, manually triggered by `/post-campaign` |
| **Who triggers it** | The orchestrator | A human running a slash command |
| **Storage shape** | Markdown pages with YAML frontmatter | JSON files, one per concept-category |
| **Substrate** | Markdown is the truth (concepts) + JSONL is the truth (records) | JSON files are the truth; HTML is derived |
| **Synthesis layer** | LLM merges N findings into M concept pages | LLM extracts campaign output into typed JSON; no merging across campaigns |
| **Cross-run accumulation** | Concept pages grow across runs (technique pages especially) | Per-campaign JSONs are immutable; `registry.json` is the cross-run index |
| **Cross-subject reuse** | Global technique pages + namespaced entity pages | One global `registry.json` with per-project entity dedup |
| **Page identity / merge logic** | Slug discipline; LLM checks existing slugs before creating new ones | Entity dedup by normalized name (lowercase, strip parens) |
| **Cross-references** | `[[slug]]` page links + `record_id` / light-source references | Globally unique IDs (E-N, C-N, P-N, DE-N, F-N, I-N) inside `registry.json` |
| **Scope of LLM output** | Free-form synthesis prose under stable section headers | Structured JSON per category, each entry self-contained |
| **Read path** | `Wiki.query(...)` returns concept pages to feed back into deep research | `retrieve_wiki_context.py` returns markdown context for a `/suggest-next` LLM call |
| **Concurrency** | Wave-based ingest with serial commit | Explicitly not safe to run in parallel |
| **Visualization** | `wiki/index.md` (markdown landing page) | Interactive HTML with tabs (graph, timeline, insights, summary) |
| **Cost / pacing** | Per wave (M/N ingest passes per run) | Once per campaign |

## Philosophical differences

### Synthesis vs. extraction

- **Spotlights** is a *living synthesis*. `paged-attention.md` is rewritten on every ingest, accumulating insight from many findings. The merge work is the point.
- **ASE** is *frozen extraction*. Each campaign deposits its own immutable JSONs; the registry is a routing layer over them, not a merged synthesis. Re-extraction only happens on a forced `/post-campaign` rerun.

### In-loop vs. post-hoc

- **Spotlights:** the wiki participates in the next module's deep research *during the same run*. Ingest fires inside the pipeline. The compounding effect is measured per-wave.
- **ASE:** the wiki helps the *next campaign*, not the current one. `/suggest-next` informs what to run next; the current campaign already finished.

### Markdown vs. JSON

- **Spotlights** bets on markdown because the consumer is an LLM and markdown is the natural format for synthesized prose.
- **ASE** bets on JSON because the consumers are scripts (`retrieve_wiki_context.py`, `visualize_*.py`) that need typed fields. The LLM is one of several consumers, not the primary one.

### Naming-discipline merge vs. ID-dedup merge

- **Spotlights** asks the LLM to choose canonical slugs (`speculative-decoding`) and check existing pages before creating new ones. Merge is judgment.
- **ASE** deterministically deduplicates entities by normalized name and assigns globally unique IDs. Merge is mechanical.

## Where the designs agree

- **Cross-objective / cross-campaign reuse is the whole point.** Both systems exist to make N+1 cheaper and smarter than 1.
- **Verifiability matters.** ASE's "every entry is self-contained" rule is the same instinct as Spotlights' `sources:` references and `verify()`.
- **Knowledge graphs over flat lists.** Both build entity → concept → … relationships, just in different shapes (markdown links vs. JSON IDs).
- **Visualization is a real artifact.** ASE has `viz/<name>.html`; Spotlights has `wiki/index.md`. Both treat the wiki as something humans browse, not just a cache.
- **Idempotent re-runs.** Both have explicit "if already indexed, skip" semantics.

## What Spotlights could borrow from ASE

1. **Frontiers / dead-ends / interactions as structured page types.** ASE separates "what failed", "what's at the edge of what we know", and "untested combinations" into their own JSON shapes. Spotlights' entity pages have an `## Open questions` section, but a richer typology — frontier or dead-end pages — would surface adjacent work more usefully. Candidate v2 page kinds alongside techniques and entities.

2. **A `/suggest-next` equivalent.** ASE turns the wiki into a forward-looking recommender: "given prior knowledge, here are 3 scored campaigns to run next." Spotlights has no "given the wiki, what's the next high-value run?" surface. With concept pages already populated, this could be Spotlights' technique-driven discovery direction.

3. **HTML visualization as a first-class output.** Markdown is great for LLM consumers; humans benefit from a graph view. ASE's per-campaign and registry HTML pages (tabs for graph / timeline / insights / summary) are user-facing in a way `wiki/index.md` isn't.

4. **Per-wave / per-run cost capture in `llm_metrics.jsonl`.** ASE captures costs on the same axes the synthesis is keyed on, so `/suggest-next` can predict cost for a similar future run. Spotlights' cost model is currently theoretical; an `llm_metrics.jsonl` sidecar would let cost be measured the way ASE does.

5. **Self-contained entries.** ASE's rule that "another agent reading this entry should understand it without looking at any other file" is a useful constraint to fold into the Spotlights ingest prompt, especially for entity pages.

## What ASE might borrow from Spotlights

- **In-loop ingest.** ASE waits until a campaign ends; for long campaigns this leaves knowledge stranded mid-flight. Wave-based ingest (or per-iteration ingest) would let later iterations benefit from earlier ones in the same campaign.
- **Cross-campaign synthesis pages.** ASE's registry routes between immutable per-campaign JSONs; a concept-keyed synthesis layer on top — "here is what we now believe about saturation detection across all 12 campaigns" — would let `/suggest-next` query an actual synthesis instead of stitching one together on the fly.
- **Two layers (raw vs. synthesized).** ASE has only one. Spotlights' records-vs-concepts split could map onto ASE as "per-campaign JSONs (raw) vs. cross-campaign synthesized markdown (concepts)."

## Honest read

These are *complementary* designs solving adjacent problems.

- **ASE** fits work structured around discrete campaigns with explicit principle/iteration semantics. Closer to a campaign archive + recommender than to a Karpathy wiki.
- **Spotlights** fits a continuous research loop where deep research recurs and benefits from warm-starts.

The biggest thing the Spotlights plan does that ASE doesn't: **the wiki is in the loop**. That's what makes it a Karpathy-style wiki, and what makes the compounding effect possible inside a single run, not just between runs. ASE's framing is "knowledge between campaigns"; Spotlights' framing is "knowledge during and between runs." Both framings are valid; the design choices follow from which one you commit to.
