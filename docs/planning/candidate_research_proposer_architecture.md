# candidate_research_proposer — Architecture

---

## Goal

Build a Python pipeline that, given `(repo_url_or_path, module_qualified_name)`, produces `changes.json` — a list of optimization proposals tied to concrete code locations in the target module. Each proposal has one of two provenance types: **research-grounded** (backed by exactly one external finding — paper / blog / PR / issue / talk — discovered in Stage 2) or **agent-novel** (proposed by Codex and Claude Code from their own training, *not* mentioned in any Stage-2 finding, on top of a candidate location). Every proposal also carries explicit reasoning emitted by the agent. The pipeline runs Stage 1 (candidate discovery via Claude↔Codex alternating review) and Stage 2 (single GPT Researcher call) in **parallel**, then Stage 3 runs in three sequential steps: 3a (finding↔candidate mapping), 3b (per-finding research-grounded proposals), and 3c (per-candidate agent-novel proposals that don't overlap Stage 3b's proposals for the same candidate).

---


## Diagram

The driver takes `(repo, module_qualified_name)`, resolves the `Module` record against [modules.json](modules.json), and fans out into two parallel discovery stages whose outputs converge on a per-finding mapping stage. Stage 3b then generates research-grounded proposals per mapped finding; Stage 3c follows sequentially, generating agent-novel proposals per candidate while reading Stage 3b's already-emitted proposals for that candidate as the "do-not-duplicate" set. A final pivot collects both into the per-candidate `changes.json`. Every artifact is keyed by `module_qualified_name`; `candidate_id` and `finding_id` are the join columns threaded through stages 3a–3c.

```
                ┌──────────────────────────────────────────┐
                │  Driver                                  │
                │  (repo, module_qualified_name) →         │
                │   Module record via modules.json         │
                └────────────────────┬─────────────────────┘
                                     │ fan-out
              ┌──────────────────────┴──────────────────────┐
              ▼                                             ▼
   ┌──────────────────────────┐                ┌──────────────────────────┐
   │  Stage 1                 │                │  Stage 2                 │
   │  Candidate discovery     │                │  Literature / blog       │
   │  (Claude ↔ Codex loop)   │                │  research                │
   │                          │                │  (GPT Researcher)        │
   │  → candidates.json       │                │  → findings.json         │
   └─────────────┬────────────┘                └─────────────┬────────────┘
                 │                                           │
                 └─────────────────────┬─────────────────────┘
                                       ▼
                       ┌──────────────────────────────┐
                       │  Stage 3a                    │
                       │  Map findings → candidates   │
                       │  (per-finding agent fan-out) │
                       │  → mapping.json              │
                       └───────────────┬──────────────┘
                                       ▼
                       ┌──────────────────────────────┐
                       │  Stage 3b                    │
                       │  Research-grounded proposals │
                       │  (per mapped finding)        │
                       │  → from_findings[]           │
                       └───────────────┬──────────────┘
                                       ▼
                       ┌──────────────────────────────┐
                       │  Stage 3c                    │
                       │  Agent-novel proposals       │
                       │  (per candidate; must not    │
                       │   overlap Stage 3b proposals │
                       │   for the same candidate)    │
                       │  → from_agents[]             │
                       └───────────────┬──────────────┘
                                       ▼
                       ┌──────────────────────────────┐
                       │  Pivot by candidate          │
                       │  → changes.json              │
                       └──────────────────────────────┘
```

---

## Output JSONs

All four canonical artifacts live directly under `~/.cache/optquest/<repo_slug>/<run_id>/`. Stage-specific scratch logs and per-agent transcripts may live in subdirectories such as `candidate_discovery/`. All schemas are checked with `pydantic >= 2.7` models published in `spotlights_engine.schemas`; the driver exports each model's `model_json_schema()` to disk for external validators.

### `candidates.json` — Stage 1 output

The audited module is identified by its `module_qualified_name` in the sibling `modules.json` artifact (a `ProjectTree`; see [src/spotlights_engine/schemas/modules.py](../src/spotlights_engine/schemas/modules.py)). Consumers re-resolve the full `Module` record via `ProjectTree.from_json(modules_json).resolve(module_qualified_name)`; we intentionally do **not** duplicate the record into `candidates.json` to keep `modules.json` the single source of truth. Every candidate's `file` MUST lie within the resolved `module.path`.

```json
{
  "module_qualified_name": "foo/bar",
  "candidates": [
    {
      "id": "cand-0001",
      "file": "src/foo/bar/scheduler.py",
      "line_start": 142,
      "line_end": 211,
      "rationale": "Called once per decode step; allocates a new list of pending requests every call; iterates O(N) over the request table to filter."
    }
  ]
}
```

Notes:
- `module_qualified_name` is the `/`-joined preorder path from `ProjectTree.walk()` (e.g. `"v1/engine/core"`), **not** the filesystem path — the latter is `Module.path` and must be obtained by resolving against `modules.json`.


### `findings.json` — Stage 2 output

Every finding has exactly five fields: `id`, `title`, `url`, `source_type`, `technique_summary`. See the Stage 2 plan for the coercion prompt that produces them.

```json
{
  "schema_version": "1.0",
  "module_qualified_name": "foo/bar",
  "findings": [
    {
      "id": "find-0007",
      "title": "PagedAttention: Memory Management for LLM Serving",
      "url": "https://arxiv.org/abs/2309.06180",
      "source_type": "paper",
      "technique_summary": "Splits KV cache into fixed-size blocks managed by an OS-style page table; eliminates contiguous-allocation fragmentation in batched decoding."
    }
  ]
}
```


### `mapping.json` — Stage 3a output (consumed by Stage 3b)

```json
{
  "module_qualified_name": "foo/bar",
  "mappings": [
    {
      "candidate_id": "cand-0001",
      "findings": [
        {
          "finding_id": "find-0007",
          "confidence": "high",
          "reasoning": "scheduler.py touches block allocation for the KV cache; the finding's paged-attention technique applies directly to the loop at line 142"
        },
        {"finding_id": "find-0011", "confidence": "medium", "reasoning": "...", "agent": "codex"}
      ]
    }
  ]
}
```


### `changes.json` — final artifact

Candidate-centric. Per candidate, two parallel lists of changes:
- `from_findings` — research-grounded (Stage 3b): each entry links back to the finding it was derived from.
- `from_agents` — agent-novel (Stage 3c): each entry records which coding agent proposed it.

```json
{
  "module_qualified_name": "foo/bar",
  "changes_per_candidate": [
    {
      "candidate_id": "cand-0001",
      "from_findings": [
        {
          "finding_id": "find-0011",
          "title": "Replace per-step pending-list allocation with a reusable ring buffer",
          "description": "Pre-allocate a fixed-capacity request slot pool; reuse across steps; track head/tail with atomic counters."
        }
      ],
      "from_agents": [
        {
          "agent": "codex",
          "title": "Hoist the format-string log call out of the hot dispatch loop",
          "description": "Switch `logger.debug(f\"dispatched {req_id} ...\")` at line 187 to `logger.debug(\"dispatched %s ...\", req_id)` or wrap in `isEnabledFor(DEBUG)` so the level filter short-circuits."
        }
      ]
    }
  ]
}
```

---
