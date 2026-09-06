# System Map — High-Level Plan

## 1. The problem this solves in Spotlights

Spotlights discovers candidates by **fanning out one agent per module**, and
each of those agents is confined to its own module:

- the prompt opens with "You are auditing ONE module of a repository"
  ([bootstrap.md](../src/spotlights_engine/candidate_discovery/prompts_data/bootstrap.md)),
  every emitted `file` "MUST be under `{module_path}`", and files under the
  module's own sub-modules are excluded as separate targets;
- containment is then enforced, not merely requested: `_drop_outside_module`
  ([validation.py:90](../src/spotlights_engine/candidate_discovery/validation.py#L90))
  silently discards any candidate whose primary file falls outside the module
  root, counting it in `dropped_outside_module`;
- the per-module calls share no state — no agent sees another module's files,
  candidates, or reasoning.

That isolation is deliberate and worth keeping: it bounds context, makes the
fan-out parallel and resumable, and keeps candidates anchored to real line
ranges. But it costs the engine three things, and all three are the same
missing fact — *what runs when, and who else runs with it*.

**(a) Candidates whose optimization unit spans two modules have nowhere to
live.** Plenty of real targets are one policy split across two directories: a
sizing or eviction heuristic owned by a cache module whose effect is consumed
by the scheduler that admits work; a dispatch seam whose registry sits in one
module and whose implementations sit in another; a config block in module A
that parameterizes an inner loop in module B. Today the agent must pick one
side, anchor there, and describe the rest in prose — or anchor on the other
side and be dropped by the containment check. When two agents each find one
half, the halves arrive as two unrelated cards, and nothing in the pipeline
knows they are one seam, so they compete with each other for rank instead of
reinforcing each other. The report schema is halfway ready for this and
discovery is not: `Candidate.locations` is a list and only `locations[0]` is
containment-checked (`primary_file`,
[schema_compat.py:48](../src/spotlights_engine/utils/schema_compat.py#L48)),
while `AgentCandidate` can emit only a single flat `file` / `line_start` /
`line_end`
([agent_schema.py:39](../src/spotlights_engine/candidate_discovery/agent_schema.py#L39)).

**(b) Candidates that live in one module but are governed by another get their
impact guessed.** A site can be perfectly self-contained and still have its
value decided elsewhere: how often its caller runs, whether the branch it sits
on is taken in the deployment the objective cares about, which module owns the
knob that puts it on the hot path at all. The agent auditing that module cannot
see any of it. The only cross-module signal it is handed is
`module.depends_on`, passed as "call-shape context" with an explicit
instruction not to propose candidates outside the module — and that list is a
sparse, agent-authored hint (13 of 34 modules in `output/vllm_lsf`, 20 of 39 in
`output/wva_lsf`), direction-free, rate-free, and mixing external packages
(`torch`) with internal qualified names. So `estimated_impact` ends up inferred
from local code shape: a tight loop looks hot whether it runs per element or
once at startup.

**(c) Every module grades on its own scale, and the ranker inherits it.** N
module agents produce N private definitions of `high`, which is why the
listwise judge's top-20 is essentially the `high` label re-sorted and why
genuinely on-path `medium` candidates sink to ranks 54, 57 and 66 (§3).

The system map supplies exactly the missing fact, once per run, to every agent:
the ordered stages, their rates, and which modules participate in each. That
turns each of the three into something tractable:

- a **stage is the name a cross-module candidate has been missing**. Two sites
  in two modules on the same stage are visibly one seam — groupable at rank
  time, and the natural anchor for a second location or a `related_modules`
  field later (§15 Q10);
- the per-module header (§4.3) tells the agent for module A which stages A runs
  in, **which other modules share those stages**, at what rate, and what the
  objective weight of those stages is — so impact for a site in A is argued
  from the stage it sits on rather than guessed about B;
- because all agents read the same stage list and the same weights, `high`
  means one thing across the whole run.

What the map deliberately does **not** do is lift the containment rule. A
two-module edit still enters the report as two candidates until discovery can
emit more than one location; the map makes the pair recognizable and rankable
together, which is the cheap 80 % (§15 Q10, §16).

## 2. Goal & scope

(Problem statement: §1.)

Add a **system map** to every engine run: a one-page, per-run description of
how work flows through the target repository at runtime — where execution
enters, the ordered stages it passes through, which modules execute at which
stage and how often, and which of those stages the run's objective actually
depends on.

The target is any repository the engine is pointed at, not a class of system.
*Unit of work* means whatever the target processes end to end: an inference
request in a serving engine, a transaction batch in an ordering service, a job
in a scheduler, a query in a database, a file in a compiler, a training step in
a trainer. The map assumes only that the repository has entry points and that
execution proceeds through identifiable stages at distinguishable rates — never
a request/response shape, a network server, or a particular domain.

The map is a layer projected onto the module tree the extractor already
produces. The tree answers *which modules exist and where they live* — a
directory-shaped inventory, one line of description and a few main files per
module, and nothing else. The map answers *what runs, in what order, how
often, and which of it the objective cares about*. It is generated **once per
run** (one agent call, not one per module) and fed to every per-module
discovery agent and to the candidate ranker, so that "high impact" is graded
against one shared ruler instead of one private ruler per module.

Examples throughout this document are drawn from the checked-in vLLM and llm-d
runs because those are the runs we have measurements for; nothing in the
contract, the schema, or the prompt is specific to them.

In scope:

- a new run-level block, `system_map/`, that builds, validates, persists, and
  renders the map;
- wiring the map into the manager between extraction and the per-module
  fan-out;
- consuming it in candidate discovery (prompt context + impact rubric), in
  the ranking skill (card fields + path weight), and in the rendered output;
- an offline experiment that measures the effect on ranked recall before the
  engine-side work is committed.

Out of scope (see §16): profiling, call-graph construction, replacing or
changing the modules extractor, changing the recall harness, and any
domain-specific vocabulary — the closed vocabularies in §4 must stay
language- and domain-neutral.

## 3. Current state (grounded)

**The discovery agent runs blind.** Both discovery prompt templates have a
`{repo_context}` slot —
[bootstrap.md](../src/spotlights_engine/candidate_discovery/prompts_data/bootstrap.md)
and
[review.md](../src/spotlights_engine/candidate_discovery/prompts_data/review.md) —
rendered from `DiscoveryConfig.repo_context_markdown`
([api.py:50](../src/spotlights_engine/candidate_discovery/api.py#L50), capped at
20 000 characters). The manager never sets it: `_build_discovery_config`
([orchestrator.py:225](../src/spotlights_engine/spotlights_manager/orchestrator.py#L225))
copies only `repo_path` / `artifacts_dir`, and the step-2 call site
([orchestrator.py:842](../src/spotlights_engine/spotlights_manager/orchestrator.py#L842))
adds only `id_segment`. So in every engine run the "Repository context" block
of every discovery prompt reads `_(none provided)_`
([prompts.py:41](../src/spotlights_engine/candidate_discovery/prompts.py#L41)).

**The tree carries no runtime information.** `Module` has `name`, `path`,
`description`, `depends_on`, `main_files`, `submodules`
([project.py:95](../src/spotlights_engine/schemas/project.py#L95)). Descriptions
are inventories of contents ("User-facing entry points: programmatic LLM, CLI,
OpenAI/... servers"). `depends_on` is a sparse, agent-authored hint rather than
a dependency graph — populated for 13 of 34 modules in
`output/vllm_lsf/artifacts/spotlights_manager/project_tree.json` and 20 of 39 in
`output/wva_lsf`, and mixing external packages (`torch`) with internal
qualified names — so it cannot be relied on for structure either. Nothing
records the rate or
the position of a module in the runtime flow — in the checked-in vLLM tree,
nothing says that `v1/worker` runs on every decode step, that `v1/kv_offload`
runs only on a prefix-cache miss, or that model loading runs once at startup;
the same gap exists for every other target.

**Impact labels are therefore uncalibrated across modules, and the ranker
inherits that.** The listwise judge in
[sort-candidates.md](../templates/commands/sort-candidates.md) ranks ~150-token
cards (label, truncated explanation, research-proposal count) and never sees
code or the execution path. Measured on saved runs, its top-20 is the "high"
label re-sorted:

| Run | labeled `high` | `high` among judge top-20 |
|---|---|---|
| `examples/vllm_subset` | 60 / 173 | 20 / 20 |
| `output/vllm_lsf` | 66 / 261 | 20 / 20 |
| `output/llmd_router_lsf` | 26 / 100 | 18 / 20 |

The recall reports show the cost: the engine *finds* expert candidates but
buries the ones labeled `medium`
([llmd_expert_recall_results.md](llmd_expert/llmd_expert_recall_results.md):
5/7 found, 2/7 in top-10, two recalled seams at ranks 54 and 66;
[vllm_expert/recall_report.md](vllm_expert/recall_report.md): 1/1 found at rank
57). Both deep-ranked llm-d seams sit on the prefix-cache stage that the run's
multi-turn hint puts squarely on the TTFT path — a fact no agent in that run
was told.

**Run sequence today** (`_run_async`,
[orchestrator.py:1847](../src/spotlights_engine/spotlights_manager/orchestrator.py#L1847)):
fingerprints → `_run_extractor_if_needed`
([:1897](../src/spotlights_engine/spotlights_manager/orchestrator.py#L1897)) →
`apply_filter`
([:1904](../src/spotlights_engine/spotlights_manager/orchestrator.py#L1904)) →
slug check → per-module tasks + `asyncio.gather`
([:1960](../src/spotlights_engine/spotlights_manager/orchestrator.py#L1960)).
The extractor step is resumable from two sidecars (`project_tree.json`,
`extractor_invocation.json`) via `read_extractor_outputs`
([persistence.py:787](../src/spotlights_engine/spotlights_manager/persistence.py#L787))
and a `manifest["extractor"]` completion flag — the pattern the map step
mirrors.

**Run-level steps have no home yet.** `PipelineStep`
([common.py:15](../src/spotlights_engine/schemas/common.py#L15)) and
`UsageStep`
([records.py:25](../src/spotlights_engine/costing/records.py#L25)) are closed
literals of per-module steps; `read_usage_records`
([persistence.py:617](../src/spotlights_engine/spotlights_manager/persistence.py#L617))
enumerates the four run steps explicitly and walks module dirs only. The old
ranking plan ([ranking_plan.md](ranking_plan.md)) hit the same gap; the
map is the first run-level step to actually land, so it pays for the plumbing.

## 4. What the system map is (contract)

### 4.1 Sections

1. **Entry points** — where work enters the system and where the process
   starts, each anchored to `file` + `symbol`. Kinds: `service` (long-running,
   accepts external calls), `job` (one-shot batch run), `cli`, `library`
   (embedded/programmatic API), `startup`, `background`.
2. **Execution lifecycle** — one line naming what a unit of work *is* for this
   repository ("one inference request", "one transaction batch", "one compiled
   file"), which fixes the meaning of the rate ladder below, followed by the
   ordered stages one such unit passes through. Each stage has an id, a
   one-line name, a **frequency** from a closed vocabulary, ≥ 1 anchor (the
   function where the stage begins), the
   participating module qualified names, and an optional `condition` ("only in
   disaggregated deployments", "only when the input has attachments").
3. **Objective → stages** — which stages the objective's metric is a sum of,
   given the workload hints, as a weight per stage in `[0, 1]` plus a short
   rationale. This is the only objective-dependent section.
4. **Module × stage table** — one row per module in the tree: stages,
   dominant frequency, `on_path ∈ {yes, conditional, no}`, derived
   `path_weight`, one-line note. Off-path modules are listed explicitly, not
   omitted.
5. **Hot paths (evidence)** — optional. Filled from a profile or the
   repository-history prior when either is supplied; empty otherwise.

Frequency vocabulary (closed), a rate ladder rather than a domain taxonomy:

| value | meaning | vLLM serving | ordering service | compiler |
|---|---|---|---|---|
| `startup` | once per process | model load, compile, warmup | key/config load, peer dial | driver init |
| `per_unit` | once per unit of work | per request | per transaction batch | per source file |
| `per_step` | once per inner-loop iteration | per scheduler step | per consensus round | per pass over the IR |
| `per_element` | once per element produced or consumed inside a unit | per token | per transaction in a batch | per statement/node |
| `conditional` | only on some paths or configurations; rate unknown | on cache miss | only when reconfiguring | only with a given flag |
| `background` | periodic, off the critical path | metrics, GC | gossip, checkpointing | cache eviction |
| `off_path` | does not run for this workload | benchmarks, LoRA admin | admin tooling | docs generation |

Rate order, used wherever a "dominant" or "highest-rate" frequency is needed
(§6C, §7.1): `off_path` < `background` < `startup` < `conditional` <
`per_unit` < `per_step` < `per_element`. A target that has no inner loop simply
never uses `per_step` / `per_element`; that is a valid map, not a defect.

### 4.2 Schema (new `schemas/system_map.py`)

```python
Frequency = Literal["startup", "per_unit", "per_step", "per_element",
                    "conditional", "background", "off_path"]

class Anchor(BaseModel):            # extra="forbid" everywhere (codex/claude strict schemas)
    file: str                       # repo-relative, must exist
    symbol: str                     # must be found in `file` (grep, not AST)
    line: int | None = None

class EntryPoint(BaseModel):
    name: str
    kind: Literal["service", "job", "cli", "library", "startup", "background"]
    anchor: Anchor
    first_stage: str                # LifecycleStage.id

class LifecycleStage(BaseModel):
    id: str                         # snake_case, target's own terms: "prefix_lookup"
    order: int                      # 1..N, a permutation
    name: str
    frequency: Frequency
    anchors: list[Anchor]           # >= 1
    modules: list[str]              # qualified names (slash form)
    condition: str | None = None

class LifecycleMap(BaseModel):      # objective-free half
    repo_name: str
    commit_sha: str | None
    unit_of_work: str               # "one inference request", "one transaction batch"

    entry_points: list[EntryPoint]
    stages: list[LifecycleStage]

class ObjectiveOverlay(BaseModel):  # objective-specific half
    objective: str                  # free text: latency, throughput, memory, cost, ...
    workload_hints: list[str]
    stage_weights: dict[str, float] # stage id -> [0, 1]; at least one > 0
    rationale: str

class ModuleRow(BaseModel):         # derived, deterministic
    module_qualified_name: str
    stages: list[str]
    frequency: Frequency
    on_path: Literal["yes", "conditional", "no"]
    path_weight: float              # max over its stages of stage_weights, × 0.5 if conditional
    note: str = ""

class SystemMap(BaseModel):
    version: Literal[1] = 1
    lifecycle: LifecycleMap
    overlay: ObjectiveOverlay
    module_rows: list[ModuleRow]
    hot_paths: list[HotPath] = []
    advisory: bool = False          # degenerate map: context only, no triage/weighting (§6C)
    issues: list[str] = []          # validator repairs, surfaced in index.md
```

Two halves on purpose: `LifecycleMap` depends only on the checkout (fingerprint
= project-tree hash + commit); `ObjectiveOverlay` depends on the run context
(fingerprint = lifecycle hash + `SpotlightContext` hash). `module_rows` is a
pure function of the two and is recomputed, never stored as agent output.

### 4.3 Markdown rendering

`render_markdown(system_map, *, for_module: str | None) -> str`, pure and
unit-tested. With `for_module` set, a header block is prepended — shape below,
filled here from the vLLM run:

```
## This module in the system map
- stages: prefix_lookup (per_unit, on cache miss), free (per_unit)
- on the objective path: yes — TTFT under the multi-turn hint depends on prefix hit rate
- NOT on: TPOT
- sharing these stages: v1/core/kv_cache_manager (prefix_lookup, free)
- immediately upstream: v1/core/sched, v1/engine/core (admission/schedule, per_step)
```

followed by the full map. The renderer enforces the 20 000-character cap of
`repo_context_markdown` by degrading in fixed order: drop `hot_paths`, drop
`note` columns, keep only rows for selected modules plus the stage owners.
A `## System map` rendering for `index.md` uses the same function with
`for_module=None`.

The `sharing these stages` / `immediately upstream` lines are the §1b fix, and
both are derived from `stages[].modules` plus stage order — no extra agent work.
They are the only cross-module information an agent receives, they are facts
from the map rather than guesses about a neighbour, and they stay inside the
containment rule: the agent learns who it runs alongside and at what rate, and
is still told to anchor every candidate in its own module.

Worked example (abridged) — the checked-in vLLM run. Stage ids, stage names and
the objective are the target's own vocabulary; only the frequency column and the
`on path` column come from the closed vocabularies of §4.1:

```markdown
# System map — vllm @ bcf2be96
unit of work: one inference request
objective: reduce median TTFT and median TPOT · hint: multi-turn agentic workload

## Entry points
- service: vllm/entrypoints/openai/api_server.py → AsyncLLM.generate
- library: vllm/entrypoints/llm.py → LLMEngine.step
- startup: model_loader, compilation, warmup — once, off the execution path

## Execution lifecycle
1. tokenize/preprocess  inputs, multimodal                  per_unit
2. admission/schedule   v1/engine/core, v1/core/sched       per_step
3. prefix-cache lookup  v1/core/kv_cache_manager, v1/kv_offload   per_unit (offload tier: on miss)
4. prefill              v1/worker, v1/attention, model_executor/layers   per_unit, chunked
5. decode               v1/worker, v1/attention, v1/sample  per_element (token)
6. detokenize/stream    v1/engine/output_processor          per_element (token)
7. free/evict           v1/core/kv_cache_manager, v1/kv_offload   per_unit

## Objective → stages
- TTFT = 2 (queue wait) + 3 + 4; under the multi-turn hint stage 3 dominates  → weights 2:0.4 3:1.0 4:0.8
- TPOT = 5 + 6 + per-step share of 2                                            → weights 5:1.0 6:0.5 2:0.3
- off-path: startup, weight loading, LoRA management, benchmarks

## Module × stage
| module                  | stages | frequency    | on path     | weight |
| v1/core                 | 2,3,7  | per_step     | yes         | 1.0    |
| v1/kv_offload           | 3,7    | per_unit     | yes         | 1.0    |
| v1/worker               | 4,5    | per_element  | yes         | 1.0    |
| distributed/kv_transfer | 4      | per_unit     | conditional | 0.4    |
| multimodal              | 1      | per_unit     | conditional | 0.2    |
| lora                    | —      | off_path     | no          | 0.0    |
```

The same shape on a non-serving target reads the same way: for an ordering
service the unit of work is a transaction batch, the objective is commit
latency or throughput, stage 2 is a consensus round at `per_step`, signature
verification is `per_element`, and gossip is `background`.

## 5. Placement: after the extractor, before the fan-out

Decision: the map is built **after** `_run_extractor_if_needed` and
`apply_filter`, **before** the per-module tasks are created — i.e. between
[orchestrator.py:1904](../src/spotlights_engine/spotlights_manager/orchestrator.py#L1904)
and
[:1960](../src/spotlights_engine/spotlights_manager/orchestrator.py#L1960).

Why after extraction:

- The map's core artifact is keyed by the same slash-form qualified names that
  discovery, the filter, the judge, and the renderer use. That vocabulary only
  exists once extraction has run; a map built earlier would need its own
  projection onto modules — the extractor's job.
- The tree is objective-independent and reused on resume; the map's overlay is
  objective-dependent. Layering the objective-dependent artifact on top keeps
  the expensive, reusable one untouched.
- The extractor has already found the source root, main files, and per-module
  descriptions. The lifecycle trace uses them as its reading guide and stays a
  single agent call.

The one argument for "before" — influencing extractor granularity when a
module straddles stages — is handled without reordering: stage anchors are
file + symbol, so a straddling module gets several stages in its row, each
with its own anchor, and the discovery agent sees sub-module precision
regardless of how coarse the tree is. If straddling ever becomes a measured
problem, feed the previous run's map into the next extraction as a hint.

## 6. The new block: `src/spotlights_engine/system_map/`

Mirror the sibling-step layout (`api.py`, `prompts.py` + `prompts_data/`,
`claude_exec.py`, `agent_schema.py`, `validation.py`, `render.py`,
`errors.py`).

### Stage A — deterministic seeds (no agent)

`seeds.py` computes, from the tree and the checkout:

- candidate entry points, from language-generic greps with no assumption about
  the kind of system: process mains (`if __name__ == "__main__"`, `func main()`,
  `fn main`, `public static void main`), declared console scripts / binaries
  (`pyproject.toml`, `setup.py`, `cmd/`, `Cargo.toml`, `package.json`),
  server/route registration, job or task entry decorators, and container
  entrypoints (`Dockerfile`, `Makefile` targets). `File.role` strings from the
  extractor are searched for the same signals (free text, not a closed
  vocabulary — §15 Q6);
- the module list with descriptions and `main_files`, rendered as the agent's
  reading guide;
- the commit sha (already resolved for the run manifest).

Seeds are hints in the prompt, not truth; the agent may reject them.

### Stage B — one agent call (Claude, read-only)

`claude_exec.py` follows
[agent_proposals/claude_exec.py](../src/spotlights_engine/agent_proposals/claude_exec.py):
`claude -p`, cleaned env, structured output against the
`AgentSystemMap` JSON schema (the agent emits `LifecycleMap` + a raw
`ObjectiveOverlay`; never `module_rows`), `AgentUsage` parsed from the stream.
Tools: read-only (`Read`, `Grep`, `Glob`, `Bash` limited to `rg`/`wc`/`git
log`), `max_turns` ≈ 40, wallclock 900 s, one strict-JSON retry as in
discovery. Schema-size guard as in discovery's `_MAX_SCHEMA_BYTES`.

The prompt (`prompts_data/system_map.md`) asks for: identify what one unit of
work is for *this* repository and say so in one line; trace from each entry
point to the point where that unit is complete; name the stage boundaries as
functions; give each stage a frequency from the closed rate ladder; list
participating modules by qualified name from the supplied list only; then, given
the objective and hints, assign stage weights with a two-sentence rationale.
Explicit rules: verify every anchor by reading the file; do not invent modules;
name stages in the repository's own vocabulary, not in borrowed
domain terms; prefer fewer, coarser stages (5–9) over a call graph; if the
repository has no inner loop, leave `per_step` / `per_element` unused rather
than inventing one.

One CLI only. A second-opinion Codex pass is not worth 2× cost for a
structural summary (same reasoning as the ranking plan); disagreement is
better spent on the discovery agents that consume the map.

### Stage C — validation and derivation (`validation.py`)

Deterministic, never raises for content problems (records `issues`):

- every `Anchor.file` exists under `repo_path`; `symbol` is found in the file
  by substring/regex; `line` (if given) is within the file's line count. A
  failing anchor is dropped; a stage with zero surviving anchors is kept and
  flagged `unanchored`;
- `unit_of_work` non-empty (an empty string is replaced by `"one unit of work"`
  with an issue, since every rate label is read relative to it);
- stage ids unique, `order` a permutation of `1..N`, `first_stage` and every
  `modules[]` / `stage_weights` key resolve; unknown module names are dropped
  with an issue; weights clipped to `[0, 1]`, at least one must be `> 0`;
- coverage: every module selected for the run has a row. Modules the agent
  never mentioned get `frequency=off_path, on_path=no, weight=0` and an issue
  ("not placed by the agent") so silence is visible rather than treated as
  fact;
- `module_rows` derived: stages = stages listing the module; frequency = the
  highest-rate stage by the §4.1 rate order; `on_path` = `yes` if any stage has
  weight > 0 and no `condition`, `conditional` if only conditioned stages carry
  weight, else `no`; `path_weight` = max stage weight, halved when
  `conditional`;
- **degenerate-map guard** (targets with no single coherent flow — a utility
  library, a toolbox monorepo, a repo the agent could not trace): if the map has
  ≤ 2 stages, or fewer than 25 % of the selected modules end up `on_path ∈
  {yes, conditional}`, record an issue and set `advisory=True` on the map.
  An advisory map is still rendered and still passed to discovery as context,
  but triage (§7.3) and the ranker's path weight (§7.2) ignore it — a map that
  failed to find the flow must not silently mark the whole repository off-path.

Fatal (raises `SystemMapValidationError`): schema parse failure after the
retry, or zero stages.

## 7. Consumers

### 7.1 Candidate discovery (the primary consumer)

Phase 1, no contract change: the manager renders `render_markdown(map,
for_module=qn)` and passes it as `repo_context_markdown` when building the
per-module `DiscoveryConfig`
([orchestrator.py:842](../src/spotlights_engine/spotlights_manager/orchestrator.py#L842)).
Both the bootstrap and every review pass see it, and discovery already
persists it per module (`layout.repo_context_path`) — provenance for free.

Phase 3, rubric change in
[bootstrap.md](../src/spotlights_engine/candidate_discovery/prompts_data/bootstrap.md)
and
[review.md](../src/spotlights_engine/candidate_discovery/prompts_data/review.md):
define `estimated_impact` relative to the map —

- `high`: the site is on the objective path at the objective's dominant
  stage(s) — the highest-weight stage, or the highest-rate one (`per_step` /
  `per_element`) when weights tie — **and** has concrete headroom;
- `medium`: on-path at `per_unit`, or `conditional`;
- `low`: off-path, startup, or headroom only under a workload the hints
  exclude.

`estimated_impact_explanation` must name the stage id. Add an optional
`stage: str` to `AgentCandidate` / `Candidate` (validated against the map's
stage ids; unknown → dropped to `None` with a drop counter, not a rejection).
This is a schema bump for `Candidate` (`extra="forbid"`) and a strict-schema
check for codex, so it is kept out of Phase 1 to make the A/B clean.

### 7.2 Ranking (`/spotlights-sort-candidates`, and the future in-engine ranker)

- Ranking card gains `stage`, `frequency`, `on_path`, `path_weight`, copied
  from `module_rows` (or the candidate's `stage` once it exists) — facts from
  the map, not the agent's self-assessment.
- Deterministic pre-score becomes
  `impact_weight × (0.25 + 0.75 × path_weight) + evidence`, so the fallback
  order already respects the objective and an off-path `high` cannot outrank an
  on-path `medium` on label alone.
- The judge prompt receives the map's objective section verbatim and is told
  to treat `on_path` as a hard prior.
- **Same-stage grouping (the §1a fix, rung (i)).** Before the listwise call,
  candidates are bucketed by stage and the bucket is presented together, so two
  halves of one spanning seam — a sizing policy in a cache module and the
  admission site that consumes it — are judged as one story instead of
  competing as two unrelated cards. Grouping only reorders what the judge sees;
  it does not merge candidate records, dedupe them, or change their ids.
- The in-engine ranking block from [ranking_plan.md](ranking_plan.md)
  consumes the same `module_rows` when it lands.

The skill reads the map from `result.json` (§7.5) and falls back to today's
behavior when the map is absent or `advisory` (§6C), so old runs and untraceable
repositories keep working.

### 7.3 Module triage (budget, not scope)

A new manager knob `system_map_triage ∈ {none, reduce, skip}` (default
`none` until measured, §14):

- `reduce`: modules with `on_path = no` run the bootstrap pass only
  (`num_review_iterations = 0`) and skip deep research;
- `skip`: they are marked `SKIPPED` with a reason.

Triage only ever reduces budget. An explicit `--include` is always honored;
triage never adds modules; and triage is disabled outright when the map is
`advisory` (§6C), so a repository whose flow the agent could not trace is never
pruned on the strength of a bad map.

### 7.4 Module deep research (optional, Phase 4)

Add the module's row and the objective section to the "Target module" block of
the research prompt
([module_deep_research/prompts.py:70](../src/spotlights_engine/module_deep_research/prompts.py#L70)),
so the literature search is steered toward the stage the objective cares
about. Low priority; the per-candidate research rework
([../docs/architecture/spotlights_deep_research_on_candidates_v2.md](../docs/architecture/spotlights_deep_research_on_candidates_v2.md))
would consume it more naturally.

### 7.5 Rendered output, `result.json`, manifests

- `index.md`: a `## System map` section right after `## Repository`
  (`_render_index`,
  [writer.py:139](../src/spotlights_engine/results_renderer/writer.py#L139)),
  plus a full `system_map.md` page next to `index.md`. Issues from §6C are
  rendered under it so a wrong row is spottable in seconds.
- `result.json` (`SpotlightReport`): optional `system_map: SystemMap | None`.
- Internal `manifest.json`: `manifest["system_map"] = {"completed",
  "duration_s", "source": "agent" | "file" | "disabled", "lifecycle_fp",
  "overlay_fp"}`.
- Public `run_manifest.json`: `run_config` gains `system_map` (mode + source);
  `by_step` gains a `system_map` cost row (§8).

## 8. Manager integration (`spotlights_manager/orchestrator.py`)

New `_run_system_map_if_needed(input, cfg, paths, manifest, tree, selected)
-> SystemMap | None`, called after `apply_filter` and before the tasks are
created. Semantics mirror `_run_extractor_if_needed`
([:631](../src/spotlights_engine/spotlights_manager/orchestrator.py#L631)):

1. `mode == "off"` → return `None`, record `source: disabled`.
2. `mode == "file"` → load and validate the user-supplied map (§10), rebuild
   `module_rows` against *this* tree, record `source: file`. Lets a user who
   knows their system skip the agent, and is what the offline experiment uses.
3. `mode == "auto"` → if both sidecars exist and the stored fingerprints match
   (lifecycle: tree hash + commit; overlay: lifecycle hash + context hash),
   reuse. If only the overlay mismatches, re-run the (cheap) overlay prompt
   against the cached lifecycle. Otherwise clear and run Stage A→C.
4. On any failure in `auto`: log, write a run-level
   `StepIssue(step="system_map", severity="warning", recoverable=True)`,
   return `None`, and continue — every consumer treats `None` as "today's
   behavior". `mode == "required"` turns the failure into a run failure for
   CI-style use.

`_run_module` gets a `system_map: SystemMap | None` parameter and threads the
per-module rendering into `_build_discovery_config(...).model_copy(update=
{"repo_context_markdown": ...})`. Triage (§7.3) is applied when building the
per-module `DiscoveryConfig` (`num_review_iterations`) and when deciding
whether to run step 3.

Fingerprints: `system_map` config joins `build_config_fingerprint`
([persistence.py:323](../src/spotlights_engine/spotlights_manager/persistence.py#L323))
so a mode change is not silently resume-compatible; a user-supplied map file's
hash joins `build_input_fingerprint`.

Costing: add `"system_map"` to `PipelineStep` and `UsageStep`; usage records
land under `paths.root / "system_map" / "usage"`; extend the run-level
aggregation that walks module dirs to also read that directory, and add the
row to `by_step`. `UsageRecord.module_qualified_name` is `min_length=1`; use
the sentinel `"(run)"` for run-level records (§15 Q2).

## 9. Persistence (`spotlights_manager/persistence.py`)

`ManagerPaths`
([persistence.py:123](../src/spotlights_engine/spotlights_manager/persistence.py#L123))
gains:

```
<artifacts>/spotlights_manager/
  system_map/
    lifecycle.json        # LifecycleMap + fingerprint
    overlay.json          # ObjectiveOverlay + fingerprint
    system_map.json       # full SystemMap (derived rows, issues) — what consumers read
    system_map.md         # rendered, for_module=None
    seeds.json            # Stage A output (debug)
    prompt.md, last_message.json, stdout/stderr on failure
    usage/                # UsageRecords
```

`write_system_map_outputs`, `read_system_map_outputs`,
`clear_system_map_artifacts` follow the extractor trio
([persistence.py:772](../src/spotlights_engine/spotlights_manager/persistence.py#L772)).
The renderer's loader reads `system_map.json` when present.

Follow-up (Phase 4): the batch extractor
([../scripts/run_modules_extractor_batch.py](../scripts/run_modules_extractor_batch.py))
can cache `lifecycle.json` next to the cached `project_tree.json`, so repos
that are re-run under many objectives pay for the trace once.

## 10. CLI / config surface

`spotlights-engine`:

```
--system-map {auto,off,required,FILE}   default: off in Phase 1, auto once §13 numbers are in
--system-map-triage {none,reduce,skip}  default: none
```

`SpotlightsManagerConfig`
([api.py:37](../src/spotlights_engine/spotlights_manager/api.py#L37)) gains
`system_map: SystemMapConfig | None` with `mode`, `path`, `triage`,
`claude_max_turns`, `wallclock_s`, `claude_model` (inherits the global model
config like the other steps). Both knobs are recorded in `run_manifest.json`
via `_run_config`
([orchestrator.py:288](../src/spotlights_engine/spotlights_manager/orchestrator.py#L288)).

`doctor` prints whether the map step will run and with which model.

## 11. Failure & degradation model

| Situation | Behavior |
|---|---|
| CLI missing / agent timeout / bad JSON twice | `auto`: warning issue, continue without map. `required`: run fails before any module starts (cheap to fail early). |
| Anchors don't resolve | Anchor dropped, stage flagged, run continues; visible in `index.md`. |
| Agent omits a selected module | Row synthesized as off-path with an issue; consumers see "not placed", not "off-path" as fact. |
| Target has no single coherent flow (utility library, toolbox monorepo), or the agent could not trace one | Degenerate-map guard (§6C): map marked `advisory` — still rendered and still used as discovery context, but no triage and no path weighting. |
| Rendered context > 20 000 chars | Degrade rendering in fixed order (§4.3); never truncate mid-table. |
| User map file fails validation | Hard error at setup (the user asked for it explicitly). |
| Tree changed on resume | Lifecycle fingerprint mismatch → rebuild; old artifacts cleared like the extractor's. |

## 12. Tests

`tests/unit/system_map/`:

- schema: closed vocabularies, `extra="forbid"`, permutation/uniqueness rules;
- validation: missing file, symbol not found, unknown module, empty weights,
  omitted selected module → synthesized row + issue; fatal on zero stages;
- derivation: `module_rows` from a fixture map (`yes` / `conditional` / `no`,
  weight halving, rate order for the dominant frequency);
- degenerate guard: ≤ 2 stages and < 25 % on-path both set `advisory`, and an
  advisory map disables triage and path weighting while still rendering;
- render: deterministic markdown, per-module header (including the
  `sharing these stages` / `immediately upstream` lines, and their absence when
  a module is alone on its stages), cap degradation order, ≤ 20 000 chars on a
  200-module synthetic tree;
- seeds: entry-point detection on `tests/fixtures` mini-repos, covering more
  than one shape — a Python service, a Go CLI/`cmd/` binary, and a
  library-only package with no `main` — so no detector assumes a server;
- prompt rendering; agent exec with a stubbed `claude` (as sibling steps do).

`tests/unit/candidate_discovery/`: `render_bootstrap` / `render_review` fill
`{repo_context}` with the map; the header names the right module.

`tests/unit/spotlights_manager/`: step ordering (map runs after filter, before
fan-out); cache hit / overlay-only rebuild / full rebuild by fingerprint;
degrade on failure in `auto`, fail in `required`; `FILE` mode bypasses the
agent; triage reduces `num_review_iterations` only for `on_path = no`;
run-level usage records aggregate into `by_step`; manifest knobs recorded.

`tests/unit/results_renderer/`: `index.md` section present/absent; issues
rendered.

`tests/integration/spotlights_manager/`: extend the fake-CLI dry run to cover
`--system-map auto` end to end.

## 13. Validation — how we know it worked

Metrics: recall@10, recall@20, and mean reciprocal rank of ground-truth
candidates, computed with the existing tooling against
`design/*_expert/` and the `runs/run-on-pr/*` ground truth. Overall recall
must not drop and the candidate count must not rise.

**Phase 0 (offline, no engine reruns).** Write the map by hand — or with the
Phase 1 agent — for `output/llmd_router_lsf` and `output/vllm_lsf`; add the
§7.2 card fields and path weight to the sort skill; re-rank the saved
`result.json` files; compare. Concrete hypotheses:

- llm-d: the prefix block-size seam (rank 54, `medium`, prefix-lookup stage)
  and the profile-handler seam (rank 66) move into the top 20 → recall@20 from
  3/7 to ≥ 5/7;
- vLLM expert: the CPU-offload dispatch seam (rank 57) moves into the top 20.

Phase 0 also carries a cheap **generality check** that costs no reruns: hand-write
the lifecycle half for one non-serving target we already have a tree for (the
`output/wva_lsf` Go controller, or the Fabric-X orderer) and confirm that every
stage lands on an existing rung of the rate ladder and that the map does not
trip the degenerate guard (§6C). If it does not fit, the vocabulary changes
before any code is written (§15 Q8).

If Phase 0 shows no movement, the map is not the lever and the engine-side
work stops at Phase 1.

**Phase 1 A/B.** Re-run discovery on `llm-d-router` with and without the map
injected (same tree, same models, `--include` unchanged), score both, and also
compare per-module candidate counts and label distributions — the map should
shift labels toward on-path modules, not inflate them.

## 14. Build order

| Phase | Work | Size | Gate |
|---|---|---|---|
| 0 | Hand-written maps for two saved runs + one non-serving target (vocabulary check); card fields + path weight in `sort-candidates.md`; measure §13. | ½ day | Numbers move; ladder fits. |
| 1 | `system_map/` package (schema, seeds, prompt, exec, validation, render); manager step; persistence; `--system-map`; consumer = discovery `{repo_context}` only; degrade-safe. Default `off`. | 2–3 days | A/B on llm-d. |
| 2 | Renderer (`index.md`, `system_map.md`), `result.json` field, costing + manifests, sort skill reads the map from `result.json`. Flip default to `auto`. | 1–2 days | — |
| 3 | Impact rubric relative to the map; `stage` field on `AgentCandidate` / `Candidate`; triage knob. | 1–2 days | Re-measure §13. |
| 4 | Deep-research consumption; lifecycle cache in the batch extractor; profile / history evidence into `hot_paths`. | later | — |

## 15. Open questions / decisions to confirm

1. **Contract placement.** Phase 1 passes the map through the infra-only
   `repo_context_markdown` (zero contract change). Promote to a
   `system_map: SystemMap | None` field on `CandidateDiscoveryInput` /
   `ModuleDeepResearchInput` in Phase 3, when the `stage` field makes it
   structured input? Recommendation: yes, at Phase 3.
2. **Run-level usage records.** Sentinel `module_qualified_name = "(run)"`
   vs. adding a `scope: Literal["module", "run"]` to `UsageRecord`.
   Recommendation: sentinel now; it keeps the record key intact.
3. **Default mode.** `off` until Phase 1 numbers exist, then `auto`.
   Confirm the flip criterion (§13) is acceptable.
4. **Scope of the trace under `--include`.** Always trace the whole repo (one
   call either way; selected modules need the surrounding stages), render only
   selected rows + stage owners. Confirm.
5. **Triage default** once measured: `none` vs `reduce`.
6. **Extractor `File.role` strings** — `File.role` is free-form agent text
   ("Attention nn.Module bound to a backend"), not a closed vocabulary, so
   Stage A greps it for entry-point signals rather than filtering on it.
   Confirm no role enum is planned.
7. **Stage granularity guidance** — 5–9 stages is a prompt rule; should the
   validator enforce an upper bound (e.g. 12) to keep the map a summary rather
   than a call graph?
8. **Rate ladder coverage** — does `per_unit` / `per_step` / `per_element`
   (§4.1) hold up on the non-serving targets we already run against (the
   `wva_lsf` Go controller, the llm-d router, the Fabric-X orderer)? Cheap
   check: hand-write the lifecycle for one of them during Phase 0 and see
   whether any stage needs a rung that does not exist.
9. **Objective vocabulary** — the overlay's `objective` is free text today.
   Keep it free text (targets differ too much), or normalize to a small set
   (latency / throughput / memory / cost) so the rubric can key off it?
   Recommendation: free text, with the workload hints doing the work.
10. **Cross-module candidates (§1a)** — how far to go beyond "same stage" in
    this plan. Three rungs, cheapest first: (i) group same-stage candidates
    from different modules in the ranker and let the judge see them together —
    needs only the `stage` field of Phase 3; (ii) add a
    `related_modules: list[str]` to `AgentCandidate`, validated against the
    stage's module list, so an agent can say "the other half is in
    `v1/core/sched`" without escaping containment; (iii) let discovery emit a
    second `CodeLocation` outside `module_path` when it is on a shared stage —
    `Candidate.locations` already allows it and only `locations[0]` is
    containment-checked, but this weakens the drop rule that keeps the fan-out
    honest. Recommendation: (i) with Phase 3, (ii) measured behind it, (iii)
    only if the recall harness shows real spanning seams still being lost.

## 16. Non-goals

- Not a call graph, a profiler, or a hotness oracle. The map records stage
  membership and a coarse frequency class; per-function hotness stays with
  future signal sources (profiles, repository history).
- Does not change the modules extractor or the tree schema.
- Does not change the recall harness or its matching rules.
- Does not add a second agent CLI to the step.
- Does not lift the per-module containment rule or let a discovery agent
  propose candidates in another module (§1). The map names the shared stage so
  a spanning seam is recognizable and rankable as one thing; merging the halves
  into a single multi-location candidate is a separate change (§15 Q10).
- Not specialized to inference serving, or to any other domain: the closed
  vocabularies stay rate- and role-based, and every domain term in a map comes
  from the target repository itself.
