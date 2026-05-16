# Find Evolve-Optimizable Code Locations

You are auditing ONE module of a repository to identify code locations that
are good candidates for **evolutionary code optimization** (in the style of
OpenEvolve / AlphaEvolve). You do NOT write fixes. You enumerate candidate
locations, each with a falsifiable rationale, a measurable metric, and a
correctness oracle.

## Module under audit
- module_qualified_name: {module_qualified_name}
- module.name:           {module_name}
- module.path:           {module_path}                   # all candidate files MUST lie under this
- module.description:    {module_description}
- module.depends_on:     {depends_on}                    # qualified names of sibling modules, NOT filesystem paths
- module.main_files:                                     # start here, in order
{main_files}
- submodules:            {submodule_names}               # nested modules under this one

## What counts as a good evolve target

Pick locations where an evolutionary loop (mutate code → run benchmark → keep
winners) can plausibly find a better implementation. Good candidates have
**all** of:

1. **Self-contained behavior.** A function, kernel, loop nest, or small region
   whose contract (inputs → outputs / side effects) is stable and testable in
   isolation.
2. **A measurable metric.** Wall-clock latency, throughput (tokens/s, req/s),
   memory footprint, cache miss rate, FLOPs, allocation count, a quality
   score, or a domain metric. Must be quantifiable from an existing or
   easily-written harness.
3. **Real headroom.** Hot paths, heuristics with magic numbers, hand-rolled
   schedules, batch/tile/block sizing, kernel launch params, prefetch/eviction
   policies, scheduling/admission control, numerical kernels,
   parsing/serialization inner loops, retry/backoff strategies.
4. **Bounded blast radius.** Correctness can be checked with existing tests,
   golden outputs, property tests, or a small custom oracle — not "audit the
   whole system."

Skip locations that are:
- pure glue, logging, config plumbing, type definitions, dataclasses — unless
  the glue/config line is the concrete registration site for a qualifying
  `plugin_seam`
- already optimal/trivial (one-liners, direct library calls)
- correctness-critical with no oracle (security checks, auth, consensus)
- so entangled that the function or seam boundary doesn't capture the
  optimization unit

## Plugin seams (`kind: "plugin_seam"`)

A separate, language-neutral category of target: not "edit lines L1–L2 of one
function" but "add a new implementation behind a stable interface, register
it at a known site." The interface and the registry already exist; the
optimization unit is a *new* sibling implementation plus one registry edit.

This pattern shows up across languages as:

- a map / dictionary / table whose values are interchangeable implementations
  of a common interface — e.g. Python `dict[str, type[X]]`, Go
  `map[string]Factory`, Rust `HashMap<&str, fn() -> Box<dyn Trait>>`, C++
  factory map, TS `Record<string, () => Impl>`
- a service-discovery / plugin mechanism — Java `ServiceLoader` +
  `META-INF/services/...`, Python entry points, .NET DI container, Go
  `init()` self-registration, Rust feature flags / `cfg`-gated modules
- a `switch` / `match` / `if`-chain dispatching on a config-driven tag
- a vtable / function-pointer table populated at startup

Emit `kind: "plugin_seam"` only when, in addition to the four criteria above:
- the interface contract is small and stable (a handful of methods/functions
  with documented invariants), and
- the selector is data-driven (config value, env var, build flag, manifest
  file) — not a hard-coded branch with no external surface.

For seam candidates, anchor the candidate at the **registration site**, not
the interface:

- `file` + `line_start` / `line_end` → the single most concrete edit a new
  implementation requires (the map entry, the service-manifest line, the
  factory registration call, the match arm). NOT the interface declaration.
- `symbol` → the registry / table / dispatch identifier.
- `current_approach` → one sentence naming the interface (with its file
  path), the existing reference implementations (with their file paths),
  and the runtime selector that picks one (config key, env var, build flag,
  manifest filename).
- `evolve_rationale` → the oracle is the interface's existing test suite
  and/or documented invariants; cite where they live.
- `metrics` → the workload-level signal a better implementation would move
  (hit rate, latency, throughput, footprint, quality score).

The evolution unit is "a new implementation file + one registry edit." The
proposer only needs to anchor the registry; the harness can follow the
chain registry → values → interface → reference implementations.

## Process

1. **Read every file in `module.main_files`** and any sibling files under
   `module.path` that look load-bearing. Use `view` / `bash` (`rg`, `wc -l`) —
   do not guess line numbers; verify them against the actual file.
2. **Rank** candidates internally by (impact × tractability). Return every
   candidate that clears the quality bar in §"What counts as a good evolve
   target" — do not cap or truncate. Equally, do NOT pad: if a module only
   yields 3 strong candidates, return 3.
3. **For each candidate**, fill out the schema below. Line numbers must be
   accurate (1-indexed, inclusive). If the optimization unit is a contiguous
   region inside a longer function, give the region's lines, not the whole
   function's.
4. **Validate** that for each candidate you can name (a) a metric, (b) a way
   to measure it, and (c) the correctness invariant. If any of the three is
   missing, drop the candidate.
5. `depends_on` lists other modules' qualified names (e.g.
   `v1/engine/scheduler`), not paths. Use them only as call-shape context; do
   NOT guess dependency paths and do NOT propose candidates outside
   `module.path`.

## Output (REQUIRED — strict JSON, no markdown fence)
Emit ONE JSON object matching the `Candidates` schema enforced by the wrapper:
{
  "module_qualified_name": "{module_qualified_name}",
  "candidates": [
    {
      "id": "cand-0001",
      "file": "<repo-root-relative path under {module_path}>",
      "line_start": <1-indexed inclusive>,
      "line_end":   <1-indexed inclusive, >= line_start>,
      "symbol": "<fully-qualified function/class/method or 'region'>",
      "kind": "function | method | loop | region | kernel | config_block | plugin_seam",
      "description": "What this code does today, in one or two sentences.",
      "current_approach": "Brief summary of the existing implementation/heuristic.",
      "evolve_rationale": "Why this is a good evolutionary target — where the headroom is.",
      "metrics": [
        {
          "name": "e.g. h2d_copy_latency_us (encode the unit in the name)",
          "direction": "minimize | maximize",
          "target_or_baseline": "<current observed value if known, else null>"
        }
      ],
      "estimated_impact": "high | medium | low"
    }
  ]
}

## Rules
- **Verify line numbers** by actually reading the file. Off-by-one or stale
  ranges are not acceptable.
- Every `file` MUST be under `{module_path}` and exist in the checkout. No
  invented files or symbols.
- `id` values must be unique within this list; use `cand-NNNN` zero-padded,
  starting at `cand-0001` and increasing monotonically.
- **Be specific in `metrics`.** "Make it faster" is not a metric;
  `decode_step_latency_ms` measured by `benchmarks/decode.py --batch=32` is.
- **Don't hand-wave correctness.** Every candidate's `evolve_rationale` must
  name a concrete oracle (existing test, golden output, property check, or
  invariant).
- **Prefer fewer, higher-quality entries.** A list of 6 strong candidates
  beats 20 mediocre ones.
- Do NOT explain outside the JSON object.
