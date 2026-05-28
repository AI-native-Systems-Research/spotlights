# Discovery Phase — Implementation Plan

The discovery phase (the `discovery` state in `ValidationStatus.phase`) produces two artifacts: `TestHarnessMap` and `ValidationWorkloadMatrix`. Both run inside `prepare()` before any change arrives.

---

## 0. Package Bootstrap

**Deliverable:** A working Python package that can be imported.

- Create `pyproject.toml` following the same structure as `spotlights-engine`: `hatchling` builder, `src/` layout, `pydantic>=2` + `spotlights-engine` as runtime deps, `anthropic` for LLM calls, `pytest`/`ruff`/`mypy` as dev deps.
- `src/spotlights_validation/__init__.py` — exports `prepare`, `start_validation`, `get_validation_status` as stubs (raise `NotImplementedError`) so the public contract exists before the internals are filled in.
- `src/spotlights_validation/schemas.py` — all Pydantic models (see Step 1).

---

## 1. Schema Layer

**File:** `src/spotlights_validation/schemas.py`

Define all models needed by discovery. Follow the `ProjectTree` pattern from `spotlights_engine/schemas/modules.py` exactly — Pydantic `BaseModel`, `to_json(path)`, `from_json(path)`, and filter helpers.

```python
class TestHarnessEntry(BaseModel):
    id: str                        # stable slug, e.g. "buildkite-basic-correctness"
    name: str                      # human label
    kind: Literal["unit", "integration", "benchmark", "correctness", "stress"]
    path: str                      # repo-relative path to script or test dir
    invoke: str                    # full shell command
    components: list[str]          # same vocabulary as TraceSummary.top_components
    output_format: Literal["pytest-json", "custom", "exit-code-only"]
    estimated_duration: int | None # seconds

class TestHarnessMap(BaseModel):
    target_version: str
    entries: list[TestHarnessEntry] = Field(default_factory=list)

    def for_components(self, components: list[str]) -> "TestHarnessMap": ...
    def of_kind(self, kind: str) -> "TestHarnessMap": ...
    def to_json(self, path: Path) -> None: ...
    @classmethod
    def from_json(cls, path: Path) -> "TestHarnessMap": ...

class WorkloadEntry(BaseModel):
    workload_id: str
    workload_class: Literal["agentic", "batch-inference", "long-context", "mixed", "stress"]
    source: Literal["discovered", "curated"]
    config_path: str | None        # repo-relative, None for curated entries
    components_exercised: list[str]

class ValidationWorkloadMatrix(BaseModel):
    target_version: str
    workloads: list[WorkloadEntry] = Field(default_factory=list)

    def for_components(self, components: list[str]) -> "ValidationWorkloadMatrix": ...
    def to_json(self, path: Path) -> None: ...
    @classmethod
    def from_json(cls, path: Path) -> "ValidationWorkloadMatrix": ...
```

`ExecutionResult`, `ValidationPlanEntry`, `ValidationPlan`, `ValidationPreparation`, `PreparationRun`, `ValidationRun`, `ValidationStatus`, and `ValidationResult` also live here — define them all now as stubs so the type system is consistent from day one.

---

## 2. Static Scanner

**File:** `src/spotlights_validation/discovery/_scanner.py`

Pure filesystem scanning — no LLM, no I/O beyond reading files. Returns raw unclassified candidates; classification happens later. Each function is independently testable with synthetic fixture trees.

```python
@dataclass
class RawHarnessCandidate:
    source: str       # "buildkite", "pytest", "script"
    name: str
    path: str
    invoke: str
    inferred_kind: str | None   # inferred heuristically, None = unknown

@dataclass
class RawWorkloadCandidate:
    config_path: str
    parameters: dict[str, Any]   # raw parsed YAML/JSON content
    inferred_class: str | None
```

Four scanners:

### 2a. BuildKite CI scanner (primary for vLLM)

- Walk `.buildkite/test_areas/*.yaml`. For each file, extract the YAML `group` (name) and each step's `command` list. This produces ~30 candidates directly mapping to BuildKite test areas (e.g., `basic_correctness.yaml` → `"pytest -v -s basic_correctness/"`).
- Key detail: BuildKite YAML uses `commands:` or `command:` keys; the scanner normalizes both.

### 2b. Benchmark script scanner

- Walk `benchmarks/*.py` (not subdirs). For each file, the invoke is `python3 {path}`. Inferred kind: `benchmark`.
- Walk `benchmarks/attention_benchmarks/configs/*.yaml`. Each config is a separate entry; invoke is `python3 benchmarks/attention_benchmarks/benchmark.py --config {config_path}`.

### 2c. Pytest root scanner (fallback / cross-check)

- Check for `pyproject.toml [tool.pytest.ini_options]` — extract `testpaths` if set, otherwise default to `tests/`.
- Walk `tests/` one level deep; each subdirectory with a `conftest.py` or `test_*.py` files becomes a candidate. Assign inferred kind from directory name: `unit` if name contains "unit", `integration` if "integration", `distributed` if "distributed", else `None`.
- This produces coarser entries (whole subdirs) as fallback when BuildKite entries are not available.

### 2d. Workload config scanner

- Walk `benchmarks/attention_benchmarks/configs/*.yaml` → parse each, capture batch_specs and backends.
- Walk `benchmarks/multi_turn/*.json` → multi-turn workload configs.
- Check `benchmarks/benchmark_latency.py`, `benchmark_throughput.py`, `benchmark_long_document_qa_throughput.py` for argparse defaults (`request_rate`, `max_tokens`) via a lightweight regex scan of the file (no import/exec).
- Infer workload class by filename keyword:
  - `throughput` → `batch-inference`
  - `long_document` / `long_context` → `long-context`
  - `multi_turn` / `agentic` → `agentic`
  - mixed batch specs (prefill+decode together) → `mixed`
  - `stress` / `adversarial` → `stress`
  - fallback → `None` (to be classified by LLM)

**Unit tests for Step 2:** Create a minimal fixture tree under `tests/fixtures/fake_vllm/` with a handful of fake `.yaml` and `.py` files. Assert the scanner produces the expected `RawHarnessCandidate` list without any real filesystem access to the actual vLLM repo.

---

## 3. Component Mapper (LLM pass)

**File:** `src/spotlights_validation/discovery/_component_mapper.py`

Takes raw candidates + a `ProjectTree` (from `modules_extractor`) and assigns the `components` field to each entry. Pure path heuristics can't reliably map `tests/kernels/` to `["attention_backend", "scheduler"]` without understanding the code structure; the `ProjectTree` provides the component vocabulary and the LLM does the mapping.

**Interface:**

```python
def map_components(
    candidates: list[RawHarnessCandidate],
    project_tree: ProjectTree,
    llm_client: anthropic.Anthropic,
) -> dict[str, list[str]]:  # candidate.name → list[component_name]
```

**Strategy:**

1. Build the component vocabulary by calling `project_tree.walk()` — get all module names as the allowed component set.
2. Batch candidates into groups of ~10 (to keep prompt size manageable).
3. For each batch, construct a prompt:
   - System: "You are mapping test scripts to software components. Respond only with JSON."
   - User: "Given these vLLM components: `{component_list}`. For each test entry, list which components it primarily tests. If none match, return an empty list."
   - Include each candidate's `name`, `path`, `invoke` as context.
4. Parse the structured JSON response. Use `claude-sonnet-4-6` with `max_tokens=1024`.
5. Fallback: if a candidate's path contains a directory name that exactly matches a component name, use that as the component without an LLM call.

**Caching:** Cache the component mapping alongside the `TestHarnessMap` (same cache key). Don't re-run the LLM pass if the cache is valid.

**Unit tests:** Mock `anthropic.Anthropic` and a fixture `ProjectTree`. Verify the function returns the expected component lists and handles partial/invalid LLM responses gracefully (return `[]` on parse failure, log a warning).

---

## 4. Harness Discovery

**File:** `src/spotlights_validation/discovery/harness_discovery.py`

Assembles Steps 2 and 3 into the public internal function.

```python
async def discover_test_harness(source_tree: Path, target_version: str) -> TestHarnessMap:
```

**Algorithm:**

1. Run all four static scanners from Step 2 against `source_tree`. Deduplicate by invoke command.
2. Heuristically assign `output_format`:
   - invoke contains `pytest` → `pytest-json`
   - invoke contains `python3 benchmarks/` → `custom`
   - shell scripts (`.sh`) → `exit-code-only`
3. Heuristically estimate `estimated_duration` from BuildKite YAML `timeout_in_minutes` field if present; otherwise use kind-based defaults (unit: 300s, integration: 900s, benchmark: 1800s, stress: None).
4. Call `map_components()` from Step 3 to fill in the `components` field. Pass `ProjectTree` loaded from `modules_extractor` output (or derive it on the fly if not cached).
5. Assign stable `id` slugs: lowercase, hyphenated, unique. E.g., `buildkite-basic-correctness`, `benchmark-throughput`, `attention-benchmark-mla-decode`.
6. Assemble and return `TestHarnessMap(target_version=target_version, entries=[...])`.

**Concurrency:** The LLM batch calls in Step 4 are parallelized with `asyncio.gather`. The function is `async` to allow this; the state boundary blocking happens in `prepare()`.

---

## 5. Workload Discovery

**File:** `src/spotlights_validation/discovery/workload_discovery.py`

```python
async def seed_workload_matrix(source_tree: Path, target_version: str) -> ValidationWorkloadMatrix:
```

**Algorithm:**

1. Run workload config scanner from Step 2d.
2. For candidates with `inferred_class = None`, run a lightweight LLM classification call (single batch): "Given these benchmark files and their parameters, classify each into one of: agentic, batch-inference, long-context, mixed, stress."
3. Map `components_exercised` using heuristics first:
   - attention backend configs → `["attention_backend"]`
   - throughput scripts → `["scheduler", "engine"]`
   - long-context scripts → `["kv_cache", "chunked_prefill"]`
   - Use `map_components()` for any that aren't obvious from name.
4. Add curated entries for workload classes that must exist but aren't discovered. For vLLM Stage 1: curated entries for `agentic` (multi-turn serving) and `batch-inference` (offline batch throughput) ensure the Stage 1 minimum of 2 distinct classes.
5. Assemble and return `ValidationWorkloadMatrix(target_version=target_version, workloads=[...])`.

---

## 6. Caching Layer

**File:** `src/spotlights_validation/discovery/_cache.py`

Prevents re-running expensive discovery (especially LLM calls) when the target hasn't changed.

```python
def load_discovery_cache(
    source_tree: Path,
    target_version: str,
    cache_dir: Path,
) -> tuple[TestHarnessMap, ValidationWorkloadMatrix] | None:
    # Returns None on cache miss

def write_discovery_cache(
    harness_map: TestHarnessMap,
    workload_matrix: ValidationWorkloadMatrix,
    target_version: str,
    cache_dir: Path,
) -> None:
```

**Cache structure:** `{cache_dir}/{sha256(str(source_tree))}/{target_version}/harness_map.json` and `workload_matrix.json`. A different `target_version` = cache miss. Uses `TestHarnessMap.to_json()` / `from_json()`.

**Default `cache_dir`:** `source_tree / ".spotlights_validation_cache"` — lives inside the target repo, gitignored.

---

## 7. Wiring into `prepare()`

**File:** `src/spotlights_validation/__init__.py`

```python
async def prepare(source_tree: Path) -> PreparationRun:
```

1. Generate `prep_id = uuid4().hex`.
2. Determine `target_version` by running `git -C source_tree rev-parse HEAD` via `asyncio.create_subprocess_exec`.
3. Launch a background `asyncio.Task` that:
   - a. Checks the cache (Step 6). If hit, skip to (d).
   - b. Calls `discover_test_harness()` and `seed_workload_matrix()` concurrently via `asyncio.gather`.
   - c. Writes to cache.
   - d. Queries Bundle B for archive context (stub for Stage 1: returns empty list).
   - e. Calls `build_validation_plan()` (planning step, out of scope for this document).
   - f. Stores `ValidationPreparation` inside the `PreparationRun` object.
4. Return `PreparationRun(prep_id=prep_id)` immediately.
5. `PreparationRun.result()` awaits the background task and returns `ValidationPreparation`.

**Phase transitions:** The background task updates a `ValidationStatus` object stored in an in-process dict keyed by `prep_id`. Phase advances `discovery → planning` after both `discover_test_harness` and `seed_workload_matrix` complete.

---

## 8. Tests

### Unit tests (no real vLLM repo, no real LLM)

| File | What it tests |
|---|---|
| `tests/unit/test_scanner.py` | Static scanner against `tests/fixtures/fake_vllm/` fixture tree. Assert correct candidate count, invoke strings, inferred kinds. |
| `tests/unit/test_component_mapper.py` | `map_components()` with a mocked Anthropic client and a minimal fixture `ProjectTree`. Assert correct batching, handling of parse errors. |
| `tests/unit/test_cache.py` | Cache write → read round-trip. Assert cache miss on different `target_version`. |
| `tests/unit/test_schemas.py` | `TestHarnessMap.for_components()`, `of_kind()`, `to_json()`/`from_json()` round-trip. |

### Integration tests (real vLLM repo, mocked LLM)

| File | What it tests |
|---|---|
| `tests/integration/test_harness_discovery.py` | `discover_test_harness(vllm_path, "HEAD")` produces a `TestHarnessMap` with ≥3 entries covering distinct kinds (unit, integration, benchmark). Verifies Stage 1 success criterion. |
| `tests/integration/test_workload_discovery.py` | `seed_workload_matrix(vllm_path, "HEAD")` produces ≥2 distinct workload classes. |
| `tests/integration/test_caching.py` | Second call with same `target_version` returns cached result without LLM calls (verify by asserting the mock LLM was called 0 times on second run). |

Integration tests read `vllm_path` from a `VLLM_REPO_PATH` env var and skip if not set.

---

## 9. Open Question Resolutions for Discovery

| Question | Decision |
|---|---|
| Static analysis vs. LLM for component mapping | Hybrid: static scanner always runs first (no LLM cost), LLM fills `components` field only. Makes the scanner testable without any LLM dependency. |
| Component vocabulary source | Use `ProjectTree` from `modules_extractor` output. If it hasn't run for the target version, run it as part of `prepare()` before the LLM pass. |
| Harness discovery refresh frequency | Invalidate on `target_version` change (commit SHA). The cache check in Step 6 handles this automatically. |
| vLLM-specific workload abstraction | Keep `workload_class` as a `Literal` enum in `WorkloadEntry` for v1. The `curated` source value is the escape hatch for classes not discoverable from repo structure. |
| Custom output format parsing | Out of scope for discovery — `output_format` in `TestHarnessEntry` declares intent; the runner implements parsers. Discovery only needs to correctly classify which format applies. |

---

## Implementation Order

Steps are sequentially dependent but each is independently deliverable:

1. **Schema layer** (Step 1) — unblocks everything else; no dependencies.
2. **Static scanner** (Step 2) — unblocks harness + workload discovery; fully testable with fixtures.
3. **Component mapper** (Step 3) — unblocks component assignment; testable with mock LLM.
4. **Harness discovery** (Step 4) + **Workload discovery** (Step 5) in parallel — both depend on 2 and 3.
5. **Caching** (Step 6) — wraps 4 and 5.
6. **`prepare()` wiring** (Step 7) — thin orchestration layer on top of 4–6.
7. **Tests** — written alongside each step, not after.
