# Discovery Phase — Implementation Plan

The discovery phase (the `discovery` state in `ValidationStatus.phase`) produces two artifacts: `TestHarnessMap` and `ValidationWorkloadMatrix`. These feed into the validation plan that determines what tests and benchmarks to run against a change.

This branch implements discovery as an **LLM-driven process using a template prompt**, not a hardcoded static scanner. The approach generalizes across target systems and languages, and incorporates multiple discovery sources including GitHub issues and pull requests.

---

## 0. Existing Foundation

The following are already implemented and inform this plan:

- **Schemas** (`src/spotlights_validation/schemas.py`): `TestHarnessMap`, `TestHarnessEntry`, `ValidationWorkloadMatrix`, `WorkloadEntry`, `ValidationPlan`, `ValidationPlanEntry`, and all result types.
- **MVP examples** (`examples/kvoffload/`, `examples/kvoffload_extended/`): Manually-produced artifacts demonstrating the expected output shape, including entries discovered from GitHub issues/PRs.
- **Runner** (`execution/runner.py`): Consumes the artifacts produced by discovery.
- **Public API contract** (`__init__.py`): `prepare()`, `start_validation()`, `get_validation_status()`.

The goal of this implementation is to automate what the MVP examples do manually: produce `TestHarnessMap`, `ValidationWorkloadMatrix`, and `ValidationPlan` given a target repo and version.

---

## 1. Discovery Prompt Template

**File:** `src/spotlights_validation/discovery/prompt_template.py`

The core of discovery is an LLM prompt that receives structured context about the target system and produces validation artifacts conforming to our schemas. This replaces a hardcoded scanner approach — the LLM reads the repo structure and available sources, then produces entries directly.

### Template placeholders

| Placeholder | Source | Description |
|---|---|---|
| `{target_repo_url}` | CLI argument | GitHub URL of the target repository |
| `{target_version}` | CLI argument or `git rev-parse HEAD` | Commit SHA, tag, or branch to validate against |
| `{source_tree_summary}` | Built at runtime | Directory listing, CI configs, test directories, benchmark scripts |
| `{component_vocabulary}` | From `ProjectTree` if available | Known components of the target system |
| `{candidate_context}` | From Candidate object | The optimization candidate: target file, symbol, kind, anomaly_refs, and evolve_rationale. Used to focus discovery on the candidate's component and inform archive queries. |
| `{change_context}` | Optional, from Change object | Affected components and code paths (for change-specific discovery) |
| `{output_artifacts_path}` | CLI argument | Where to write the produced JSON artifacts |
| `{existing_artifacts}` | Loaded from prior run if present | Base harness map and workload matrix to extend (not duplicate) |
| `{github_discovery_results}` | From GitHub search step | Issues and PRs relevant to the discovery scope |

### Template structure

The prompt instructs the LLM to:

1. **Scan the source tree** for test infrastructure: CI configs (any CI system — not limited to BuildKite), test directories, benchmark scripts, workload configs, `__main__` entry points.
2. **Identify invocation commands** without assuming pytest or Python — support any executable test script, Makefile target, shell script, or language-specific test runner.
3. **Map components** from path heuristics and file content.
4. **Produce structured JSON** conforming to `TestHarnessEntry` and `WorkloadEntry` schemas.

### Output format

The LLM returns JSON matching the `TestHarnessMap` and `ValidationWorkloadMatrix` schemas. The template includes the schema shapes inline as output format instructions.

---

## 2. GitHub Discovery (Issues & PRs)

**File:** `src/spotlights_validation/discovery/github_discovery.py`

Searches GitHub for test cases, workloads, benchmarks, and regression signals relevant to the discovery scope. This extends the base discovery with entries that exist in the project's issue tracker but may not be obvious from the source tree alone.

### Interface

```python
@dataclass
class GitHubDiscoveryResult:
    issues_inspected: list[dict]      # number, url, title
    prs_inspected: list[dict]
    harness_candidates: list[dict]    # raw entries extracted from issues/PRs
    workload_candidates: list[dict]
    skipped: list[dict]               # ref + reason

async def discover_from_github(
    repo_url: str,
    target_version: str,
    scope_keywords: list[str],
    source_tree: Path | None = None,
) -> GitHubDiscoveryResult:
```

### Algorithm

1. **Search** — Query GitHub REST API (or `gh` CLI) for issues and PRs matching scope keywords. Multiple queries for coverage (e.g., component names, feature names, known failure modes).
2. **Filter** — Apply trustworthiness heuristics:
   - Skip issues with no maintainer engagement.
   - Skip authors with no track record of accepted issues.
   - Prefer merged PRs over open issues.
3. **Extract** — For each qualifying issue/PR, extract:
   - Test cases: reproduction scripts, pytest invocations, test file paths.
   - Workloads: model names, request patterns, concurrency levels, sequence lengths.
   - Benchmarks: script invocations, reported metric values, measurement conditions.
   - Regression signals: metric worsening → becomes `halt_on_failure: true` candidate.
   - Environment constraints: hardware requirements, flags.
4. **Version gate** — Verify that referenced paths exist at `target_version` using `git ls-tree`. Entries referencing code that doesn't exist at the target version are omitted.
5. **Deduplicate** — Entries already present in base artifacts get `source_refs` annotation rather than a new entry.

---

## 3. Discovery Orchestrator

**File:** `src/spotlights_validation/discovery/orchestrator.py`

Coordinates the full discovery flow: source tree analysis, GitHub search, LLM-based artifact generation, and verification.

### Interface

```python
async def run_discovery(
    target_repo_url: str,
    target_version: str,
    source_tree: Path,
    output_path: Path,
    candidate: Candidate,
    change_context: dict | None = None,
    existing_artifacts_path: Path | None = None,
) -> DiscoveryOutput:
```

```python
@dataclass
class DiscoveryOutput:
    harness_map: TestHarnessMap
    workload_matrix: ValidationWorkloadMatrix
    discovery_summary: dict             # queries run, entries found, sources
    verification_report: VerificationReport
```

### Algorithm

1. **Build source tree summary** — List CI config files, test directories, benchmark scripts, workload configs, `__main__` executables. Language-agnostic: look for any recognizable test/CI patterns.
2. **Load component vocabulary** — From `ProjectTree` if available, otherwise derive from directory structure.
3. **Extract candidate context** — From the `Candidate` object, extract the target component (`file`, `symbol`, `kind`), `anomaly_refs`, and `evolve_rationale`. This focuses GitHub search keywords and informs plan prioritization.
4. **Run GitHub discovery** (Step 2) — Produce `GitHubDiscoveryResult`. Scope keywords include the candidate's file path, symbol name, and component.
5. **Assemble prompt context** — Fill the template (Step 1) with all gathered context including `{candidate_context}`.
6. **Call LLM** — Send the assembled prompt. Parse structured JSON response into `TestHarnessMap` and `ValidationWorkloadMatrix`.
7. **Verify artifacts** (Step 4) — Run the verification pass on produced entries.
8. **Write outputs** — Save artifacts to `output_path`.

### Concurrency

GitHub search and source tree scanning run in parallel (`asyncio.gather`). The LLM call happens after both complete since it needs their outputs as context.

---

## 4. Artifact Verification

**File:** `src/spotlights_validation/discovery/verification.py`

After artifacts are generated (whether by LLM or manually), verify they are sound before use. This implements the approval process described in the design.

### Checks

```python
@dataclass
class VerificationReport:
    entries_verified: int
    entries_failed: int
    issues: list[VerificationIssue]

@dataclass
class VerificationIssue:
    entry_id: str
    issue_type: Literal["path_missing", "not_runnable", "duplicate", "version_mismatch"]
    detail: str

async def verify_artifacts(
    harness_map: TestHarnessMap,
    workload_matrix: ValidationWorkloadMatrix,
    source_tree: Path,
    target_version: str,
) -> VerificationReport:
```

### Verification rules

1. **Path existence** — Every `TestHarnessEntry.path` must exist at `source_tree` at `target_version`. Use `git ls-tree` for version-gated verification, or filesystem check for local trees.
2. **Runnability** — The `invoke` command references an executable that exists (test runner binary, script file, Makefile target). Does not execute the command, only verifies the entry point.
3. **Duplicate detection** — No two entries run the same underlying check via different wrappers. Compare by normalized invoke command and path.
4. **Version match** — `harness_map.target_version` and `workload_matrix.target_version` match the expected `target_version`.
5. **Candidate relevance** — Each entry must be relevant to the `Candidate` (covers the candidate's component, file, or symbol). If a `Change` object is available, entries must also be relevant to the change's affected code paths. Entries with no demonstrable connection to the candidate (or change) are flagged as irrelevant.

Entries that fail verification are flagged but not automatically removed — the caller decides whether to filter them out or surface them for manual review.

---

## 5. Validation Plan Creation

**File:** `src/spotlights_validation/discovery/plan_builder.py`

Takes verified discovery artifacts and produces a `ValidationPlan`. This step bridges discovery and execution.

### Interface

```python
async def build_validation_plan(
    harness_map: TestHarnessMap,
    workload_matrix: ValidationWorkloadMatrix,
    archive_context: list[str] | None = None,
    change: Change | None = None,
) -> ValidationPlan:
```

### Strategy

The plan is always prioritized based on the candidate's component (from `Candidate.file` and `Candidate.symbol`):
- Entries covering the candidate's component get higher priority.
- Entries derived from regression reports on that component get `halt_on_failure: true`.
- Workloads exercising the candidate's component are paired with relevant harness entries.

When `change` is additionally provided (Change Validation Discovery phase), the plan is further refined based on the specific code paths affected by the change — which may be narrower than the candidate's component scope.

When neither candidate component matching nor change context applies to an entry, it receives default priority ordering by kind: correctness > unit > integration > benchmark > stress.

---

## 6. Caching Layer

**File:** `src/spotlights_validation/discovery/cache.py`

Prevents re-running expensive discovery (especially LLM calls and GitHub API queries) when the target hasn't changed.

```python
def load_discovery_cache(
    source_tree: Path,
    target_version: str,
    cache_dir: Path,
) -> DiscoveryOutput | None:

def write_discovery_cache(
    output: DiscoveryOutput,
    target_version: str,
    cache_dir: Path,
) -> None:
```

**Cache key:** `{cache_dir}/{target_version}/`. A different `target_version` = cache miss.

**Default `cache_dir`:** `source_tree / ".spotlights_validation_cache"`.

---

## 7. CLI Integration

**File:** `src/spotlights_validation/cli.py` (extend existing)

```bash
# Full automated discovery
python -m spotlights_validation.cli discover \
    --source-tree /path/to/target \
    --target-version v0.18.0 \
    --repo-url https://github.com/vllm-project/vllm \
    --candidate artifacts/candidate.json \  # Candidate from Bundle C
    --out-dir artifacts/ \
    --include-github               # opt-in to GitHub issue/PR discovery
    --scope-keywords "kv offload,cpu offload,swap"  # narrows GitHub search

# Plan creation from existing artifacts
python -m spotlights_validation.cli plan \
    --harness-map artifacts/harness_map.json \
    --workload-matrix artifacts/workload_matrix.json \
    --candidate artifacts/candidate.json \
    --change-repo /path/to/change \
    --out artifacts/validation_plan.json
```

---

## 8. Wiring into `prepare()`

**File:** `src/spotlights_validation/__init__.py`

```python
async def prepare(source_tree: Path, candidate: Candidate, artifacts_dir: Path | None = None) -> PreparationRun:
```

1. If `artifacts_dir` is provided and contains valid cached artifacts, load them (MVP path — manual artifacts).
2. Otherwise, run `run_discovery()` with the `candidate` to produce artifacts live. The candidate's component info focuses GitHub search and informs prioritization.
3. Run `build_validation_plan()` to produce the base plan, prioritized toward the candidate's component.
4. Store `ValidationPreparation` (including the `candidate`) with all outputs.
5. Return `PreparationRun` immediately; background task handles the work.

Phase transitions: `discovery → planning` after both harness map and workload matrix are produced and verified.

---

## 9. Tests

### Unit tests (no real target repo, no real LLM, no network)

| File | What it tests |
|---|---|
| `tests/unit/test_prompt_template.py` | Template rendering with various placeholder combinations. Output prompt contains all expected sections. |
| `tests/unit/test_verification.py` | Verification logic against fixture artifacts with known issues (missing paths, duplicates). |
| `tests/unit/test_plan_builder.py` | Plan prioritization logic: with/without Change, halt_on_failure assignment, component matching. |
| `tests/unit/test_cache.py` | Cache write → read round-trip. Cache miss on different `target_version`. |
| `tests/unit/test_schemas.py` | `TestHarnessMap.for_components()`, `of_kind()`, `to_json()`/`from_json()` round-trip. |

### Integration tests (real target repo, mocked LLM)

| File | What it tests |
|---|---|
| `tests/integration/test_discovery_orchestrator.py` | `run_discovery()` against a real source tree produces valid `TestHarnessMap` with ≥3 entries covering distinct kinds. |
| `tests/integration/test_github_discovery.py` | `discover_from_github()` returns structured candidates. Uses recorded API responses (VCR/cassettes) to avoid live API calls in CI. |
| `tests/integration/test_verification.py` | Verification against real source tree catches intentionally invalid paths. |
| `tests/integration/test_caching.py` | Second call with same `target_version` returns cached result without LLM calls. |

Integration tests read `target_repo_path` from a `TARGET_REPO_PATH` env var and skip if not set.

---

## 10. Design Decisions

| Question | Decision |
|---|---|
| Hardcoded scanners vs. LLM-driven discovery | LLM-driven via template prompt. Generalizes across target systems and languages; not limited to Python/pytest or vLLM-specific CI (BuildKite). |
| GitHub issues/PRs as discovery source | First-class source with trustworthiness filtering and version gating. Extends base discovery with regression signals and reproduction scripts from community reports. |
| Discovery scope (Python/pytest only?) | No. The template prompt and verification are language-agnostic. Support any executable: shell scripts, Makefile targets, `__main__` scripts, language-specific test runners. |
| `__main__` scripts as test harness entries | Yes, included as a valid invocation pattern. The `invoke` field can be `python path/to/script.py` or any other command. |
| Artifact verification | Automated check after generation; issues reported but not auto-removed. Human-in-the-loop review possible via CLI output. |
| Component vocabulary source | `ProjectTree` from `modules_extractor` when available; falls back to directory-structure inference. |
| Cache invalidation | On `target_version` change (commit SHA). Same version = cache hit. |

---

## Implementation Order

Steps are sequentially dependent but each is independently deliverable:

1. **Prompt template** (Step 1) — Defines the discovery contract; no dependencies.
2. **Verification** (Step 4) — Testable with fixture artifacts; no LLM or network needed.
3. **GitHub discovery** (Step 2) — Independent module; testable with recorded responses.
4. **Plan builder** (Step 5) — Depends only on schemas; testable with fixture data.
5. **Orchestrator** (Step 3) — Assembles 1–4 into the full flow.
6. **Caching** (Step 6) — Wraps the orchestrator.
7. **CLI integration** (Step 7) — Thin layer on top of orchestrator.
8. **`prepare()` wiring** (Step 8) — Connects discovery to the public API.
