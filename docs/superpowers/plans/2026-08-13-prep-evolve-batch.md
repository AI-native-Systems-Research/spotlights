# prep-evolve batch-first CLI — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Simplify `prep-evolve` so `--result` accepts a run directory, `--candidate`/`--module`/`--out` become optional (candidate omitted = all candidates), and bundles land in a uniform `<base>/evolve/<module>/<candidate>/<evolver>/` tree.

**Architecture:** Keep the existing extract → validate → render → materialize pipeline. Add two pure resolver helpers (a result-path locator and a candidate-id scanner), rewrite the `prep_evolve` orchestrator to loop over a selection (one candidate, or all) with per-candidate skip-and-warn resilience, and replace the flat bundle-dir name with a nested path built from `slug_for(qn)`.

**Tech Stack:** Python 3, Pydantic v2 models, `argparse` CLI, `pytest`. Existing module: `src/spotlights_engine/prep_evolve/`.

## Global Constraints

- **Design spec:** `docs/superpowers/specs/2026-08-13-prep-evolve-batch-design.md` — all behavior below is defined there.
- **Module slug:** the `<module>` path segment MUST be `slug_for(qn)` from `spotlights_engine.utils.id_helpers` (slash-form qn with any char outside `[A-Za-z0-9._-]` → `_`), matching the existing `modules/<slug>.md` convention. Never hand-roll the flattening.
- **Bundle layout (always):** `<base>/evolve/<slug_for(qn)>/<candidate_id>/<evolver>/`. Base = `--out` if given, else the run directory (the folder containing `result.json`).
- **Bundle contents unchanged:** skydiscover = `seed.<ext>` + `config.yaml` + `README.md`; coral = `task.yaml` + `README.md`; nous = `campaign.yaml` (minimal, no README). Do not alter adapters.
- **Errors:** raise the existing `PrepEvolveError` subclasses in `errors.py`. `SelectionError` for bad/missing selections, `RepoResolutionError` for repo issues, `StalenessError` from `validate_target`.
- **Run tests with:** `python -m pytest tests/unit/prep_evolve/ -q` from the repo root.

---

## File Structure

- `src/spotlights_engine/prep_evolve/resolve.py` — add `ResultLocation` + `resolve_result_location`, `CandidateSelection` + `find_candidate` + `iter_candidates`. (Existing resolvers stay.)
- `src/spotlights_engine/prep_evolve/api.py` — rewrite `prep_evolve`; add `_bundle_dir`; extend `SkippedEvolver`; make `PrepEvolveInput.module/candidate/out` optional; drop `_bundle_dir_name`/`_path_segment`/`_candidate_id`/`_BUNDLE_SEGMENT_RE`; simplify `_materialize`.
- `src/spotlights_engine/prep_evolve/cli.py` — make `--module`/`--candidate`/`--out` optional; add batch summary output.
- `tests/unit/prep_evolve/test_extract_resolve.py` — add resolver tests (Tasks 1–2).
- `tests/unit/prep_evolve/test_materialize_cli.py` — add new-behavior tests and update path/skip-dependent tests (Tasks 3–4).
- `docs/prep-evolve.md` and `README.md` — usage updates (Task 5).

---

## Task 1: Result-path locator (directory-or-file `--result`)

**Files:**
- Modify: `src/spotlights_engine/prep_evolve/resolve.py`
- Test: `tests/unit/prep_evolve/test_extract_resolve.py`

**Interfaces:**
- Produces: `ResultLocation(result_json: Path, run_dir: Path, index: Path | None)` and `resolve_result_location(result: Path) -> ResultLocation`. When `result` is a directory, `result_json = result/"result.json"` (SelectionError if absent) and `run_dir = result`; when a file, `result_json = result` and `run_dir = result.parent`; `index` is `run_dir/"index.md"` when that file exists, else `None`. Nonexistent `result` → SelectionError.

- [ ] **Step 1: Write the failing tests**

Add to `tests/unit/prep_evolve/test_extract_resolve.py`:

```python
from pathlib import Path

import pytest

from spotlights_engine.prep_evolve.errors import SelectionError
from spotlights_engine.prep_evolve.resolve import resolve_result_location

from . import _fixtures as fx


def test_resolve_result_location_directory(tmp_path: Path) -> None:
    fx.write_result(tmp_path)          # writes tmp_path/result.json
    fx.write_index(tmp_path, tmp_path)  # writes tmp_path/index.md
    loc = resolve_result_location(tmp_path)
    assert loc.result_json == tmp_path / "result.json"
    assert loc.run_dir == tmp_path
    assert loc.index == tmp_path / "index.md"


def test_resolve_result_location_file_no_index(tmp_path: Path) -> None:
    rj = fx.write_result(tmp_path)
    loc = resolve_result_location(rj)
    assert loc.result_json == rj
    assert loc.run_dir == tmp_path
    assert loc.index is None  # no index.md written


def test_resolve_result_location_dir_missing_result_json(tmp_path: Path) -> None:
    with pytest.raises(SelectionError):
        resolve_result_location(tmp_path)  # empty dir


def test_resolve_result_location_missing_path(tmp_path: Path) -> None:
    with pytest.raises(SelectionError):
        resolve_result_location(tmp_path / "nope")
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/unit/prep_evolve/test_extract_resolve.py -q -k resolve_result_location`
Expected: FAIL with `ImportError: cannot import name 'resolve_result_location'`.

- [ ] **Step 3: Implement `ResultLocation` + `resolve_result_location`**

In `resolve.py`, after the `LoadedResult` dataclass, add:

```python
@dataclass
class ResultLocation:
    """Where a run's artifacts live, resolved from a directory or file `--result`."""

    result_json: Path
    run_dir: Path
    index: Path | None


def resolve_result_location(result: Path) -> ResultLocation:
    """Resolve `--result` (a run directory or a `result.json` file).

    A directory must contain `result.json`; the run dir is the directory itself.
    A file is used directly; the run dir is its parent. `index` points at a
    sibling `index.md` when present (the `--repo` fallback), else `None`.
    """
    result = Path(result).expanduser()
    if result.is_dir():
        result_json = result / "result.json"
        if not result_json.is_file():
            raise SelectionError(f"no result.json in directory: {result}")
        run_dir = result
    elif result.is_file():
        result_json = result
        run_dir = result.parent
    else:
        raise SelectionError(f"--result path not found: {result}")

    index = run_dir / "index.md"
    return ResultLocation(
        result_json=result_json,
        run_dir=run_dir,
        index=index if index.is_file() else None,
    )
```

Add `"ResultLocation"` and `"resolve_result_location"` to `resolve.py`'s `__all__`.

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/unit/prep_evolve/test_extract_resolve.py -q -k resolve_result_location`
Expected: PASS (4 passed).

- [ ] **Step 5: Commit**

```bash
git add src/spotlights_engine/prep_evolve/resolve.py tests/unit/prep_evolve/test_extract_resolve.py
git commit -m "feat(prep-evolve): resolve --result from a run directory or file"
```

---

## Task 2: Candidate-id scanner (drop `--module` dependency + batch enumeration)

**Files:**
- Modify: `src/spotlights_engine/prep_evolve/resolve.py`
- Test: `tests/unit/prep_evolve/test_extract_resolve.py`

**Interfaces:**
- Consumes: `LoadedResult` (from `load_result`), `Candidates` (from `schemas.candidate`).
- Produces:
  - `CandidateSelection(qn: str, run: dict, candidate: Candidate)` — a resolved selection carrying the module qn, its raw run dict, and the validated candidate.
  - `find_candidate(loaded: LoadedResult, candidate_id: str) -> CandidateSelection` — scans every module run for a candidate whose `id` matches; SelectionError if none.
  - `iter_candidates(loaded: LoadedResult) -> list[CandidateSelection]` — every candidate across all runs, in `module_runs` order then per-module candidate order. Runs with no/invalid `candidates` block are skipped (tolerated).

- [ ] **Step 1: Write the failing tests**

Add to `tests/unit/prep_evolve/test_extract_resolve.py`:

```python
from spotlights_engine.prep_evolve.resolve import (
    find_candidate,
    iter_candidates,
    load_result,
)


def _loaded_with_second_candidate(tmp_path: Path):
    """A LoadedResult whose single module run has two candidates."""
    payload = fx.make_result_dict()
    run = payload["module_runs"]["v1/attention"]
    second = dict(run["candidates"]["candidates"][0])
    second["id"] = "cand-v1_attention-0003"
    run["candidates"]["candidates"].append(second)
    result_json = tmp_path / "result.json"
    result_json.write_text(__import__("json").dumps(payload), encoding="utf-8")
    return load_result(result_json)


def test_find_candidate_locates_module(tmp_path: Path) -> None:
    loaded = load_result(fx.write_result(tmp_path))
    sel = find_candidate(loaded, "cand-v1_attention-0002")
    assert sel.qn == "v1/attention"
    assert sel.candidate.id == "cand-v1_attention-0002"
    assert isinstance(sel.run, dict)


def test_find_candidate_missing(tmp_path: Path) -> None:
    loaded = load_result(fx.write_result(tmp_path))
    with pytest.raises(SelectionError):
        find_candidate(loaded, "cand-does-not-exist-9999")


def test_iter_candidates_enumerates_all(tmp_path: Path) -> None:
    loaded = _loaded_with_second_candidate(tmp_path)
    ids = [s.candidate.id for s in iter_candidates(loaded)]
    assert ids == ["cand-v1_attention-0002", "cand-v1_attention-0003"]
    assert all(s.qn == "v1/attention" for s in iter_candidates(loaded))
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/unit/prep_evolve/test_extract_resolve.py -q -k "find_candidate or iter_candidates"`
Expected: FAIL with `ImportError: cannot import name 'find_candidate'`.

- [ ] **Step 3: Implement the scanner**

In `resolve.py`, add near the other resolvers:

```python
@dataclass
class CandidateSelection:
    """A resolved (module qn, raw run, candidate) triple."""

    qn: str
    run: dict
    candidate: Candidate


def _candidates_or_none(run: dict) -> Candidates | None:
    """Validate a run's `candidates` block, tolerating absent/invalid blocks.

    Batch enumeration must not abort on one module whose discovery failed, so a
    missing or non-validating block yields `None` (the run is skipped) rather
    than raising.
    """
    raw = run.get("candidates")
    if not raw:
        return None
    try:
        return Candidates.model_validate(raw)
    except Exception:  # noqa: BLE001 - tolerate one bad run during enumeration
        return None


def find_candidate(loaded: LoadedResult, candidate_id: str) -> CandidateSelection:
    """Locate a candidate by id across all module runs (module is inferred)."""
    for qn, run in loaded.module_runs_raw.items():
        if not isinstance(run, dict):
            continue
        cands = _candidates_or_none(run)
        if cands is None:
            continue
        for cand in cands.candidates:
            if cand.id == candidate_id:
                return CandidateSelection(qn=qn, run=run, candidate=cand)
    raise SelectionError(
        f"candidate {candidate_id!r} not found in any module run of result.json"
    )


def iter_candidates(loaded: LoadedResult) -> list[CandidateSelection]:
    """Every candidate across all module runs, in document order."""
    out: list[CandidateSelection] = []
    for qn, run in loaded.module_runs_raw.items():
        if not isinstance(run, dict):
            continue
        cands = _candidates_or_none(run)
        if cands is None:
            continue
        for cand in cands.candidates:
            out.append(CandidateSelection(qn=qn, run=run, candidate=cand))
    return out
```

Add `"CandidateSelection"`, `"find_candidate"`, `"iter_candidates"` to `__all__`.

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/unit/prep_evolve/test_extract_resolve.py -q -k "find_candidate or iter_candidates"`
Expected: PASS (3 passed).

- [ ] **Step 5: Commit**

```bash
git add src/spotlights_engine/prep_evolve/resolve.py tests/unit/prep_evolve/test_extract_resolve.py
git commit -m "feat(prep-evolve): locate/enumerate candidates by id, inferring the module"
```

---

## Task 3: Rewrite the orchestrator (optional inputs, nested layout, single + batch, skip-and-warn)

**Files:**
- Modify: `src/spotlights_engine/prep_evolve/api.py`
- Test: `tests/unit/prep_evolve/test_materialize_cli.py`

**Interfaces:**
- Consumes: `resolve_result_location`, `find_candidate`, `iter_candidates`, `CandidateSelection` (Tasks 1–2); existing `load_result`, `resolve_repo_path`, `resolve_module`, `resolve_findings`, `ensure_repo_dir`, `validate_candidate_target`, `validate_scope_file`, `capture_revision`, `build_spec`, `infer_direction`, `build_adapter`, `normalize_evolver`, `slug_for`.
- Produces:
  - `PrepEvolveInput` with `result: Path`, `module: str | None = None`, `candidate: str | None = None`, `out: Path | None = None` (other fields unchanged).
  - `SkippedEvolver(evolver: str, reason: str, candidate_id: str | None = None, module_qualified_name: str | None = None)`.
  - `_bundle_dir(base: Path, qn: str, candidate_id: str, evolver: str) -> Path` returning `base/"evolve"/slug_for(qn)/candidate_id/evolver`.
  - `_materialize(bundle_path: Path, files: list[GeneratedFile]) -> list[str]` (no `force` param; keeps the path-escape guard; always writes/overwrites).
  - `prep_evolve(input, config) -> PrepEvolveResult` — single candidate when `input.candidate` set (loud errors), all candidates when `None` (per-candidate skip-and-warn).

- [ ] **Step 1: Write the failing tests**

Add these to `tests/unit/prep_evolve/test_materialize_cli.py`:

```python
def test_new_bundle_layout(tmp_path: Path) -> None:
    repo = fx.make_repo(tmp_path)
    out = tmp_path / "bundles"
    result = prep_evolve(_input(tmp_path, repo, evolver="coral", out=out), _CFG)
    bundle = Path(result.bundles[0].path)
    assert bundle == out / "evolve" / "v1_attention" / "cand-v1_attention-0002" / "coral"


def test_out_defaults_to_run_dir(tmp_path: Path) -> None:
    repo = fx.make_repo(tmp_path)
    result_json = fx.write_result(tmp_path)
    result = prep_evolve(
        _input(tmp_path, repo, result=result_json, evolver="coral", out=None), _CFG
    )
    bundle = Path(result.bundles[0].path)
    # run dir is tmp_path (the folder holding result.json)
    assert bundle == tmp_path / "evolve" / "v1_attention" / "cand-v1_attention-0002" / "coral"


def test_result_directory_resolves_repo_from_index(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    fx.write_result(run_dir)          # run/result.json
    repo = fx.make_repo(tmp_path)
    fx.write_index(run_dir, repo)      # run/index.md carrying Repo path
    result = prep_evolve(
        _input(tmp_path, repo, result=run_dir, repo=None, evolver="coral", out=None),
        _CFG,
    )
    bundle = Path(result.bundles[0].path)
    assert bundle == run_dir / "evolve" / "v1_attention" / "cand-v1_attention-0002" / "coral"


def test_module_inferred_when_omitted(tmp_path: Path) -> None:
    repo = fx.make_repo(tmp_path)
    result = prep_evolve(_input(tmp_path, repo, evolver="coral", module=None), _CFG)
    assert result.bundles[0].evolver == "coral"


def test_module_mismatch_errors(tmp_path: Path) -> None:
    repo = fx.make_repo(tmp_path)
    with pytest.raises(SelectionError):
        prep_evolve(_input(tmp_path, repo, evolver="coral", module="v1/nope"), _CFG)


def test_batch_all_candidates(tmp_path: Path) -> None:
    repo = fx.make_repo(tmp_path)
    payload = fx.make_result_dict()
    run = payload["module_runs"]["v1/attention"]
    second = dict(run["candidates"]["candidates"][0])
    second["id"] = "cand-v1_attention-0003"
    run["candidates"]["candidates"].append(second)
    result_json = tmp_path / "result.json"
    result_json.write_text(json.dumps(payload), encoding="utf-8")

    result = prep_evolve(
        _input(tmp_path, repo, result=result_json, evolver="coral",
               candidate=None, out=tmp_path / "b"),
        _CFG,
    )
    ids = sorted(Path(b.path).parent.name for b in result.bundles)
    assert ids == ["cand-v1_attention-0002", "cand-v1_attention-0003"]


def test_batch_skips_invalid_candidate(tmp_path: Path) -> None:
    repo = fx.make_repo(tmp_path)
    payload = fx.make_result_dict()
    run = payload["module_runs"]["v1/attention"]
    bogus = dict(run["candidates"]["candidates"][0])
    bogus["id"] = "cand-v1_attention-0099"
    bogus["locations"] = [
        {"file": "pkg/attn/missing.py",
         "spans": [{"line_start": 1, "line_end": 2, "symbol": "x", "kind": "function"}]}
    ]
    run["candidates"]["candidates"].append(bogus)
    result_json = tmp_path / "result.json"
    result_json.write_text(json.dumps(payload), encoding="utf-8")

    result = prep_evolve(
        _input(tmp_path, repo, result=result_json, evolver="coral",
               candidate=None, out=tmp_path / "b"),
        _CFG,
    )
    assert len(result.bundles) == 1
    assert Path(result.bundles[0].path).parent.name == "cand-v1_attention-0002"
    assert any(s.candidate_id == "cand-v1_attention-0099" for s in result.skipped)
```

Also update the three existing tests that assumed the old flat name / force-raise:

Replace `test_bundle_dir_name_sanitizes_repo_name` entirely with:

```python
def test_bundle_layout_ignores_repo_name(tmp_path: Path) -> None:
    # The new nested layout does not embed the repo name; a hostile repo name
    # never reaches the path.
    repo = fx.make_repo(tmp_path)
    payload = fx.make_result_dict()
    payload["report"]["project_tree"]["repository"]["name"] = "../demo repo"
    result_json = tmp_path / "custom_result.json"
    result_json.write_text(json.dumps(payload), encoding="utf-8")

    out = tmp_path / "bundles"
    result = prep_evolve(
        _input(tmp_path, repo, result=result_json, evolver="coral", out=out), _CFG
    )
    bundle = Path(result.bundles[0].path)
    assert bundle == out / "evolve" / "v1_attention" / "cand-v1_attention-0002" / "coral"
```

Replace `test_force_overwrites_bundle_without_prior_manifest` body path with the new layout:

```python
def test_force_overwrites_bundle_without_prior_manifest(tmp_path: Path) -> None:
    repo = fx.make_repo(tmp_path)
    out = tmp_path / "bundles"
    bundle = out / "evolve" / "v1_attention" / "cand-v1_attention-0002" / "coral"
    bundle.mkdir(parents=True)
    result = prep_evolve(_input(tmp_path, repo, evolver="coral", out=out, force=True), _CFG)
    assert result.bundles
    assert (bundle / "task.yaml").exists()
```

Replace `test_force_required_for_existing_bundle` (now skip semantics, not raise):

```python
def test_existing_bundle_skipped_without_force(tmp_path: Path) -> None:
    repo = fx.make_repo(tmp_path)
    first = prep_evolve(_input(tmp_path, repo, evolver="coral"), _CFG)
    assert first.bundles
    # Re-run without --force: the existing bundle is skipped, not overwritten,
    # and not an error.
    again = prep_evolve(_input(tmp_path, repo, evolver="coral"), _CFG)
    assert again.bundles == []
    assert any("exists" in s.reason for s in again.skipped)
    # With --force it is regenerated.
    forced = prep_evolve(_input(tmp_path, repo, evolver="coral", force=True), _CFG)
    assert forced.bundles
```

Update `test_generated_path_escape_rejected_before_writing` to drop the `force` kwarg:

```python
def test_generated_path_escape_rejected_before_writing(tmp_path: Path) -> None:
    bundle = tmp_path / "bundle"
    with pytest.raises(BundleExistsError):
        _materialize(bundle, [GeneratedFile(path="../escape.txt", text="x")])
    assert not bundle.exists()
    assert not (tmp_path / "escape.txt").exists()
```

Add `SelectionError` to the imports at the top of the test file:

```python
from spotlights_engine.prep_evolve.errors import (
    BundleExistsError,
    ScopeError,
    SelectionError,
    UnsupportedEvolverError,
)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/unit/prep_evolve/test_materialize_cli.py -q`
Expected: FAIL — new tests error on optional kwargs / missing `_bundle_dir` behavior; updated tests fail against old flat paths.

- [ ] **Step 3: Update `PrepEvolveInput` and `SkippedEvolver`**

In `api.py`, change `PrepEvolveInput` fields:

```python
class PrepEvolveInput(BaseModel):
    model_config = ConfigDict(extra="forbid", arbitrary_types_allowed=True)

    result: Path
    evolver: str
    module: str | None = None  # slash-form qn; inferred from candidate when omitted
    candidate: str | None = None  # omit to process every candidate in result.json
    out: Path | None = None  # base dir; defaults to the run directory
    index: Path | None = None
    repo: str | None = None
    scope: Scope = "candidate"
    direction: Direction | None = None
    model: str | None = None
    force: bool = False
```

Extend `SkippedEvolver`:

```python
class SkippedEvolver(BaseModel):
    model_config = ConfigDict(extra="forbid")

    evolver: str
    reason: str
    candidate_id: str | None = None
    module_qualified_name: str | None = None
```

- [ ] **Step 4: Add `_bundle_dir`, simplify `_materialize`, remove dead helpers**

At the top of `api.py`, add the import and remove the now-unused regex:

```python
from spotlights_engine.utils.id_helpers import slug_for
```

Delete `_BUNDLE_SEGMENT_RE`, `_path_segment`, `_bundle_dir_name`, and `_candidate_id` (they encoded the old flat name). Add:

```python
def _bundle_dir(base: Path, qn: str, candidate_id: str, evolver: str) -> Path:
    """Nested bundle path mirroring the modules/ tree: evolve/<slug>/<cand>/<evolver>/."""
    return base / "evolve" / slug_for(qn) / candidate_id / evolver
```

Also update the module docstring at the top of `api.py` (lines 1–7), which still describes the old writer contract ("a bundle dir is refused unless `--force`"). Replace its second sentence with: "The materializer is one central writer: it always writes generated files, overwriting existing ones. The orchestrator decides whether to skip an existing bundle (skipped unless `--force`)." Keep the last sentence about the shared `README.md`.

Change `_materialize` to drop the exists/force guard (the orchestrator decides), keeping the escape guard:

```python
def _materialize(bundle_path: Path, files: list[GeneratedFile]) -> list[str]:
    """Write `files` into `bundle_path`, overwriting existing files.

    The bundle is a flat set of generator-owned files. Existing-bundle handling
    (skip vs overwrite) is decided by the caller; this writer always writes.
    Path-escaping generated paths are rejected before any write.
    """
    bundle_root = bundle_path.resolve()
    destinations = [(gf, _generated_destination(bundle_path, bundle_root, gf.path)) for gf in files]

    bundle_path.mkdir(parents=True, exist_ok=True)

    written: list[str] = []
    for gf, dest in destinations:
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_text(gf.text, encoding="utf-8")
        written.append(gf.path)
    return written
```

- [ ] **Step 5: Rewrite `prep_evolve` and add the per-candidate helper**

Replace the body of `prep_evolve` and add `_process_candidate`:

```python
def prep_evolve(input: PrepEvolveInput, config: PrepEvolveConfig | None = None) -> PrepEvolveResult:
    config = config or PrepEvolveConfig()
    captured_at = config.captured_at or _now_iso()
    warnings: list[str] = []
    skipped: list[SkippedEvolver] = []

    # 0. normalize / expand evolver.
    try:
        evolver_keys = normalize_evolver(input.evolver)
    except KeyError as exc:
        raise UnsupportedEvolverError(
            f"unknown --evolver {exc.args[0]!r}; expected one of "
            "skydiscover, coral, nous, agentic-strategy-evolution, all"
        ) from exc

    # 1. locate + load result.
    location = resolve_result_location(input.result)
    loaded = load_result(location.result_json)

    # 2. resolve repo path (explicit --repo/--index win over the run dir's index.md).
    repo_path = resolve_repo_path(input.repo, input.index or location.index)
    ensure_repo_dir(repo_path)

    # 3. output base: --out, else the run directory.
    base = input.out or location.run_dir

    # 4. scope guard (single-evolver requests fail loudly; `all` skips+warns).
    if input.scope == "module-main-files":
        if input.evolver == "skydiscover":
            raise ScopeError(
                "skydiscover does not support --scope module-main-files "
                "(single-file only); choose coral/nous or --scope candidate"
            )
        if "skydiscover" in evolver_keys:
            evolver_keys = [k for k in evolver_keys if k != "skydiscover"]
            skipped.append(
                SkippedEvolver(
                    evolver="skydiscover",
                    reason="skydiscover is single-file; skipped for --scope module-main-files",
                )
            )

    # 5. direction (run-wide; warn once).
    direction = input.direction or infer_direction(loaded.context.objective)
    if input.direction is None:
        warnings.append(
            f"direction inferred as {direction!r} from the objective; pass --direction to override"
        )

    # 6. build the selection.
    batch = input.candidate is None
    if batch:
        selections = iter_candidates(loaded)
    else:
        sel = find_candidate(loaded, input.candidate)
        if input.module is not None and input.module != sel.qn:
            raise SelectionError(
                f"--module {input.module!r} does not match candidate "
                f"{input.candidate!r}, which is in module {sel.qn!r}; omit --module"
            )
        selections = [sel]

    # 7. process each selection.
    bundles: list[BundleResult] = []
    for sel in selections:
        try:
            b, s = _process_candidate(
                sel, loaded, repo_path, base, evolver_keys, input, captured_at, direction, batch
            )
            bundles.extend(b)
            skipped.extend(s)
        except PrepEvolveError as exc:
            if not batch:
                raise
            skipped.append(
                SkippedEvolver(
                    evolver=input.evolver,
                    reason=str(exc),
                    candidate_id=sel.candidate.id,
                    module_qualified_name=sel.qn,
                )
            )

    return PrepEvolveResult(bundles=bundles, warnings=warnings, skipped=skipped)


def _process_candidate(
    sel,
    loaded,
    repo_path: Path,
    base: Path,
    evolver_keys: list[str],
    input: PrepEvolveInput,
    captured_at: str,
    direction: Direction,
    batch: bool,
) -> tuple[list[BundleResult], list[SkippedEvolver]]:
    """Render + materialize every requested evolver bundle for one candidate."""
    module = resolve_module(loaded.project_tree, sel.qn)
    findings = resolve_findings(sel.run, sel.qn)

    # live-target validation (raises StalenessError; caught per-candidate in batch).
    validated = validate_candidate_target(repo_path, sel.candidate)
    for t_file in (
        {mf.path for mf in module.main_files} if input.scope == "module-main-files" else set()
    ):
        if t_file != primary_file(sel.candidate):
            validate_scope_file(repo_path, t_file)

    revision = capture_revision(repo_path, captured_at)
    spec = build_spec(
        loaded=loaded,
        module=module,
        qn=sel.qn,
        candidate=sel.candidate,
        findings=findings,
        repo_path=str(repo_path),
        validated=validated,
        revision=revision,
        scope=input.scope,
        direction=direction,
    )

    bundles: list[BundleResult] = []
    skips: list[SkippedEvolver] = []
    for key in evolver_keys:
        adapter = build_adapter(key, input.model)
        ok, reason = adapter.supports(spec)
        if not ok:
            # Batch and `--evolver all` skip incompatible candidates; a single,
            # explicit evolver request fails loudly (spec §5).
            if batch or input.evolver == "all":
                skips.append(
                    SkippedEvolver(
                        evolver=key,
                        reason=reason,
                        candidate_id=sel.candidate.id,
                        module_qualified_name=sel.qn,
                    )
                )
                continue
            raise UnsupportedEvolverError(f"{key} cannot run this selection: {reason}")

        bundle_path = _bundle_dir(base, sel.qn, sel.candidate.id, key)
        if bundle_path.exists() and not input.force:
            skips.append(
                SkippedEvolver(
                    evolver=key,
                    reason="bundle already exists (use --force to overwrite)",
                    candidate_id=sel.candidate.id,
                    module_qualified_name=sel.qn,
                )
            )
            continue

        native_files = adapter.render(spec)
        minimal = getattr(adapter, "minimal_bundle", False)
        all_files = native_files if minimal else native_files + _always_emitted(spec, key)
        written = _materialize(bundle_path, all_files)
        bundles.append(BundleResult(evolver=key, path=str(bundle_path), files=sorted(written)))

    return bundles, skips
```

Update the imports at the top of `api.py` to pull the new resolvers:

```python
from spotlights_engine.prep_evolve.resolve import (
    find_candidate,
    iter_candidates,
    load_result,
    resolve_findings,
    resolve_module,
    resolve_repo_path,
    resolve_result_location,
)
```

(The old `resolve_candidate`, `resolve_candidates`, `resolve_module_run` imports are no longer used by `api.py`; remove them from this import block. They remain exported by `resolve.py` for other callers.)

- [ ] **Step 6: Run the full prep_evolve test module**

Run: `python -m pytest tests/unit/prep_evolve/test_materialize_cli.py -q`
Expected: PASS. If `test_all_skips_skydiscover_for_multi_file` or `test_skydiscover_scope_error_single_request` fail, re-check the scope-guard block was preserved verbatim in Step 5.

- [ ] **Step 7: Run the whole prep_evolve suite**

Run: `python -m pytest tests/unit/prep_evolve/ -q`
Expected: PASS.

- [ ] **Step 8: Commit**

```bash
git add src/spotlights_engine/prep_evolve/api.py tests/unit/prep_evolve/test_materialize_cli.py
git commit -m "feat(prep-evolve): batch-first orchestrator with nested evolve/ layout"
```

---

## Task 4: CLI — optional flags and batch summary

**Files:**
- Modify: `src/spotlights_engine/prep_evolve/cli.py`
- Test: `tests/unit/prep_evolve/test_materialize_cli.py`

**Interfaces:**
- Consumes: `PrepEvolveInput` (module/candidate/out optional), `PrepEvolveResult` (`bundles`, `warnings`, `skipped` with the new `candidate_id`/`module_qualified_name`).
- Produces: `main(argv) -> int` — 0 when ≥1 bundle written, else 1. Prints one line per written bundle to stdout, then a `prep-evolve: N bundle(s) written, M skipped` summary and one line per skip to stderr.

- [ ] **Step 1: Write the failing tests**

Add to `tests/unit/prep_evolve/test_materialize_cli.py`:

```python
def test_cli_batch_no_candidate(tmp_path: Path, capsys) -> None:
    repo = fx.make_repo(tmp_path)
    result_json = fx.write_result(tmp_path)
    rc = prep_main(
        ["--result", str(result_json), "--repo", str(repo),
         "--evolver", "coral", "--out", str(tmp_path / "out")]
    )
    assert rc == 0
    err = capsys.readouterr().err
    assert "bundle(s) written" in err


def test_cli_minimal_flags_directory(tmp_path: Path, capsys) -> None:
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    fx.write_result(run_dir)
    repo = fx.make_repo(tmp_path)
    fx.write_index(run_dir, repo)
    # No --module, --candidate, --repo, or --out.
    rc = prep_main(["--result", str(run_dir), "--evolver", "coral"])
    assert rc == 0
    assert (run_dir / "evolve" / "v1_attention").is_dir()
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/unit/prep_evolve/test_materialize_cli.py -q -k "cli_batch_no_candidate or cli_minimal_flags"`
Expected: FAIL — argparse still marks `--module`/`--candidate`/`--out` required.

- [ ] **Step 3: Make flags optional in `_build_argparser`**

In `cli.py`, change these three arguments:

```python
    p.add_argument(
        "--module",
        default=None,
        help="Slash-form qualified name (optional; inferred from --candidate).",
    )
    p.add_argument(
        "--candidate",
        default=None,
        help="Candidate id, e.g. cand-....; omit to process every candidate.",
    )
    p.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Bundles base dir (optional; defaults to the run directory).",
    )
```

Update the `--result` help to note directory support:

```python
    p.add_argument(
        "--result",
        type=Path,
        required=True,
        help="Path to result.json, or a run directory containing it.",
    )
```

- [ ] **Step 4: Emit the batch summary in `main`**

Replace the reporting block at the end of `main` (everything after the `prep_evolve(...)` try/except) with:

```python
    for w in result.warnings:
        print(f"warning: {w}", file=sys.stderr)

    for b in result.bundles:
        print(f"{b.evolver}: {b.path} ({len(b.files)} files)")

    print(
        f"prep-evolve: {len(result.bundles)} bundle(s) written, "
        f"{len(result.skipped)} skipped",
        file=sys.stderr,
    )
    for s in result.skipped:
        who = (
            f"{s.module_qualified_name}/{s.candidate_id} "
            if s.candidate_id
            else ""
        )
        print(f"  skipped {who}({s.evolver}): {s.reason}", file=sys.stderr)

    return 0 if result.bundles else 1
```

- [ ] **Step 5: Run the CLI tests**

Run: `python -m pytest tests/unit/prep_evolve/test_materialize_cli.py -q -k cli`
Expected: PASS (all `cli` tests, including the existing `test_cli_main_success`, `test_top_level_cli_dispatch`, `test_cli_main_clean_error`).

- [ ] **Step 6: Run the whole prep_evolve suite**

Run: `python -m pytest tests/unit/prep_evolve/ -q`
Expected: PASS.

- [ ] **Step 7: Commit**

```bash
git add src/spotlights_engine/prep_evolve/cli.py tests/unit/prep_evolve/test_materialize_cli.py
git commit -m "feat(prep-evolve): optional --module/--candidate/--out and batch summary"
```

---

## Task 5: Documentation

**Files:**
- Modify: `docs/prep-evolve.md`
- Modify: `README.md` (the `prep-evolve` usage snippet, if present)

**Interfaces:** none (docs only).

- [ ] **Step 1: Update the usage section in `docs/prep-evolve.md`**

Replace the `### Usage` code block and the paragraph that follows it (lines describing `--module`/`--candidate`/`--repo`) with:

````markdown
### Usage

`prep-evolve` consumes a completed run and a target repo. Point `--result` at the
run directory (it finds `result.json` and reads the repo path from `index.md`);
omit `--candidate` to generate a bundle for every candidate:

```bash
# every candidate → spotlights-out/evolve/<module>/<candidate>/skydiscover/
spotlights-engine prep-evolve --result ./spotlights-out --evolver skydiscover

# one candidate
spotlights-engine prep-evolve --result ./spotlights-out \
  --candidate cand-vllm_v1_kv_offload-0002 --evolver skydiscover
```

`--result` accepts a run directory or a `result.json` file. `--candidate` takes
the candidate id from the module page and is optional (omitting it processes all
candidates); the module is inferred from the id, so `--module` is no longer
required. The target repo is resolved from `--repo`, else from the `Repo path:`
line in the run directory's `index.md`. Bundles are written to
`<base>/evolve/<module>/<candidate>/<evolver>/`, where `<base>` is `--out` when
given, else the run directory. Candidates an evolver cannot handle are skipped
with a warning, and existing bundles are skipped unless `--force` is set.
````

- [ ] **Step 2: Update the Flags table in `docs/prep-evolve.md`**

In the `### Flags` table, change the rows for `--result`, `--module`, `--candidate`, and `--out` to:

```markdown
| `--result` | (required) | A finished run's `result.json`, or the run directory containing it. |
| `--module` | (inferred) | Slash-form qualified name; inferred from the candidate id when omitted. |
| `--candidate` | (all) | Candidate id, e.g. `cand-…-0002`. Omit to process every candidate. |
| `--out` | (run dir) | Base directory; the `evolve/…` tree is written under it. Defaults to the run directory. |
```

- [ ] **Step 3: Update the "What lands on disk" section**

Replace the first paragraph of `### What lands on disk` with:

```markdown
Bundles are written to `<base>/evolve/<module>/<candidate>/<evolver>/`, mirroring
the `modules/` tree. Alongside the evolver-native files, every bundle (except the
single-file Nous campaign) includes:
```

- [ ] **Step 4: Update the README snippet**

Run: `grep -n "prep-evolve" README.md`

If a `prep-evolve` example block exists, replace its command with:

```bash
spotlights-engine prep-evolve --result ./spotlights-out --evolver skydiscover
```

If no such block exists, skip this step.

- [ ] **Step 5: Commit**

```bash
git add docs/prep-evolve.md README.md
git commit -m "docs(prep-evolve): document batch-first, run-dir-aware usage"
```

---

## Self-Review

**1. Spec coverage:**
- §1 `--result` dir/file → Task 1 + Task 3 (repo-from-index wiring). ✓
- §2 `--candidate` optional / batch → Task 2 (`iter_candidates`) + Task 3 (selection). ✓
- §3 `--module` dropped, inferred, validated on mismatch → Task 2 (`find_candidate`) + Task 3 (mismatch SelectionError). ✓
- §4 uniform layout, `--out` relocates base → Task 3 (`_bundle_dir`, base default). ✓
- §5 batch resilience (skip unsupported/staleness, existing-bundle skip, summary) → Task 3 (skip-and-warn, per-candidate try/except) + Task 4 (summary). ✓
- Bundle contents unchanged → adapters untouched. ✓
- Backward compat (`--module` accepted+validated, file `--result` works, `--candidate` reproduces single) → Task 3. ✓

**2. Placeholder scan:** No TBD/TODO; every code and test step shows full content. ✓

**3. Type consistency:** `resolve_result_location`/`ResultLocation`, `find_candidate`/`iter_candidates`/`CandidateSelection`, `_bundle_dir(base, qn, candidate_id, evolver)`, `SkippedEvolver(..., candidate_id, module_qualified_name)`, and `_materialize(bundle_path, files)` are used with identical signatures across tasks. `PrepEvolveInput` optional fields (`module`/`candidate`/`out`) are consumed consistently in Task 3 and set optionally by the CLI in Task 4. ✓

**Note for the implementer:** existing single-candidate behavior changes in two visible ways — output path (now nested `evolve/…` instead of the flat `repo__module__cand__evolver` dir) and existing-bundle handling (now skipped-with-warning instead of raising `BundleExistsError`). Both are intended by the spec and the affected tests are updated in Task 3.
