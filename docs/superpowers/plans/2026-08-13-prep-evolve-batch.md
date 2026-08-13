# prep-evolve batch-first CLI — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Simplify `prep-evolve` so `--result` accepts any run artifact (run dir, `result.json`, `index.md`, a `sorted/` dir, or a `sorted_candidates.json`/`.md`), `--candidate`/`--module`/`--out` become optional (candidate omitted = all candidates, or the ranking's top-N via `--top-n` when the source is sorted), and bundles land in a uniform `<base>/evolve/<module>/<candidate>/<evolver>/` tree.

**Architecture:** Keep the existing extract → validate → render → materialize pipeline. Add three pure resolver helpers (a result-path locator that self-locates `result.json`/`index.md`/ranking from any accepted form, a candidate-id scanner, and a ranking reader), rewrite the `prep_evolve` orchestrator to loop over a selection (one candidate, all candidates, or the ranked top-N) with per-candidate skip-and-warn resilience, and replace the flat bundle-dir name with a nested path built from `slug_for(qn)`.

**Tech Stack:** Python 3, Pydantic v2 models, `argparse` CLI, `pytest`. Existing module: `src/spotlights_engine/prep_evolve/`.

## Global Constraints

- **Design spec:** `docs/superpowers/specs/2026-08-13-prep-evolve-batch-design.md` — all behavior below is defined there.
- **Module slug:** the `<module>` path segment MUST be `slug_for(qn)` from `spotlights_engine.utils.id_helpers` (slash-form qn with any char outside `[A-Za-z0-9._-]` → `_`), matching the existing `modules/<slug>.md` convention. Never hand-roll the flattening.
- **Bundle layout (always):** `<base>/evolve/<slug_for(qn)>/<candidate_id>/<evolver>/`. Base = `--out` if given, else the run directory (the folder containing `result.json`). For a sorted source the run directory is the parent of `sorted/`, so bundles land at `<run>/evolve/…`, never inside `sorted/`.
- **Candidate data always from `result.json`:** a sorted source supplies only the order + top-N cut; each ranked id is resolved back to its full candidate in `result.json`.
- **`--top-n` is sorted-only:** `--top-n <N>` with a plain source (run dir / `result.json` / `index.md`) is a `SelectionError`. Default is `all` (no cut).
- **Bundle contents unchanged:** skydiscover = `seed.<ext>` + `config.yaml` + `README.md`; coral = `task.yaml` + `README.md`; nous = `campaign.yaml` (minimal, no README). Do not alter adapters.
- **Errors:** raise the existing `PrepEvolveError` subclasses in `errors.py`. `SelectionError` for bad/missing selections, `RepoResolutionError` for repo issues, `StalenessError` from `validate_target`.
- **Run tests with:** `python -m pytest tests/unit/prep_evolve/ -q` from the repo root.

---

## File Structure

- `src/spotlights_engine/prep_evolve/resolve.py` — add `ResultLocation` + `resolve_result_location` (all `--result` forms + ranking source), `CandidateSelection` + `find_candidate` + `iter_candidates`, and `load_ranking`. (Existing resolvers stay.)
- `src/spotlights_engine/prep_evolve/api.py` — rewrite `prep_evolve` (single / all / ranked top-N selection); add `_bundle_dir`; extend `SkippedEvolver`; make `PrepEvolveInput.module/candidate/out` optional and add `top_n`; drop `_bundle_dir_name`/`_path_segment`/`_candidate_id`/`_BUNDLE_SEGMENT_RE`; simplify `_materialize`.
- `src/spotlights_engine/prep_evolve/cli.py` — make `--module`/`--candidate`/`--out` optional; add `--top-n`; add batch summary output.
- `tests/unit/prep_evolve/test_extract_resolve.py` — add resolver tests (Tasks 1–2).
- `tests/unit/prep_evolve/test_materialize_cli.py` — add new-behavior tests and update path/skip-dependent tests (Tasks 3–4).
- `docs/prep-evolve.md` and `README.md` — usage updates (Task 5).

---

## Task 1: Result-path locator (every `--result` form + ranking source)

**Files:**
- Modify: `src/spotlights_engine/prep_evolve/resolve.py`
- Modify: `tests/unit/prep_evolve/_fixtures.py` (add `write_sorted`)
- Test: `tests/unit/prep_evolve/test_extract_resolve.py`

**Interfaces:**
- Produces: `ResultLocation(result_json: Path, run_dir: Path, index: Path | None, ranking: Path | None)` and `resolve_result_location(result: Path) -> ResultLocation`.
  - Directory containing `result.json` → `run_dir = result`, no ranking.
  - Directory containing `sorted_candidates.json`/`.md` (a `sorted/` dir) → `run_dir = result.parent`, `ranking = result/"sorted_candidates.json"`.
  - File `index.md` → `run_dir = parent`, `result_json = parent/"result.json"`.
  - File `sorted_candidates.json` → `run_dir = parent.parent`, `ranking = the file`.
  - File `sorted_candidates.md` → `run_dir = parent.parent`, `ranking = parent/"sorted_candidates.json"`.
  - Any other file → treated as the `result.json` itself (back-compat): `run_dir = parent`, no ranking.
  - `result_json` (whether derived as `run_dir/"result.json"` or the file itself) must exist, else SelectionError. `index` = `run_dir/"index.md"` when present, else `None`. A sorted source with no `sorted_candidates.json` is a SelectionError. Nonexistent `result` → SelectionError.

- [ ] **Step 1: Add the `write_sorted` fixture helper**

Add to `tests/unit/prep_evolve/_fixtures.py` (after `write_index`):

```python
def write_sorted(run_dir: Path, ids: list[str], *, md: bool = True) -> Path:
    """Write a run/sorted/ dir with sorted_candidates.json (+ optional .md).

    Returns the sorted directory. `ids` are written in rank order.
    """
    sorted_dir = run_dir / "sorted"
    sorted_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "source_result": "result.json",
        "method": "listwise sub-agent judge",
        "total_ranked": len(ids),
        "candidates": [
            {"rank": i + 1, "id": cid, "module_qualified_name": "v1/attention"}
            for i, cid in enumerate(ids)
        ],
    }
    (sorted_dir / "sorted_candidates.json").write_text(json.dumps(payload), encoding="utf-8")
    if md:
        (sorted_dir / "sorted_candidates.md").write_text(
            "# Sorted candidates\n", encoding="utf-8"
        )
    return sorted_dir
```

- [ ] **Step 2: Write the failing tests**

Add to `tests/unit/prep_evolve/test_extract_resolve.py`:

```python
from pathlib import Path

import pytest

from spotlights_engine.prep_evolve.errors import SelectionError
from spotlights_engine.prep_evolve.resolve import resolve_result_location

from . import _fixtures as fx


def test_resolve_location_run_directory(tmp_path: Path) -> None:
    fx.write_result(tmp_path)
    fx.write_index(tmp_path, tmp_path)
    loc = resolve_result_location(tmp_path)
    assert loc.result_json == tmp_path / "result.json"
    assert loc.run_dir == tmp_path
    assert loc.index == tmp_path / "index.md"
    assert loc.ranking is None


def test_resolve_location_result_json_file(tmp_path: Path) -> None:
    rj = fx.write_result(tmp_path)
    loc = resolve_result_location(rj)
    assert loc.result_json == rj
    assert loc.run_dir == tmp_path
    assert loc.index is None
    assert loc.ranking is None


def test_resolve_location_arbitrary_file_is_result_json(tmp_path: Path) -> None:
    # A non-standard filename is still treated as the result.json itself.
    payload = fx.make_result_dict()
    rj = tmp_path / "custom_result.json"
    rj.write_text(__import__("json").dumps(payload), encoding="utf-8")
    loc = resolve_result_location(rj)
    assert loc.result_json == rj
    assert loc.run_dir == tmp_path


def test_resolve_location_index_md(tmp_path: Path) -> None:
    fx.write_result(tmp_path)
    index = fx.write_index(tmp_path, tmp_path)
    loc = resolve_result_location(index)
    assert loc.result_json == tmp_path / "result.json"
    assert loc.run_dir == tmp_path
    assert loc.index == index
    assert loc.ranking is None


def test_resolve_location_sorted_dir(tmp_path: Path) -> None:
    fx.write_result(tmp_path)
    sorted_dir = fx.write_sorted(tmp_path, ["cand-v1_attention-0002"])
    loc = resolve_result_location(sorted_dir)
    assert loc.result_json == tmp_path / "result.json"
    assert loc.run_dir == tmp_path
    assert loc.ranking == sorted_dir / "sorted_candidates.json"


def test_resolve_location_sorted_json_file(tmp_path: Path) -> None:
    fx.write_result(tmp_path)
    sorted_dir = fx.write_sorted(tmp_path, ["cand-v1_attention-0002"])
    loc = resolve_result_location(sorted_dir / "sorted_candidates.json")
    assert loc.result_json == tmp_path / "result.json"
    assert loc.run_dir == tmp_path
    assert loc.ranking == sorted_dir / "sorted_candidates.json"


def test_resolve_location_sorted_md_file(tmp_path: Path) -> None:
    fx.write_result(tmp_path)
    sorted_dir = fx.write_sorted(tmp_path, ["cand-v1_attention-0002"])
    loc = resolve_result_location(sorted_dir / "sorted_candidates.md")
    assert loc.ranking == sorted_dir / "sorted_candidates.json"
    assert loc.run_dir == tmp_path


def test_resolve_location_sorted_md_without_json(tmp_path: Path) -> None:
    fx.write_result(tmp_path)
    sorted_dir = fx.write_sorted(tmp_path, ["cand-v1_attention-0002"], md=True)
    (sorted_dir / "sorted_candidates.json").unlink()
    with pytest.raises(SelectionError):
        resolve_result_location(sorted_dir / "sorted_candidates.md")


def test_resolve_location_dir_without_artifacts(tmp_path: Path) -> None:
    with pytest.raises(SelectionError):
        resolve_result_location(tmp_path)  # empty dir


def test_resolve_location_index_without_result_json(tmp_path: Path) -> None:
    index = fx.write_index(tmp_path, tmp_path)  # no result.json alongside
    with pytest.raises(SelectionError):
        resolve_result_location(index)


def test_resolve_location_missing_path(tmp_path: Path) -> None:
    with pytest.raises(SelectionError):
        resolve_result_location(tmp_path / "nope")
```

- [ ] **Step 3: Run tests to verify they fail**

Run: `python -m pytest tests/unit/prep_evolve/test_extract_resolve.py -q -k resolve_location`
Expected: FAIL with `ImportError: cannot import name 'resolve_result_location'`.

- [ ] **Step 4: Implement `ResultLocation` + `resolve_result_location`**

In `resolve.py`, after the `LoadedResult` dataclass, add:

```python
@dataclass
class ResultLocation:
    """Where a run's artifacts live, resolved from any accepted `--result` form.

    `ranking` is the `sorted_candidates.json` when `--result` pointed at a sorted
    source (a `sorted/` dir or a `sorted_candidates.{json,md}`), else `None`.
    """

    result_json: Path
    run_dir: Path
    index: Path | None
    ranking: Path | None


def _require_ranking(sorted_dir: Path) -> Path:
    ranking = sorted_dir / "sorted_candidates.json"
    if not ranking.is_file():
        raise SelectionError(
            f"{sorted_dir} has no sorted_candidates.json to read the ranking from"
        )
    return ranking


def resolve_result_location(result: Path) -> ResultLocation:
    """Resolve `--result` (any run artifact) into result.json/run dir/index/ranking.

    Accepts a run directory, a `result.json` (or any other file, treated as the
    result.json itself), an `index.md`, a `sorted/` directory, or a
    `sorted_candidates.{json,md}`. Sorted sources live one level below the run
    dir, so the run dir is resolved by going up; the candidate *data* is always
    read from `result.json` regardless (a ranking only orders/filters it).
    """
    result = Path(result).expanduser()
    ranking: Path | None = None
    result_json_override: Path | None = None  # set when --result IS the result.json

    if result.is_dir():
        if (result / "result.json").is_file():
            run_dir = result
        elif (result / "sorted_candidates.json").is_file() or (
            result / "sorted_candidates.md"
        ).is_file():
            run_dir = result.parent
            ranking = _require_ranking(result)
        else:
            raise SelectionError(
                f"directory has no result.json or sorted_candidates.json: {result}"
            )
    elif result.is_file():
        if result.name == "index.md":
            run_dir = result.parent
        elif result.name == "sorted_candidates.json":
            run_dir = result.parent.parent
            ranking = result
        elif result.name == "sorted_candidates.md":
            run_dir = result.parent.parent
            ranking = _require_ranking(result.parent)
        else:  # any other file: treat as the result.json itself (back-compat)
            run_dir = result.parent
            result_json_override = result
    else:
        raise SelectionError(f"--result path not found: {result}")

    result_json = result_json_override or (run_dir / "result.json")
    if not result_json.is_file():
        raise SelectionError(f"no result.json for this run: expected {result_json}")

    index = run_dir / "index.md"
    return ResultLocation(
        result_json=result_json,
        run_dir=run_dir,
        index=index if index.is_file() else None,
        ranking=ranking,
    )
```

Add `"ResultLocation"` and `"resolve_result_location"` to `resolve.py`'s `__all__`.

- [ ] **Step 5: Run tests to verify they pass**

Run: `python -m pytest tests/unit/prep_evolve/test_extract_resolve.py -q -k resolve_location`
Expected: PASS (11 passed).

- [ ] **Step 6: Commit**

```bash
git add src/spotlights_engine/prep_evolve/resolve.py tests/unit/prep_evolve/test_extract_resolve.py tests/unit/prep_evolve/_fixtures.py
git commit -m "feat(prep-evolve): resolve --result from any run artifact incl. sorted sources"
```

---

## Task 2: Candidate-id scanner + ranking reader

**Files:**
- Modify: `src/spotlights_engine/prep_evolve/resolve.py`
- Test: `tests/unit/prep_evolve/test_extract_resolve.py`

**Interfaces:**
- Consumes: `LoadedResult` (from `load_result`), `Candidates` (from `schemas.candidate`).
- Produces:
  - `CandidateSelection(qn: str, run: dict, candidate: Candidate)` — a resolved selection carrying the module qn, its raw run dict, and the validated candidate.
  - `find_candidate(loaded: LoadedResult, candidate_id: str) -> CandidateSelection` — scans every module run for a candidate whose `id` matches; SelectionError if none.
  - `iter_candidates(loaded: LoadedResult) -> list[CandidateSelection]` — every candidate across all runs, in `module_runs` order then per-module candidate order. Runs with no/invalid `candidates` block are skipped (tolerated).
  - `load_ranking(path: Path) -> list[str]` — ordered candidate ids from a `sorted_candidates.json` (sorted by `rank` ascending). SelectionError if unreadable, not an object, or has no candidate ids.

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


def test_load_ranking_orders_by_rank(tmp_path: Path) -> None:
    from spotlights_engine.prep_evolve.resolve import load_ranking

    sorted_dir = fx.write_sorted(tmp_path, ["cand-a-0001", "cand-b-0002", "cand-c-0003"])
    ids = load_ranking(sorted_dir / "sorted_candidates.json")
    assert ids == ["cand-a-0001", "cand-b-0002", "cand-c-0003"]


def test_load_ranking_empty_errors(tmp_path: Path) -> None:
    from spotlights_engine.prep_evolve.resolve import load_ranking

    bad = tmp_path / "sorted_candidates.json"
    bad.write_text('{"candidates": []}', encoding="utf-8")
    with pytest.raises(SelectionError):
        load_ranking(bad)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/unit/prep_evolve/test_extract_resolve.py -q -k "find_candidate or iter_candidates or load_ranking"`
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


def load_ranking(path: Path) -> list[str]:
    """Ordered candidate ids from a `sorted_candidates.json` (rank ascending).

    Only the id order is used; the candidate *data* is read from `result.json`.
    """
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise SelectionError(f"could not read ranking {path}: {exc}") from exc
    cands = raw.get("candidates") if isinstance(raw, dict) else None
    if not isinstance(cands, list):
        raise SelectionError(f"ranking {path} has no candidates list")
    ranked = sorted(
        (c for c in cands if isinstance(c, dict) and c.get("id")),
        key=lambda c: c.get("rank", 1_000_000),
    )
    ids = [c["id"] for c in ranked]
    if not ids:
        raise SelectionError(f"ranking {path} lists no candidate ids")
    return ids
```

Add `"CandidateSelection"`, `"find_candidate"`, `"iter_candidates"`, `"load_ranking"` to `__all__`.

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/unit/prep_evolve/test_extract_resolve.py -q -k "find_candidate or iter_candidates or load_ranking"`
Expected: PASS (5 passed).

- [ ] **Step 5: Commit**

```bash
git add src/spotlights_engine/prep_evolve/resolve.py tests/unit/prep_evolve/test_extract_resolve.py
git commit -m "feat(prep-evolve): candidate-by-id scanner + sorted_candidates ranking reader"
```

---

## Task 3: Rewrite the orchestrator (optional inputs, nested layout, single + batch, skip-and-warn)

**Files:**
- Modify: `src/spotlights_engine/prep_evolve/api.py`
- Test: `tests/unit/prep_evolve/test_materialize_cli.py`

**Interfaces:**
- Consumes: `resolve_result_location`, `find_candidate`, `iter_candidates`, `load_ranking`, `CandidateSelection` (Tasks 1–2); existing `load_result`, `resolve_repo_path`, `resolve_module`, `resolve_findings`, `ensure_repo_dir`, `validate_candidate_target`, `validate_scope_file`, `capture_revision`, `build_spec`, `infer_direction`, `build_adapter`, `normalize_evolver`, `slug_for`, `SelectionError`.
- Produces:
  - `PrepEvolveInput` with `result: Path`, `module: str | None = None`, `candidate: str | None = None`, `out: Path | None = None`, `top_n: int | None = None` (other fields unchanged).
  - `SkippedEvolver(evolver: str, reason: str, candidate_id: str | None = None, module_qualified_name: str | None = None)`.
  - `_bundle_dir(base: Path, qn: str, candidate_id: str, evolver: str) -> Path` returning `base/"evolve"/slug_for(qn)/candidate_id/evolver`.
  - `_materialize(bundle_path: Path, files: list[GeneratedFile]) -> list[str]` (no `force` param; keeps the path-escape guard; always writes/overwrites).
  - `prep_evolve(input, config) -> PrepEvolveResult` — selection is: `input.candidate` set → that one (loud errors); else sorted source (`location.ranking`) → ranked ids (top-N cut) resolved via `find_candidate`, per-id skip-and-warn; else plain batch → `iter_candidates`. `top_n` with a plain source is a `SelectionError`.

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


def _two_candidate_run(tmp_path: Path) -> Path:
    """Write a result.json with two candidates; return its path."""
    payload = fx.make_result_dict()
    run = payload["module_runs"]["v1/attention"]
    second = dict(run["candidates"]["candidates"][0])
    second["id"] = "cand-v1_attention-0003"
    run["candidates"]["candidates"].append(second)
    result_json = tmp_path / "result.json"
    result_json.write_text(json.dumps(payload), encoding="utf-8")
    return result_json


def test_sorted_source_uses_ranking_order(tmp_path: Path) -> None:
    repo = fx.make_repo(tmp_path)
    _two_candidate_run(tmp_path)
    # Ranking lists 0003 first, then 0002.
    sorted_dir = fx.write_sorted(
        tmp_path, ["cand-v1_attention-0003", "cand-v1_attention-0002"]
    )
    result = prep_evolve(
        _input(tmp_path, repo, result=sorted_dir, candidate=None,
               evolver="coral", out=tmp_path / "b"),
        _CFG,
    )
    order = [Path(b.path).parent.name for b in result.bundles]
    assert order == ["cand-v1_attention-0003", "cand-v1_attention-0002"]


def test_top_n_limits_ranked_selection(tmp_path: Path) -> None:
    repo = fx.make_repo(tmp_path)
    _two_candidate_run(tmp_path)
    sorted_dir = fx.write_sorted(
        tmp_path, ["cand-v1_attention-0003", "cand-v1_attention-0002"]
    )
    result = prep_evolve(
        _input(tmp_path, repo, result=sorted_dir, candidate=None, top_n=1,
               evolver="coral", out=tmp_path / "b"),
        _CFG,
    )
    names = [Path(b.path).parent.name for b in result.bundles]
    assert names == ["cand-v1_attention-0003"]


def test_ranked_id_missing_in_result_is_skipped(tmp_path: Path) -> None:
    repo = fx.make_repo(tmp_path)
    fx.write_result(tmp_path)  # only cand-v1_attention-0002 exists
    sorted_dir = fx.write_sorted(
        tmp_path, ["cand-v1_attention-0002", "cand-ghost-0009"]
    )
    result = prep_evolve(
        _input(tmp_path, repo, result=sorted_dir, candidate=None,
               evolver="coral", out=tmp_path / "b"),
        _CFG,
    )
    assert len(result.bundles) == 1
    assert any(s.candidate_id == "cand-ghost-0009" for s in result.skipped)


def test_top_n_without_ranking_errors(tmp_path: Path) -> None:
    repo = fx.make_repo(tmp_path)
    result_json = fx.write_result(tmp_path)
    with pytest.raises(SelectionError):
        prep_evolve(
            _input(tmp_path, repo, result=result_json, candidate=None, top_n=5,
                   evolver="coral", out=tmp_path / "b"),
            _CFG,
        )
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
    top_n: int | None = Field(default=None, ge=1)  # sorted sources only; None = all
    index: Path | None = None
    repo: str | None = None
    scope: Scope = "candidate"
    direction: Direction | None = None
    model: str | None = None
    force: bool = False
```

(`Field` is already imported at the top of `api.py`.)

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
    #    - explicit --candidate → that one (top_n/ranking ignored)
    #    - sorted source        → ranked ids (top-N cut), each resolved in result.json
    #    - plain batch          → every candidate in document order
    batch = input.candidate is None
    if not batch:
        sel = find_candidate(loaded, input.candidate)
        if input.module is not None and input.module != sel.qn:
            raise SelectionError(
                f"--module {input.module!r} does not match candidate "
                f"{input.candidate!r}, which is in module {sel.qn!r}; omit --module"
            )
        selections = [sel]
    elif location.ranking is not None:
        ranked_ids = load_ranking(location.ranking)
        if input.top_n is not None:
            ranked_ids = ranked_ids[: input.top_n]
        selections = []
        for cid in ranked_ids:
            try:
                selections.append(find_candidate(loaded, cid))
            except SelectionError as exc:
                skipped.append(
                    SkippedEvolver(evolver=input.evolver, reason=str(exc), candidate_id=cid)
                )
    else:
        if input.top_n is not None:
            raise SelectionError(
                "--top-n requires a ranked source; point --result at a sorted/ "
                "directory or a sorted_candidates.json (a plain result.json has no ranking)"
            )
        selections = iter_candidates(loaded)

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

Update the imports at the top of `api.py` to pull the new resolvers, and add `SelectionError` to the `errors` import:

```python
from spotlights_engine.prep_evolve.errors import (
    BundleExistsError,
    PrepEvolveError,
    ScopeError,
    SelectionError,
    UnsupportedEvolverError,
)
from spotlights_engine.prep_evolve.resolve import (
    find_candidate,
    iter_candidates,
    load_ranking,
    load_result,
    resolve_findings,
    resolve_module,
    resolve_repo_path,
    resolve_result_location,
)
```

(The old `resolve_candidate`, `resolve_candidates`, `resolve_module_run` imports are no longer used by `api.py`; remove them from this import block. They remain exported by `resolve.py` for other callers. `PrepEvolveError` must be present in the `errors` import because the per-candidate loop catches it — verify it's imported; add it if the current file only imports the subclasses.)

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
- Consumes: `PrepEvolveInput` (module/candidate/out optional, `top_n`), `PrepEvolveResult` (`bundles`, `warnings`, `skipped` with the new `candidate_id`/`module_qualified_name`).
- Produces: `main(argv) -> int` — 0 when ≥1 bundle written, else 1; 2 on a bad `--top-n` value or a `PrepEvolveError`. `--top-n` accepts `all` (default → `top_n=None`) or a positive integer (→ `top_n=<int>`); anything else prints an error and returns 2. Prints one line per written bundle to stdout, then a `prep-evolve: N bundle(s) written, M skipped` summary and one line per skip to stderr.

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


def test_cli_top_n_from_sorted(tmp_path: Path) -> None:
    repo = fx.make_repo(tmp_path)
    _two_candidate_run(tmp_path)
    sorted_dir = fx.write_sorted(
        tmp_path, ["cand-v1_attention-0003", "cand-v1_attention-0002"]
    )
    rc = prep_main(
        ["--result", str(sorted_dir), "--evolver", "coral",
         "--top-n", "1", "--out", str(tmp_path / "out")]
    )
    assert rc == 0
    # Only the top-ranked candidate's bundle exists.
    evolve = tmp_path / "out" / "evolve" / "v1_attention"
    assert (evolve / "cand-v1_attention-0003").is_dir()
    assert not (evolve / "cand-v1_attention-0002").exists()


def test_cli_bad_top_n_returns_2(tmp_path: Path, capsys) -> None:
    repo = fx.make_repo(tmp_path)
    result_json = fx.write_result(tmp_path)
    rc = prep_main(
        ["--result", str(result_json), "--evolver", "coral",
         "--top-n", "banana", "--out", str(tmp_path / "out")]
    )
    assert rc == 2
    assert "--top-n" in capsys.readouterr().err
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/unit/prep_evolve/test_materialize_cli.py -q -k "cli_batch_no_candidate or cli_minimal_flags or cli_top_n or cli_bad_top_n"`
Expected: FAIL — argparse still marks `--module`/`--candidate`/`--out` required and does not know `--top-n`.

- [ ] **Step 3: Make flags optional and add `--top-n` in `_build_argparser`**

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

Update the `--result` help to note the extra accepted forms:

```python
    p.add_argument(
        "--result",
        type=Path,
        required=True,
        help=(
            "A finished run's result.json, or the run directory / index.md / "
            "sorted/ dir / sorted_candidates.{json,md} that self-locates it."
        ),
    )
```

Add the `--top-n` argument (kept as a raw string here; parsed in `main` so a bad
value returns 2 like other errors rather than argparse's own exit):

```python
    p.add_argument(
        "--top-n",
        dest="top_n",
        default="all",
        help=(
            "For a sorted --result: build only the top N ranked candidates "
            "('all' = every ranked candidate, the default)."
        ),
    )
```

- [ ] **Step 4: Parse `--top-n` and build the input in `main`**

In `main`, after `args = _build_argparser().parse_args(argv)` and before constructing `PrepEvolveInput`, parse `--top-n` and thread it through:

```python
    raw_top_n = str(args.top_n).strip().lower()
    if raw_top_n == "all":
        top_n: int | None = None
    else:
        try:
            top_n = int(raw_top_n)
        except ValueError:
            print(
                f"prep-evolve: --top-n must be a positive integer or 'all', "
                f"got {args.top_n!r}",
                file=sys.stderr,
            )
            return 2
        if top_n < 1:
            print(
                f"prep-evolve: --top-n must be >= 1, got {top_n}",
                file=sys.stderr,
            )
            return 2

    inp = PrepEvolveInput(
        result=args.result,
        index=args.index,
        module=args.module,
        candidate=args.candidate,
        repo=args.repo,
        evolver=args.evolver,
        out=args.out,
        top_n=top_n,
        scope=args.scope,
        direction=args.direction,
        model=args.model,
        force=args.force,
    )
```

- [ ] **Step 5: Emit the batch summary in `main`**

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

- [ ] **Step 6: Run the CLI tests**

Run: `python -m pytest tests/unit/prep_evolve/test_materialize_cli.py -q -k cli`
Expected: PASS (all `cli` tests, including the existing `test_cli_main_success`, `test_top_level_cli_dispatch`, `test_cli_main_clean_error`).

- [ ] **Step 7: Run the whole prep_evolve suite**

Run: `python -m pytest tests/unit/prep_evolve/ -q`
Expected: PASS.

- [ ] **Step 8: Commit**

```bash
git add src/spotlights_engine/prep_evolve/cli.py tests/unit/prep_evolve/test_materialize_cli.py
git commit -m "feat(prep-evolve): optional --module/--candidate/--out, --top-n, batch summary"
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

# top 10 of a ranking → still under spotlights-out/evolve/… (not inside sorted/)
spotlights-engine prep-evolve --result ./spotlights-out/sorted \
  --evolver skydiscover --top-n 10
```

`--result` accepts any run artifact and self-locates the rest: a run directory, a
`result.json` file, an `index.md`, a `sorted/` directory, or a
`sorted_candidates.json`/`.md`. `--candidate` takes the candidate id from the
module page and is optional (omitting it processes all candidates); the module is
inferred from the id, so `--module` is no longer required. When `--result` points
at a sorted source, omitting `--candidate` processes the ranked candidates in rank
order, and `--top-n <N>` builds only the top N (default `all`); `--top-n` with a
plain `result.json` is an error. The target repo is resolved from `--repo`, else
from the `Repo path:` line in the run directory's `index.md`. Bundles are written
to `<base>/evolve/<module>/<candidate>/<evolver>/`, where `<base>` is `--out` when
given, else the run directory. Candidates an evolver cannot handle are skipped
with a warning, and existing bundles are skipped unless `--force` is set.
````

- [ ] **Step 2: Update the Flags table in `docs/prep-evolve.md`**

In the `### Flags` table, change the rows for `--result`, `--module`, `--candidate`, and `--out`, and add a `--top-n` row after `--candidate`:

```markdown
| `--result` | (required) | A finished run's `result.json`, the run directory containing it, an `index.md`, a `sorted/` dir, or a `sorted_candidates.{json,md}`. |
| `--module` | (inferred) | Slash-form qualified name; inferred from the candidate id when omitted. |
| `--candidate` | (all) | Candidate id, e.g. `cand-…-0002`. Omit to process every candidate. |
| `--top-n` | `all` | For a sorted `--result`: build only the top N ranked candidates. Error with a plain source. |
| `--out` | (run dir) | Base directory; the `evolve/…` tree is written under it. Defaults to the run directory. |
```

Also update the last paragraph of the Flags section (the re-run guard) so it matches the new skip-not-fail semantics:

```markdown
Re-runs resume cleanly: an existing bundle directory is skipped (not an error)
unless `--force` is set. With `--force`, every generated file in that bundle is
overwritten — including a hand-edited evaluator/grader — so copy out any evaluator
work you want to keep before re-running.
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
- §1 `--result` accepts every form (run dir, `result.json`/any file, `index.md`, `sorted/` dir, `sorted_candidates.{json,md}`) + ranking source → Task 1 (`resolve_result_location`, 11 tests) + Task 3 (repo-from-index wiring, sorted run-dir default). ✓
- §2 `--candidate` optional / batch → Task 2 (`iter_candidates`) + Task 3 (selection). ✓
- §2 `--top-n` for ranked sources: sorted source → ranked order, top-N cut, ranked-id-missing skip-with-warning; `--top-n` with plain source is an error → Task 2 (`load_ranking`) + Task 3 (ranking branch, `top_n` field, `test_sorted_source_uses_ranking_order`/`test_top_n_limits_ranked_selection`/`test_ranked_id_missing_in_result_is_skipped`/`test_top_n_without_ranking_errors`) + Task 4 (`--top-n` flag, `all`→None parse, rc 2 on bad value). ✓
- §3 `--module` dropped, inferred, validated on mismatch → Task 2 (`find_candidate`) + Task 3 (mismatch SelectionError). ✓
- §4 uniform layout, `--out` relocates base (sorted source → run dir is parent of `sorted/`, bundles never inside `sorted/`) → Task 3 (`_bundle_dir`, base default from `location.run_dir`). ✓
- §5 batch resilience (skip unsupported/staleness, existing-bundle skip, summary) → Task 3 (skip-and-warn, per-candidate try/except) + Task 4 (summary). ✓
- Bundle contents unchanged → adapters untouched. ✓
- Backward compat (`--module` accepted+validated, file `--result` works, `--candidate` reproduces single) → Task 1 (arbitrary-file → result.json) + Task 3. ✓

**2. Placeholder scan:** No TBD/TODO; every code and test step shows full content. ✓

**3. Type consistency:** `ResultLocation(result_json, run_dir, index, ranking)`/`resolve_result_location`, `find_candidate`/`iter_candidates`/`load_ranking`/`CandidateSelection`, `_bundle_dir(base, qn, candidate_id, evolver)`, `SkippedEvolver(..., candidate_id, module_qualified_name)`, and `_materialize(bundle_path, files)` are used with identical signatures across tasks. `PrepEvolveInput` optional fields (`module`/`candidate`/`out`/`top_n`) are consumed consistently in Task 3 and set optionally by the CLI in Task 4 (`--top-n` parsed `all`→`None`, else positive int). The `write_sorted` fixture (Task 1) is reused by Tasks 2–4. ✓

**Note for the implementer:** existing single-candidate behavior changes in two visible ways — output path (now nested `evolve/…` instead of the flat `repo__module__cand__evolver` dir) and existing-bundle handling (now skipped-with-warning instead of raising `BundleExistsError`). Both are intended by the spec and the affected tests are updated in Task 3.
