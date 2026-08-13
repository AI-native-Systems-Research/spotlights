"""Loading and resolution helpers for prep-evolve.

This module is the only place that knows how to:
- read a `result.json` (CLI sidecar `SpotlightsManagerResult` shape, which
  nests `project_tree`/`context` under `report`, or the legacy flat shape)
  into the core fields we need,
- resolve the target repo path from `--repo` or the rendered `index.md`,
- locate a module run + candidate by slash-form qualified name, raising clear
  errors on a miss.

Everything here is pure over its inputs (no bundle writes, no live-repo reads
beyond the optional `index.md` text passed in by the caller).
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path

from spotlights_engine.prep_evolve.errors import (
    RepoResolutionError,
    SelectionError,
)
from spotlights_engine.schemas.candidate import Candidate, Candidates
from spotlights_engine.schemas.common import SpotlightContext
from spotlights_engine.schemas.finding import Finding
from spotlights_engine.schemas.project import Module, ProjectTree


@dataclass
class LoadedResult:
    """The core slice of a Spotlights result the extractor consumes.

    `module_runs_raw` keeps the raw per-module dicts (keyed by slash-form qn) so
    we can pull `candidates`/`findings` without depending on the manager's
    telemetry-rich `ModuleRun` wrapper being importable.
    """

    project_tree: ProjectTree
    context: SpotlightContext
    module_runs_raw: dict[str, dict]


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


def load_result(path: Path) -> LoadedResult:
    """Parse `result.json` into a `LoadedResult`.

    Accepts the current CLI sidecar (`SpotlightsManagerResult`), which nests
    `project_tree`/`context` under `report` and keeps `module_runs` top-level,
    as well as the legacy flat shape (`project_tree`/`context`/`module_runs`
    all top-level). Only those three fields are read; extra keys are ignored.
    """
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise SelectionError(f"result.json not found: {path}") from exc
    except json.JSONDecodeError as exc:
        raise SelectionError(f"result.json is not valid JSON: {path}: {exc}") from exc

    if not isinstance(raw, dict):
        raise SelectionError(f"result.json must be a JSON object: {path}")

    # `project_tree`/`context` moved under `report` (SpotlightReport); fall back
    # to the top level for legacy sidecars. `module_runs` stays top-level.
    report = raw.get("report")
    tree_context_src = report if isinstance(report, dict) else raw

    for key, src in (
        ("project_tree", tree_context_src),
        ("context", tree_context_src),
        ("module_runs", raw),
    ):
        if key not in src:
            raise SelectionError(f"result.json missing required field {key!r}: {path}")

    try:
        project_tree = ProjectTree.model_validate(tree_context_src["project_tree"])
        context = SpotlightContext.model_validate(tree_context_src["context"])
    except Exception as exc:  # noqa: BLE001 - surface as a clean selection error
        raise SelectionError(
            f"result.json project_tree/context did not validate: {type(exc).__name__}: {exc}"
        ) from exc

    module_runs = raw["module_runs"]
    if not isinstance(module_runs, dict):
        raise SelectionError("result.json module_runs must be an object")

    return LoadedResult(
        project_tree=project_tree,
        context=context,
        module_runs_raw=module_runs,
    )


def resolve_module(tree: ProjectTree, qn: str) -> Module:
    """Resolve a `Module` by its slash-form qualified name (the canonical key)."""
    module = tree.resolve(qn)
    if module is None:
        available = ", ".join(node_qn for node_qn, _ in tree.walk())
        raise SelectionError(
            f"module {qn!r} not found in project_tree. Available modules: {available or '(none)'}"
        )
    return module


def resolve_module_run(loaded: LoadedResult, qn: str) -> dict:
    """Return the raw module-run dict keyed by slash-form qn."""
    run = loaded.module_runs_raw.get(qn)
    if run is None:
        available = ", ".join(sorted(loaded.module_runs_raw)) or "(none)"
        raise SelectionError(
            f"no module run for {qn!r} in module_runs. Available runs: {available}"
        )
    if not isinstance(run, dict):
        raise SelectionError(f"module run for {qn!r} is not an object")
    return run


def resolve_candidates(run: dict, qn: str) -> Candidates:
    """Validate and return the `Candidates` object from a raw module run."""
    raw = run.get("candidates")
    if not raw:
        raise SelectionError(f"module run {qn!r} has no candidates (discovery may have failed)")
    try:
        return Candidates.model_validate(raw)
    except Exception as exc:  # noqa: BLE001
        raise SelectionError(
            f"candidates for {qn!r} did not validate: {type(exc).__name__}: {exc}"
        ) from exc


def resolve_candidate(candidates: Candidates, candidate_id: str) -> Candidate:
    """Find a candidate by id, raising a clear error listing available ids."""
    for cand in candidates.candidates:
        if cand.id == candidate_id:
            return cand
    available = ", ".join(c.id for c in candidates.candidates) or "(none)"
    raise SelectionError(
        f"candidate {candidate_id!r} not found in module "
        f"{candidates.module_qualified_name!r}. Available: {available}"
    )


def resolve_findings(run: dict, qn: str) -> list[Finding]:
    """Validate and return the module's findings list (may be empty)."""
    raw = run.get("findings") or []
    findings: list[Finding] = []
    for entry in raw:
        try:
            findings.append(Finding.model_validate(entry))
        except Exception as exc:  # noqa: BLE001
            raise SelectionError(
                f"finding in module {qn!r} did not validate: {type(exc).__name__}: {exc}"
            ) from exc
    return findings


_REPO_PATH_RE = re.compile(r"^-\s*\*\*Repo path:\*\*\s*(.+?)\s*$", re.MULTILINE)


def parse_repo_path_from_index(index_text: str) -> str | None:
    """Extract `- **Repo path:** <path>` from a rendered `index.md`.

    Returns `None` when the marker is absent or carries the renderer's
    `_(unknown)_` placeholder.
    """
    match = _REPO_PATH_RE.search(index_text)
    if not match:
        return None
    value = match.group(1).strip()
    if not value or value == "_(unknown)_":
        return None
    return value


def resolve_repo_path(
    repo_arg: str | None,
    index_path: Path | None,
) -> Path:
    """Resolve the target repo: `--repo` wins, else parse `--index`, else error.

    The returned path must be an existing directory.
    """
    candidate: str | None = None
    source = ""
    if repo_arg:
        candidate = repo_arg
        source = "--repo"
    elif index_path is not None:
        try:
            text = index_path.read_text(encoding="utf-8")
        except OSError as exc:
            raise RepoResolutionError(f"could not read --index {index_path}: {exc}") from exc
        candidate = parse_repo_path_from_index(text)
        source = f"--index ({index_path})"

    if not candidate:
        raise RepoResolutionError(
            "could not resolve target repo path: pass --repo, or --index "
            "pointing at a rendered index.md that carries a 'Repo path:' line."
        )

    path = Path(candidate).expanduser()
    if not path.is_dir():
        raise RepoResolutionError(f"resolved repo path from {source} is not a directory: {path}")
    return path.resolve()


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


def _iter_module_candidates(loaded: LoadedResult):
    """Yield (qn, run, candidate) for every candidate across all module runs.

    Runs that are not dicts or whose `candidates` block is missing/invalid are
    skipped (tolerated during enumeration).
    """
    for qn, run in loaded.module_runs_raw.items():
        if not isinstance(run, dict):
            continue
        cands = _candidates_or_none(run)
        if cands is None:
            continue
        for cand in cands.candidates:
            yield qn, run, cand


def find_candidate(loaded: LoadedResult, candidate_id: str) -> CandidateSelection:
    """Locate a candidate by id across all module runs (module is inferred)."""
    for qn, run, cand in _iter_module_candidates(loaded):
        if cand.id == candidate_id:
            return CandidateSelection(qn=qn, run=run, candidate=cand)
    raise SelectionError(
        f"candidate {candidate_id!r} not found in any module run of result.json"
    )


def iter_candidates(loaded: LoadedResult) -> list[CandidateSelection]:
    """Every candidate across all module runs, in document order."""
    return [
        CandidateSelection(qn=qn, run=run, candidate=cand)
        for qn, run, cand in _iter_module_candidates(loaded)
    ]


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


__all__ = [
    "CandidateSelection",
    "LoadedResult",
    "ResultLocation",
    "find_candidate",
    "iter_candidates",
    "load_result",
    "load_ranking",
    "parse_repo_path_from_index",
    "resolve_candidate",
    "resolve_candidates",
    "resolve_findings",
    "resolve_module",
    "resolve_module_run",
    "resolve_repo_path",
    "resolve_result_location",
]
