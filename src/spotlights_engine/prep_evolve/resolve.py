"""Loading and resolution helpers for prep-evolve.

This module is the only place that knows how to:
- read a `result.json` (CLI sidecar `SpotlightsManagerResult` shape or the
  architecture-shaped `SpotlightsResult`) into the core fields we need,
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


def load_result(path: Path) -> LoadedResult:
    """Parse `result.json` into a `LoadedResult`.

    Accepts both the CLI sidecar (`SpotlightsManagerResult`, which carries
    `extractor_invocation`/`per_module_telemetry`/...) and the architecture
    `SpotlightsResult` shape. Only `project_tree`, `context`, and `module_runs`
    are read; extra keys are ignored.
    """
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise SelectionError(f"result.json not found: {path}") from exc
    except json.JSONDecodeError as exc:
        raise SelectionError(f"result.json is not valid JSON: {path}: {exc}") from exc

    if not isinstance(raw, dict):
        raise SelectionError(f"result.json must be a JSON object: {path}")

    for key in ("project_tree", "context", "module_runs"):
        if key not in raw:
            raise SelectionError(
                f"result.json missing required field {key!r}: {path}"
            )

    try:
        project_tree = ProjectTree.model_validate(raw["project_tree"])
        context = SpotlightContext.model_validate(raw["context"])
    except Exception as exc:  # noqa: BLE001 - surface as a clean selection error
        raise SelectionError(
            f"result.json project_tree/context did not validate: "
            f"{type(exc).__name__}: {exc}"
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
            f"module {qn!r} not found in project_tree. "
            f"Available modules: {available or '(none)'}"
        )
    return module


def resolve_module_run(loaded: LoadedResult, qn: str) -> dict:
    """Return the raw module-run dict keyed by slash-form qn."""
    run = loaded.module_runs_raw.get(qn)
    if run is None:
        available = ", ".join(sorted(loaded.module_runs_raw)) or "(none)"
        raise SelectionError(
            f"no module run for {qn!r} in module_runs. "
            f"Available runs: {available}"
        )
    if not isinstance(run, dict):
        raise SelectionError(f"module run for {qn!r} is not an object")
    return run


def resolve_candidates(run: dict, qn: str) -> Candidates:
    """Validate and return the `Candidates` object from a raw module run."""
    raw = run.get("candidates")
    if not raw:
        raise SelectionError(
            f"module run {qn!r} has no candidates (discovery may have failed)"
        )
    try:
        return Candidates.model_validate(raw)
    except Exception as exc:  # noqa: BLE001
        raise SelectionError(
            f"candidates for {qn!r} did not validate: "
            f"{type(exc).__name__}: {exc}"
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
                f"finding in module {qn!r} did not validate: "
                f"{type(exc).__name__}: {exc}"
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
            raise RepoResolutionError(
                f"could not read --index {index_path}: {exc}"
            ) from exc
        candidate = parse_repo_path_from_index(text)
        source = f"--index ({index_path})"

    if not candidate:
        raise RepoResolutionError(
            "could not resolve target repo path: pass --repo, or --index "
            "pointing at a rendered index.md that carries a 'Repo path:' line."
        )

    path = Path(candidate).expanduser()
    if not path.is_dir():
        raise RepoResolutionError(
            f"resolved repo path from {source} is not a directory: {path}"
        )
    return path.resolve()


__all__ = [
    "LoadedResult",
    "load_result",
    "parse_repo_path_from_index",
    "resolve_candidate",
    "resolve_candidates",
    "resolve_findings",
    "resolve_module",
    "resolve_module_run",
    "resolve_repo_path",
]
