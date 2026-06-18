"""Stage C: `prep_evolve(input, config) -> PrepEvolveResult`.

Orchestrates extract → validate → render → materialize. The materializer is one
central writer: a bundle dir is refused unless `--force` is set, and with
`--force` every generated file is overwritten. Besides each evolver's native
config it emits a single shared `README.md`.
"""

from __future__ import annotations

import re
from datetime import UTC, datetime
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field

from spotlights_engine.prep_evolve.adapters import (
    build_adapter,
    normalize_evolver,
)
from spotlights_engine.prep_evolve.adapters.base import GeneratedFile
from spotlights_engine.prep_evolve.errors import (
    BundleExistsError,
    ScopeError,
    UnsupportedEvolverError,
)
from spotlights_engine.prep_evolve.extract import Direction, Scope, build_spec, infer_direction
from spotlights_engine.prep_evolve.resolve import (
    load_result,
    resolve_candidate,
    resolve_candidates,
    resolve_findings,
    resolve_module,
    resolve_module_run,
    resolve_repo_path,
)
from spotlights_engine.prep_evolve.spec import EvolveSpec, Target
from spotlights_engine.prep_evolve.templates import render_template
from spotlights_engine.prep_evolve.validate_target import (
    capture_revision,
    ensure_repo_dir,
    validate_candidate_target,
    validate_scope_file,
)

_BUNDLE_SEGMENT_RE = re.compile(r"[^A-Za-z0-9._-]+")


class PrepEvolveInput(BaseModel):
    model_config = ConfigDict(extra="forbid", arbitrary_types_allowed=True)

    result: Path
    module: str  # slash-form qn
    candidate: str
    evolver: str
    out: Path
    index: Path | None = None
    repo: str | None = None
    scope: Scope = "candidate"
    direction: Direction | None = None
    model: str | None = None
    force: bool = False


class PrepEvolveConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    captured_at: str | None = None  # injected timestamp; default = now (UTC)


class BundleResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    evolver: str
    path: str
    files: list[str]


class SkippedEvolver(BaseModel):
    model_config = ConfigDict(extra="forbid")

    evolver: str
    reason: str


class PrepEvolveResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    bundles: list[BundleResult] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
    skipped: list[SkippedEvolver] = Field(default_factory=list)


def _now_iso() -> str:
    return datetime.now(UTC).isoformat()


def _path_segment(value: str, default: str) -> str:
    segment = _BUNDLE_SEGMENT_RE.sub("_", value).strip("._-")
    return segment or default


def _bundle_dir_name(spec: EvolveSpec, evolver: str) -> str:
    # Collapse path separators and other unsafe characters so the bundle is a
    # single directory, not a nested or escapable path.
    repo_seg = _path_segment(spec.run.repo_name, "repo")
    module_seg = _path_segment(spec.module.qualified_name, "module")
    candidate_seg = _path_segment(_candidate_id(spec), "candidate")
    evolver_seg = _path_segment(evolver, "evolver")
    return f"{repo_seg}__{module_seg}__{candidate_seg}__{evolver_seg}"


def _candidate_id(spec: EvolveSpec) -> str:
    for t in spec.targets:
        if t.scope_kind == "candidate" and t.candidate_id:
            return t.candidate_id
    return "candidate"


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

    # 1. load result.
    loaded = load_result(input.result)

    # 2. resolve repo path.
    repo_path = resolve_repo_path(input.repo, input.index)
    ensure_repo_dir(repo_path)

    # 3. resolve module run + candidate.
    module = resolve_module(loaded.project_tree, input.module)
    run = resolve_module_run(loaded, input.module)
    candidates = resolve_candidates(run, input.module)
    candidate = resolve_candidate(candidates, input.candidate)
    findings = resolve_findings(run, input.module)

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

    # 5. live-target validation (before any render/write).
    validated = validate_candidate_target(repo_path, candidate)
    for t_file in (
        {mf.path for mf in module.main_files} if input.scope == "module-main-files" else set()
    ):
        if t_file != candidate.file:
            validate_scope_file(repo_path, t_file)

    revision = capture_revision(repo_path, captured_at)

    direction = input.direction or infer_direction(loaded.context.objective)
    if input.direction is None:
        warnings.append(
            f"direction inferred as {direction!r} from the objective; pass --direction to override"
        )

    spec = build_spec(
        loaded=loaded,
        module=module,
        qn=input.module,
        candidate=candidate,
        findings=findings,
        repo_path=str(repo_path),
        validated=validated,
        revision=revision,
        scope=input.scope,
        direction=direction,
    )

    # Render + materialize per evolver.
    bundles: list[BundleResult] = []
    for key in evolver_keys:
        adapter = build_adapter(key, input.model)
        ok, reason = adapter.supports(spec)
        if not ok:
            if input.evolver == "all":
                skipped.append(SkippedEvolver(evolver=key, reason=reason))
                continue
            raise UnsupportedEvolverError(f"{key} cannot run this selection: {reason}")

        native_files = adapter.render(spec)
        # Minimal bundles (e.g. nous: a single self-contained campaign.yaml) are
        # just their native config — no shared README.
        minimal = getattr(adapter, "minimal_bundle", False)
        all_files = native_files if minimal else native_files + _always_emitted(spec, key)
        bundle_path = input.out / _bundle_dir_name(spec, key)
        written = _materialize(bundle_path, all_files, force=input.force)
        bundles.append(
            BundleResult(
                evolver=key,
                path=str(bundle_path),
                files=sorted(written),
            )
        )

    return PrepEvolveResult(bundles=bundles, warnings=warnings, skipped=skipped)


# --- always-emitted files -------------------------------------------------


def _scope_files_md(spec: EvolveSpec) -> str:
    lines = []
    for t in spec.targets:
        if t.scope_kind == "candidate" and t.line_start is not None:
            lines.append(f"- `{t.file}`:{t.line_start}–{t.line_end} (`{t.symbol}`)")
        else:
            role = f" — {t.role}" if t.role else ""
            lines.append(f"- `{t.file}` (whole file{role})")
    return "\n".join(lines)


def _run_command(evolver: str, ext: str = ".py") -> str:
    if evolver == "skydiscover":
        return f"skydiscover-run seed{ext} evaluator.py -c config.yaml"
    if evolver == "coral":
        return "coral start --config task.yaml"
    if evolver == "nous":
        return "NOUS_CAMPAIGN_PARENT=$PWD/nous_runs nous run campaign.yaml"
    return "(see bundle files)"


def _evaluator_file(evolver: str) -> str:
    if evolver == "skydiscover":
        return "evaluator.py"
    if evolver == "coral":
        return "eval/grader.py"
    return "ground_truth in campaign.yaml"


def _candidate_target(spec: EvolveSpec) -> Target:
    for t in spec.targets:
        if t.scope_kind == "candidate":
            return t
    return spec.targets[0]


def _always_emitted(spec: EvolveSpec, evolver: str) -> list[GeneratedFile]:
    cand = _candidate_target(spec)
    ext = Path(cand.file).suffix.lower() or ".py"
    readme = render_template(
        "readme.md.tmpl",
        repo_name=spec.run.repo_name,
        module_qn=spec.module.qualified_name,
        candidate_id=cand.candidate_id or "(unknown)",
        evolver=evolver,
        run_command=_run_command(evolver, ext),
        correctness_oracle=", ".join(cand.oracles.correctness) or "(none parsed — add one)",
        performance_oracle=cand.oracles.performance or "(none parsed — see objective)",
        direction=spec.objective.direction,
        evaluator_file=_evaluator_file(evolver),
        scope_files=_scope_files_md(spec),
        objective=spec.objective.goal,
    )
    return [GeneratedFile(path="README.md", text=readme)]


# --- materialization / writer --------------------------------------------


def _materialize(
    bundle_path: Path,
    files: list[GeneratedFile],
    *,
    force: bool,
) -> list[str]:
    """Write `files` into `bundle_path`.

    The bundle is a flat set of generator-owned files. An existing bundle dir is
    refused unless `--force` is set; with `--force` every file is overwritten.
    Returns the bundle-relative paths written.
    """
    if bundle_path.exists() and not force:
        raise BundleExistsError(
            f"bundle dir already exists: {bundle_path}. Re-run with --force to "
            "overwrite generated files."
        )

    bundle_root = bundle_path.resolve()
    destinations = [(gf, _generated_destination(bundle_path, bundle_root, gf.path)) for gf in files]

    bundle_path.mkdir(parents=True, exist_ok=True)

    written: list[str] = []
    for gf, dest in destinations:
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_text(gf.text, encoding="utf-8")
        written.append(gf.path)
    return written


def _generated_destination(bundle_path: Path, bundle_root: Path, path: str) -> Path:
    """Resolve a generated file path and reject bundle path escapes."""
    rel = Path(path)
    if rel.is_absolute() or not rel.parts or ".." in rel.parts:
        raise BundleExistsError(f"invalid generated file path: {path!r}")

    dest = bundle_path / rel
    resolved_dest = dest.parent.resolve() / dest.name
    if not resolved_dest.is_relative_to(bundle_root):
        raise BundleExistsError(f"generated file path escapes bundle directory: {path!r}")
    return dest


__all__ = [
    "BundleResult",
    "PrepEvolveConfig",
    "PrepEvolveInput",
    "PrepEvolveResult",
    "SkippedEvolver",
    "prep_evolve",
]
