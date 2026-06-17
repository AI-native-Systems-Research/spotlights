"""Stage C: `prep_evolve(input, config) -> PrepEvolveResult`.

Orchestrates extract → validate → render → materialize. The materializer is one
central writer that enforces `--force`/manifest ownership rules and always
emits `evolve_spec.json`, `findings_digest.md`, `generated_files.json`, and
`README.md`.
"""

from __future__ import annotations

import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field

from spotlights_engine.prep_evolve.adapters import (
    build_adapter,
    normalize_evolver,
)
from spotlights_engine.prep_evolve.adapters.base import GeneratedFile
from spotlights_engine.prep_evolve.digest import render_digest
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

_MANIFEST_NAME = "generated_files.json"


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


def _sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _bundle_dir_name(spec: EvolveSpec, evolver: str) -> str:
    # The qn is slash-form (`v1/attention`); collapse separators to `_` so the
    # bundle is a single directory, not a nested path.
    module_seg = spec.module.qualified_name.replace("/", "_")
    return (
        f"{spec.run.repo_name}__{module_seg}__"
        f"{_candidate_id(spec)}__{evolver}"
    )


def _candidate_id(spec: EvolveSpec) -> str:
    for t in spec.targets:
        if t.scope_kind == "candidate" and t.candidate_id:
            return t.candidate_id
    return "candidate"


def prep_evolve(
    input: PrepEvolveInput, config: PrepEvolveConfig | None = None
) -> PrepEvolveResult:
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
                    reason="skydiscover is single-file; skipped for "
                    "--scope module-main-files",
                )
            )

    # 5. live-target validation (before any render/write).
    validated = validate_candidate_target(repo_path, candidate)
    for t_file in {
        mf.path for mf in module.main_files
    } if input.scope == "module-main-files" else set():
        if t_file != candidate.file:
            validate_scope_file(repo_path, t_file)

    revision = capture_revision(repo_path, captured_at)

    direction = input.direction or infer_direction(loaded.context.objective)
    if input.direction is None:
        warnings.append(
            f"direction inferred as {direction!r} from the objective; "
            "pass --direction to override"
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
        all_files = native_files + _always_emitted(spec, key)
        bundle_path = input.out / _bundle_dir_name(spec, key)
        written = _materialize(bundle_path, all_files, force=input.force, warnings=warnings)
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
        return (
            "NOUS_CAMPAIGN_PARENT=$PWD/nous_runs nous run campaign.yaml "
            "--bundle bundle.yaml"
        )
    return "(see bundle files)"


def _evaluator_file(evolver: str) -> str:
    if evolver == "skydiscover":
        return "evaluator.py"
    if evolver == "coral":
        return "grader/src/spotlights_evolve_grader/grader.py"
    return "ground_truth in campaign.yaml + optional bundle.yaml"


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
        correctness_oracle=", ".join(cand.oracles.correctness)
        or "(none parsed — add one)",
        performance_oracle=cand.oracles.performance or "(none parsed — see objective)",
        direction=spec.objective.direction,
        evaluator_file=_evaluator_file(evolver),
        scope_files=_scope_files_md(spec),
        objective=spec.objective.goal,
    )
    return [
        GeneratedFile(
            path="evolve_spec.json",
            text=spec.model_dump_json(indent=2) + "\n",
            overwrite="always",
        ),
        GeneratedFile(
            path="findings_digest.md",
            text=render_digest(spec),
            overwrite="always",
        ),
        GeneratedFile(path="README.md", text=readme, overwrite="always"),
    ]


# --- materialization / writer --------------------------------------------


def _load_prior_manifest(bundle_path: Path) -> dict[str, dict] | None:
    manifest_path = bundle_path / _MANIFEST_NAME
    if not manifest_path.exists():
        return None
    try:
        raw = json.loads(manifest_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return None
    files = raw.get("files") if isinstance(raw, dict) else None
    if not isinstance(files, dict):
        return None
    return files


def _materialize(
    bundle_path: Path,
    files: list[GeneratedFile],
    *,
    force: bool,
    warnings: list[str],
) -> list[str]:
    """Write `files` into `bundle_path`, honoring --force/manifest ownership.

    Returns the bundle-relative paths actually present in the (new) manifest.
    """
    exists = bundle_path.exists()
    prior = _load_prior_manifest(bundle_path) if exists else None

    if exists and not force:
        raise BundleExistsError(
            f"bundle dir already exists: {bundle_path}. Re-run with --force to "
            "rewrite generator-owned files."
        )
    if exists and force and prior is None:
        raise BundleExistsError(
            f"--force refused: {bundle_path} has no {_MANIFEST_NAME}; the "
            "generator cannot safely determine ownership of existing files."
        )

    bundle_path.mkdir(parents=True, exist_ok=True)

    # Build the new manifest as we go.
    new_manifest: dict[str, dict] = {}

    for gf in files:
        dest = bundle_path / gf.path
        new_hash = _sha256(gf.text)
        prior_entry = prior.get(gf.path) if prior else None

        decision = _decide_write(dest, gf, prior_entry, warnings)
        if decision == "write":
            dest.parent.mkdir(parents=True, exist_ok=True)
            dest.write_text(gf.text, encoding="utf-8")
            new_manifest[gf.path] = {
                "path": gf.path,
                "sha256": new_hash,
                "overwrite": gf.overwrite,
            }
        elif decision == "preserve":
            # Carry forward the prior GENERATED hash so future force-runs still
            # recognize the file as user-modified.
            carried = (prior_entry or {}).get("sha256", new_hash)
            new_manifest[gf.path] = {
                "path": gf.path,
                "sha256": carried,
                "overwrite": gf.overwrite,
            }
        # decision == "skip" -> not generator-owned; leave out of manifest.

    # Write the manifest last, atomically; exclude itself from its own listing.
    manifest_path = bundle_path / _MANIFEST_NAME
    payload = {"version": "1", "files": new_manifest}
    tmp = manifest_path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    tmp.replace(manifest_path)

    return list(new_manifest.keys()) + [_MANIFEST_NAME]


def _decide_write(
    dest: Path,
    gf: GeneratedFile,
    prior_entry: dict | None,
    warnings: list[str],
) -> str:
    """Return 'write' | 'preserve' | 'skip' for one file."""
    if not dest.exists():
        return "write"

    # File exists on disk.
    if prior_entry is None:
        # Pre-existing, non-owned file -> never clobber.
        warnings.append(
            f"left pre-existing non-owned file untouched: {gf.path}"
        )
        return "skip"

    if gf.overwrite == "always":
        return "write"

    # preserve_if_modified: rewrite only if the on-disk file still matches the
    # prior generated hash (i.e. the user hasn't edited it).
    current_hash = _sha256(dest.read_text(encoding="utf-8"))
    prior_hash = prior_entry.get("sha256")
    if current_hash == prior_hash:
        return "write"
    warnings.append(
        f"preserved user-modified file (not overwritten): {gf.path}"
    )
    return "preserve"


__all__ = [
    "BundleResult",
    "PrepEvolveConfig",
    "PrepEvolveInput",
    "PrepEvolveResult",
    "SkippedEvolver",
    "prep_evolve",
]
