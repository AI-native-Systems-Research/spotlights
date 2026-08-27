"""Stage C: `prep_evolve(input, config) -> PrepEvolveResult`.

Orchestrates extract → validate → render → materialize. The materializer is one
central writer: it always writes generated files, overwriting existing ones. The
orchestrator decides whether to skip an existing bundle (skipped unless
`--force`). Besides each evolver's native config it emits a single shared
`README.md`.
"""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field

from spotlights_engine.prep_evolve.adapters import (
    build_adapter,
    normalize_evolver,
)
from spotlights_engine.prep_evolve.adapters.base import GeneratedFile
from spotlights_engine.prep_evolve.errors import (
    GeneratedPathError,
    PrepEvolveError,
    ScopeError,
    SelectionError,
    UnsupportedEvolverError,
)
from spotlights_engine.prep_evolve.extract import Direction, Scope, build_spec, infer_direction
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
from spotlights_engine.prep_evolve.spec import EvolveSpec, Target
from spotlights_engine.prep_evolve.templates import render_template
from spotlights_engine.prep_evolve.validate_target import (
    capture_revision,
    ensure_repo_dir,
    validate_candidate_target,
    validate_scope_file,
)
from spotlights_engine.utils.id_helpers import slug_for
from spotlights_engine.utils.schema_compat import primary_file


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
    candidate_id: str | None = None
    module_qualified_name: str | None = None


class PrepEvolveResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    bundles: list[BundleResult] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
    skipped: list[SkippedEvolver] = Field(default_factory=list)


def _now_iso() -> str:
    return datetime.now(UTC).isoformat()


def _bundle_dir(base: Path, qn: str, candidate_id: str, evolver: str) -> Path:
    """Nested bundle path mirroring the modules/ tree: evolve/<slug>/<cand>/<evolver>/."""
    return base / "evolve" / slug_for(qn) / candidate_id / evolver


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

        all_files = adapter.render(spec) + _always_emitted(spec, key)
        written = _materialize(bundle_path, all_files)
        bundles.append(BundleResult(evolver=key, path=str(bundle_path), files=sorted(written)))

    return bundles, skips


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
        return "grader/grader.py"
    return "campaign.yaml"


# --- the evaluation gap, per evolver -------------------------------------
#
# skydiscover and coral leave a `# TODO` in hand-written evaluator code; nous
# needs no evaluator code but leaves `ground_truth.pass_condition` as a TODO in
# campaign.yaml. Same gap, different shape — so the README's gap section and the
# in-scope lead-in are resolved per evolver rather than shared verbatim.


def _gap_heading(evolver: str) -> str:
    if evolver == "nous":
        return "The evaluation gap — you must define the pass condition"
    return "The evaluation gap — you must complete the evaluator"


def _gap_intro(evolver: str) -> str:
    if evolver == "nous":
        return (
            "This campaign is **launchable but unscored**. `campaign.yaml` names the\n"
            "metric and which way it should move, but `ground_truth.pass_condition` is a\n"
            "`TODO`. Nous will run and report numbers; nothing decides whether\n"
            "those numbers count as a win until you write that rule."
        )
    return (
        "This bundle ships a **launchable but degenerate** evaluator: the correctness\n"
        "gate is wired from what Spotlights parsed, but the performance measurement is a\n"
        "`# TODO`. Until you implement it, scores are not meaningful."
    )


def _pass_condition_example(spec: EvolveSpec, cand: Target) -> str:
    """The body of the worked `pass_condition` block scalar (4-space indented).

    Never names a metric the spec does not have: with no parsed performance
    oracle the example stays generic rather than inventing one. A performance
    oracle may name several comma-separated metrics, so the subject is phrased to
    keep the singular verb grammatical either way.
    """
    metrics = [m.strip() for m in (cand.oracles.performance or "").split(",") if m.strip()]
    if not metrics:
        return (
            "    the primary metric moves in the intended direction versus the\n"
            "    seed baseline, with all correctness checks passing"
        )
    verb = "decreases" if spec.objective.direction == "minimize" else "increases"
    subject = metrics[0] if len(metrics) == 1 else f"every one of {', '.join(metrics)}"
    return (
        f"    {subject} {verb} by at least 5% versus the seed baseline,\n"
        "    with all correctness checks passing"
    )


def _gap_action(evolver: str, spec: EvolveSpec, cand: Target) -> str:
    if evolver != "nous":
        return (
            f"Complete the measurement in `{_evaluator_file(evolver)}` before trusting any result."
        )
    return (
        "Replace the `pass_condition` TODO under `ground_truth` in `campaign.yaml`\n"
        "before trusting any result. Spotlights knows the metric and its direction, not\n"
        "how much movement counts as a win. A concrete rule for this candidate looks\n"
        "like:\n"
        "\n"
        "```yaml\n"
        "ground_truth:\n"
        "  pass_condition: >-\n"
        f"{_pass_condition_example(spec, cand)}\n"
        "```\n"
        "\n"
        "Choose the threshold from your own benchmark's noise floor."
    )


def _scope_lead_in(evolver: str) -> str:
    return "Only these files may change:"


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
        gap_heading=_gap_heading(evolver),
        gap_intro=_gap_intro(evolver),
        gap_action=_gap_action(evolver, spec, cand),
        scope_lead_in=_scope_lead_in(evolver),
        scope_files=_scope_files_md(spec),
        objective=spec.objective.goal,
    )
    return [GeneratedFile(path="README.md", text=readme)]


# --- materialization / writer --------------------------------------------


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


def _generated_destination(bundle_path: Path, bundle_root: Path, path: str) -> Path:
    """Resolve a generated file path and reject bundle path escapes."""
    rel = Path(path)
    if rel.is_absolute() or not rel.parts or ".." in rel.parts:
        raise GeneratedPathError(f"invalid generated file path: {path!r}")

    dest = bundle_path / rel
    resolved_dest = dest.parent.resolve() / dest.name
    if not resolved_dest.is_relative_to(bundle_root):
        raise GeneratedPathError(f"generated file path escapes bundle directory: {path!r}")
    return dest


__all__ = [
    "BundleResult",
    "PrepEvolveConfig",
    "PrepEvolveInput",
    "PrepEvolveResult",
    "SkippedEvolver",
    "prep_evolve",
]
