"""Signal-pipeline orchestrator.

Implements the run loop:

1. `_check_layout()` — precondition gate (origin/main layout present).
2. Mkdir `run_dir`; persist `input.json` if absent.
3. Apply `--inject` specs in stage order, each fully validated *before* any write.
4. For each stage in the requested selection: skip if complete (per-shape rule)
   under `--resume`, else run.
5. Return `SignalPipelineResult`.

Resume rules and `--inject` validation tables are pinned in the approved
plan at `/Users/idanfr/.claude/plans/humble-plotting-cook.md`. Anything
here that drifts from that doc is a bug.
"""

from __future__ import annotations

import json
import shutil
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from spotlights_engine.signal_pipeline.layout import (
    ALL_STAGES,
    RunDirLayout,
    StageId,
    StageShape,
    artifact_hash,
    atomic_write_json,
    atomic_write_text,
    validate_entry_id,
)
from spotlights_engine.signal_pipeline.schemas import (
    SignalPipelineInput,
    SignalPipelineResult,
)


__all__ = [
    "InjectSpec",
    "PipelineLayoutError",
    "PipelinePreconditionError",
    "InjectValidationError",
    "SignalPipelineInput",
    "SignalPipelineResult",
    "StageId",
    "StageSelection",
    "run_pipeline",
]


# ── Errors ───────────────────────────────────────────────────────────────


class PipelinePreconditionError(RuntimeError):
    """Raised when the runner can't start (layout missing, run dir invalid)."""


class PipelineLayoutError(RuntimeError):
    """Raised when an upstream artifact is missing for the requested stage range."""


class InjectValidationError(ValueError):
    """Raised when an `--inject` spec fails any validation check.

    Hard error; partial application doesn't happen — the runner validates
    every inject before writing anything.
    """


# ── Layout precondition ──────────────────────────────────────────────────


def _check_layout() -> None:
    """Verify the origin/main layout is importable.

    Called as the very first thing in `run_pipeline`. Stage modules also
    lazy-import their main-only dependencies inside `run` / `run_one` so a
    missing layout fails here with a clear message rather than as an
    obscure ImportError elsewhere.
    """
    try:
        from spotlights_engine.modules_extractor import extract  # noqa: F401
        from spotlights_engine.schemas import (  # noqa: F401
            candidate,
            common,
            project,
        )
    except ImportError as e:
        raise PipelinePreconditionError(
            "signal_pipeline requires the origin/main layout — rebase the branch first"
        ) from e


# ── Stage selection (CLI: --from-stage / --to-stage / --only-stage) ──────


@dataclass(frozen=True)
class StageSelection:
    """Which stages to run; orthogonal to `resume`.

    Defaults to all 5 stages. `--only-stage 03` is the same shape as
    `--from-stage 03 --to-stage 03`; the implementation order doc pairs
    `--only-stage` with `--no-resume`, but that's the CLI layer's call.
    """

    from_stage: StageId = "01"
    to_stage: StageId = "05"

    @classmethod
    def all(cls) -> "StageSelection":
        return cls()

    @classmethod
    def only(cls, stage: StageId) -> "StageSelection":
        return cls(from_stage=stage, to_stage=stage)

    def __post_init__(self) -> None:
        if self.from_stage > self.to_stage:  # lexicographic over "01".."05" is correct
            raise ValueError(
                f"from_stage {self.from_stage!r} > to_stage {self.to_stage!r}"
            )

    def selected(self) -> list[StageId]:
        return [s for s in ALL_STAGES if self.from_stage <= s <= self.to_stage]

    def upstream_required(self) -> list[StageId]:
        """Stages strictly before `from_stage` — must be already complete."""
        return [s for s in ALL_STAGES if s < self.from_stage]


# ── Inject specs ─────────────────────────────────────────────────────────


@dataclass(frozen=True)
class InjectSpec:
    """One `--inject` argument, parsed.

    Three legal shapes:
    - single-artifact stage: `stage="03", id_=None, source=<file>`
    - fan-out whole-dir:     `stage="04", id_=None, source=<dir>`
    - fan-out per-id:        `stage="04", id_="cand-0001", source=<file>`
    """

    stage: StageId
    id_: str | None
    source: Path

    @classmethod
    def parse(cls, raw: str) -> "InjectSpec":
        if "=" not in raw:
            raise InjectValidationError(
                f"--inject value must be NN=path or NN/<id>=path; got {raw!r}"
            )
        lhs, rhs = raw.split("=", 1)
        if "/" in lhs:
            stage, id_ = lhs.split("/", 1)
            if not id_:
                raise InjectValidationError(f"empty id in --inject {raw!r}")
            validate_entry_id(id_)
        else:
            stage, id_ = lhs, None
        if stage not in ALL_STAGES:
            raise InjectValidationError(
                f"unknown stage {stage!r} in --inject {raw!r}; valid: {ALL_STAGES}"
            )
        return cls(stage=stage, id_=id_, source=Path(rhs))


# ── Status file ──────────────────────────────────────────────────────────


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


class StageStatus(BaseModel):
    model_config = ConfigDict(extra="forbid")

    state: str = "pending"  # "done" | "pending" | other free-form for now
    started_at: str | None = None
    ended_at: str | None = None
    error: str | None = None
    issues: list[str] = Field(default_factory=list)


class PipelineStatus(BaseModel):
    model_config = ConfigDict(extra="forbid")

    stages: dict[str, StageStatus] = Field(default_factory=dict)

    def get(self, stage: StageId) -> StageStatus:
        return self.stages.get(stage, StageStatus())

    def set(self, stage: StageId, status: StageStatus) -> None:
        self.stages[stage] = status


def _load_status(layout: RunDirLayout) -> PipelineStatus:
    if not layout.status_path.exists():
        return PipelineStatus()
    try:
        return PipelineStatus.model_validate_json(layout.status_path.read_text())
    except (json.JSONDecodeError, ValidationError):
        # Corrupt status file shouldn't block recovery — start fresh and
        # let the per-stage completion checks decide.
        return PipelineStatus()


def _save_status(layout: RunDirLayout, status: PipelineStatus) -> None:
    atomic_write_text(
        layout.status_path,
        status.model_dump_json(indent=2) + "\n",
    )


# ── Artifact load / completeness ─────────────────────────────────────────


def _load_single_artifact(spec: Any, layout: RunDirLayout) -> Any:
    """Read + parse a single-artifact stage's canonical file."""
    path = layout.stage_artifact(spec.stage_id, shape="single")
    raw = json.loads(path.read_text(encoding="utf-8"))
    return spec.parse_artifact(raw)


def _load_fanout_dir(spec: Any, layout: RunDirLayout) -> dict[str, Any]:
    """Assemble a fan-out stage's collected view as `{id: parsed_item}`,
    sorted by id (deterministic for hashing).

    Skips `_manifest.json`; tolerates extra files only if their stems are
    valid entry ids that parse against `parse_item` — anything else is a
    schema error surfaced to the caller.
    """
    fanout_dir = layout.stage_artifact(spec.stage_id, shape="fanout")
    result: dict[str, Any] = {}
    for entry in sorted(fanout_dir.iterdir(), key=lambda p: p.name):
        if entry.name == "_manifest.json":
            continue
        if entry.suffix != ".json" or not entry.is_file():
            continue
        id_ = entry.stem
        validate_entry_id(id_)
        raw = json.loads(entry.read_text(encoding="utf-8"))
        result[id_] = spec.parse_item(raw)
    return result


def _load_manifest(layout: RunDirLayout, stage: StageId) -> dict[str, Any] | None:
    p = layout.fanout_manifest(stage)
    if not p.exists():
        return None
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return None


def _is_complete(
    spec: Any,
    layout: RunDirLayout,
    status: PipelineStatus,
    upstream: dict[StageId, Any],
) -> tuple[bool, list[str]]:
    """Per-shape completion criteria from the plan.

    Returns `(complete, issues)`. `issues` is informational — populated
    even when complete, e.g. orphan-file warnings on fan-out stages.
    """
    issues: list[str] = []
    if spec.shape == "single":
        path = layout.stage_artifact(spec.stage_id, shape="single")
        if not path.exists():
            return False, issues
        try:
            raw = json.loads(path.read_text(encoding="utf-8"))
            spec.parse_artifact(raw)
        except (json.JSONDecodeError, ValidationError, TypeError) as e:
            return False, [f"artifact failed to parse: {e}"]
        if status.get(spec.stage_id).state != "done":
            return False, issues
        return True, issues

    # Fan-out
    fanout_dir = layout.stage_artifact(spec.stage_id, shape="fanout")
    if not fanout_dir.exists():
        return False, issues
    manifest = _load_manifest(layout, spec.stage_id)
    if manifest is None:
        return False, ["manifest missing or invalid JSON"]
    upstream_payload = upstream.get(spec.upstream_for_hash)  # type: ignore[arg-type]
    if upstream_payload is None:
        # Upstream not loaded — runner shouldn't reach here for fan-out
        # without first loading upstream, but guard anyway.
        return False, ["upstream payload not loaded"]
    expected_hash = artifact_hash(upstream_payload)
    actual_hash = manifest.get(spec.upstream_hash_field)
    if actual_hash != expected_hash:
        return False, [
            f"upstream hash mismatch ({spec.upstream_hash_field}): "
            f"manifest={actual_hash!r}, current={expected_hash!r}"
        ]
    expected_ids = set(spec.ids_from_upstream(upstream_payload))
    covered_ids = set(manifest.get("covered_ids", []))
    if covered_ids != expected_ids:
        missing = expected_ids - covered_ids
        orphans = covered_ids - expected_ids
        msg_parts = []
        if missing:
            msg_parts.append(f"missing ids: {sorted(missing)}")
        if orphans:
            msg_parts.append(f"orphan ids: {sorted(orphans)}")
        return False, ["covered_ids mismatch — " + "; ".join(msg_parts)]
    # Each entry parses
    for id_ in expected_ids:
        entry_path = layout.fanout_entry(spec.stage_id, id_)
        if not entry_path.exists():
            return False, [f"entry file missing: {entry_path.name}"]
        try:
            raw = json.loads(entry_path.read_text(encoding="utf-8"))
            spec.parse_item(raw)
        except (json.JSONDecodeError, ValidationError, TypeError) as e:
            return False, [f"entry {id_} failed to parse: {e}"]
    # Orphan files (filenames not in expected_ids) are non-fatal — flagged
    # but don't block completion.
    for entry in fanout_dir.iterdir():
        if entry.name == "_manifest.json" or entry.suffix != ".json":
            continue
        if entry.stem not in expected_ids:
            issues.append(f"orphan file in fan-out dir: {entry.name}")
    if status.get(spec.stage_id).state != "done":
        return False, issues
    return True, issues


# ── Inject application ───────────────────────────────────────────────────


def _validate_and_apply_inject(
    inj: InjectSpec,
    *,
    spec: Any,
    layout: RunDirLayout,
    status: PipelineStatus,
    upstream_loader,
) -> None:
    """Validate then apply one inject. Raises `InjectValidationError` on any failure.

    `upstream_loader(stage_id) -> Any | None` returns the parsed upstream
    payload if available, else None.
    """
    if not inj.source.exists():
        raise InjectValidationError(
            f"--inject source does not exist: {inj.source}"
        )

    if spec.shape == "single":
        if inj.id_ is not None:
            raise InjectValidationError(
                f"stage {inj.stage} is single-artifact; per-id inject not allowed"
            )
        if inj.source.is_dir():
            raise InjectValidationError(
                f"stage {inj.stage} is single-artifact; expected a file, got a directory: {inj.source}"
            )
        try:
            raw = json.loads(inj.source.read_text(encoding="utf-8"))
            spec.parse_artifact(raw)
        except (json.JSONDecodeError, ValidationError, TypeError) as e:
            raise InjectValidationError(
                f"--inject {inj.stage}={inj.source}: payload failed validation: {e}"
            ) from e
        target = layout.stage_artifact(inj.stage, shape="single")
        atomic_write_json(target, raw)
        _set_done(status, inj.stage)
        return

    # Fan-out — check the inject's *shape* (per-id vs whole-dir) before we
    # need to know the upstream id set. Shape mismatches give clearer errors
    # than "upstream missing" when the user typed `04=file.json` instead of
    # `04/<id>=file.json` or `04=dir/`.
    if inj.id_ is None and not inj.source.is_dir():
        raise InjectValidationError(
            f"--inject {_inject_label(inj)}: stage {inj.stage} is fan-out and the "
            f"value isn't `<id>`-qualified; expected a directory, got a file. Use "
            f"`--inject {inj.stage}/<id>=file` for per-id replacement or "
            f"`--inject {inj.stage}=dir/` for whole-dir."
        )
    if inj.id_ is not None and inj.source.is_dir():
        raise InjectValidationError(
            f"--inject {_inject_label(inj)}: per-id inject expects a file, got a directory"
        )

    upstream_payload = upstream_loader(spec.upstream_for_hash)
    if upstream_payload is None:
        raise InjectValidationError(
            f"--inject {_inject_label(inj)}: stage {inj.stage} is fan-out but its "
            f"upstream stage {spec.upstream_for_hash} has no artifact yet — inject the "
            f"upstream first (or run it) so the id set is known"
        )
    expected_ids = list(spec.ids_from_upstream(upstream_payload))
    expected_set = set(expected_ids)

    if inj.id_ is not None:
        if inj.id_ not in expected_set:
            raise InjectValidationError(
                f"--inject {_inject_label(inj)}: id {inj.id_!r} not in upstream of "
                f"stage {inj.stage}; valid ids: {sorted(expected_set)}"
            )
        try:
            raw = json.loads(inj.source.read_text(encoding="utf-8"))
            spec.parse_item(raw)
        except (json.JSONDecodeError, ValidationError, TypeError) as e:
            raise InjectValidationError(
                f"--inject {_inject_label(inj)}: payload failed validation: {e}"
            ) from e
        target = layout.fanout_entry(inj.stage, inj.id_)
        atomic_write_json(target, raw)
        # Refresh manifest from current upstream + on-disk entries.
        _refresh_fanout_manifest(spec, layout, upstream_payload)
        return

    # Whole-dir form
    if not inj.source.is_dir():
        raise InjectValidationError(
            f"--inject {_inject_label(inj)}: stage {inj.stage} is fan-out and the "
            f"value isn't `<id>`-qualified; expected a directory, got a file. Use "
            f"`--inject {inj.stage}/<id>=file` for per-id replacement or "
            f"`--inject {inj.stage}=dir/` for whole-dir."
        )
    src_files = sorted(p for p in inj.source.iterdir() if p.is_file() and p.suffix == ".json")
    src_ids = {p.stem for p in src_files if p.name != "_manifest.json"}
    missing = expected_set - src_ids
    orphans = src_ids - expected_set
    if missing:
        raise InjectValidationError(
            f"--inject {_inject_label(inj)}: missing files for upstream ids: {sorted(missing)}"
        )
    if orphans:
        raise InjectValidationError(
            f"--inject {_inject_label(inj)}: orphan files for non-upstream ids: {sorted(orphans)} "
            f"(remove them or fix the upstream first)"
        )
    # All filenames map to expected ids. Schema-validate each.
    parsed: dict[str, Any] = {}
    for p in src_files:
        if p.name == "_manifest.json":
            continue
        try:
            raw = json.loads(p.read_text(encoding="utf-8"))
            spec.parse_item(raw)
            parsed[p.stem] = raw
        except (json.JSONDecodeError, ValidationError, TypeError) as e:
            raise InjectValidationError(
                f"--inject {_inject_label(inj)}: file {p.name} failed validation: {e}"
            ) from e
    # Atomically replace the dir contents.
    fanout_dir = layout.stage_artifact(inj.stage, shape="fanout")
    if fanout_dir.exists():
        shutil.rmtree(fanout_dir)
    fanout_dir.mkdir(parents=True)
    for id_, raw in parsed.items():
        atomic_write_json(layout.fanout_entry(inj.stage, id_), raw)
    _refresh_fanout_manifest(spec, layout, upstream_payload)
    _set_done(status, inj.stage)


def _inject_label(inj: InjectSpec) -> str:
    lhs = inj.stage if inj.id_ is None else f"{inj.stage}/{inj.id_}"
    return f"{lhs}={inj.source}"


def _refresh_fanout_manifest(spec: Any, layout: RunDirLayout, upstream_payload: Any) -> None:
    expected_ids = list(spec.ids_from_upstream(upstream_payload))
    fanout_dir = layout.stage_artifact(spec.stage_id, shape="fanout")
    on_disk_ids = sorted(
        p.stem
        for p in fanout_dir.iterdir()
        if p.is_file() and p.suffix == ".json" and p.name != "_manifest.json"
    )
    covered = [i for i in expected_ids if i in set(on_disk_ids)]
    manifest: dict[str, Any] = {
        spec.upstream_hash_field: artifact_hash(upstream_payload),
        "covered_ids": covered,
    }
    atomic_write_json(layout.fanout_manifest(spec.stage_id), manifest)


# ── Stage execution ──────────────────────────────────────────────────────


def _set_done(status: PipelineStatus, stage: StageId) -> None:
    entry = status.get(stage)
    status.set(
        stage,
        StageStatus(
            state="done",
            started_at=entry.started_at,
            ended_at=_now_iso(),
            error=None,
            issues=entry.issues,
        ),
    )


def _load_upstreams(
    spec: Any, registry: dict[StageId, Any], layout: RunDirLayout
) -> dict[StageId, Any]:
    """Load every artifact this stage names as an upstream.

    For single-artifact upstreams: parsed payload.
    For fan-out upstreams: `dict[id, parsed_item]` (keys sorted).
    """
    out: dict[StageId, Any] = {}
    for up_id in spec.upstream:
        up_spec = registry[up_id]
        if up_spec.shape == "single":
            path = layout.stage_artifact(up_id, shape="single")
            if not path.exists():
                raise PipelineLayoutError(
                    f"stage {spec.stage_id} requires upstream {up_id} but "
                    f"{path.name} does not exist (run that stage first or use --inject)"
                )
            out[up_id] = _load_single_artifact(up_spec, layout)
        else:
            fanout_dir = layout.stage_artifact(up_id, shape="fanout")
            if not fanout_dir.exists():
                raise PipelineLayoutError(
                    f"stage {spec.stage_id} requires upstream {up_id} but "
                    f"{fanout_dir.name}/ does not exist"
                )
            out[up_id] = _load_fanout_dir(up_spec, layout)
    # Fan-out stages also need the hash-upstream loaded (if not already).
    if spec.shape == "fanout" and spec.upstream_for_hash not in out:
        # Currently `upstream_for_hash` always equals (or is the only entry of)
        # `spec.upstream` for stages 04/05, but guard against drift.
        raise PipelineLayoutError(
            f"stage {spec.stage_id} declares upstream_for_hash={spec.upstream_for_hash!r} "
            f"but it isn't in its `upstream` tuple; fix the StageSpec"
        )
    return out


def _run_single(spec: Any, ctx_factory) -> None:
    ctx = ctx_factory(spec)
    payload = spec.run(ctx)
    raw = (
        payload.model_dump(mode="json", by_alias=True, exclude_none=False)
        if isinstance(payload, BaseModel)
        else _coerce_for_json(payload)
    )
    atomic_write_json(
        ctx.layout.stage_artifact(spec.stage_id, shape="single"),
        raw,
    )


def _coerce_for_json(payload: Any) -> Any:
    """Recursively coerce pydantic models inside lists/dicts to JSON-friendly form."""
    if isinstance(payload, BaseModel):
        return payload.model_dump(mode="json", by_alias=True, exclude_none=False)
    if isinstance(payload, list):
        return [_coerce_for_json(p) for p in payload]
    if isinstance(payload, dict):
        return {k: _coerce_for_json(v) for k, v in payload.items()}
    return payload


def _run_fanout(
    spec: Any,
    layout: RunDirLayout,
    ctx_factory,
    upstream_payload: Any,
    no_resume: bool,
) -> list[str]:
    """Run a fan-out stage. Returns the list of issues to record."""
    fanout_dir = layout.stage_artifact(spec.stage_id, shape="fanout")
    if no_resume and fanout_dir.exists():
        shutil.rmtree(fanout_dir)
    fanout_dir.mkdir(parents=True, exist_ok=True)

    expected_ids = list(spec.ids_from_upstream(upstream_payload))
    expected_set = set(expected_ids)

    # Identify which entries already exist and parse cleanly — those we keep.
    keep: set[str] = set()
    if not no_resume:
        for id_ in expected_ids:
            entry_path = layout.fanout_entry(spec.stage_id, id_)
            if not entry_path.exists():
                continue
            try:
                raw = json.loads(entry_path.read_text(encoding="utf-8"))
                spec.parse_item(raw)
                keep.add(id_)
            except (json.JSONDecodeError, ValidationError, TypeError):
                # Corrupt entry — drop it and re-run.
                entry_path.unlink()

    # Run missing ids.
    ctx = ctx_factory(spec)
    for id_ in expected_ids:
        if id_ in keep:
            continue
        payload = spec.run_one(ctx, id_)
        raw = (
            payload.model_dump(mode="json", by_alias=True, exclude_none=False)
            if isinstance(payload, BaseModel)
            else _coerce_for_json(payload)
        )
        atomic_write_json(layout.fanout_entry(spec.stage_id, id_), raw)

    # Manifest written last per the atomicity discipline (per-entry first).
    _refresh_fanout_manifest(spec, layout, upstream_payload)

    # Surface orphans as issues (we don't auto-delete).
    issues: list[str] = []
    for entry in fanout_dir.iterdir():
        if entry.name == "_manifest.json" or entry.suffix != ".json":
            continue
        if entry.stem not in expected_set:
            issues.append(f"orphan file in fan-out dir: {entry.name}")
    return issues


# ── Public entrypoint ────────────────────────────────────────────────────


def run_pipeline(
    input: SignalPipelineInput,
    *,
    run_dir: Path,
    stages: StageSelection | None = None,
    resume: bool = True,
    inject: Iterable[InjectSpec] | None = None,
) -> SignalPipelineResult:
    """Run (a subset of) the signal-based discovery pipeline.

    See `/Users/idanfr/.claude/plans/humble-plotting-cook.md` Phase 3 for
    the resume / inject / stage-selection semantics.
    """
    _check_layout()

    # Imported here, after `_check_layout`, so the registry's stage modules
    # (which import from `schemas.candidate` etc.) only resolve once layout
    # is confirmed.
    from spotlights_engine.signal_pipeline.stages import STAGES
    from spotlights_engine.signal_pipeline.stages._types import StageContext

    sel = stages or StageSelection.all()
    layout = RunDirLayout(run_dir.resolve())
    layout.root.mkdir(parents=True, exist_ok=True)

    # Persist the input contract (idempotent).
    if not layout.input_path.exists():
        atomic_write_json(
            layout.input_path,
            input.model_dump(mode="json", by_alias=True, exclude_none=False),
        )

    status = _load_status(layout)

    # ── Validate + apply injects (in stage order) ───────────────────────
    if inject:
        # Sort by stage so an inject of stage 03 lands before a fan-out
        # inject of stage 04 that depends on it.
        for inj in sorted(inject, key=lambda i: (i.stage, i.id_ or "")):
            spec = STAGES[inj.stage]
            _validate_and_apply_inject(
                inj,
                spec=spec,
                layout=layout,
                status=status,
                upstream_loader=lambda up_id, _layout=layout, _registry=STAGES: (
                    _try_load(up_id, _registry, _layout)
                ),
            )
            _save_status(layout, status)

    # ── Pre-flight: every stage strictly before from_stage must be done ─
    for up_id in sel.upstream_required():
        spec = STAGES[up_id]
        upstreams = _try_load_upstreams(spec, STAGES, layout)
        if upstreams is None:
            raise PipelineLayoutError(
                f"--from-stage {sel.from_stage} requires stage {up_id} to be "
                f"complete on disk (its upstream is missing)"
            )
        complete, _issues = _is_complete(spec, layout, status, upstreams)
        if not complete:
            raise PipelineLayoutError(
                f"--from-stage {sel.from_stage} requires stage {up_id} to be "
                f"complete; it isn't (run earlier stages first or remove "
                f"--from-stage)"
            )

    # ── Run loop ────────────────────────────────────────────────────────
    completed: list[str] = []
    skipped: list[str] = []
    aggregate_issues: list[str] = []

    for stage_id in sel.selected():
        spec = STAGES[stage_id]
        upstreams = _load_upstreams(spec, STAGES, layout)

        if resume:
            complete, issues = _is_complete(spec, layout, status, upstreams)
            if complete:
                skipped.append(stage_id)
                aggregate_issues.extend(issues)
                continue

        # Mark in-progress before doing work.
        status.set(
            stage_id,
            StageStatus(
                state="in_progress",
                started_at=_now_iso(),
                ended_at=None,
            ),
        )
        _save_status(layout, status)

        log_dir = layout.stage_log_dir(stage_id)
        log_dir.mkdir(parents=True, exist_ok=True)
        ctx_for = lambda s, _layout=layout, _input=input, _ups=upstreams, _log=log_dir: (
            StageContext(
                stage_id=s.stage_id,
                layout=_layout,
                signal_input=_input,
                upstream=_ups,
                log_dir=_log,
            )
        )

        if spec.shape == "single":
            _run_single(spec, ctx_for)
            issues: list[str] = []
        else:
            issues = _run_fanout(
                spec,
                layout,
                ctx_for,
                upstream_payload=upstreams[spec.upstream_for_hash],
                no_resume=not resume,
            )

        # Mark done.
        status.set(
            stage_id,
            StageStatus(
                state="done",
                started_at=status.get(stage_id).started_at,
                ended_at=_now_iso(),
                error=None,
                issues=issues,
            ),
        )
        _save_status(layout, status)
        completed.append(stage_id)
        aggregate_issues.extend(issues)

    return SignalPipelineResult(
        run_dir=layout.root,
        completed_stages=completed,
        skipped_stages=skipped,
        issues=aggregate_issues,
    )


def _try_load(stage_id: StageId, registry: dict[StageId, Any], layout: RunDirLayout) -> Any | None:
    spec = registry[stage_id]
    if spec.shape == "single":
        path = layout.stage_artifact(stage_id, shape="single")
        if not path.exists():
            return None
        try:
            return _load_single_artifact(spec, layout)
        except (json.JSONDecodeError, ValidationError, TypeError):
            return None
    fanout_dir = layout.stage_artifact(stage_id, shape="fanout")
    if not fanout_dir.exists():
        return None
    try:
        return _load_fanout_dir(spec, layout)
    except (json.JSONDecodeError, ValidationError, TypeError):
        return None


def _try_load_upstreams(
    spec: Any, registry: dict[StageId, Any], layout: RunDirLayout
) -> dict[StageId, Any] | None:
    """Like `_load_upstreams` but returns None if any upstream is missing/corrupt
    instead of raising — used by the from-stage pre-flight check."""
    out: dict[StageId, Any] = {}
    for up_id in spec.upstream:
        loaded = _try_load(up_id, registry, layout)
        if loaded is None:
            return None
        out[up_id] = loaded
    return out
