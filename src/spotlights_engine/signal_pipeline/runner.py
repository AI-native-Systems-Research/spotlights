"""Signal-pipeline orchestrator.

Implements the run loop:

1. `_check_layout()` — precondition gate (origin/main layout present).
2. Mkdir `run_dir`; persist `input.json` if absent.
3. Apply `--inject` specs in stage order, each fully validated *before* any write.
4. For each stage in the requested selection: skip if complete (per-shape rule)
   under `--resume`, else run.
5. Return `SignalPipelineResult`.
"""

from __future__ import annotations

import concurrent.futures
import hashlib
import json
import shutil
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Iterable

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from spotlights_engine.signal_pipeline.event_formatter import StageEventFormatter
from spotlights_engine.signal_pipeline.signal_summary import emit_signal_summary
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
from spotlights_engine.signal_pipeline.spotlight_report import emit_spotlight_report


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


def _fmt_stage_duration(seconds: float) -> str:
    seconds = max(0.0, float(seconds))
    if seconds < 60:
        return f"{seconds:.1f}s"
    m, s = divmod(int(round(seconds)), 60)
    return f"{m}:{s:02d}"


class StageStatus(BaseModel):
    model_config = ConfigDict(extra="forbid")

    state: str = "pending"  # "done" | "pending" | other free-form for now
    started_at: str | None = None
    ended_at: str | None = None
    error: str | None = None
    issues: list[str] = Field(default_factory=list)
    # Populated post-stage from `_logs/<stage>/**/meta.json` written by
    # `claude_subprocess.run_claude`. Stage 02 goes through main's
    # `modules_extractor` (no meta.json), so these stay None for it until
    # main grows equivalent metadata.
    model: str | None = None
    cost_usd: float | None = None
    duration_s: float | None = None


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


# ── Effective model resolution ───────────────────────────────────────────


def _resolve_stage_model(spec: Any, signal_input: SignalPipelineInput) -> str | None:
    """Pick the model id for one stage's `claude -p` invocation.

    Precedence (highest first):
      1. `StageSpec.model` — pinned per-stage in `stages/sNN.py`.
      2. `SignalPipelineInput.model` — set via `--model <id>` for this run.
      3. `signal_pipeline.DEFAULT_MODEL` — project-wide constant.
      4. None → `claude -p` falls back to its own default (Opus today).

    Reads `DEFAULT_MODEL` via attribute lookup so monkeypatching the
    package attribute in tests behaves as expected.
    """
    if spec.model is not None:
        return spec.model
    if signal_input.model is not None:
        return signal_input.model
    from spotlights_engine import signal_pipeline as _sp_pkg

    return getattr(_sp_pkg, "DEFAULT_MODEL", None)


# ── Meta aggregation (model + cost from claude_subprocess) ───────────────


def _aggregate_meta(log_dir: Path) -> dict[str, Any]:
    """Walk a stage's log dir, sum cost across `meta.json` files, dedupe models.

    Each `claude -p` invocation writes one `meta.json` (see
    `claude_subprocess._write_meta`). Fan-out stages produce multiple
    (`<id>/meta.json`) — we sum cost and collapse models to a unique
    list. Stages that don't go through `claude_subprocess` (stage 02
    via `modules_extractor`) just won't have any files; result is empty.
    """
    if not log_dir.exists():
        return {}
    models: list[str] = []
    cost_total: float = 0.0
    saw_cost = False
    for meta_path in sorted(log_dir.rglob("meta.json")):
        try:
            data = json.loads(meta_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        m = data.get("model")
        if isinstance(m, str) and m not in models:
            models.append(m)
        c = data.get("cost_usd")
        if isinstance(c, (int, float)):
            cost_total += float(c)
            saw_cost = True
    out: dict[str, Any] = {}
    if models:
        # Single-model run is the common case — flatten for readability.
        out["model"] = models[0] if len(models) == 1 else ",".join(models)
    if saw_cost:
        out["cost_usd"] = round(cost_total, 4)
    return out


def _aggregate_run_cost(status: "PipelineStatus") -> float | None:
    """Sum `cost_usd` across every stage that recorded one.

    Returns None if no stage carried cost (e.g. the run only exercised the
    pre-cooked-JSON branches that don't go through claude_subprocess).
    """
    total = 0.0
    saw_any = False
    for stage_status in status.stages.values():
        if stage_status.cost_usd is not None:
            total += float(stage_status.cost_usd)
            saw_any = True
    return round(total, 4) if saw_any else None


def _compute_run_id(layout: RunDirLayout) -> str:
    """Deterministic, resume-stable run id derived from `input.json`.

    Mirrors the DR pipeline's `_run_id_from_manifest`
    (`spotlights_manager/orchestrator.py`): hash the run's content-addressed
    inputs and prefix with `run-`. Same input -> same id across resumes
    (resume-stable for log correlation); different inputs in the same
    artifacts dir get different ids (so two consecutive runs into the
    same `--artifacts-dir` are no longer indistinguishable as they were
    when `run_id` was just `layout.root.name`).

    Falls back to a hash of the artifacts-dir path so the field stays
    non-empty when called before `input.json` has been persisted (e.g.
    in a partial run that errored very early); `RunInfo.run_id` requires
    `min_length=1`, so an empty string is unsafe.
    """
    input_path = layout.input_path
    if input_path.exists():
        try:
            payload = json.loads(input_path.read_text(encoding="utf-8"))
            canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"))
            return "run-" + hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:16]
        except (OSError, ValueError):
            pass
    fallback_seed = str(layout.root.resolve())
    return "run-" + hashlib.sha256(fallback_seed.encode("utf-8")).hexdigest()[:16]


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
    # Each fan-out item is a separate `claude -p` invocation, so flush the
    # progress formatter between items to prevent burst dedup from collapsing
    # cross-item events that happen to share a signature.
    on_event = ctx.on_event
    flush = getattr(on_event, "flush", None) if on_event is not None else None
    for id_ in expected_ids:
        if id_ in keep:
            continue
        if flush is not None:
            flush()
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
    output_folder: Path | None = None,
    stages: StageSelection | None = None,
    resume: bool = True,
    inject: Iterable[InjectSpec] | None = None,
    on_event: Callable[[str], None] | None = None,
) -> SignalPipelineResult:
    """Run (a subset of) the signal-based discovery pipeline.

    `output_folder` is where the human-readable rollup (`findings.{json,md}`)
    lands. Defaults to `<run_dir>/report/` so the whole run still tars/shares
    as a single directory; pass an external path when publishing the rollup
    independently of the artifacts.
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
    resolved_output_folder = (
        output_folder.resolve() if output_folder is not None else layout.root / "report"
    )
    run_started_at = _now_iso()

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

    # ── Pre-flight: every declared upstream of a selected stage must be
    # complete, *unless* that upstream is itself in the selection (we'll
    # run it ourselves in this invocation).
    selected_set = set(sel.selected())
    needed_upstream: set[StageId] = set()
    for stage_id in sel.selected():
        needed_upstream.update(STAGES[stage_id].upstream)
    needed_upstream -= selected_set

    for up_id in sorted(needed_upstream):
        spec = STAGES[up_id]
        upstreams = _try_load_upstreams(spec, STAGES, layout)
        if upstreams is None:
            raise PipelineLayoutError(
                f"selection {sel.from_stage}..{sel.to_stage} requires stage "
                f"{up_id} to be complete on disk (its upstream is missing)"
            )
        complete, _issues = _is_complete(spec, layout, status, upstreams)
        if not complete:
            raise PipelineLayoutError(
                f"selection {sel.from_stage}..{sel.to_stage} requires stage "
                f"{up_id} to be complete; it isn't (run earlier stages first "
                f"or extend the selection)"
            )

    # ── Run loop (topological, parallel) ────────────────────────────────
    # Stages whose declared upstreams are disjoint (e.g. 01 and 02) run
    # concurrently in the same batch; stages with dependencies on other
    # selected stages wait for their upstreams to land. For the canonical
    # 01..05 selection the schedule is {01, 02} → {03} → {04} → {05}.
    #
    # `status` mutation is protected by a lock; on-disk writes are atomic
    # via tmp+replace already. Stage artifact paths are disjoint so file
    # writes don't contend.
    completed: list[str] = []
    skipped: list[str] = []
    aggregate_issues: list[str] = []
    status_lock = threading.Lock()

    sink: Callable[[str], None] = (
        on_event if on_event is not None else lambda s: print(s, flush=True)
    )

    selected = sel.selected()
    selected_set: set[StageId] = set(selected)

    def _execute_one_stage(stage_id: StageId) -> tuple[str, StageId, list[str]]:
        """Run a single stage end-to-end: resume-check, run, status update.

        Returns (`outcome`, stage_id, issues) where outcome is "skipped"
        or "completed". Raises on failure — the scheduler catches and
        re-raises after the rest of the batch finishes.
        """
        spec = STAGES[stage_id]
        upstreams = _load_upstreams(spec, STAGES, layout)

        if resume:
            complete, issues = _is_complete(spec, layout, status, upstreams)
            if complete:
                sink(f"[{stage_id}] SKIP — already complete")
                return "skipped", stage_id, issues

        with status_lock:
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

        formatter = StageEventFormatter(stage_id, sink=sink)
        sink(f"[{stage_id}] START — {spec.name} ({spec.shape})")
        stage_t0 = time.monotonic()

        ctx_for = (
            lambda s, _layout=layout, _input=input, _ups=upstreams,
            _log=log_dir, _on=formatter: StageContext(
                stage_id=s.stage_id,
                layout=_layout,
                signal_input=_input,
                upstream=_ups,
                log_dir=_log,
                on_event=_on,
                model=_resolve_stage_model(s, _input),
            )
        )

        try:
            if spec.shape == "single":
                _run_single(spec, ctx_for)
                issues = []
            else:
                issues = _run_fanout(
                    spec,
                    layout,
                    ctx_for,
                    upstream_payload=upstreams[spec.upstream_for_hash],
                    no_resume=not resume,
                )
        except Exception as exc:
            formatter.flush()
            stage_elapsed = time.monotonic() - stage_t0
            sink(
                f"[{stage_id}] FAILED in {_fmt_stage_duration(stage_elapsed)}: "
                f"{type(exc).__name__}: {str(exc)[:200]}"
            )
            with status_lock:
                status.set(
                    stage_id,
                    StageStatus(
                        state="failed",
                        started_at=status.get(stage_id).started_at,
                        ended_at=_now_iso(),
                        error=f"{type(exc).__name__}: {exc}",
                    ),
                )
                _save_status(layout, status)
            raise

        formatter.flush()
        stage_elapsed = time.monotonic() - stage_t0
        meta = _aggregate_meta(log_dir)
        sink(f"[{stage_id}] STAGE DONE in {_fmt_stage_duration(stage_elapsed)}")

        with status_lock:
            status.set(
                stage_id,
                StageStatus(
                    state="done",
                    started_at=status.get(stage_id).started_at,
                    ended_at=_now_iso(),
                    error=None,
                    issues=issues,
                    model=meta.get("model"),
                    cost_usd=meta.get("cost_usd"),
                    duration_s=round(stage_elapsed, 3),
                ),
            )
            _save_status(layout, status)
        return "completed", stage_id, issues

    settled: set[StageId] = set()  # selected stages already completed/skipped this run
    max_workers = min(len(selected_set), 4) if selected_set else 1
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as ex:
        while len(settled) < len(selected_set):
            ready = [
                sid for sid in selected
                if sid not in settled
                and all(
                    up not in selected_set or up in settled
                    for up in STAGES[sid].upstream
                )
            ]
            if not ready:
                # Should be unreachable: STAGES is a DAG and selection is
                # contiguous over a topological order. Defensive guard.
                raise PipelineLayoutError(
                    f"scheduler deadlock: no runnable stage in {sorted(selected_set - settled)}"
                )

            futures = {ex.submit(_execute_one_stage, sid): sid for sid in ready}
            first_error: BaseException | None = None
            for fut in concurrent.futures.as_completed(futures):
                try:
                    outcome, stage_id, issues = fut.result()
                except BaseException as e:  # noqa: BLE001 — re-raise after batch drains
                    if first_error is None:
                        first_error = e
                    continue
                settled.add(stage_id)
                if outcome == "completed":
                    completed.append(stage_id)
                else:
                    skipped.append(stage_id)
                aggregate_issues.extend(issues)
            if first_error is not None:
                # Sibling stages in the batch were allowed to finish so their
                # work isn't lost; preserve "stop on first error" semantics
                # by re-raising once the batch drains.
                raise first_error

    # Best-effort findings rollup — joins stage 03 candidates with stage 04
    # change specs into a single human-readable view. Skipped silently if the
    # candidates artifact isn't on disk yet (e.g. selection ended before 03).
    try:
        emit_signal_summary(layout, resolved_output_folder)
    except Exception as exc:  # noqa: BLE001 — never fail the pipeline on rollup
        aggregate_issues.append(f"findings rollup failed: {exc}")

    # Best-effort SpotlightReport emission — produces the unified report at
    # `<run_dir>/spotlight_report.json` per docs/specs/spotlight_report.md.
    # Skipped silently when the run didn't get far enough to have the upstream
    # artifacts (signals + project_tree + candidates); a partial run that
    # ended before stage 04 still produces a valid report with empty
    # `proposals` lists. Errors here are recorded but never abort the run.
    try:
        cost_total = _aggregate_run_cost(status)
        emit_spotlight_report(
            layout,
            run_id=_compute_run_id(layout),
            started_at=run_started_at,
            finished_at=_now_iso(),
            cost_usd=cost_total,
        )
    except Exception as exc:  # noqa: BLE001
        aggregate_issues.append(f"spotlight_report emission failed: {exc}")

    # Sort to keep the result deterministic across runs — under the
    # parallel scheduler these lists' append order depends on which
    # future finishes first, but the stage IDs are zero-padded so
    # lexicographic == declaration order.
    return SignalPipelineResult(
        run_dir=layout.root,
        output_folder=resolved_output_folder,
        completed_stages=sorted(completed),
        skipped_stages=sorted(skipped),
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
