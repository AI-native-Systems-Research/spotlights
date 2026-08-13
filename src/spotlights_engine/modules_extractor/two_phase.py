"""Two-phase module extraction (Idea 4): deterministic skeleton, LLM enrichment.

Orchestrates the four sequential stages described in
`design/module_extraction_fix_impl_plan.md` (the pipeline historically had a
Stage 4 semantic review, since removed; assembly keeps its Stage-5 numbering):

1. Claude + Python — repository metadata and source root.
2. Python — deterministic directory inventory / skeleton.
3. Claude + Python — enrichment, strict filesystem validation, coverage gate.
5. Python — deterministic final `ProjectTree` assembly and re-validation.

The public return type (`ExtractionRunResult`) and the serialized `ProjectTree`
schema are unchanged. When `artifacts_dir` is set, every stage persists its
inputs and outputs *as it runs*, so a finished (or failed) run is fully
reconstructable from disk. An artifact-write failure on an otherwise successful
path raises `ExtractorArtifactError`; a write failure while handling an existing
stage error is logged and never masks the original error.
"""

from __future__ import annotations

import asyncio
from collections.abc import Callable, Coroutine
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, TypeVar

from spotlights_engine.modules_extractor.agent import (
    ExtractionInvocation,
    ExtractionRunResult,
)
from spotlights_engine.modules_extractor.claude_stage import (
    ClaudeStageResult,
    add_optional_float,
    add_optional_int,
    atomic_write_json,
    atomic_write_text,
    notify,
    run_structured_claude_stage,
)
from spotlights_engine.modules_extractor.coverage import (
    CoverageReport,
    CrossArtifactError,
    compute_coverage,
    forced_repository_level_files,
    validate_enriched_tree,
    validate_source_root_decision,
)
from spotlights_engine.modules_extractor.errors import (
    ExtractorArtifactError,
    ExtractorCoverageError,
    ExtractorValidationError,
)
from spotlights_engine.modules_extractor.prompts import (
    render_enrich_prompt,
    render_enrich_shard_prompt,
    render_identify_source_root_prompt,
)
from spotlights_engine.modules_extractor.sharding import (
    EnrichShard,
    ShardPlan,
    branch_coverage,
    covers_entire_skeleton,
    derive_enrich_shards,
    has_several_source_roots,
    merge_fragments,
    owning_shard,
    validate_promotion_parent,
    validate_shard_scope,
    validate_spine_main_files,
)
from spotlights_engine.modules_extractor.skeleton import (
    ALGORITHM_PROFILE_VERSION,
    build_skeleton,
    compute_fingerprint,
    scan_source_files,
)
from spotlights_engine.modules_extractor.stage_schemas import (
    EnrichedTree,
    ExcludedSourcePath,
    Skeleton,
    SourceRootDecision,
)
from spotlights_engine.schemas.project import (
    ProjectTree,
)

if TYPE_CHECKING:
    from spotlights_engine.modules_extractor.extractor import ExtractorConfig

_STAGE_REPAIR_PAYLOAD_MAX_CHARS = 60_000

_T = TypeVar("_T")


def _run_event_loop(coro: Coroutine[Any, Any, _T]) -> _T:
    """Drive `coro` to completion from synchronous code.

    `run_two_phase_extraction` is a synchronous API, but not every caller
    reaches it from a bare thread: `spotlights_manager.orchestrator` runs step 1
    from *inside* `asyncio.run(_run_async(...))`. `asyncio.run` refuses to nest
    (`RuntimeError: asyncio.run() cannot be called from a running event loop`),
    so when this thread already owns a running loop the coroutine is handed to a
    dedicated worker thread with a loop of its own. The calling thread blocks
    either way, which is exactly what a synchronous entrypoint promises.
    """
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(coro)
    with ThreadPoolExecutor(
        max_workers=1, thread_name_prefix="extractor-shards"
    ) as pool:
        return pool.submit(asyncio.run, coro).result()


# ── Telemetry accumulation ────────────────────────────────────────────────


@dataclass
class _StageSession:
    stage: str
    provider: str
    session_id: str | None
    input_tokens: int | None
    output_tokens: int | None
    cache_read: int | None
    cache_create: int | None
    cost_usd: float | None
    duration_s: float


@dataclass
class _Telemetry:
    sessions: list[_StageSession] = field(default_factory=list)
    total_duration_s: float = 0.0
    total_cost_usd: float | None = None
    total_input_tokens: int | None = None
    total_output_tokens: int | None = None
    accepted_session_id: str | None = None

    def add_claude(
        self, stage: str, result: ClaudeStageResult[Any]
    ) -> None:
        t = result.telemetry
        usage = t.usage
        self.sessions.append(
            _StageSession(
                stage=stage,
                provider="claude",
                session_id=t.session_id,
                input_tokens=t.input_tokens,
                output_tokens=t.output_tokens,
                cache_read=usage.cache_read if usage is not None else None,
                cache_create=usage.cache_create if usage is not None else None,
                cost_usd=t.cost_usd,
                duration_s=t.duration_s,
            )
        )
        self.total_duration_s += t.duration_s
        self.total_cost_usd = add_optional_float(self.total_cost_usd, t.cost_usd)
        self.total_input_tokens = add_optional_int(
            self.total_input_tokens, t.input_tokens
        )
        self.total_output_tokens = add_optional_int(
            self.total_output_tokens, t.output_tokens
        )

    def as_dict(self) -> dict[str, Any]:
        return {
            "accepted_session_id": self.accepted_session_id,
            "total_duration_s": self.total_duration_s,
            "total_cost_usd": self.total_cost_usd,
            "total_input_tokens": self.total_input_tokens,
            "total_output_tokens": self.total_output_tokens,
            "sessions": [
                {
                    "stage": s.stage,
                    "provider": s.provider,
                    "session_id": s.session_id,
                    "input_tokens": s.input_tokens,
                    "output_tokens": s.output_tokens,
                    "cache_read": s.cache_read,
                    "cache_create": s.cache_create,
                    "cost_usd": s.cost_usd,
                    "duration_s": s.duration_s,
                }
                for s in self.sessions
            ],
        }


# ── Artifact persistence helpers ──────────────────────────────────────────


def _required_write_json(base: Path | None, rel: str, obj: object) -> None:
    """Write `obj` on an otherwise-successful path. A failure here is itself an
    extraction failure (`ExtractorArtifactError`)."""
    if base is None:
        return
    try:
        atomic_write_json(base / rel, obj)
    except OSError as exc:
        raise ExtractorArtifactError(
            f"failed to write required artifact {rel}: {exc}", artifact=rel
        ) from exc


def _required_write_text(base: Path | None, rel: str, text: str) -> None:
    if base is None:
        return
    try:
        atomic_write_text(base / rel, text)
    except OSError as exc:
        raise ExtractorArtifactError(
            f"failed to write required artifact {rel}: {exc}", artifact=rel
        ) from exc


def _best_effort_write_json(
    base: Path | None, rel: str, obj: object, on_event: Callable | None
) -> None:
    """Write during error handling: never mask the in-flight stage error."""
    if base is None:
        return
    try:
        atomic_write_json(base / rel, obj)
    except OSError as exc:  # noqa: BLE001 - must not mask the original error
        notify(on_event, f"extractor: failed to persist {rel} during error handling: {exc}")


def _persist_sessions(base: Path | None, telemetry: _Telemetry) -> None:
    """sessions.json is rewritten atomically after every invocation."""
    _required_write_json(base, "sessions.json", telemetry.as_dict())


# ── Public entrypoint ─────────────────────────────────────────────────────


def run_two_phase_extraction(
    repo_path: Path,
    *,
    config: ExtractorConfig,
    on_event: Callable[[str], None] | None,
    artifacts_dir: Path | None,
) -> ExtractionRunResult:
    """Run the full four-stage two-phase extraction over `repo_path`.

    `artifacts_dir` is the `modules_extractor/` run directory (already created
    by the caller) or `None` to disable persistence.
    """
    base = artifacts_dir
    telemetry = _Telemetry()

    # Repo-wide safe source scan + fingerprint, shared by Stage 1 and Stage 5.
    repo_scan = scan_source_files(repo_path, "")
    repo_fingerprint = compute_fingerprint(
        repo_path, repo_scan.files, extra_paths=repo_scan.skipped_symlinks
    )

    # ── Stage 1 — source-root decision ────────────────────────────────────
    decision = _stage1_source_root(
        repo_path,
        base=base,
        config=config,
        telemetry=telemetry,
        detected_source_files=repo_scan.files,
        repo_fingerprint=repo_fingerprint,
        on_event=on_event,
    )
    repository = decision.repository
    _persist_sessions(base, telemetry)

    # ── Stage 2 — deterministic skeleton ──────────────────────────────────
    skeleton = _stage2_skeleton(
        repo_path, decision, base=base, on_event=on_event
    )

    # ── Stage 3 — enrichment + coverage gate ──────────────────────────────
    stage3 = _stage3_enrich(
        repo_path,
        repository,
        skeleton,
        base=base,
        config=config,
        telemetry=telemetry,
        on_event=on_event,
    )
    _persist_sessions(base, telemetry)
    enriched, coverage = stage3.enriched, stage3.coverage

    # ── Stage 5 — deterministic assembly ──────────────────────────────────
    tree = _stage5_assemble(
        repo_path,
        repository,
        decision,
        skeleton,
        enriched,
        base=base,
        detected_source_files=repo_scan.files,
        repo_fingerprint=repo_fingerprint,
        on_event=on_event,
    )

    # Final top-level artifacts.
    _required_write_json(base, "enriched_tree.json", enriched.model_dump(mode="json"))
    _required_write_json(base, "coverage.json", coverage.model_dump(mode="json"))
    _persist_sessions(base, telemetry)

    invocation = ExtractionInvocation(
        session_id=telemetry.accepted_session_id,
        duration_s=telemetry.total_duration_s,
        cost_usd=telemetry.total_cost_usd,
        input_tokens=telemetry.total_input_tokens,
        output_tokens=telemetry.total_output_tokens,
    )
    return ExtractionRunResult(
        project_tree=tree,
        invocation=invocation,
        raw_payload=tree.model_dump_json(indent=2),
    )


# ── Stage 1 ───────────────────────────────────────────────────────────────


def _inject_forced_repository_level_files(
    decision: SourceRootDecision, detected_source_files: list[str]
) -> SourceRootDecision:
    """Deterministically add the mechanically-forced `repository_level_file`
    exclusions the Stage-1 model may have missed.

    A source file sitting directly at the selected root can only ever be
    excluded (the directory-only schema cannot represent it), so requiring the
    model to enumerate every one is fragile — the real vLLM run failed because a
    single root-level script (`build_vllm_ppc64le.sh`) was left unclassified.
    We compute that forced set and merge in only the files not already covered
    by an existing exclusion (the model's own classifications, and ancestor
    directory exclusions, are preserved). Semantic exclusions stay the model's
    job. The result is idempotent: files already excluded are never re-added.
    """
    source_root = decision.repository.source_root
    forced = forced_repository_level_files(source_root, detected_source_files)
    if not forced:
        return decision

    existing = {e.path for e in decision.excluded_source_paths}
    covered_by_dir = [e.path for e in decision.excluded_source_paths]

    def _already_covered(path: str) -> bool:
        if path in existing:
            return True
        return any(path.startswith(ex + "/") for ex in covered_by_dir)

    additions = [
        ExcludedSourcePath(
            path=path,
            reason="repository_level_file",
            explanation=(
                "source file directly at the source root; the directory-only "
                "module schema cannot represent it (auto-classified)"
            ),
        )
        for path in forced
        if not _already_covered(path)
    ]
    if not additions:
        return decision

    return decision.model_copy(
        update={
            "excluded_source_paths": [
                *decision.excluded_source_paths,
                *additions,
            ]
        }
    )


def _stage1_source_root(
    repo_path: Path,
    *,
    base: Path | None,
    config: ExtractorConfig,
    telemetry: _Telemetry,
    detected_source_files: list[str],
    repo_fingerprint: str,
    on_event: Callable[[str], None] | None,
) -> SourceRootDecision:
    stage_dir = base / "01_source_root" if base is not None else None
    if stage_dir is not None:
        stage_dir.mkdir(parents=True, exist_ok=True)
    base_prompt = render_identify_source_root_prompt()
    if stage_dir is not None:
        _required_write_text(base, "01_source_root/prompt.md", base_prompt)

    prompt = base_prompt
    last_errors: str | None = None
    for attempt in range(1, 3):  # initial + one repair
        attempt_dir = (
            stage_dir / f"attempt_{attempt:02d}" if stage_dir is not None else None
        )
        if attempt == 2 and last_errors is not None:
            prompt = _repair_prompt(base_prompt, last_errors, previous_payload=None)

        result = run_structured_claude_stage(
            output_type=SourceRootDecision,
            repo_path=repo_path,
            prompt=prompt,
            stage_name="source_root",
            attempt_dir=attempt_dir,
            claude_bin=config.claude_bin,
            max_turns=config.source_root_max_turns,
            timeout_s=_effective_timeout(config.source_root_timeout_s, config.timeout_s),
            on_event=on_event,
        )
        telemetry.add_claude("01_source_root", result)
        _persist_sessions(base, telemetry)

        if result.parsed is None:
            last_errors = str(result.validation_error)
            _write_attempt_validation(
                attempt_dir, ok=False, kind="pydantic", detail=last_errors
            )
            if attempt == 1:
                notify(on_event, "extractor: stage-1 parse failed; one repair")
                continue
            raise ExtractorValidationError(
                f"source-root decision failed to parse after 2 attempts: "
                f"{result.validation_error}",
                stage="source_root",
            )

        decision = _inject_forced_repository_level_files(
            result.parsed, detected_source_files
        )
        try:
            validate_source_root_decision(decision, repo_path, detected_source_files)
        except CrossArtifactError as exc:
            last_errors = str(exc)
            _write_attempt_validation(
                attempt_dir, ok=False, kind="filesystem", detail=last_errors
            )
            if attempt == 1:
                notify(on_event, "extractor: stage-1 filesystem validation failed; one repair")
                continue
            raise ExtractorValidationError(
                f"source-root decision failed filesystem validation: {exc}",
                stage="source_root",
            ) from exc

        _write_attempt_validation(attempt_dir, ok=True, kind=None, detail=None)
        _required_write_json(
            base,
            "01_source_root/validation.json",
            {
                "ok": True,
                "repo_fingerprint": repo_fingerprint,
                "algorithm_profile": ALGORITHM_PROFILE_VERSION,
                "detected_source_files": sorted(detected_source_files),
            },
        )
        _required_write_json(
            base, "01_source_root/decision.json", decision.model_dump(mode="json")
        )
        _required_write_json(
            base, "source_root_decision.json", decision.model_dump(mode="json")
        )
        return decision

    raise AssertionError("unreachable: stage-1 attempt loop exhausted")


def _write_attempt_validation(
    attempt_dir: Path | None, *, ok: bool, kind: str | None, detail: str | None
) -> None:
    if attempt_dir is None:
        return
    try:
        atomic_write_json(
            attempt_dir / "validation.json",
            {"ok": ok, "kind": kind, "detail": detail},
        )
    except OSError:
        pass


# ── Stage 2 ───────────────────────────────────────────────────────────────


def _split_exclusions(
    repo_path: Path, decision: SourceRootDecision
) -> tuple[frozenset[str], frozenset[str]]:
    dirs: set[str] = set()
    files: set[str] = set()
    for ex in decision.excluded_source_paths:
        if (repo_path / ex.path).is_dir():
            dirs.add(ex.path)
        else:
            files.add(ex.path)
    return frozenset(dirs), frozenset(files)


def _stage2_skeleton(
    repo_path: Path,
    decision: SourceRootDecision,
    *,
    base: Path | None,
    on_event: Callable[[str], None] | None,
) -> Skeleton:
    excluded_dirs, excluded_files = _split_exclusions(repo_path, decision)
    source_root = decision.repository.source_root
    skeleton = build_skeleton(
        repo_path,
        source_root,
        excluded_dirs=excluded_dirs,
        excluded_files=excluded_files,
    )
    _required_write_json(
        base,
        "02_skeleton/inputs.json",
        {
            "source_root": source_root,
            "excluded_dirs": sorted(excluded_dirs),
            "excluded_files": sorted(excluded_files),
            "algorithm_profile": ALGORITHM_PROFILE_VERSION,
        },
    )
    _required_write_json(
        base, "02_skeleton/skeleton.json", skeleton.model_dump(mode="json")
    )
    _required_write_json(
        base,
        "02_skeleton/walk_audit.json",
        {
            "ignored": skeleton.ignored,
            "excluded": skeleton.excluded,
            "organizational_only": skeleton.organizational_only,
            "skipped_symlinks": skeleton.skipped_symlinks,
        },
    )
    _required_write_json(base, "skeleton.json", skeleton.model_dump(mode="json"))
    notify(
        on_event,
        f"extractor: skeleton — {len(skeleton.required_paths())} required, "
        f"{len(skeleton.all_paths())} inventoried",
    )
    return skeleton


# ── Stage 3 ───────────────────────────────────────────────────────────────


def _effective_timeout(specific: int | None, fallback: int) -> int:
    """`None` on a stage-specific deadline means "inherit `timeout_s`"."""
    return specific if specific is not None else fallback


@dataclass
class _Stage3Result:
    """Stage-3 output: the accepted enriched tree and its coverage report.

    `plan` is `None` in `enrich_sharding="single"` mode, which reproduces
    today's monolithic Stage 3 verbatim.
    """

    enriched: EnrichedTree
    coverage: CoverageReport
    plan: ShardPlan | None = None
    fragments: dict[str, EnrichedTree] = field(default_factory=dict)
    accepted_sessions: dict[str, str | None] = field(default_factory=dict)


@dataclass
class _ShardOutcome:
    shard: EnrichShard
    fragment: EnrichedTree
    session_id: str | None
    attempts: int


class _CancelFlag:
    """Set by the first shard to fail, so a shard that wins the semaphore in the
    same tick does not start a subprocess that is already known to be pointless."""

    def __init__(self) -> None:
        self.cancelled = False


def _stage3_enrich(
    repo_path: Path,
    repository: Any,
    skeleton: Skeleton,
    *,
    base: Path | None,
    config: ExtractorConfig,
    telemetry: _Telemetry,
    on_event: Callable[[str], None] | None,
) -> _Stage3Result:
    """Enrich the skeleton, sharded by top-level branch unless mode is `single`."""
    stage_dir = base / "03_enrich" if base is not None else None
    if stage_dir is not None:
        stage_dir.mkdir(parents=True, exist_ok=True)

    # Several top-level source folders must always be enriched per-folder in
    # parallel, never lumped into one monolithic pass — even when the caller
    # asked for "single". A single top-level folder keeps today's single pass.
    if config.enrich_sharding == "single" and not has_several_source_roots(skeleton):
        return _stage3_enrich_single(
            repo_path,
            repository,
            skeleton,
            base=base,
            config=config,
            telemetry=telemetry,
            on_event=on_event,
        )
    return _stage3_enrich_sharded(
        repo_path,
        repository,
        skeleton,
        base=base,
        config=config,
        telemetry=telemetry,
        on_event=on_event,
    )


def _stage3_enrich_single(
    repo_path: Path,
    repository: Any,
    skeleton: Skeleton,
    *,
    base: Path | None,
    config: ExtractorConfig,
    telemetry: _Telemetry,
    on_event: Callable[[str], None] | None,
) -> _Stage3Result:
    """The monolithic fallback: one Claude call over the whole repository.

    Kept byte-for-byte compatible with the pre-sharding implementation for A/B
    comparison and for small repos where one call is cheaper.
    """
    repo_data = repository.model_dump(mode="json")
    skel_data = skeleton.model_dump(mode="json")
    base_prompt = render_enrich_prompt(
        repository=repo_data, skeleton=skel_data
    )
    _required_write_text(base, "03_enrich/prompt.md", base_prompt)
    _required_write_json(
        base,
        "03_enrich/data_blocks.json",
        {"repository": repo_data, "skeleton": skel_data},
    )

    prompt = base_prompt
    for attempt in range(1, 3):
        attempt_rel = f"03_enrich/attempt_{attempt:02d}"
        attempt_dir = base / attempt_rel if base is not None else None
        result: ClaudeStageResult[EnrichedTree] = run_structured_claude_stage(
            output_type=EnrichedTree,
            repo_path=repo_path,
            prompt=prompt,
            stage_name="enrich",
            attempt_dir=attempt_dir,
            claude_bin=config.claude_bin,
            max_turns=config.max_turns,
            timeout_s=config.timeout_s,
            on_event=on_event,
        )
        telemetry.add_claude("03_enrich", result)
        _persist_sessions(base, telemetry)

        if result.parsed is None:
            detail = str(result.validation_error)
            _required_write_json(
                base,
                f"{attempt_rel}/validation.json",
                {"ok": False, "kind": "pydantic", "detail": detail},
            )
            if attempt == 1:
                notify(on_event, "extractor: stage-3 parse failed; one repair")
                prompt = _repair_prompt(
                    base_prompt, detail, previous_payload=result.raw_payload
                )
                continue
            raise ExtractorValidationError(
                f"enriched tree failed to parse after 2 attempts: "
                f"{result.validation_error}",
                stage="enrich",
            )

        enriched = result.parsed
        _required_write_json(
            base, f"{attempt_rel}/enriched_tree.json", enriched.model_dump(mode="json")
        )

        validation_error: str | None = None
        try:
            validate_enriched_tree(enriched, repo_path, repository, skeleton)
        except CrossArtifactError as exc:
            validation_error = str(exc)

        coverage = compute_coverage(enriched, skeleton)
        _required_write_json(
            base, f"{attempt_rel}/coverage.json", coverage.model_dump(mode="json")
        )
        _required_write_json(
            base,
            f"{attempt_rel}/validation.json",
            {
                "ok": validation_error is None,
                "kind": "cross_artifact" if validation_error else None,
                "detail": validation_error,
            },
        )

        if validation_error is not None:
            if attempt == 1:
                notify(on_event, "extractor: stage-3 validation failed; one repair")
                prompt = _repair_prompt(
                    base_prompt, validation_error, previous_payload=result.raw_payload
                )
                continue
            raise ExtractorValidationError(
                f"enriched tree failed cross-artifact validation: {validation_error}",
                stage="enrich",
            )

        if coverage.missing:
            if attempt == 1:
                notify(
                    on_event,
                    f"extractor: stage-3 missing {len(coverage.missing)} "
                    "required paths; one repair",
                )
                prompt = _repair_prompt(
                    base_prompt,
                    "Missing required paths (must be emitted or folded): "
                    + ", ".join(coverage.missing),
                    previous_payload=result.raw_payload,
                )
                continue
            raise ExtractorCoverageError(
                f"{len(coverage.missing)} required paths not covered",
                missing=coverage.missing,
                stage="enrich",
            )

        telemetry.accepted_session_id = result.telemetry.session_id
        return _Stage3Result(enriched=enriched, coverage=coverage)

    raise AssertionError("unreachable: stage-3 attempt loop exhausted")


# ── Stage 3 — sharded ─────────────────────────────────────────────────────


def _scope_dict(shard: EnrichShard) -> dict[str, Any]:
    return {
        "key": shard.key,
        "root_path": shard.root_path,
        "owns_root": shard.owns_root,
        "is_subshard": shard.is_subshard,
        "parent_key": shard.parent_key,
        "depth": shard.depth,
        "promoted_children": list(shard.promoted_children),
        "promotion_parent": shard.promotion_parent,
    }


def _shard_plan_dict(plan: ShardPlan) -> dict[str, Any]:
    return {
        "branch_roots": dict(plan.branch_roots),
        "not_split_reasons": dict(plan.not_split_reasons),
        "shards": [
            {
                **_scope_dict(s),
                "required_nodes": len(s.subtree.required_paths()),
                "inventory_nodes": len(s.subtree.all_paths()),
            }
            for s in plan.shards
        ],
    }


def _validate_shard_fragment(
    shard: EnrichShard,
    fragment: EnrichedTree,
    *,
    repo_path: Path,
    repository: Any,
) -> str | None:
    """Subtree-scoped cross-artifact validation. Returns the error text or None.

    The shard-scoped knob is what lets the *unchanged* validator run against a
    slice: a spine defers Rule 4 for its promotion parent, because the children
    that guarantee it are pruned from its subtree. That is the promotion parent
    and nothing else — a chain-split spine holds chain nodes above it whose
    children are all present, and their Rule 4 is decided here or never.
    """
    try:
        validate_shard_scope(shard, fragment)
        validate_enriched_tree(
            fragment,
            repo_path,
            repository,
            shard.subtree,
            rule4_exempt_paths=shard.rule4_exempt_paths or None,
        )
        validate_spine_main_files(shard, fragment)
        validate_promotion_parent(shard, fragment)
    except CrossArtifactError as exc:
        return str(exc)
    return None


async def _run_one_enrich_shard(
    shard: EnrichShard,
    *,
    sem: asyncio.Semaphore,
    cancel: _CancelFlag,
    repo_path: Path,
    repository: Any,
    skeleton: Skeleton,
    plan: ShardPlan,
    base: Path | None,
    config: ExtractorConfig,
    telemetry: _Telemetry,
    on_event: Callable[[str], None] | None,
) -> _ShardOutcome:
    """One shard: enrich its subtree, with the base plan's *one* bounded repair
    scoped to this shard rather than to the whole tree.

    Only the blocking subprocess call goes through `asyncio.to_thread`; telemetry
    is mutated here, on the event-loop thread, so `_Telemetry` needs no lock.
    """
    async with sem:
        if cancel.cancelled:
            raise asyncio.CancelledError(
                f"shard {shard.key!r} cancelled after a sibling failed"
            )
        try:
            return await _enrich_shard_attempts(
                shard,
                repo_path=repo_path,
                repository=repository,
                skeleton=skeleton,
                plan=plan,
                base=base,
                config=config,
                telemetry=telemetry,
                on_event=on_event,
            )
        except BaseException:
            cancel.cancelled = True
            raise


async def _enrich_shard_attempts(
    shard: EnrichShard,
    *,
    repo_path: Path,
    repository: Any,
    skeleton: Skeleton,
    plan: ShardPlan,
    base: Path | None,
    config: ExtractorConfig,
    telemetry: _Telemetry,
    on_event: Callable[[str], None] | None,
) -> _ShardOutcome:
    shard_rel = f"03_enrich/{shard.key}"
    repo_data = repository.model_dump(mode="json")
    subtree_data = shard.subtree.model_dump(mode="json")
    scope = _scope_dict(shard)
    base_prompt = render_enrich_shard_prompt(
        repository=repo_data,
        subtree=subtree_data,
        scope=scope,
    )
    _required_write_json(base, f"{shard_rel}/scope.json", scope)
    _required_write_json(base, f"{shard_rel}/subtree_skeleton.json", subtree_data)
    _required_write_json(
        base,
        f"{shard_rel}/data_blocks.json",
        {
            "repository": repo_data,
            "subtree": subtree_data,
            "scope": scope,
        },
    )
    _required_write_text(base, f"{shard_rel}/prompt.md", base_prompt)
    _required_write_json(
        base, f"{shard_rel}/schema.json", EnrichedTree.model_json_schema()
    )

    # A shard whose scope IS the whole skeleton is today's single call and keeps
    # today's deadline; every real slice gets the shorter per-shard budget.
    timeout_s = (
        config.timeout_s
        if covers_entire_skeleton(shard, skeleton)
        else _effective_timeout(config.enrich_timeout_s, config.timeout_s)
    )

    prompt = base_prompt
    for attempt in range(1, 3):
        attempt_rel = f"{shard_rel}/attempt_{attempt:02d}"
        attempt_dir = base / attempt_rel if base is not None else None
        stage_tag = f"03_enrich[{shard.key}]" + ("#repair" if attempt == 2 else "")
        result: ClaudeStageResult[EnrichedTree] = await asyncio.to_thread(
            run_structured_claude_stage,
            output_type=EnrichedTree,
            repo_path=repo_path,
            prompt=prompt,
            stage_name=f"enrich:{shard.key}",
            attempt_dir=attempt_dir,
            claude_bin=config.claude_bin,
            max_turns=config.max_turns,
            timeout_s=timeout_s,
            on_event=on_event,
        )
        telemetry.add_claude(stage_tag, result)

        if result.parsed is None:
            detail = str(result.validation_error)
            _required_write_json(
                base,
                f"{attempt_rel}/validation.json",
                {"ok": False, "kind": "pydantic", "detail": detail},
            )
            _persist_sessions(base, telemetry)
            if attempt == 1:
                notify(
                    on_event,
                    f"extractor: shard {shard.key} parse failed; one repair",
                )
                prompt = _repair_prompt(
                    base_prompt, detail, previous_payload=result.raw_payload
                )
                continue
            raise ExtractorValidationError(
                f"shard {shard.key!r} failed to parse after 2 attempts: "
                f"{result.validation_error}",
                stage="enrich",
                shard=shard.key,
            )

        fragment = result.parsed
        _required_write_json(
            base, f"{attempt_rel}/enriched_tree.json", fragment.model_dump(mode="json")
        )

        validation_error = _validate_shard_fragment(
            shard,
            fragment,
            repo_path=repo_path,
            repository=repository,
        )
        coverage = compute_coverage(fragment, shard.subtree)
        _required_write_json(
            base, f"{attempt_rel}/coverage.json", coverage.model_dump(mode="json")
        )
        _required_write_json(
            base,
            f"{attempt_rel}/validation.json",
            {
                "ok": validation_error is None,
                "kind": "cross_artifact" if validation_error else None,
                "detail": validation_error,
            },
        )
        _persist_sessions(base, telemetry)

        if validation_error is not None:
            if attempt == 1:
                notify(
                    on_event,
                    f"extractor: shard {shard.key} validation failed; one repair",
                )
                prompt = _repair_prompt(
                    base_prompt, validation_error, previous_payload=result.raw_payload
                )
                continue
            raise ExtractorValidationError(
                f"shard {shard.key!r} failed subtree validation: {validation_error}",
                stage="enrich",
                shard=shard.key,
            )

        if coverage.missing:
            if attempt == 1:
                notify(
                    on_event,
                    f"extractor: shard {shard.key} missing "
                    f"{len(coverage.missing)} required paths; one repair",
                )
                prompt = _repair_prompt(
                    base_prompt,
                    "Missing required paths (must be emitted or folded): "
                    + ", ".join(coverage.missing),
                    previous_payload=result.raw_payload,
                )
                continue
            raise ExtractorCoverageError(
                f"shard {shard.key!r}: {len(coverage.missing)} required paths "
                "not covered",
                missing=coverage.missing,
                stage="enrich",
                shard=shard.key,
            )

        _required_write_json(
            base, f"{shard_rel}/fragment.json", fragment.model_dump(mode="json")
        )
        return _ShardOutcome(
            shard=shard,
            fragment=fragment,
            session_id=result.telemetry.session_id,
            attempts=attempt,
        )

    raise AssertionError("unreachable: shard attempt loop exhausted")


async def _run_enrich_shards_async(
    shards: list[EnrichShard], *, sem: asyncio.Semaphore, **kwargs: Any
) -> list[_ShardOutcome]:
    """Run every shard under a bounded-concurrency semaphore.

    `asyncio.gather` propagates the first exception but does **not** cancel the
    still-running siblings, so cancel them explicitly. That saves *cost* — a
    shard still queued on the semaphore never launches a subprocess — but not
    wall clock: a shard already inside `asyncio.to_thread` keeps running, and
    `asyncio.run` teardown joins the default executor's threads, so
    `enrich_timeout_s` is the real bound on how long a doomed run takes to exit.
    """
    cancel = _CancelFlag()
    tasks = [
        asyncio.create_task(
            _run_one_enrich_shard(s, sem=sem, cancel=cancel, **kwargs)
        )
        for s in shards
    ]
    try:
        return list(await asyncio.gather(*tasks))
    except BaseException:
        cancel.cancelled = True
        for t in tasks:
            t.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
        raise


def _stage3_enrich_sharded(
    repo_path: Path,
    repository: Any,
    skeleton: Skeleton,
    *,
    base: Path | None,
    config: ExtractorConfig,
    telemetry: _Telemetry,
    on_event: Callable[[str], None] | None,
) -> _Stage3Result:
    plan = derive_enrich_shards(skeleton, config)
    _required_write_json(base, "03_enrich/shards.json", _shard_plan_dict(plan))
    notify(
        on_event,
        f"extractor: stage-3 sharded into {len(plan.shards)} shard(s) "
        f"across {len(plan.branch_roots)} top-level branch(es)",
    )

    async def _run() -> list[_ShardOutcome]:
        return await _run_enrich_shards_async(
            plan.shards,
            sem=asyncio.Semaphore(config.max_parallel_enrich_shards),
            repo_path=repo_path,
            repository=repository,
            skeleton=skeleton,
            plan=plan,
            base=base,
            config=config,
            telemetry=telemetry,
            on_event=on_event,
        )

    try:
        outcomes = _run_event_loop(_run())
    except BaseException:
        # A write failure while handling a shard error must never mask it.
        _best_effort_write_json(base, "sessions.json", telemetry.as_dict(), on_event)
        raise
    _persist_sessions(base, telemetry)

    fragments = {o.shard.key: o.fragment for o in outcomes}
    accepted = {o.shard.key: o.session_id for o in outcomes}

    enriched, coverage = _merge_and_gate(
        repo_path,
        repository,
        skeleton,
        plan,
        fragments,
        base=base,
        on_event=on_event,
    )
    primary = plan.primary_shard()
    telemetry.accepted_session_id = (
        accepted.get(primary.key) if primary is not None else None
    )
    return _Stage3Result(
        enriched=enriched,
        coverage=coverage,
        plan=plan,
        fragments=fragments,
        accepted_sessions=accepted,
    )


def _merge_and_gate(
    repo_path: Path,
    repository: Any,
    skeleton: Skeleton,
    plan: ShardPlan,
    fragments: dict[str, EnrichedTree],
    *,
    base: Path | None,
    on_event: Callable[[str], None] | None,
) -> tuple[EnrichedTree, CoverageReport]:
    """Per-branch precheck, deterministic merge, then the **unchanged** full
    cross-artifact validation and coverage gate on the merged tree."""
    pairs = [(s, fragments[s.key]) for s in plan.shards if s.key in fragments]

    # Per-branch precheck: a cheap, loud integrity check on this plan's own
    # partition invariant plus a blame localizer. Its firing indicates a
    # `sharding.py` bug, not a bad model response.
    for branch_key in plan.branch_keys():
        branch_pairs = [
            (s, f) for s, f in pairs if plan.branch_key_of(s) == branch_key
        ]
        report = branch_coverage(plan, branch_key, branch_pairs)
        _required_write_json(
            base,
            f"03_enrich/branches/{branch_key}/coverage.json",
            report.model_dump(mode="json"),
        )
        if report.missing:
            owners = {
                p: (
                    owning_shard(p, [s for s, _ in branch_pairs]) or branch_pairs[0][0]
                ).key
                for p in report.missing
            }
            raise ExtractorCoverageError(
                f"branch {branch_key!r} is missing {len(report.missing)} required "
                f"path(s) after re-assembly: {owners}",
                missing=report.missing,
                stage="enrich",
                branch=branch_key,
            )

    merged = merge_fragments(pairs, skeleton=skeleton, plan=plan)
    _required_write_json(
        base, "03_enrich/merged/enriched_tree.json", merged.model_dump(mode="json")
    )

    validation_error: str | None = None
    try:
        validate_enriched_tree(merged, repo_path, repository, skeleton)
    except CrossArtifactError as exc:
        validation_error = str(exc)
    coverage = compute_coverage(merged, skeleton)
    _required_write_json(
        base,
        "03_enrich/merged/validation.json",
        {
            "ok": validation_error is None,
            "kind": "cross_artifact" if validation_error else None,
            "detail": validation_error,
        },
    )
    _required_write_json(
        base, "03_enrich/merged/coverage.json", coverage.model_dump(mode="json")
    )
    _required_write_json(
        base, "03_enrich/enriched_tree.json", merged.model_dump(mode="json")
    )

    if validation_error is not None:
        raise ExtractorValidationError(
            f"merged enriched tree failed cross-artifact validation: "
            f"{validation_error}",
            stage="enrich",
        )
    if coverage.missing:
        raise ExtractorCoverageError(
            f"{len(coverage.missing)} required paths not covered",
            missing=coverage.missing,
            stage="enrich",
        )
    notify(
        on_event,
        f"extractor: merged {len(pairs)} shard fragment(s) into "
        f"{len(merged.modules)} top-level module(s)",
    )
    return merged, coverage


# ── Stage 5 ───────────────────────────────────────────────────────────────


def _stage5_assemble(
    repo_path: Path,
    repository: Any,
    decision: SourceRootDecision,
    skeleton: Skeleton,
    enriched: EnrichedTree,
    *,
    base: Path | None,
    detected_source_files: list[str],
    repo_fingerprint: str,
    on_event: Callable[[str], None] | None,
) -> ProjectTree:
    tree = ProjectTree.model_validate(
        {
            "repository": repository.model_dump(),
            "modules": enriched.modules_as_project_tree_dicts(),
        }
    )

    # Re-run the full validation + coverage gate on the final tree.
    validate_enriched_tree(enriched, repo_path, repository, skeleton)
    coverage = compute_coverage(enriched, skeleton)
    if coverage.missing:
        raise ExtractorCoverageError(
            f"{len(coverage.missing)} required paths not covered at assembly",
            missing=coverage.missing,
            stage="assemble",
        )

    # Re-run Stage-1 exclusion validation and Stage-2 walk; require both
    # fingerprints to match (defense against concurrent mutation).
    fresh_scan = scan_source_files(repo_path, "")
    validate_source_root_decision(decision, repo_path, fresh_scan.files)
    fresh_repo_fingerprint = compute_fingerprint(
        repo_path, fresh_scan.files, extra_paths=fresh_scan.skipped_symlinks
    )
    if fresh_repo_fingerprint != repo_fingerprint:
        raise ExtractorValidationError(
            "repository source content changed during extraction "
            "(repo-wide fingerprint mismatch)",
            stage="assemble",
        )

    excluded_dirs, excluded_files = _split_exclusions(repo_path, decision)
    fresh_skeleton = build_skeleton(
        repo_path,
        decision.repository.source_root,
        excluded_dirs=excluded_dirs,
        excluded_files=excluded_files,
    )
    if fresh_skeleton.inventory_fingerprint != skeleton.inventory_fingerprint:
        raise ExtractorValidationError(
            "repository inventory changed during extraction "
            "(skeleton fingerprint mismatch)",
            stage="assemble",
        )

    if base is not None:
        try:
            tree.to_json(base / "project_tree.json")
        except OSError as exc:
            raise ExtractorArtifactError(
                f"failed to write project_tree.json: {exc}",
                artifact="project_tree.json",
            ) from exc
    notify(on_event, f"extractor: assembled {len(tree.modules)} top-level modules")
    return tree


# ── Prompt helpers ────────────────────────────────────────────────────────


def _bounded(text: str) -> str:
    if len(text) > _STAGE_REPAIR_PAYLOAD_MAX_CHARS:
        return text[:_STAGE_REPAIR_PAYLOAD_MAX_CHARS] + "\n...<truncated>"
    return text


def _repair_prompt(
    base_prompt: str, errors: str, *, previous_payload: str | None
) -> str:
    parts = [
        base_prompt,
        "\n\n## Repair after local validation failure\n",
        "Your previous output was rejected. Fix the exact problems below and "
        "return ONE complete valid JSON object in the same schema.\n\n",
        "Errors:\n",
        _bounded(errors),
    ]
    if previous_payload is not None:
        parts.append("\n\nPrevious output:\n")
        parts.append(_bounded(previous_payload))
    return "".join(parts) + "\n"


__all__ = ["run_two_phase_extraction"]
