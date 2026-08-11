"""Two-phase module extraction (Idea 4): deterministic skeleton, LLM enrichment.

Orchestrates the five sequential stages described in
`design/module_extraction_fix_impl_plan.md`:

1. Claude + Python — repository metadata and source root.
2. Python — deterministic directory inventory / skeleton.
3. Claude + Python — enrichment, strict filesystem validation, coverage gate.
4. Codex (+ optional Claude) — semantic review and one bounded revision.
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
import json
import time
from collections.abc import Callable, Coroutine
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, TypeVar

from pydantic import ValidationError

from spotlights_engine.module_deep_research.codex_exec import (
    CodexExecClient,
    CodexExecOptions,
    CodexExecResult,
    CodexExecTimeout,
)
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
    validate_enriched_tree,
    validate_source_root_decision,
)
from spotlights_engine.modules_extractor.errors import (
    ExtractorArtifactError,
    ExtractorCoverageError,
    ExtractorReviewError,
    ExtractorValidationError,
)
from spotlights_engine.modules_extractor.prompts import (
    render_enrich_prompt,
    render_enrich_shard_prompt,
    render_identify_source_root_prompt,
    render_review_prompt,
    render_review_shard_prompt,
)
from spotlights_engine.modules_extractor.sharding import (
    EnrichShard,
    ShardPlan,
    branch_coverage,
    covers_entire_skeleton,
    derive_enrich_shards,
    derive_review_shards,
    merge_fragments,
    owning_shard,
    validate_shard_depends_on,
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
    ReviewArtifact,
    ReviewIssue,
    ReviewReport,
    Skeleton,
    SourceRootDecision,
)
from spotlights_engine.schemas.project import ProjectTree, _qualified_name

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

    def add_codex(
        self, stage: str, result: CodexExecResult | None, duration_s: float
    ) -> None:
        usage = result.usage if result is not None else None
        cost = usage.cli_reported_cost_usd if usage is not None else None
        inp = usage.input if usage is not None else None
        out = usage.output if usage is not None else None
        self.sessions.append(
            _StageSession(
                stage=stage,
                provider="codex",
                session_id=None,
                input_tokens=inp,
                output_tokens=out,
                cache_read=usage.cache_read if usage is not None else None,
                cache_create=usage.cache_create if usage is not None else None,
                cost_usd=cost,
                duration_s=duration_s,
            )
        )
        self.total_duration_s += duration_s
        self.total_cost_usd = add_optional_float(self.total_cost_usd, cost)
        self.total_input_tokens = add_optional_int(self.total_input_tokens, inp)
        self.total_output_tokens = add_optional_int(self.total_output_tokens, out)

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
    """Run the full five-stage two-phase extraction over `repo_path`.

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

    # ── Stage 4 — review + optional revision ──────────────────────────────
    enriched, coverage, review = _stage4_review(
        repo_path,
        repository,
        skeleton,
        stage3,
        base=base,
        config=config,
        telemetry=telemetry,
        on_event=on_event,
    )
    _persist_sessions(base, telemetry)

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
    _required_write_json(base, "review.json", review.model_dump(mode="json"))
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

        decision = result.parsed
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
    """Stage-3 output plus everything Stage 4 needs to work per-branch.

    `plan` is `None` in `enrich_sharding="single"` mode, which reproduces
    today's monolithic Stage 3 (and monolithic Stage 4) verbatim.
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

    if config.enrich_sharding == "single":
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
    top_level_qns = _top_level_qns(skeleton)
    base_prompt = render_enrich_prompt(
        repository=repo_data, skeleton=skel_data, top_level_qns=top_level_qns
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


def _top_level_qns(skeleton: Skeleton) -> list[str]:
    return sorted(
        _qualified_name(n.path, skeleton.source_root) for n in skeleton.nodes
    )


def _scope_dict(shard: EnrichShard) -> dict[str, Any]:
    return {
        "key": shard.key,
        "root_path": shard.root_path,
        "owns_root": shard.owns_root,
        "is_subshard": shard.is_subshard,
        "parent_key": shard.parent_key,
        "depth": shard.depth,
        "promoted_children": list(shard.promoted_children),
    }


def _shard_plan_dict(plan: ShardPlan) -> dict[str, Any]:
    return {
        "top_level_qns": list(plan.top_level_qns),
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
    plan: ShardPlan,
) -> str | None:
    """Subtree-scoped cross-artifact validation. Returns the error text or None.

    The shard-scoped knobs are what let the *unchanged* validator run against a
    slice: dependency resolution is deferred to the merged tree (a shard
    legitimately names siblings it cannot see), the vocabulary check replaces it
    locally, and a spine defers Rule 4 for its own root because the children
    that guarantee it are pruned from its subtree.
    """
    try:
        validate_shard_scope(shard, fragment)
        validate_shard_depends_on(
            shard, fragment, carries_depends_on=plan.carries_depends_on(shard)
        )
        validate_enriched_tree(
            fragment,
            repo_path,
            repository,
            shard.subtree,
            skip_dependency_resolution=True,
            allowed_internal_qns=set(plan.top_level_qns),
            rule4_exempt_paths={shard.root_path} if shard.is_spine else None,
        )
        validate_spine_main_files(shard, fragment)
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
        top_level_qns=plan.top_level_qns,
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
            "top_level_qns": list(plan.top_level_qns),
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
            plan=plan,
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


# ── Stage 4 ───────────────────────────────────────────────────────────────


def _stage4_review(
    repo_path: Path,
    repository: Any,
    skeleton: Skeleton,
    stage3: _Stage3Result,
    *,
    base: Path | None,
    config: ExtractorConfig,
    telemetry: _Telemetry,
    on_event: Callable[[str], None] | None,
) -> tuple[EnrichedTree, CoverageReport, ReviewArtifact]:
    stage_dir = base / "04_review" if base is not None else None
    if stage_dir is not None:
        stage_dir.mkdir(parents=True, exist_ok=True)

    if stage3.plan is None:
        return _stage4_review_single(
            repo_path,
            repository,
            skeleton,
            stage3.enriched,
            base=base,
            config=config,
            telemetry=telemetry,
            on_event=on_event,
        )
    return _stage4_review_sharded(
        repo_path,
        repository,
        skeleton,
        stage3,
        base=base,
        config=config,
        telemetry=telemetry,
        on_event=on_event,
    )


def _stage4_review_single(
    repo_path: Path,
    repository: Any,
    skeleton: Skeleton,
    enriched: EnrichedTree,
    *,
    base: Path | None,
    config: ExtractorConfig,
    telemetry: _Telemetry,
    on_event: Callable[[str], None] | None,
) -> tuple[EnrichedTree, CoverageReport, ReviewArtifact]:
    """The monolithic review path, paired with `enrich_sharding="single"`."""
    review = _run_codex_review(
        repo_path,
        skeleton,
        enriched,
        base=base,
        rel="04_review/attempt_01",
        config=config,
        telemetry=telemetry,
        on_event=on_event,
    )

    # Recompute current coverage for the accepted (pre-revision) tree.
    coverage = compute_coverage(enriched, skeleton)

    if review.status == "completed" and review.report and not review.report.ok:
        revised = _run_review_revision(
            repo_path,
            repository,
            skeleton,
            enriched,
            review.report,
            base=base,
            config=config,
            telemetry=telemetry,
            on_event=on_event,
        )
        if revised is not None:
            enriched, coverage = revised
            telemetry.accepted_session_id = telemetry.sessions[-1].session_id

    if config.fail_on_review_issues:
        rereview = _run_codex_review(
            repo_path,
            skeleton,
            enriched,
            base=base,
            rel="04_review/re_review",
            config=config,
            telemetry=telemetry,
            on_event=on_event,
        )
        if rereview.status == "completed" and rereview.report and not rereview.report.ok:
            raise ExtractorReviewError(
                f"strict review found {len(rereview.report.issues)} remaining issue(s)",
                stage="review",
            )
        review = rereview

    _required_write_json(base, "review.json", review.model_dump(mode="json"))
    return enriched, coverage, review


@dataclass
class _CodexOutcome:
    """One Codex review invocation, before telemetry is recorded.

    Telemetry is deliberately *not* mutated inside this function: with sharded
    review it runs on a worker thread via `asyncio.to_thread`, and `_Telemetry`
    is only safe to mutate from the event-loop thread.
    """

    artifact: ReviewArtifact
    result: CodexExecResult | None
    duration_s: float


def _run_codex_review(
    repo_path: Path,
    skeleton: Skeleton,
    enriched: EnrichedTree,
    *,
    base: Path | None,
    rel: str,
    config: ExtractorConfig,
    telemetry: _Telemetry,
    on_event: Callable[[str], None] | None,
) -> ReviewArtifact:
    prompt = render_review_prompt(
        skeleton=skeleton.model_dump(mode="json"),
        enriched=enriched.model_dump(mode="json"),
    )
    outcome = _invoke_codex_review(
        repo_path, prompt, base=base, rel=rel, config=config, on_event=on_event
    )
    telemetry.add_codex(rel.split("/")[-1], outcome.result, outcome.duration_s)
    _persist_sessions(base, telemetry)
    return outcome.artifact


def _invoke_codex_review(
    repo_path: Path,
    prompt: str,
    *,
    base: Path | None,
    rel: str,
    config: ExtractorConfig,
    on_event: Callable[[str], None] | None,
) -> _CodexOutcome:
    """Blocking Codex review of one prompt. Never raises: every failure mode is
    an honest `status="skipped"` artifact, because review is advisory."""
    review_dir = base / rel if base is not None else None
    if review_dir is not None:
        review_dir.mkdir(parents=True, exist_ok=True)
    _required_write_text(base, f"{rel}/prompt.md", prompt)
    _required_write_json(base, f"{rel}/schema.json", ReviewReport.model_json_schema())

    extra_args: tuple[str, ...] = ()
    if config.codex_reasoning_effort:
        extra_args = ("-c", f'model_reasoning_effort="{config.codex_reasoning_effort}"')

    last_message_path = (
        review_dir / "review_last_message.json" if review_dir is not None else None
    )
    options = CodexExecOptions(
        cwd=repo_path,
        codex_bin=config.codex_bin,
        model=config.codex_model,
        sandbox="read-only",
        approval="never",
        search=False,
        skip_git_repo_check=True,
        ephemeral=True,
        json_events=True,
        output_last_message=last_message_path,
        timeout_seconds=config.codex_timeout_s,
        extra_args=extra_args,
    )
    client = CodexExecClient(options)

    start = time.monotonic()
    result: CodexExecResult | None = None
    error: str | None = None
    stdout = ""
    stderr = ""
    final_message: str | None = None
    try:
        result = client.run(prompt, check=False)
        stdout, stderr = result.stdout, result.stderr
        final_message = result.final_message
        duration_s = time.monotonic() - start
    except CodexExecTimeout as exc:
        duration_s = exc.duration_s
        stdout, stderr = exc.partial_stdout, exc.partial_stderr
        final_message = exc.final_message
        error = f"codex review timed out after {exc.duration_s:.1f}s"
    except (OSError, FileNotFoundError) as exc:
        duration_s = time.monotonic() - start
        error = f"codex executable failed to start: {exc}"
    except Exception as exc:  # noqa: BLE001 - review failure is a skip, not a crash
        duration_s = time.monotonic() - start
        error = f"codex review raised: {exc}"

    # Persist raw evidence before parsing.
    if base is not None:
        _best_effort_write_json(
            base, f"{rel}/stream.jsonl", {"stdout": stdout}, on_event
        )
        try:
            atomic_write_text((base / rel) / "stderr.log", stderr or "")
            if final_message is not None:
                atomic_write_text((base / rel) / "last_message.json", final_message)
        except OSError:
            pass

    def _skip(reason: str) -> _CodexOutcome:
        artifact = ReviewArtifact(status="skipped", error=reason)
        _write_review_status(base, rel, artifact)
        notify(on_event, f"extractor: review skipped — {reason}")
        return _CodexOutcome(artifact=artifact, result=result, duration_s=duration_s)

    if error is not None:
        return _skip(error)

    assert result is not None
    if result.returncode != 0:
        return _skip(f"codex exit={result.returncode}")

    if not final_message or not final_message.strip():
        return _skip("codex produced no final message")

    try:
        report = ReviewReport.model_validate_json(_strip_json(final_message))
    except (ValidationError, json.JSONDecodeError, ValueError) as exc:
        return _skip(f"review JSON invalid: {exc}")

    artifact = ReviewArtifact(status="completed", report=report)
    _required_write_json(base, f"{rel}/review_report.json", report.model_dump(mode="json"))
    _write_review_status(base, rel, artifact)
    return _CodexOutcome(artifact=artifact, result=result, duration_s=duration_s)


def _write_review_status(base: Path | None, rel: str, artifact: ReviewArtifact) -> None:
    _required_write_json(
        base,
        f"{rel}/status.json",
        {"status": artifact.status, "error": artifact.error},
    )


def _run_review_revision(
    repo_path: Path,
    repository: Any,
    skeleton: Skeleton,
    enriched: EnrichedTree,
    report: ReviewReport,
    *,
    base: Path | None,
    config: ExtractorConfig,
    telemetry: _Telemetry,
    on_event: Callable[[str], None] | None,
) -> tuple[EnrichedTree, CoverageReport] | None:
    """Run one Claude enrichment revision from reviewer notes. Returns the new
    (tree, coverage) only if it re-passes every gate; otherwise None (advisory)
    or raises `ExtractorReviewError` (strict)."""
    rel = "04_review/revision"
    attempt_dir = base / rel if base is not None else None
    notes = "\n".join(
        f"- [{i.kind}] {i.path}: {i.detail}" for i in report.issues
    )
    base_prompt = render_enrich_prompt(
        repository=repository.model_dump(mode="json"),
        skeleton=skeleton.model_dump(mode="json"),
        # Without this the TOP_LEVEL_MODULES block renders as `[]`, and the
        # prompt calls that "the complete, authoritative vocabulary" — i.e. it
        # would tell the model no internal `depends_on` target is legal at all.
        top_level_qns=_top_level_qns(skeleton),
    )
    prompt = (
        f"{base_prompt}\n\n## Reviewer notes to address\n\n"
        "A reviewer flagged the following semantic issues in your previous tree. "
        "Return a full replacement EnrichedTree that addresses them while keeping "
        "every required path emitted or folded:\n\n"
        f"{notes}\n\n"
        "Previous tree:\n"
        f"{_bounded(enriched.model_dump_json())}\n"
    )

    result: ClaudeStageResult[EnrichedTree] = run_structured_claude_stage(
        output_type=EnrichedTree,
        repo_path=repo_path,
        prompt=prompt,
        stage_name="revision",
        attempt_dir=attempt_dir,
        claude_bin=config.claude_bin,
        max_turns=config.max_turns,
        timeout_s=config.timeout_s,
        on_event=on_event,
    )
    telemetry.add_claude("04_revision", result)
    _persist_sessions(base, telemetry)

    def _reject(detail: str) -> tuple[EnrichedTree, CoverageReport] | None:
        _required_write_json(
            base, f"{rel}/validation.json", {"ok": False, "detail": detail}
        )
        if config.fail_on_review_issues:
            raise ExtractorReviewError(
                f"review-driven revision failed validation: {detail}",
                stage="review",
            )
        notify(on_event, "extractor: revision invalid; keeping pre-review tree")
        return None

    if result.parsed is None:
        return _reject(str(result.validation_error))

    revised = result.parsed
    _required_write_json(
        base, f"{rel}/enriched_tree.json", revised.model_dump(mode="json")
    )
    try:
        validate_enriched_tree(revised, repo_path, repository, skeleton)
    except CrossArtifactError as exc:
        _required_write_json(
            base,
            f"{rel}/coverage.json",
            compute_coverage(revised, skeleton).model_dump(mode="json"),
        )
        return _reject(str(exc))

    coverage = compute_coverage(revised, skeleton)
    _required_write_json(base, f"{rel}/coverage.json", coverage.model_dump(mode="json"))
    if coverage.missing:
        return _reject("missing required paths: " + ", ".join(coverage.missing))

    _required_write_json(base, f"{rel}/validation.json", {"ok": True, "detail": None})
    notify(on_event, "extractor: accepted review-driven revision")
    return revised, coverage


# ── Stage 4 — sharded ─────────────────────────────────────────────────────


def _branch_slice(tree: EnrichedTree, root: str) -> EnrichedTree:
    """One branch's slice of the merged tree: its top-level module + its folds."""
    return EnrichedTree(
        modules=[m for m in tree.modules if m.path.strip("/") == root],
        folds=[
            f
            for f in tree.folds
            if f.path == root or f.path.startswith(root + "/")
        ],
    )


def _merge_reviews(
    results: list[tuple[str, ReviewArtifact]],
) -> tuple[ReviewArtifact, list[dict[str, Any]]]:
    """Merge per-branch reviews into one `ReviewArtifact` plus a skip ledger.

    `ok` is *derived* from the merged issue list rather than computed
    independently: `ReviewReport._ok_agrees_with_issues` rejects any other
    combination, so deriving it is what keeps the merged report constructible at
    all. A fully-skipped review stays `status="skipped"` — never a false
    `ok=true`.

    The skip ledger is returned separately, not attached to `ReviewArtifact`:
    that model is `extra="forbid"` and its dump is the public `review.json`, so
    the ledger lives in `04_review/merged_review.json` instead and the public
    schema is genuinely unchanged.
    """
    skipped = [
        {"key": key, "error": art.error}
        for key, art in results
        if art.status == "skipped"
    ]
    completed = [art for _, art in results if art.status == "completed"]
    if not completed:
        return (
            ReviewArtifact(
                status="skipped",
                error=(
                    f"all {len(results)} branch review(s) skipped"
                    if results
                    else "no branches to review"
                ),
            ),
            skipped,
        )
    issues = sorted(
        (i for art in completed if art.report for i in art.report.issues),
        key=lambda i: (i.path, i.kind),
    )
    return (
        ReviewArtifact(
            status="completed",
            report=ReviewReport(ok=not issues, issues=issues),
        ),
        skipped,
    )


def _run_branch_reviews(
    repo_path: Path,
    enriched: EnrichedTree,
    review_shards: list[EnrichShard],
    *,
    base: Path | None,
    subdir: str,
    stage_prefix: str,
    config: ExtractorConfig,
    telemetry: _Telemetry,
    on_event: Callable[[str], None] | None,
) -> list[tuple[str, ReviewArtifact]]:
    """One Codex review per top-level branch, under a bounded semaphore.

    A branch review that times out, exits nonzero, or returns invalid JSON
    becomes that shard's `skipped` artifact and does not fail the run; other
    branches are unaffected.
    """

    async def _one(
        shard: EnrichShard, sem: asyncio.Semaphore
    ) -> tuple[str, ReviewArtifact]:
        async with sem:
            prompt = render_review_shard_prompt(
                subtree=shard.subtree.model_dump(mode="json"),
                fragment=_branch_slice(enriched, shard.root_path).model_dump(
                    mode="json"
                ),
                scope=_scope_dict(shard),
            )
            rel = f"04_review/{shard.key}/{subdir}"
            outcome = await asyncio.to_thread(
                _invoke_codex_review,
                repo_path,
                prompt,
                base=base,
                rel=rel,
                config=config,
                on_event=on_event,
            )
            telemetry.add_codex(
                f"{stage_prefix}[{shard.key}]", outcome.result, outcome.duration_s
            )
            return shard.key, outcome.artifact

    async def _run() -> list[tuple[str, ReviewArtifact]]:
        sem = asyncio.Semaphore(config.max_parallel_review_shards)
        return list(
            await asyncio.gather(
                *[asyncio.create_task(_one(s, sem)) for s in review_shards]
            )
        )

    try:
        results = _run_event_loop(_run())
    except BaseException:
        _best_effort_write_json(base, "sessions.json", telemetry.as_dict(), on_event)
        raise
    _persist_sessions(base, telemetry)
    return results


def _revise_branch_shards(
    repo_path: Path,
    repository: Any,
    plan: ShardPlan,
    branch_key: str,
    issues: list[ReviewIssue],
    fragments: dict[str, EnrichedTree],
    *,
    base: Path | None,
    config: ExtractorConfig,
    telemetry: _Telemetry,
    on_event: Callable[[str], None] | None,
) -> tuple[dict[str, EnrichedTree], dict[str, str | None]] | None:
    """One bounded revision of exactly the shards that own the flagged paths.

    Issue attribution is **total**: `ReviewIssue.path` is an unvalidated free
    string, so an issue matching no shard scope is attributed to the branch's
    spine (or un-split top-level shard), which owns the branch as a whole.
    Nothing is silently dropped — the attribution is recorded on disk.

    Returns the revised fragments and their sessions, or None when the revision
    was discarded (advisory mode). Never replaces a valid fragment with an
    invalid revision.
    """
    branch_shards = plan.shards_of_branch(branch_key)
    branch_root = plan.branch_roots[branch_key]
    fallback = next(
        (s for s in branch_shards if s.owns_root and s.root_path == branch_root),
        branch_shards[0],
    )

    by_shard: dict[str, list[ReviewIssue]] = {}
    attribution: list[dict[str, str]] = []
    for issue in issues:
        owner = owning_shard(issue.path.strip("/"), branch_shards) or fallback
        by_shard.setdefault(owner.key, []).append(issue)
        attribution.append(
            {
                "path": issue.path,
                "kind": issue.kind,
                "shard": owner.key,
                "unmapped": str(
                    owning_shard(issue.path.strip("/"), branch_shards) is None
                ),
            }
        )

    rel_root = f"04_review/{branch_key}/revision"
    _required_write_json(
        base,
        f"{rel_root}/attribution.json",
        {"branch": branch_key, "issues": attribution},
    )

    revised: dict[str, EnrichedTree] = {}
    sessions: dict[str, str | None] = {}
    for shard_key in sorted(by_shard):
        shard = next(s for s in branch_shards if s.key == shard_key)
        outcome = _revise_one_shard(
            repo_path,
            repository,
            plan,
            shard,
            by_shard[shard_key],
            fragments[shard_key],
            base=base,
            rel=f"{rel_root}/{shard_key}",
            config=config,
            telemetry=telemetry,
            on_event=on_event,
        )
        if outcome is None:
            if config.fail_on_review_issues:
                raise ExtractorReviewError(
                    f"review-driven revision of shard {shard_key!r} failed "
                    "validation",
                    stage="review",
                    shard=shard_key,
                )
            notify(
                on_event,
                f"extractor: revision of shard {shard_key} invalid; keeping "
                "the pre-review fragments for branch " + branch_key,
            )
            return None
        revised[shard_key] = outcome[0]
        sessions[shard_key] = outcome[1]
    return revised, sessions


def _revise_one_shard(
    repo_path: Path,
    repository: Any,
    plan: ShardPlan,
    shard: EnrichShard,
    issues: list[ReviewIssue],
    previous: EnrichedTree,
    *,
    base: Path | None,
    rel: str,
    config: ExtractorConfig,
    telemetry: _Telemetry,
    on_event: Callable[[str], None] | None,
) -> tuple[EnrichedTree, str | None] | None:
    notes = "\n".join(f"- [{i.kind}] {i.path}: {i.detail}" for i in issues)
    base_prompt = render_enrich_shard_prompt(
        repository=repository.model_dump(mode="json"),
        subtree=shard.subtree.model_dump(mode="json"),
        scope=_scope_dict(shard),
        top_level_qns=plan.top_level_qns,
    )
    prompt = (
        f"{base_prompt}\n\n## Reviewer notes to address\n\n"
        "A reviewer flagged the following semantic issues in your previous "
        "fragment. Return a full replacement EnrichedTree for this same scope "
        "that addresses them while keeping every required path in scope emitted "
        "or folded:\n\n"
        f"{notes}\n\n"
        "Previous fragment:\n"
        f"{_bounded(previous.model_dump_json())}\n"
    )
    attempt_dir = base / rel if base is not None else None
    result: ClaudeStageResult[EnrichedTree] = run_structured_claude_stage(
        output_type=EnrichedTree,
        repo_path=repo_path,
        prompt=prompt,
        stage_name=f"revision:{shard.key}",
        attempt_dir=attempt_dir,
        claude_bin=config.claude_bin,
        max_turns=config.max_turns,
        timeout_s=_effective_timeout(config.enrich_timeout_s, config.timeout_s),
        on_event=on_event,
    )
    telemetry.add_claude(f"04_revision[{shard.key}]", result)
    _persist_sessions(base, telemetry)

    if result.parsed is None:
        _required_write_json(
            base,
            f"{rel}/validation.json",
            {"ok": False, "detail": str(result.validation_error)},
        )
        return None

    revised = result.parsed
    _required_write_json(
        base, f"{rel}/enriched_tree.json", revised.model_dump(mode="json")
    )
    detail = _validate_shard_fragment(
        shard, revised, repo_path=repo_path, repository=repository, plan=plan
    )
    coverage = compute_coverage(revised, shard.subtree)
    _required_write_json(base, f"{rel}/coverage.json", coverage.model_dump(mode="json"))
    if detail is None and coverage.missing:
        detail = "missing required paths: " + ", ".join(coverage.missing)
    _required_write_json(
        base, f"{rel}/validation.json", {"ok": detail is None, "detail": detail}
    )
    if detail is not None:
        return None
    return revised, result.telemetry.session_id


def _stage4_review_sharded(
    repo_path: Path,
    repository: Any,
    skeleton: Skeleton,
    stage3: _Stage3Result,
    *,
    base: Path | None,
    config: ExtractorConfig,
    telemetry: _Telemetry,
    on_event: Callable[[str], None] | None,
) -> tuple[EnrichedTree, CoverageReport, ReviewArtifact]:
    plan = stage3.plan
    assert plan is not None
    enriched, coverage = stage3.enriched, stage3.coverage
    fragments = dict(stage3.fragments)
    accepted = dict(stage3.accepted_sessions)

    review_shards = derive_review_shards(skeleton)
    _required_write_json(
        base,
        "04_review/shards.json",
        {"shards": [_scope_dict(s) for s in review_shards]},
    )

    results = _run_branch_reviews(
        repo_path,
        enriched,
        review_shards,
        base=base,
        subdir="attempt_01",
        stage_prefix="04_review",
        config=config,
        telemetry=telemetry,
        on_event=on_event,
    )
    review, skipped = _merge_reviews(results)

    # Review-driven revision, scoped to the shards that own the issue paths —
    # never the whole tree, and not necessarily the whole branch.
    revised_any = False
    for branch_key, artifact in sorted(results, key=lambda kv: kv[0]):
        if artifact.status != "completed" or artifact.report is None:
            continue
        if not artifact.report.issues:
            continue
        outcome = _revise_branch_shards(
            repo_path,
            repository,
            plan,
            branch_key,
            artifact.report.issues,
            fragments,
            base=base,
            config=config,
            telemetry=telemetry,
            on_event=on_event,
        )
        if outcome is None:
            continue
        new_fragments, new_sessions = outcome
        candidate = {**fragments, **new_fragments}
        branch_pairs = [
            (s, candidate[s.key])
            for s in plan.shards_of_branch(branch_key)
            if s.key in candidate
        ]
        precheck = branch_coverage(plan, branch_key, branch_pairs)
        if precheck.missing:
            if config.fail_on_review_issues:
                raise ExtractorReviewError(
                    f"revised branch {branch_key!r} is missing "
                    f"{len(precheck.missing)} required path(s)",
                    stage="review",
                )
            notify(
                on_event,
                f"extractor: revised branch {branch_key} is incomplete; "
                "keeping the pre-review fragments",
            )
            continue
        fragments = candidate
        accepted.update(new_sessions)
        revised_any = True

    if revised_any:
        try:
            enriched, coverage = _merge_and_gate(
                repo_path,
                repository,
                skeleton,
                plan,
                fragments,
                base=base,
                on_event=on_event,
            )
        except (ExtractorValidationError, ExtractorCoverageError) as exc:
            if config.fail_on_review_issues:
                raise ExtractorReviewError(
                    f"review-driven revision failed the full gate: {exc}",
                    stage="review",
                ) from exc
            notify(
                on_event,
                "extractor: revised tree failed the full gate; restoring the "
                "pre-review tree",
            )
            fragments = dict(stage3.fragments)
            accepted = dict(stage3.accepted_sessions)
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

    if config.fail_on_review_issues:
        rereviews = _run_branch_reviews(
            repo_path,
            enriched,
            review_shards,
            base=base,
            subdir="re_review",
            stage_prefix="04_re_review",
            config=config,
            telemetry=telemetry,
            on_event=on_event,
        )
        review, skipped = _merge_reviews(rereviews)
        if review.status == "completed" and review.report and not review.report.ok:
            raise ExtractorReviewError(
                f"strict review found {len(review.report.issues)} remaining "
                "issue(s)",
                stage="review",
            )

    _required_write_json(
        base,
        "04_review/merged_review.json",
        {**review.model_dump(mode="json"), "skipped": skipped},
    )
    _required_write_json(base, "review.json", review.model_dump(mode="json"))
    return enriched, coverage, review


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


def _strip_json(text: str) -> str:
    """Best-effort extraction of a JSON object from a model final message.

    Handles a bare object as well as one wrapped in a ```json fence.
    """
    s = text.strip()
    if s.startswith("```"):
        s = s.split("\n", 1)[-1] if "\n" in s else s
        if s.endswith("```"):
            s = s[: s.rfind("```")]
        s = s.strip()
        if s.startswith("json"):
            s = s[len("json"):].strip()
    start = s.find("{")
    end = s.rfind("}")
    if start != -1 and end != -1 and end > start:
        return s[start : end + 1]
    return s


__all__ = ["run_two_phase_extraction"]
