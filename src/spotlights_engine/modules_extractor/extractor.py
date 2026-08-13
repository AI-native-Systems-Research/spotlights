"""Top-level orchestration for the modules extractor.

Implements step 1 of the deep-research pipeline (see
`docs/architecture/spotlights_deep_research_path_architecture.md`):
take a `ModulesExtractorInput`, drive a Claude Code subprocess over the repo
with the `extraction.md` prompt, and return a `ProjectTree` validated against
the shared schema.

Two entrypoints, mirroring `candidate_discovery`:

- `extract(input)` is the architecture-shaped contract — synchronous,
  takes `ModulesExtractorInput`, returns a `ProjectTree`.
- `extract_with_telemetry(input, *, config)` is the runtime-rich form: same
  logic, plus an optional `artifacts_dir` for run logs and tunable agent
  budgets, returning telemetry alongside the tree.
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from spotlights_engine.modules_extractor.agent import (
    ExtractionInvocation,
    run_extraction,
)
from spotlights_engine.modules_extractor.errors import ExtractorSetupError
from spotlights_engine.modules_extractor.prompts import EXTRACTION_PROMPT
from spotlights_engine.schemas.pipeline import ModulesExtractorInput
from spotlights_engine.schemas.project import ProjectTree


class ExtractorConfig(BaseModel):
    """Runtime/infra config for the modules extractor.

    Holds only fields outside the architectural `ModulesExtractorInput`
    contract: optional `artifacts_dir` for persisting prompt/schema/raw
    output, tunable agent budgets, and the two-phase strategy knobs.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    artifacts_dir: Path | None = None
    claude_bin: str = "claude"
    max_turns: int = Field(default=60, ge=1)
    # Per-Claude-stage subprocess deadline. The two-phase Stage-3 enrichment is a
    # single repo-wide call that must explore and emit the full module tree; on a
    # large monorepo (e.g. vLLM: ~345 required modules across vllm/, rust/, csrc/)
    # the 1800s default was too small and the enrichment was killed mid-generation
    # before writing enriched_tree.json/coverage.json. 5400s gives that call the
    # headroom to finish while every stage still enforces a hard subprocess
    # timeout.
    timeout_s: int = Field(default=5400, ge=1)

    # Two-phase (Idea 4) strategy. Default on: the deterministic-skeleton +
    # LLM-enrichment path is the standard extractor now; set False for the
    # legacy single-shot path.
    two_phase: bool = True
    source_root_max_turns: int = Field(default=15, ge=1)
    codex_bin: str = "codex"
    codex_model: str | None = None
    codex_reasoning_effort: (
        Literal["minimal", "low", "medium", "high", "xhigh"] | None
    ) = None
    codex_timeout_s: int = Field(default=900, ge=1)
    fail_on_review_issues: bool = False

    # ── Stage-3/4 sharding strategy ───────────────────────────────────────
    # Enrichment is sharded by top-level skeleton node and run under a bounded
    # concurrency executor; see `sharding.py` and
    # `design/module_extraction_fix_impl__top_level_plan.md`. Always-on rather
    # than threshold-gated: a repo with one top-level source-bearing node
    # degenerates to exactly one shard = today's single call, so there is one
    # code path instead of a mode switch whose branches diverge exactly where
    # bugs hide (merge, telemetry, artifacts).
    #
    #   "auto"            top-level sharding + size-gated sub-sharding
    #   "top_level_only"  shard by top-level, never sub-shard
    #   "single"          force today's monolithic single call (A/B, small repos)
    enrich_sharding: Literal["auto", "top_level_only", "single"] = "auto"
    max_parallel_enrich_shards: int = Field(default=5, ge=1)
    max_parallel_review_shards: int = Field(default=5, ge=1)

    # Stage-specific deadlines that decouple from the coarse `timeout_s`.
    # `None` means "inherit `timeout_s`". The fields are `int | None` rather
    # than ints with a truthy default on purpose: with a truthy default,
    # `enrich_timeout_s or timeout_s` would never fall through, so a caller who
    # only bumped `timeout_s` would be silently ignored.
    source_root_timeout_s: int | None = Field(default=None, ge=1)
    # Per-shard enrichment deadline. Shards are far smaller than the whole repo,
    # so this is deliberately below the 5400s monolithic budget — but a shard
    # whose scope IS the entire skeleton still gets `timeout_s` (see
    # `sharding.covers_entire_skeleton`), so the degenerate single-shard repo
    # does not newly time out.
    enrich_timeout_s: int | None = Field(default=1800, ge=1)

    # Recursive sub-sharding of an oversized top-level branch. `weight` is a
    # node's required-node count.
    enrich_subshard_threshold: int = Field(default=40, ge=1)
    enrich_subshard_child_min: int = Field(default=2, ge=1)
    enrich_subshard_max_depth: int = Field(default=2, ge=1)
    # Hard ceiling on total derived enrich shards. `ge=2` so the cap can never
    # force a single-child spine, which Stage-5 Rule 4 would reject with no
    # possible recovery on the pure-Python merge.
    enrich_max_shards: int = Field(default=24, ge=2)


class ExtractorResult(BaseModel):
    """Architectural output (`project_tree`) plus run telemetry."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    project_tree: ProjectTree
    invocation: ExtractionInvocation


def extract_with_telemetry(
    input: ModulesExtractorInput,
    *,
    config: ExtractorConfig | None = None,
    on_event: Callable[[str], None] | None = None,
) -> ExtractorResult:
    """Runtime-rich extraction. Persists artifacts when `artifacts_dir` is set."""
    cfg = config or ExtractorConfig()
    repo = input.repo_path
    if not repo.exists() or not repo.is_dir():
        raise ExtractorSetupError(
            f"repo_path does not exist or is not a directory: {repo}",
            repo_path=str(repo),
        )

    artifacts = cfg.artifacts_dir
    if artifacts is not None:
        resolved_repo = repo.resolve(strict=False)
        resolved_artifacts = artifacts.resolve(strict=False)
        try:
            resolved_artifacts.relative_to(resolved_repo)
        except ValueError:
            pass
        else:
            raise ExtractorSetupError(
                f"artifacts_dir resolves inside repo_path: {artifacts}",
                artifacts_dir=str(artifacts),
                repo_path=str(repo),
            )
        run_dir = artifacts / "modules_extractor"
        if run_dir.exists():
            raise ExtractorSetupError(
                f"pre-existing modules_extractor run dir; resume is not supported: {run_dir}",
                run_dir=str(run_dir),
            )
        run_dir.mkdir(parents=True)
    else:
        run_dir = None

    if cfg.two_phase:
        # Lazy import: two_phase pulls in codex/skeleton machinery that the
        # legacy path (and the Stage-02 safety-order test) must not require at
        # module top.
        from spotlights_engine.modules_extractor.two_phase import (
            run_two_phase_extraction,
        )

        run = run_two_phase_extraction(
            repo,
            config=cfg,
            on_event=on_event,
            artifacts_dir=run_dir,
        )
        return ExtractorResult(
            project_tree=run.project_tree, invocation=run.invocation
        )

    run_kwargs: dict = dict(
        repo_path=repo,
        prompt=EXTRACTION_PROMPT,
        claude_bin=cfg.claude_bin,
        max_turns=cfg.max_turns,
        timeout_s=cfg.timeout_s,
        artifacts_dir=run_dir,
    )
    if on_event is not None:
        run_kwargs["on_event"] = on_event
    run = run_extraction(**run_kwargs)

    return ExtractorResult(project_tree=run.project_tree, invocation=run.invocation)


def extract(
    input: ModulesExtractorInput,
    *,
    config: ExtractorConfig | None = None,
    on_event: Callable[[str], None] | None = None,
) -> ProjectTree:
    """Architecture-shaped entrypoint for step 1 (`ModulesExtractor`).

    Mirrors the per-step contract: input is `ModulesExtractorInput`, output is
    the structural `ProjectTree`. Telemetry flows through `extract_with_telemetry`
    for callers that want it.

    `on_event`, when given, is forwarded to the underlying `claude -p` reader
    thread (one summary line per stream-json event). Default is silent.
    """
    return extract_with_telemetry(
        input, config=config, on_event=on_event
    ).project_tree


__all__ = [
    "ExtractorConfig",
    "ExtractorResult",
    "extract",
    "extract_with_telemetry",
]
