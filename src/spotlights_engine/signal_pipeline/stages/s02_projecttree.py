"""Stage 02 — ProjectTree extraction.

Delegates to `spotlights_engine.modules_extractor.extract` from origin/main
(LLM-backed, runs `claude -p` on the subject repo). Each invocation places
its modules-extractor working tree under a fresh timestamped subdir of the
stage's `log_dir`, sidestepping main's "pre-existing run dir is unsupported"
guard so re-runs don't collide.

⚠ Doc divergence: the flow doc labels this stage "deterministic, no LLM"
but the schema's `description` / `role` fields require an LLM. Resolution
belongs to the flow-doc owner.

The `modules_extractor` import is local to `_extract_project_tree` (not
at module top) so a missing origin/main layout fails fast in the runner's
`_check_layout()` rather than as an opaque ImportError at package import.

## Cross-run cache

Stage 02 is the most expensive stage that doesn't depend on telemetry —
running it against the same subject checkout produces a byte-identical
ProjectTree (modulo schema drift). The cache lives at
`~/.cache/spotlights-engine/projecttree/pt-<repo-hash>-<git-sha>[-<dirty-hash>].v1.json`
and short-circuits extraction when the (resolved repo path, git HEAD,
porcelain status) tuple matches. Bypass with `--no-projecttree-cache`
or `SignalPipelineInput.projecttree_cache=False`.

Cache misses cleanly when:
- subject_root isn't a git repo (no SHA available)
- HEAD has moved
- working tree is dirty in a way that hashes differently (porcelain output
  is part of the key)
- ProjectTree schema has bumped past v1 (filename suffix)
"""

from __future__ import annotations

import hashlib
import json
import re
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from pydantic import ValidationError

from spotlights_engine.schemas.project import ProjectTree
from spotlights_engine.signal_pipeline.stages._types import StageContext, StageSpec


_CACHE_SCHEMA_VERSION = "v1"


def parse_artifact(raw: Any) -> ProjectTree:
    return ProjectTree.model_validate(raw)


def _ts_subdir() -> str:
    """Filesystem-safe UTC ISO timestamp for a per-invocation subdir name."""
    iso = datetime.now(timezone.utc).isoformat(timespec="seconds")
    return re.sub(r"[:+]", "-", iso)


# ── Cross-run ProjectTree cache ──────────────────────────────────────────


def _cache_root() -> Path:
    return Path.home() / ".cache" / "spotlights-engine" / "projecttree"


def _git(subject_root: Path, *args: str) -> str | None:
    """Run a git command in `subject_root`; return stripped stdout or None
    if git isn't available, the dir isn't a repo, or the command fails."""
    try:
        result = subprocess.run(
            ["git", "-C", str(subject_root), *args],
            capture_output=True,
            text=True,
            check=False,
        )
    except (FileNotFoundError, OSError):
        return None
    if result.returncode != 0:
        return None
    return result.stdout.strip()


def _cache_key(subject_root: Path) -> str | None:
    """Cache filename for this subject_root, or None if the dir isn't a
    git repo (cache disabled in that case — no stable identity)."""
    head = _git(subject_root, "rev-parse", "HEAD")
    if not head:
        return None
    repo_hash = hashlib.sha256(
        str(subject_root.resolve()).encode("utf-8")
    ).hexdigest()[:12]
    porcelain = _git(subject_root, "status", "--porcelain")
    suffix = ""
    if porcelain:
        # Dirty tree — fold the working-tree state into the key so an
        # uncommitted edit doesn't return a cache built before the edit.
        dirty_hash = hashlib.sha256(porcelain.encode("utf-8")).hexdigest()[:8]
        suffix = f"-{dirty_hash}"
    return f"pt-{repo_hash}-{head}{suffix}.{_CACHE_SCHEMA_VERSION}.json"


def _load_cache(subject_root: Path) -> ProjectTree | None:
    """Cache hit → parsed ProjectTree. Cache miss / not-a-repo / corrupt → None.
    Errors are silent on purpose — callers fall back to extraction."""
    key = _cache_key(subject_root)
    if key is None:
        return None
    path = _cache_root() / key
    if not path.exists():
        return None
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
        return ProjectTree.model_validate(raw)
    except (OSError, json.JSONDecodeError, ValidationError):
        return None


def _save_cache(subject_root: Path, tree: ProjectTree) -> None:
    """Persist a fresh ProjectTree under its cache key. Best-effort:
    OSErrors (e.g. permission denied on ~/.cache) are swallowed so the
    pipeline run never fails on cache write."""
    key = _cache_key(subject_root)
    if key is None:
        return
    cache_dir = _cache_root()
    try:
        cache_dir.mkdir(parents=True, exist_ok=True)
        (cache_dir / key).write_text(
            tree.model_dump_json(indent=2, by_alias=True) + "\n",
            encoding="utf-8",
        )
    except OSError:
        pass


def _extract_project_tree(
    subject_root: Path, log_dir: Path, on_event=None, use_cache: bool = True
) -> ProjectTree:
    """Real extraction path. Factored for ease of monkeypatching in tests.

    The conftest in `tests/unit/signal_pipeline/` swaps this out with a
    placeholder so runner state-machine tests don't fire Claude.

    When `use_cache=True` (the default), check the cross-run cache first
    and short-circuit on hit; on miss, extract and write the result back.
    Set `use_cache=False` to force a fresh extraction (the result still
    overwrites the cache).
    """
    if use_cache:
        cached = _load_cache(subject_root)
        if cached is not None:
            if on_event is not None:
                on_event("projecttree-cache: HIT — skipping extraction")
            return cached

    # Lazy import per the safety order in the approved plan.
    from spotlights_engine.modules_extractor import ExtractorConfig, extract
    from spotlights_engine.schemas.pipeline import ModulesExtractorInput

    artifacts_dir = log_dir / _ts_subdir()
    tree = extract(
        ModulesExtractorInput(repo_path=subject_root),
        config=ExtractorConfig(artifacts_dir=artifacts_dir),
        on_event=on_event,
    )
    _save_cache(subject_root, tree)
    return tree


def run(ctx: StageContext) -> ProjectTree:
    return _extract_project_tree(
        ctx.signal_input.subject_root,
        ctx.log_dir,
        ctx.on_event,
        use_cache=ctx.signal_input.projecttree_cache,
    )


SPEC = StageSpec(
    stage_id="02",
    name="projecttree",
    shape="single",
    upstream=(),  # ProjectTree extraction takes subject_root only, no signals dep
    parse_artifact=parse_artifact,
    run=run,
)


__all__ = ["SPEC"]
