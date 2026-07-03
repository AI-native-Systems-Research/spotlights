"""Stage 1 run-directory path conventions.

One place that knows the layout of `<artifacts_dir>/candidate_discovery/`.
Spec §3 is the authoritative reference.
"""

from __future__ import annotations

from pathlib import Path


def candidate_discovery_root(artifacts_dir: Path) -> Path:
    return artifacts_dir / "candidate_discovery"


def schema_path(artifacts_dir: Path) -> Path:
    return candidate_discovery_root(artifacts_dir) / "candidates.schema.json"


def repo_context_path(artifacts_dir: Path) -> Path:
    return candidate_discovery_root(artifacts_dir) / "repo_context.md"


def iter_dir(artifacts_dir: Path, n: int, agent: str) -> Path:
    # Iteration 0 is the bootstrap pair: one dir per agent
    # (`iter_0_bootstrap_claude_code`, `iter_0_bootstrap_codex`). Both keep the
    # `bootstrap` marker so tooling matching the `iter_0_bootstrap*` prefix still
    # finds the seed dirs. Review dirs stay `iter_N_<agent>`.
    tag = f"bootstrap_{agent}" if n == 0 else agent
    return candidate_discovery_root(artifacts_dir) / f"iter_{n}_{tag}"


def merge_dir(artifacts_dir: Path) -> Path:
    """Dir for the deterministic bootstrap merge (renumber + dedup) output."""
    return candidate_discovery_root(artifacts_dir) / "iter_0_merge"


def iterations_jsonl(artifacts_dir: Path) -> Path:
    return candidate_discovery_root(artifacts_dir) / "iterations.jsonl"


def final_artifact(artifacts_dir: Path) -> Path:
    return artifacts_dir / "candidates.json"


__all__ = [
    "candidate_discovery_root",
    "final_artifact",
    "iter_dir",
    "iterations_jsonl",
    "merge_dir",
    "repo_context_path",
    "schema_path",
]
