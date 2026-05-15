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


def iter_dir(artifacts_dir: Path, n: int, agent: str) -> Path:
    tag = "bootstrap" if n == 0 else agent
    return candidate_discovery_root(artifacts_dir) / f"iter_{n}_{tag}"


def iterations_jsonl(artifacts_dir: Path) -> Path:
    return candidate_discovery_root(artifacts_dir) / "iterations.jsonl"


def final_artifact(artifacts_dir: Path) -> Path:
    return artifacts_dir / "candidates.json"


__all__ = [
    "candidate_discovery_root",
    "final_artifact",
    "iter_dir",
    "iterations_jsonl",
    "schema_path",
]
