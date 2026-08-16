"""Shared filesystem validation for assignment-based module extraction.

Stage-3 assignment and metadata validation lives in :mod:`assignments`. This
module owns the Stage-1 source-root gate and the filesystem helpers used by
that validator.
"""

from __future__ import annotations

from pathlib import Path, PurePosixPath

from spotlights_engine.modules_extractor.stage_schemas import SourceRootDecision


class CrossArtifactError(ValueError):
    """A persisted/model artifact disagrees with another artifact or the repo."""


def _resolves_inside(repo_path: Path, rel: str) -> bool:
    repo_resolved = repo_path.resolve(strict=False)
    target = (repo_path / rel).resolve(strict=False)
    try:
        target.relative_to(repo_resolved)
    except ValueError:
        return False
    return True


def _has_symlink_component(repo_path: Path, rel: str) -> bool:
    current = repo_path
    for segment in PurePosixPath(rel).parts:
        current = current / segment
        if current.is_symlink():
            return True
    return False


def _is_real_dir(repo_path: Path, rel: str) -> bool:
    path = repo_path / rel
    return path.is_dir() and not _has_symlink_component(repo_path, rel)


def _is_real_file(repo_path: Path, rel: str) -> bool:
    path = repo_path / rel
    return path.is_file() and not _has_symlink_component(repo_path, rel)


def _dir_has_direct_file(repo_path: Path, module_path: str) -> bool:
    directory = repo_path / module_path
    try:
        entries = list(directory.iterdir())
    except (OSError, NotADirectoryError):
        return False
    return any(entry.is_file() and not entry.is_symlink() for entry in entries)


def _is_ancestor(ancestor: str, descendant: str) -> bool:
    return descendant == ancestor or descendant.startswith(ancestor + "/")


def forced_repository_level_files(
    source_root: str, detected_source_files: list[str]
) -> list[str]:
    """Return source files directly at the modeled root.

    The directory-only public schema cannot represent these as modules, so
    Stage 1 classifies them deterministically as repository-level files.
    """
    if source_root:
        prefix = source_root + "/"
        return sorted(
            path
            for path in detected_source_files
            if path.startswith(prefix) and "/" not in path[len(prefix) :]
        )
    return sorted(path for path in detected_source_files if "/" not in path)


def validate_source_root_decision(
    decision: SourceRootDecision,
    repo_path: Path,
    detected_source_files: list[str],
) -> None:
    """Validate a Stage-1 decision against the current repository."""
    source_root = decision.repository.source_root
    if source_root and not _is_real_dir(repo_path, source_root):
        raise CrossArtifactError(
            f"source_root {source_root!r} is not an existing non-symlink "
            "directory inside the repository"
        )

    exclusions = [entry.path for entry in decision.excluded_source_paths]
    if len(set(exclusions)) != len(exclusions):
        raise CrossArtifactError(f"duplicate excluded_source_paths: {exclusions}")

    for excluded in exclusions:
        if not _resolves_inside(repo_path, excluded):
            raise CrossArtifactError(f"exclusion escapes repository: {excluded!r}")
        if not (repo_path / excluded).exists():
            raise CrossArtifactError(f"exclusion path does not exist: {excluded!r}")
        if source_root and excluded == source_root:
            raise CrossArtifactError(
                f"exclusion cannot equal source_root: {excluded!r}"
            )
        if source_root and _is_ancestor(excluded, source_root):
            raise CrossArtifactError(
                f"exclusion {excluded!r} contains the source_root {source_root!r}"
            )

    for path in exclusions:
        for other in exclusions:
            if path != other and _is_ancestor(other, path):
                raise CrossArtifactError(
                    f"exclusions must form an antichain; {path!r} is nested "
                    f"under {other!r}"
                )

    for excluded in exclusions:
        if not any(
            path == excluded or path.startswith(excluded + "/")
            for path in detected_source_files
        ):
            raise CrossArtifactError(
                f"exclusion {excluded!r} covers no detected source file"
            )


__all__ = [
    "CrossArtifactError",
    "forced_repository_level_files",
    "validate_source_root_decision",
]
