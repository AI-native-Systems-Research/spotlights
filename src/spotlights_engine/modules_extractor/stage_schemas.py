"""Strict inter-stage models for the two-phase modules extractor.

These are the contracts that flow *between* the deterministic and LLM stages
(Idea 4 in `design/module_extraction_fix.md`). They are deliberately stricter
than the public `ProjectTree`/`Module` schema:

- `SourceRootDecision` (Stage 1) nests the canonical `Repository` model rather
  than duplicating its fields, and rejects unknown repository keys.
- `Skeleton`/`SkeletonNode` (Stage 2) are the deterministic inventory.
- `EnrichedTree` and friends (Stage 3) forbid extra fields, require non-empty
  bounded descriptions, and at most 5 unique `main_files` — none of which the
  public `Module` model enforces. The matching *lower* bound of 1 lives in
  `coverage.validate_enriched_tree`, which can see the filesystem and so can
  exempt a pure container directory that owns no file of its own.

The public serialized `ProjectTree` schema is unchanged; Stage 5 converts an
`EnrichedTree` back into `ProjectTree` dicts via `modules_as_project_tree_dicts`.
"""

from __future__ import annotations

from pathlib import PurePosixPath
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from spotlights_engine.schemas.project import Repository, _normalize_module_segment

# Bounds — generous enough for honest descriptions, tight enough that a
# runaway model can't blow the argv/prompt budget.
_MAX_DESCRIPTION = 2000
_MAX_ROLE = 500
_MAX_REASON = 500
_MAX_EXPLANATION = 1000

ExclusionReason = Literal[
    "centralized_tests",
    "docs",
    "examples",
    "benchmarks",
    "tooling",
    "generated",
    "vendored",
    "not_product_source",
    "repository_level_file",
]

_REPOSITORY_KEYS = frozenset(Repository.model_fields.keys())


def _structural_relpath(raw: str) -> str:
    """Structural (non-filesystem) check of a repo-relative POSIX path.

    Rejects empty, absolute, backslash-aliased, and `..`-containing paths.
    Filesystem existence/symlink checks happen later in `coverage.py` /
    `skeleton.py`; this only enforces the lexical contract.
    """
    stripped = raw.strip()
    if not stripped:
        raise ValueError("path must be non-empty")
    if "\\" in stripped:
        raise ValueError(f"path must use POSIX separators, got backslash: {raw!r}")
    if stripped.startswith("/"):
        raise ValueError(f"path must be repo-relative, got absolute: {raw!r}")
    norm = stripped.strip("/")
    parts = [seg for seg in norm.split("/") if seg not in ("", ".")]
    if any(seg == ".." for seg in parts):
        raise ValueError(f"path must not contain '..': {raw!r}")
    if not parts:
        raise ValueError(f"path normalized to empty: {raw!r}")
    return "/".join(parts)


# ── Stage 1 — source-root decision ───────────────────────────────────────


class ExcludedSourcePath(BaseModel):
    """One audited semantic exclusion of a source-bearing path."""

    model_config = ConfigDict(extra="forbid")

    path: str
    reason: ExclusionReason
    explanation: str = Field(min_length=1, max_length=_MAX_EXPLANATION)

    @model_validator(mode="after")
    def _normalize_path(self) -> ExcludedSourcePath:
        object.__setattr__(self, "path", _structural_relpath(self.path))
        return self


class SourceRootDecision(BaseModel):
    """Stage-1 model output: canonical `Repository` + audited exclusions."""

    model_config = ConfigDict(extra="forbid")

    repository: Repository
    excluded_source_paths: list[ExcludedSourcePath] = Field(default_factory=list)

    @model_validator(mode="before")
    @classmethod
    def _reject_unknown_repository_keys(cls, data: Any) -> Any:
        """The nested `Repository` uses Pydantic's default (ignore) extra
        behavior, so an unknown repository key would be silently dropped.
        Reject it here, before it can be discarded, keeping the stage contract
        strict without touching the public `Repository` schema."""
        if isinstance(data, dict):
            repo = data.get("repository")
            if isinstance(repo, dict):
                unknown = set(repo) - _REPOSITORY_KEYS
                if unknown:
                    raise ValueError(
                        f"unknown repository field(s): {sorted(unknown)}"
                    )
        return data

    @model_validator(mode="after")
    def _validate_repository_fields(self) -> SourceRootDecision:
        if not self.repository.name.strip():
            raise ValueError("repository.name must be non-empty")
        if not self.repository.summary.strip():
            raise ValueError("repository.summary must be non-empty")
        return self


# ── Stage 2 — deterministic skeleton ─────────────────────────────────────


class SkeletonNode(BaseModel):
    """One source-bearing directory in the deterministic inventory."""

    model_config = ConfigDict(extra="forbid")

    path: str
    direct_source_file_count: int = Field(ge=0)
    source_child_count: int = Field(ge=0)
    representative_files: list[str] = Field(default_factory=list, max_length=5)
    required: bool
    required_reasons: list[str] = Field(default_factory=list)
    children: list[SkeletonNode] = Field(default_factory=list)


class Skeleton(BaseModel):
    """The full deterministic inventory produced by Stage 2."""

    model_config = ConfigDict(extra="forbid")

    source_root: str
    nodes: list[SkeletonNode] = Field(default_factory=list)
    ignored: list[str] = Field(default_factory=list)
    excluded: list[str] = Field(default_factory=list)
    organizational_only: list[str] = Field(default_factory=list)
    skipped_symlinks: list[str] = Field(default_factory=list)
    inventory_fingerprint: str

    def iter_nodes(self):
        """Preorder traversal over every `SkeletonNode`."""

        def _walk(nodes: list[SkeletonNode]):
            for n in nodes:
                yield n
                yield from _walk(n.children)

        yield from _walk(self.nodes)

    def all_paths(self) -> set[str]:
        return {n.path for n in self.iter_nodes()}

    def required_paths(self) -> set[str]:
        return {n.path for n in self.iter_nodes() if n.required}


# ── Stage 3 — enrichment ─────────────────────────────────────────────────


class EnrichedFile(BaseModel):
    model_config = ConfigDict(extra="forbid")

    path: str
    role: str = Field(min_length=1, max_length=_MAX_ROLE)

    @model_validator(mode="after")
    def _normalize_path(self) -> EnrichedFile:
        object.__setattr__(self, "path", _structural_relpath(self.path))
        return self


def _default_name_from_path(data: Any) -> Any:
    if isinstance(data, dict) and not data.get("name") and data.get("path"):
        basename = PurePosixPath(str(data["path"]).strip("/")).name
        data = {**data, "name": _normalize_module_segment(basename)}
    return data


def _check_main_files(main_files: list[EnrichedFile]) -> None:
    # The upper bound is structural and enforced here; the lower bound is not,
    # because it depends on the filesystem. A *pure container* directory (only
    # sub-directories — a Go `cmd/`, a namespace package) has no file of its own
    # to cite once its children are emitted, so it legally cites none.
    # `validate_enriched_tree` holds every other module to ≥1, where it can see
    # the directory and say so precisely.
    if len(main_files) > 5:
        raise ValueError(
            f"main_files must have at most 5 entries, got {len(main_files)}"
        )
    paths = [f.path for f in main_files]
    if len(set(paths)) != len(paths):
        raise ValueError(f"main_files paths must be unique: {paths}")


class EnrichedSubmodule(BaseModel):
    """A nested module."""

    model_config = ConfigDict(extra="forbid")

    name: str = Field(pattern=r"^[a-z][a-z0-9_]*$")
    path: str
    description: str = Field(min_length=1, max_length=_MAX_DESCRIPTION)
    main_files: list[EnrichedFile]
    submodules: list[EnrichedSubmodule] = Field(default_factory=list)

    @model_validator(mode="before")
    @classmethod
    def _name_default(cls, data: Any) -> Any:
        return _default_name_from_path(data)

    @model_validator(mode="after")
    def _validate(self) -> EnrichedSubmodule:
        _structural_relpath(self.path)
        _check_main_files(self.main_files)
        return self


class EnrichedTopModule(BaseModel):
    """A top-level module."""

    model_config = ConfigDict(extra="forbid")

    name: str = Field(pattern=r"^[a-z][a-z0-9_]*$")
    path: str
    description: str = Field(min_length=1, max_length=_MAX_DESCRIPTION)
    main_files: list[EnrichedFile]
    submodules: list[EnrichedSubmodule] = Field(default_factory=list)

    @model_validator(mode="before")
    @classmethod
    def _name_default(cls, data: Any) -> Any:
        return _default_name_from_path(data)

    @model_validator(mode="after")
    def _validate(self) -> EnrichedTopModule:
        _structural_relpath(self.path)
        _check_main_files(self.main_files)
        return self


class FoldRecord(BaseModel):
    """A directory folded into an emitted ancestor's `main_files`."""

    model_config = ConfigDict(extra="forbid")

    path: str
    into: str
    reason: str = Field(min_length=1, max_length=_MAX_REASON)
    evidence_files: list[str] = Field(min_length=1)

    @model_validator(mode="after")
    def _validate(self) -> FoldRecord:
        object.__setattr__(self, "path", _structural_relpath(self.path))
        object.__setattr__(self, "into", _structural_relpath(self.into))
        norm_evidence = [_structural_relpath(f) for f in self.evidence_files]
        if len(set(norm_evidence)) != len(norm_evidence):
            raise ValueError(
                f"fold evidence_files must be unique: {self.evidence_files}"
            )
        object.__setattr__(self, "evidence_files", norm_evidence)
        return self


def _submodule_to_dict(sub: EnrichedSubmodule) -> dict[str, Any]:
    out: dict[str, Any] = {
        "name": sub.name,
        "path": sub.path,
        "description": sub.description,
        "main_files": [{"path": f.path, "role": f.role} for f in sub.main_files],
    }
    if sub.submodules:
        out["submodules"] = [_submodule_to_dict(s) for s in sub.submodules]
    return out


class EnrichedTree(BaseModel):
    """Stage-3 output: strict module tree + fold ledger."""

    model_config = ConfigDict(extra="forbid")

    modules: list[EnrichedTopModule] = Field(default_factory=list)
    folds: list[FoldRecord] = Field(default_factory=list)

    def iter_all_modules(self):
        """Yield every emitted module/submodule (top-level and nested)."""

        def _walk(mods) -> Any:
            for m in mods:
                yield m
                yield from _walk(m.submodules)

        yield from _walk(self.modules)

    def emitted_paths(self) -> set[str]:
        return {_structural_relpath(m.path) for m in self.iter_all_modules()}

    def modules_as_project_tree_dicts(self) -> list[dict[str, Any]]:
        """Convert the top-level modules to `ProjectTree`-shaped dicts."""
        out: list[dict[str, Any]] = []
        for m in self.modules:
            d: dict[str, Any] = {
                "name": m.name,
                "path": m.path,
                "description": m.description,
                "main_files": [
                    {"path": f.path, "role": f.role} for f in m.main_files
                ],
            }
            if m.submodules:
                d["submodules"] = [_submodule_to_dict(s) for s in m.submodules]
            out.append(d)
        return out


SkeletonNode.model_rebuild()
EnrichedSubmodule.model_rebuild()
EnrichedTopModule.model_rebuild()


__all__ = [
    "EnrichedFile",
    "EnrichedSubmodule",
    "EnrichedTopModule",
    "EnrichedTree",
    "ExcludedSourcePath",
    "ExclusionReason",
    "FoldRecord",
    "Skeleton",
    "SkeletonNode",
    "SourceRootDecision",
]
