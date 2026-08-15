"""Strict inter-stage models for assignment-based module extraction.

Stage 1 selects the source root, Stage 2 builds a deterministic skeleton,
Stage 3A labels every skeleton path ``MODULE`` or ``PART``, and Stage 3B adds
metadata for the resolved modules. Stage 5 derives the unchanged public
``ProjectTree`` from those assignment artifacts.
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from spotlights_engine.modules_extractor.constants import (
    MAX_MAIN_FILES,
    MAX_REPRESENTATIVE_FILES,
)
from spotlights_engine.schemas.project import Repository

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
    """Normalize a lexical repo-relative POSIX path without filesystem I/O."""
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
    """Stage-1 model output: canonical repository metadata and exclusions."""

    model_config = ConfigDict(extra="forbid")

    repository: Repository
    excluded_source_paths: list[ExcludedSourcePath] = Field(default_factory=list)

    @model_validator(mode="before")
    @classmethod
    def _reject_unknown_repository_keys(cls, data: Any) -> Any:
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


class SkeletonNode(BaseModel):
    """One source-bearing directory in the deterministic inventory."""

    model_config = ConfigDict(extra="forbid")

    path: str
    direct_source_file_count: int = Field(ge=0)
    subtree_source_file_count: int = Field(default=0, ge=0)
    source_child_count: int = Field(ge=0)
    representative_files: list[str] = Field(
        default_factory=list, max_length=MAX_REPRESENTATIVE_FILES
    )
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
        def _walk(nodes: list[SkeletonNode]):
            for node in nodes:
                yield node
                yield from _walk(node.children)

        yield from _walk(self.nodes)

    def all_paths(self) -> set[str]:
        return {node.path for node in self.iter_nodes()}

    def required_paths(self) -> set[str]:
        return {node.path for node in self.iter_nodes() if node.required}


class EnrichedFile(BaseModel):
    """One representative file and its role in a resolved module."""

    model_config = ConfigDict(extra="forbid")

    path: str
    role: str = Field(min_length=1, max_length=_MAX_ROLE)

    @model_validator(mode="after")
    def _normalize_path(self) -> EnrichedFile:
        object.__setattr__(self, "path", _structural_relpath(self.path))
        return self


AssignmentLabel = Literal["MODULE", "PART"]
AssignmentOrigin = Literal[
    "top_level_anchor", "size", "independent", "part", "qn_collision_part"
]


def _normalize_mapping_keys(raw: Any, field_name: str) -> Any:
    """Normalize mapping keys and reject normalization collisions."""
    if not isinstance(raw, dict):
        return raw
    out: dict[str, Any] = {}
    first_raw: dict[str, Any] = {}
    for key, value in raw.items():
        norm = _structural_relpath(str(key))
        if norm in out:
            raise ValueError(
                f"{field_name} keys {first_raw[norm]!r} and {key!r} both "
                f"normalize to {norm!r}"
            )
        out[norm] = value
        first_raw[norm] = key
    return out


class ModuleDecision(BaseModel):
    """The decision record accompanying one ``MODULE`` label."""

    model_config = ConfigDict(extra="forbid")

    keep_reason: str | None = Field(default=None, max_length=_MAX_REASON)

    @model_validator(mode="after")
    def _strip_reason(self) -> ModuleDecision:
        if self.keep_reason is not None:
            stripped = self.keep_reason.strip()
            if not stripped:
                raise ValueError("keep_reason, when present, must be non-empty")
            object.__setattr__(self, "keep_reason", stripped)
        return self


class AssignmentTree(BaseModel):
    """Raw Stage-3A output: one label for every skeleton path in scope."""

    model_config = ConfigDict(extra="forbid")

    assignments: dict[str, AssignmentLabel]
    module_decisions: dict[str, ModuleDecision]

    @model_validator(mode="before")
    @classmethod
    def _normalize_keys(cls, data: Any) -> Any:
        if isinstance(data, dict):
            data = {**data}
            for field_name in ("assignments", "module_decisions"):
                if field_name in data:
                    data[field_name] = _normalize_mapping_keys(
                        data[field_name], field_name
                    )
        return data

    def module_paths(self) -> set[str]:
        return {p for p, label in self.assignments.items() if label == "MODULE"}

    def part_paths(self) -> set[str]:
        return {p for p, label in self.assignments.items() if label == "PART"}


class ResolvedAssignmentTree(BaseModel):
    """Canonical Stage-3A artifact with deterministically derived maps."""

    model_config = ConfigDict(extra="forbid")

    assignments: dict[str, AssignmentLabel]
    module_decisions: dict[str, ModuleDecision]
    owners: dict[str, str]
    origins: dict[str, AssignmentOrigin]
    collision_precedence: dict[str, str] = Field(default_factory=dict)
    merge_threshold: int = Field(ge=0)

    @model_validator(mode="before")
    @classmethod
    def _normalize_keys(cls, data: Any) -> Any:
        if isinstance(data, dict):
            data = {**data}
            for field_name in (
                "assignments",
                "module_decisions",
                "owners",
                "origins",
                "collision_precedence",
            ):
                if field_name in data:
                    data[field_name] = _normalize_mapping_keys(
                        data[field_name], field_name
                    )
        return data

    @model_validator(mode="after")
    def _check_key_agreement(self) -> ResolvedAssignmentTree:
        keys = set(self.assignments)
        if set(self.owners) != keys:
            raise ValueError("owners keys must equal assignments keys")
        if set(self.origins) != keys:
            raise ValueError("origins keys must equal assignments keys")
        modules = {p for p, label in self.assignments.items() if label == "MODULE"}
        if set(self.module_decisions) != modules:
            raise ValueError(
                "module_decisions keys must equal MODULE-labeled paths exactly"
            )
        return self

    def module_paths(self) -> set[str]:
        return {p for p, label in self.assignments.items() if label == "MODULE"}


class ModuleInfo(BaseModel):
    """Stage-3B metadata for one final module territory."""

    model_config = ConfigDict(extra="forbid")

    description: str = Field(min_length=1, max_length=_MAX_DESCRIPTION)
    main_files: list[EnrichedFile] = Field(max_length=MAX_MAIN_FILES)

    @model_validator(mode="after")
    def _unique_main_files(self) -> ModuleInfo:
        paths = [f.path for f in self.main_files]
        if len(set(paths)) != len(paths):
            raise ValueError(f"main_files paths must be unique: {paths}")
        return self


class ModuleMetadataBatch(BaseModel):
    """One Stage-3B response for exactly the requested module set."""

    model_config = ConfigDict(extra="forbid")

    modules: dict[str, ModuleInfo]

    @model_validator(mode="before")
    @classmethod
    def _normalize_keys(cls, data: Any) -> Any:
        if isinstance(data, dict) and "modules" in data:
            data = {
                **data,
                "modules": _normalize_mapping_keys(data["modules"], "modules"),
            }
        return data


class ModuleMetadataMap(BaseModel):
    """The checked global Stage-3B union across accepted batches."""

    model_config = ConfigDict(extra="forbid")

    modules: dict[str, ModuleInfo]

    @model_validator(mode="before")
    @classmethod
    def _normalize_keys(cls, data: Any) -> Any:
        if isinstance(data, dict) and "modules" in data:
            data = {
                **data,
                "modules": _normalize_mapping_keys(data["modules"], "modules"),
            }
        return data


SkeletonNode.model_rebuild()

__all__ = [
    "AssignmentLabel",
    "AssignmentOrigin",
    "AssignmentTree",
    "EnrichedFile",
    "ExcludedSourcePath",
    "ExclusionReason",
    "ModuleDecision",
    "ModuleInfo",
    "ModuleMetadataBatch",
    "ModuleMetadataMap",
    "ResolvedAssignmentTree",
    "Skeleton",
    "SkeletonNode",
    "SourceRootDecision",
]
