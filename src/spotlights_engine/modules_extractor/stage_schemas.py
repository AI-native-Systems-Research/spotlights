"""Strict inter-stage models for the two-phase modules extractor.

These are the contracts that flow *between* the deterministic and LLM stages
(Idea 4 in `design/module_extraction_fix.md`). They are deliberately stricter
than the public `ProjectTree`/`Module` schema:

- `SourceRootDecision` (Stage 1) nests the canonical `Repository` model rather
  than duplicating its fields, and rejects unknown repository keys.
- `Skeleton`/`SkeletonNode` (Stage 2) are the deterministic inventory.
- Stage 3 has **two contracts** while the `ExtractorConfig.contract` migration
  flag exists (`design/module_extractor_simplified.md`):
  - the tree contract: `EnrichedTree` and friends — a module tree with a
    `folds[]` ledger. Forbids extra fields, requires non-empty bounded
    descriptions, and at most `MAX_MAIN_FILES` unique `main_files`; the
    matching *lower* bound of 1 lives in `coverage.validate_enriched_tree`,
    which can see the filesystem and so can exempt a pure container directory
    that owns no file of its own.
  - the assignment contract: Stage 3A returns an exhaustive
    `AssignmentTree` (one `MODULE`/`PART` label per skeleton path plus a
    decision per module), code resolves it into a `ResolvedAssignmentTree`,
    and Stage 3B returns `ModuleMetadataBatch`es (descriptions + main files
    for already-final module territories) united into a `ModuleMetadataMap`.

The public serialized `ProjectTree` schema is unchanged; Stage 5 converts an
`EnrichedTree` (or resolved assignments + metadata, via `derive.py`) back into
`ProjectTree` dicts.
"""

from __future__ import annotations

from pathlib import PurePosixPath
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from spotlights_engine.modules_extractor.constants import (
    MAX_MAIN_FILES,
    MAX_REPRESENTATIVE_FILES,
)
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
    # Non-init source files in the whole subtree (this node + descendants).
    # Defaulted so skeletons serialized before the field existed still load.
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
    if len(main_files) > MAX_MAIN_FILES:
        raise ValueError(
            f"main_files must have at most {MAX_MAIN_FILES} entries, "
            f"got {len(main_files)}"
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
        object.__setattr__(self, "path", _structural_relpath(self.path))
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
        object.__setattr__(self, "path", _structural_relpath(self.path))
        _check_main_files(self.main_files)
        return self


class FoldRecord(BaseModel):
    """A directory folded into an emitted ancestor module.

    `evidence_files` must name ≥1 real source file under the folded path
    (validated by Rule 8 in `coverage._validate_folds`); the evidence is not
    required to appear in the target's `main_files`.
    """

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


# ── Stage 3 — assignment contract ─────────────────────────────────────────

AssignmentLabel = Literal["MODULE", "PART"]

AssignmentOrigin = Literal[
    "top_level_anchor", "size", "independent", "part", "qn_collision_part"
]


def _normalize_mapping_keys(raw: Any, field_name: str) -> Any:
    """Normalize a raw dict's keys with `_structural_relpath`, collision-safe.

    Two distinct raw keys that normalize to the same path (`a/b` and `a//b`)
    are rejected instead of silently overwriting one — a Pydantic **parse**
    failure, not a separate validation class. Exact duplicate raw JSON object
    keys were already collapsed by the standard JSON parser; "exactly once"
    means the post-parse mapping checked here.
    """
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
    """The per-module decision record accompanying a `MODULE` label.

    `keep_reason` is required (non-null) only for a non-anchor module at or
    below the merge threshold — enforced by V3 in `assignments.py`, which can
    see the skeleton counts. Here it is merely stripped, bounded, and non-empty
    when present.
    """

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
    """Raw Stage-3A model output: one label for every skeleton path in scope.

    - `assignments` values are the closed `MODULE`/`PART` enum, never paths.
    - `module_decisions` keys must equal the `MODULE`-labeled paths exactly
      (the bijection is enforced by V3, which owns label semantics).
    - No descriptions, main files, folds, or evidence exist in this pass.
    """

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
    """Canonical post-validation Stage-3A artifact: labels + derived maps.

    `owners` maps every path to its owning module (a `MODULE` owns itself; a
    `PART` is owned by its deepest strict ancestor labeled `MODULE`).
    `origins` records why each path carries its label; `collision_precedence`
    maps each forced-`PART` normalized-name-collision loser to its winning
    path. The derived maps are audit data: every load must recompute and
    compare them against the full skeleton (`derive.verify_resolved_assignments`)
    rather than trust the persisted values.
    """

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
    # `max_length` (not a custom validator) so the generated Claude JSON
    # Schema carries `maxItems` and prompt/schema cannot drift.
    main_files: list[EnrichedFile] = Field(max_length=MAX_MAIN_FILES)

    @model_validator(mode="after")
    def _unique_main_files(self) -> ModuleInfo:
        paths = [f.path for f in self.main_files]
        if len(set(paths)) != len(paths):
            raise ValueError(f"main_files paths must be unique: {paths}")
        return self


class ModuleMetadataBatch(BaseModel):
    """One Stage-3B response: metadata for exactly the requested module set."""

    model_config = ConfigDict(extra="forbid")

    modules: dict[str, ModuleInfo]

    @model_validator(mode="before")
    @classmethod
    def _normalize_keys(cls, data: Any) -> Any:
        if isinstance(data, dict) and "modules" in data:
            data = {**data, "modules": _normalize_mapping_keys(data["modules"], "modules")}
        return data


class ModuleMetadataMap(BaseModel):
    """The checked global Stage-3B union across accepted batches."""

    model_config = ConfigDict(extra="forbid")

    modules: dict[str, ModuleInfo]

    @model_validator(mode="before")
    @classmethod
    def _normalize_keys(cls, data: Any) -> Any:
        if isinstance(data, dict) and "modules" in data:
            data = {**data, "modules": _normalize_mapping_keys(data["modules"], "modules")}
        return data


__all__ = [
    "AssignmentLabel",
    "AssignmentOrigin",
    "AssignmentTree",
    "EnrichedFile",
    "EnrichedSubmodule",
    "EnrichedTopModule",
    "EnrichedTree",
    "ExcludedSourcePath",
    "ExclusionReason",
    "FoldRecord",
    "ModuleDecision",
    "ModuleInfo",
    "ModuleMetadataBatch",
    "ModuleMetadataMap",
    "ResolvedAssignmentTree",
    "Skeleton",
    "SkeletonNode",
    "SourceRootDecision",
]
