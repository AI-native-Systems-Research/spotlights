"""Schema for the structured module map produced by the modules extractor.

See [docs/modules_extractor.md](../../../../docs/modules_extractor.md) for the
narrative spec; this file is the canonical definition.
"""

from __future__ import annotations

import json
import re
from collections.abc import Iterable
from pathlib import Path, PurePosixPath
from typing import Any

from pydantic import BaseModel, Field, model_validator

_NAME_PATTERN = r"^[a-z][a-z0-9_]*$"
_INVALID_SEGMENT_CHARS = re.compile(r"[^a-z0-9_]")


def _normalize_module_segment(raw: str) -> str:
    """Normalize one path segment into a valid `_NAME_PATTERN` token.

    Lowercase, replace any character outside `[a-z0-9_]` with `_`, and
    guarantee a leading letter (prefixing `m_` when the result would start
    with a digit or underscore — e.g. `2d` -> `m_2d`). Used both for the
    `Module.name` default and for every qualified-name path segment. Because
    segments carry no `/`, the slash-form qualified name is unambiguous and its
    per-module slug (`/` -> `_`) stays collision-free.
    """
    cleaned = _INVALID_SEGMENT_CHARS.sub("_", raw.strip().lower())
    if not cleaned or not cleaned[0].isalpha():
        cleaned = "m_" + cleaned
    return cleaned


def _normalize_source_root(raw: str) -> str:
    """Normalize a repo-relative `source_root`: strip slashes, collapse `./`.

    Returns `""` for the repo root. Rejects absolute paths and `..` segments.
    """
    stripped = raw.strip()
    if stripped.startswith("/"):
        raise ValueError(f"source_root must be repo-relative, got absolute: {raw!r}")
    parts: list[str] = []
    for seg in stripped.split("/"):
        if seg in ("", "."):
            continue
        if seg == "..":
            raise ValueError(f"source_root must not contain '..': {raw!r}")
        parts.append(seg)
    return "/".join(parts)


def _posix_relpath(path: str, source_root: str) -> str:
    """Repo-relative POSIX path of `path` below `source_root`.

    `source_root` must already be normalized. Returns `""` when `path` equals
    `source_root`; raises when `path` is not under `source_root`.
    """
    norm = path.strip().strip("/")
    if not source_root:
        return norm
    if norm == source_root:
        return ""
    prefix = source_root + "/"
    if not norm.startswith(prefix):
        raise ValueError(
            f"module path {path!r} is not under source_root {source_root!r}"
        )
    return norm[len(prefix):]


def _qualified_name(path: str, source_root: str) -> str:
    """Qualified name = `source_root`-relative path, normalized per segment.

    Raises `ValueError` for an empty relative path (`path == source_root`),
    rather than returning `""`.
    """
    rel = _posix_relpath(path, source_root)
    parts = [seg for seg in rel.split("/") if seg]
    if not parts:
        raise ValueError(
            f"module path {path!r} equals source_root {source_root!r}; "
            "the source-root directory is not itself a module"
        )
    return "/".join(_normalize_module_segment(seg) for seg in parts)


class File(BaseModel):
    path: str
    role: str


class Module(BaseModel):
    name: str = Field(pattern=_NAME_PATTERN)
    path: str
    description: str = ""
    depends_on: list[str] = Field(default_factory=list)
    main_files: list[File] = Field(default_factory=list)
    submodules: list["Module"] = Field(default_factory=list)

    @model_validator(mode="before")
    @classmethod
    def _default_name_from_path(cls, data: Any) -> Any:
        if isinstance(data, dict) and not data.get("name") and data.get("path"):
            basename = PurePosixPath(str(data["path"]).strip("/")).name
            data = {**data, "name": _normalize_module_segment(basename)}
        return data

    @model_validator(mode="after")
    def _check_name_matches_basename(self) -> "Module":
        basename = PurePosixPath(self.path.strip("/")).name if self.path.strip() else ""
        if basename:
            expected = _normalize_module_segment(basename)
            if self.name != expected:
                raise ValueError(
                    f"Module.name {self.name!r} disagrees with the normalized "
                    f"basename of path {self.path!r} (expected {expected!r}); the "
                    "last qualified-name segment must equal Module.name"
                )
        return self

    @model_validator(mode="after")
    def _sort_submodules(self) -> "Module":
        self.submodules.sort(key=lambda m: m.name)
        return self


class Repository(BaseModel):
    name: str
    summary: str
    source_root: str = ""
    external_dependencies: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def _normalize_source_root_field(self) -> "Repository":
        self.source_root = _normalize_source_root(self.source_root)
        return self


class ProjectTree(BaseModel):
    repository: Repository
    modules: list[Module] = Field(default_factory=list)

    @model_validator(mode="before")
    @classmethod
    def _infer_source_root_when_omitted(cls, data: Any) -> Any:
        """Deterministic fallback: if `repository.source_root` was *omitted*,
        infer `"src"` when every module path lives under a top-level `src/`,
        else `""`. An explicit `source_root` (including `""`) is respected, so
        a root-layout repo can opt in without triggering the `src/` guess.
        """
        if not isinstance(data, dict):
            return data
        repo = data.get("repository")
        if not isinstance(repo, dict) or "source_root" in repo:
            return data

        def _paths(modules: Any) -> Iterable[str]:
            if not isinstance(modules, list):
                return
            for m in modules:
                if isinstance(m, dict) and isinstance(m.get("path"), str):
                    yield m["path"]
                    yield from _paths(m.get("submodules"))

        paths = [p.strip().strip("/") for p in _paths(data.get("modules"))]
        # Require every module to be strictly *under* src/. A module whose path
        # is literally "src" is treated as root-layout (qn "src") rather than
        # inferring source_root="src", which the path/qn validator would then
        # reject as "path equals source_root".
        inferred = (
            "src" if paths and all(p.startswith("src/") for p in paths) else ""
        )
        data = {**data, "repository": {**repo, "source_root": inferred}}
        return data

    @model_validator(mode="after")
    def _sort_modules(self) -> "ProjectTree":
        self.modules.sort(key=lambda m: m.name)
        return self

    @model_validator(mode="after")
    def _validate_paths_and_qns(self) -> "ProjectTree":
        """Enforce the invariants `walk()` and the qualified-name keys rely on.

        Every module path must be repo-relative, non-empty, under
        `source_root` (and not equal to it), and every child path must be
        nested under its parent. Qualified names are computed and checked for
        duplicates so normalization collisions (`foo-bar` vs `foo_bar`) fail
        fast instead of silently colliding downstream.
        """
        source_root = self.repository.source_root
        seen_qns: dict[str, str] = {}

        def _validate(modules: list[Module], parent_path: str | None) -> None:
            for m in modules:
                norm = m.path.strip().strip("/")
                if not norm:
                    raise ValueError(f"module path must be non-empty: {m.path!r}")
                if m.path.strip().startswith("/"):
                    raise ValueError(
                        f"module path must be repo-relative, got absolute: {m.path!r}"
                    )
                if any(seg == ".." for seg in norm.split("/")):
                    raise ValueError(
                        f"module path must not contain '..': {m.path!r}"
                    )
                if source_root:
                    if norm == source_root:
                        raise ValueError(
                            f"module path equals source_root {source_root!r}; the "
                            f"source-root directory is not itself a module: {m.path!r}"
                        )
                    if not norm.startswith(source_root + "/"):
                        raise ValueError(
                            f"module path {m.path!r} is not under "
                            f"source_root {source_root!r}"
                        )
                if parent_path is not None and not norm.startswith(
                    parent_path.strip().strip("/") + "/"
                ):
                    raise ValueError(
                        f"child module path {m.path!r} is not nested under "
                        f"parent path {parent_path!r}"
                    )
                qn = _qualified_name(m.path, source_root)
                if qn in seen_qns:
                    raise ValueError(
                        f"duplicate qualified name {qn!r} derived from paths "
                        f"{seen_qns[qn]!r} and {m.path!r} (normalization collision?)"
                    )
                seen_qns[qn] = m.path
                _validate(m.submodules, m.path)

        _validate(self.modules, None)
        return self

    def walk(self) -> Iterable[tuple[str, Module]]:
        """Preorder traversal yielding `(qualified_name, module)` for every module.

        The qualified name is the module's `path` relative to
        `repository.source_root`, normalized per segment. The `_validate_paths_and_qns`
        validator guarantees every path yields a non-empty, unique qn, so this is total.
        """
        source_root = self.repository.source_root

        def _walk(modules: list[Module]) -> Iterable[tuple[str, Module]]:
            for m in modules:
                yield _qualified_name(m.path, source_root), m
                yield from _walk(m.submodules)

        yield from _walk(self.modules)

    def leaves(self) -> Iterable[tuple[str, Module]]:
        """Yield `(qualified_name, module)` for every leaf module, in `walk()` order."""
        for qn, m in self.walk():
            if not m.submodules:
                yield qn, m

    def resolve(self, qualified_name: str) -> Module | None:
        """Look up a module by its qualified name.

        A bare `name` (no `/`) is accepted only when exactly one module in the
        tree carries it.
        """
        for qn, m in self.walk():
            if qn == qualified_name:
                return m
        if "/" not in qualified_name:
            matches = [m for _, m in self.walk() if m.name == qualified_name]
            if len(matches) == 1:
                return matches[0]
        return None

    def to_json(self, path: Path) -> None:
        """Serialize this tree to `path` as indented JSON (UTF-8, trailing newline)."""
        path.write_text(self.model_dump_json(indent=2) + "\n", encoding="utf-8")

    @classmethod
    def from_json(cls, path: Path) -> "ProjectTree":
        """Load a `ProjectTree` from a JSON file written by `to_json`."""
        return cls.model_validate(json.loads(path.read_text(encoding="utf-8")))


Module.model_rebuild()


__all__ = ["File", "Module", "ProjectTree", "Repository"]
