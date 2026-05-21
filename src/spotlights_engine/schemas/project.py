"""Schema for the structured module map produced by the modules extractor.

See [docs/modules_extractor.md](../../../../docs/modules_extractor.md) for the
narrative spec; this file is the canonical definition.
"""

from __future__ import annotations

import json
from collections.abc import Iterable
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field, model_validator

_NAME_PATTERN = r"^[a-z][a-z0-9_]*$"


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
            data = {**data, "name": Path(data["path"]).name}
        return data

    @model_validator(mode="after")
    def _sort_submodules(self) -> "Module":
        self.submodules.sort(key=lambda m: m.name)
        return self


class Repository(BaseModel):
    name: str
    summary: str
    external_dependencies: list[str] = Field(default_factory=list)


class ProjectTree(BaseModel):
    repository: Repository
    modules: list[Module] = Field(default_factory=list)

    @model_validator(mode="after")
    def _sort_modules(self) -> "ProjectTree":
        self.modules.sort(key=lambda m: m.name)
        return self

    def walk(self) -> Iterable[tuple[str, Module]]:
        """Preorder traversal yielding `(qualified_name, module)` for every module."""

        def _walk(modules: list[Module], prefix: str) -> Iterable[tuple[str, Module]]:
            for m in modules:
                qn = f"{prefix}/{m.name}" if prefix else m.name
                yield qn, m
                yield from _walk(m.submodules, qn)

        yield from _walk(self.modules, "")

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
