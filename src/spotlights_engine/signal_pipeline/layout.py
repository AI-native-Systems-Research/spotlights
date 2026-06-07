"""On-disk layout, atomic writes, and canonical artifact hashing.

Mirrors the idiom in `spotlights_engine.spotlights_manager.persistence` —
write a sibling `.tmp` then `os.replace` for atomicity. Ownership is
separate (this is its own pipeline) so we copy the pattern rather than
import.

The hashing helper here is the **single source of truth** for
`upstream_<…>_hash` in fan-out manifests. All resume / inject decisions
must use `artifact_hash`.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel


StageId = Literal["01", "02", "03", "04", "05"]
ALL_STAGES: tuple[StageId, ...] = ("01", "02", "03", "04", "05")
StageShape = Literal["single", "fanout"]


# ── Atomic IO ────────────────────────────────────────────────────────────


def atomic_write_text(path: Path, text: str) -> None:
    """Write `text` to `path` atomically via a sibling `.tmp` file + replace."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(text, encoding="utf-8")
    os.replace(tmp, path)


def atomic_write_json(path: Path, payload: Any) -> None:
    """Pretty-printed (indent=2, sort_keys=True) JSON write — for human-readable artifacts.

    NOTE: this is the *storage* form, not the *hash* form. `artifact_hash`
    has its own canonical compact serialization rules.
    """
    atomic_write_text(path, json.dumps(payload, indent=2, sort_keys=True) + "\n")


# ── Canonical hashing (Phase 2 of the plan) ──────────────────────────────


def _canonicalize(payload: Any) -> str:
    """Produce the exact string fed to SHA-256.

    Rules (pinned — manifests are only valid under these rules):
    1. Pydantic models → `model_dump(mode="json", by_alias=True, exclude_none=False)`.
       Lists of models map element-wise. Plain dicts/lists pass through.
    2. JSON-encode with `sort_keys=True, ensure_ascii=False, separators=(",", ":")`.
       No indentation, no trailing newline, deterministic key order.
    3. List order is preserved — a deliberate reorder forces fan-out re-evaluation.
    """
    if isinstance(payload, BaseModel):
        obj: Any = payload.model_dump(mode="json", by_alias=True, exclude_none=False)
    elif isinstance(payload, list):
        obj = [
            p.model_dump(mode="json", by_alias=True, exclude_none=False)
            if isinstance(p, BaseModel)
            else p
            for p in payload
        ]
    elif isinstance(payload, dict):
        obj = {
            k: (
                v.model_dump(mode="json", by_alias=True, exclude_none=False)
                if isinstance(v, BaseModel)
                else v
            )
            for k, v in payload.items()
        }
    else:
        obj = payload
    return json.dumps(obj, sort_keys=True, ensure_ascii=False, separators=(",", ":"))


def artifact_hash(payload: Any) -> str:
    """Stable SHA-256 hex digest of a stage artifact's logical content.

    Used everywhere a fan-out manifest reads or writes `upstream_<…>_hash`.
    Do not introduce ad-hoc digesting elsewhere.
    """
    return hashlib.sha256(_canonicalize(payload).encode("utf-8")).hexdigest()


# ── Run-dir layout ───────────────────────────────────────────────────────


# Stage filename prefixes. Single-artifact stages get a `.json` suffix;
# fan-out stages get a directory under the same prefix.
_STAGE_FILENAMES: dict[StageId, str] = {
    "01": "01_signals",
    "02": "02_projecttree",
    "03": "03_candidates",
    "04": "04_changes",
    "05": "05_results",
}


# Loose validation for fan-out entry IDs to avoid path-traversal surprises.
# `Candidate.id` matches `^cand-\d{4}$` per `schemas.candidate`; `Change.change_id`
# is free-form. Allow `[A-Za-z0-9._-]` and reject `..` segments.
_ENTRY_ID = re.compile(r"^[A-Za-z0-9._-]+$")


def validate_entry_id(id_: str) -> None:
    if not _ENTRY_ID.match(id_) or ".." in id_:
        raise ValueError(f"invalid fan-out entry id: {id_!r}")


@dataclass(frozen=True)
class RunDirLayout:
    """Path resolver for one signal-pipeline run."""

    root: Path

    @property
    def status_path(self) -> Path:
        return self.root / "status.json"

    @property
    def input_path(self) -> Path:
        return self.root / "input.json"

    @property
    def logs_root(self) -> Path:
        return self.root / "_logs"

    def stage_log_dir(self, stage: StageId) -> Path:
        return self.logs_root / _STAGE_FILENAMES[stage]

    def stage_artifact(self, stage: StageId, *, shape: StageShape) -> Path:
        """Return the canonical artifact path for a stage.

        - single → file at `<root>/<NN>_<name>.json`
        - fanout → directory at `<root>/<NN>_<name>/`
        """
        base = self.root / _STAGE_FILENAMES[stage]
        return base.with_suffix(".json") if shape == "single" else base

    def fanout_manifest(self, stage: StageId) -> Path:
        return self.stage_artifact(stage, shape="fanout") / "_manifest.json"

    def fanout_entry(self, stage: StageId, id_: str) -> Path:
        validate_entry_id(id_)
        return self.stage_artifact(stage, shape="fanout") / f"{id_}.json"


__all__ = [
    "ALL_STAGES",
    "RunDirLayout",
    "StageId",
    "StageShape",
    "artifact_hash",
    "atomic_write_json",
    "atomic_write_text",
    "validate_entry_id",
]
