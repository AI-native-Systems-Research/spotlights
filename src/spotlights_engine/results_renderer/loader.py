"""Disk → in-memory model for the results renderer.

Reuses `spotlights_manager.persistence` so the read path stays consistent
with the orchestrator's resume code.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from spotlights_engine.modules_extractor.agent import ExtractionInvocation
from spotlights_engine.schemas.legacy.common import SpotlightContext
from spotlights_engine.schemas.legacy.project import ProjectTree
from spotlights_engine.spotlights_manager.persistence import (
    LoadedModuleState,
    ManagerPaths,
    read_extractor_outputs,
    read_manifest,
    read_module_state,
    slug_for,
)

from spotlights_engine.results_renderer.errors import (
    RendererLoadError,
    RendererSetupError,
)


@dataclass
class LoadedRun:
    """Everything the renderer needs from a single run dir."""

    paths: ManagerPaths
    manifest: dict[str, Any]
    project_tree: ProjectTree
    extractor_invocation: ExtractionInvocation
    context: SpotlightContext | None
    modules: dict[str, LoadedModuleState]
    warnings: list[str] = field(default_factory=list)


def load_run(artifacts_dir: Path) -> LoadedRun:
    """Read `<artifacts_dir>/spotlights_manager/` into a `LoadedRun`.

    Raises `RendererSetupError` if the run dir is missing entirely, and
    `RendererLoadError` if required artifacts (manifest JSON validity, both
    extractor sidecars) are missing or unreadable.
    """
    paths = ManagerPaths(artifacts_dir)
    if not paths.root.exists():
        raise RendererSetupError(
            f"spotlights_manager run dir missing under {artifacts_dir}: "
            f"{paths.root} does not exist"
        )

    warnings: list[str] = []

    manifest = _read_manifest_or_raise(paths)

    tree, invocation = read_extractor_outputs(paths)
    if tree is None or invocation is None:
        missing = []
        if tree is None:
            missing.append(str(paths.project_tree_path))
        if invocation is None:
            missing.append(str(paths.extractor_invocation_path))
        raise RendererLoadError(
            "extractor outputs missing — cannot render module pages: "
            + ", ".join(missing)
        )

    context = _read_context(manifest, warnings)

    module_qns = _resolve_module_qns(manifest, paths, warnings)

    modules: dict[str, LoadedModuleState] = {}
    for qn in module_qns:
        try:
            modules[qn] = read_module_state(paths.for_module(qn))
        except Exception as exc:  # noqa: BLE001
            warnings.append(
                f"failed to read module state for {qn!r}: "
                f"{type(exc).__name__}: {exc}"
            )

    # Prefer the qualified name carried by the checkpoint when it disagrees
    # with the manifest key (rare; only matters for the manifest-less fallback
    # below where keys are slugs, not qns).
    rekeyed: dict[str, LoadedModuleState] = {}
    for qn, state in modules.items():
        cp_qn = (
            state.checkpoint.module_qualified_name
            if state.checkpoint is not None
            else qn
        )
        rekeyed[cp_qn] = state
    modules = rekeyed

    return LoadedRun(
        paths=paths,
        manifest=manifest,
        project_tree=tree,
        extractor_invocation=invocation,
        context=context,
        modules=modules,
        warnings=warnings,
    )


def _read_manifest_or_raise(paths: ManagerPaths) -> dict[str, Any]:
    if not paths.manifest_path.exists():
        return {}
    try:
        return read_manifest(paths) or {}
    except json.JSONDecodeError as exc:
        raise RendererLoadError(
            f"manifest is not valid JSON: {paths.manifest_path}: {exc}"
        ) from exc


def _read_context(
    manifest: dict[str, Any], warnings: list[str]
) -> SpotlightContext | None:
    raw = manifest.get("context")
    if raw is None:
        warnings.append(
            "manifest has no 'context' field (pre-existing run); rendering "
            "with a placeholder"
        )
        return None
    try:
        return SpotlightContext.model_validate(raw)
    except Exception as exc:  # noqa: BLE001
        warnings.append(
            f"manifest 'context' did not validate as SpotlightContext: "
            f"{type(exc).__name__}: {exc}"
        )
        return None


def _resolve_module_qns(
    manifest: dict[str, Any],
    paths: ManagerPaths,
    warnings: list[str],
) -> list[str]:
    """Module qualified-name list. Manifest is authoritative; fall back to
    walking `modules_root` if absent (in which case the on-disk slugs are
    treated as qns and the per-module checkpoint will correct it later)."""
    manifest_modules = manifest.get("modules") if manifest else None
    if isinstance(manifest_modules, dict) and manifest_modules:
        return list(manifest_modules.keys())

    if not paths.modules_root.exists():
        return []
    warnings.append(
        "manifest has no module list; falling back to scanning "
        f"{paths.modules_root}"
    )
    qns: list[str] = []
    for child in sorted(paths.modules_root.iterdir()):
        if child.is_dir():
            qns.append(child.name)
    return qns


__all__ = [
    "LoadedRun",
    "load_run",
    "slug_for",
]
