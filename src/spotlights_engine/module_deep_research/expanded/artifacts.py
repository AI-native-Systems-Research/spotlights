"""Artifact writer for expanded module deep research."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

from pydantic import BaseModel

from spotlights_engine.module_deep_research.expanded.models import ExpandedResearchReport
from spotlights_engine.schemas.pipeline import ModuleDeepResearchOutput


def write_expanded_artifacts(
    *,
    artifacts_dir: Path | None,
    report: ExpandedResearchReport,
    output: ModuleDeepResearchOutput,
) -> None:
    """Write sidecar artifacts without affecting the public Spotlight output."""
    if artifacts_dir is None:
        return
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    _write_json(artifacts_dir / "expanded_research_report.json", report)
    _write_json(artifacts_dir / "module_deep_research.expanded.json", output)
    _write_json(artifacts_dir / "prompt_evolution.json", report.prompt_evolution)
    _write_json(artifacts_dir / "merged_papers.json", report.merged_papers)
    _write_json(artifacts_dir / "before_after.json", report.comparison)
    _write_json(artifacts_dir / "coverage_ui.json", report.coverage)
    _write_json(artifacts_dir / "agent_outputs.json", report.agent_outputs)


def _write_json(path: Path, payload: Any) -> None:
    if isinstance(payload, BaseModel):
        serializable = payload.model_dump(mode="json")
    elif isinstance(payload, list):
        serializable = [
            item.model_dump(mode="json") if isinstance(item, BaseModel) else item
            for item in payload
        ]
    else:
        serializable = payload
    text = json.dumps(serializable, indent=2, sort_keys=True) + "\n"
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(text, encoding="utf-8")
    os.replace(tmp, path)
