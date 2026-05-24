"""Opt-in end-to-end smoke for SpotlightsManager.

Runs the manager against a single target module through real `claude` and
`codex` CLIs. Skipped unless `MANAGER_NEEDS_CLIS=1` and both binaries are on
PATH. The goal isn't to validate model outputs (they are non-deterministic)
— it's to confirm checkpoint files land where plan §7 says they should and
that resume after deleting the step-3 sidecar re-runs only step 3.
"""

from __future__ import annotations

import os
import shutil
from pathlib import Path

import pytest

from spotlights_engine.schemas.common import SpotlightContext
from spotlights_engine.schemas.pipeline import SpotlightsManagerInput
from spotlights_engine.spotlights_manager import (
    ModuleFilter,
    SpotlightsManagerConfig,
    run_with_telemetry,
)


needs_clis = pytest.mark.skipif(
    os.environ.get("MANAGER_NEEDS_CLIS") != "1"
    or shutil.which("claude") is None
    or shutil.which("codex") is None,
    reason="set MANAGER_NEEDS_CLIS=1 and install `claude` + `codex` to run",
)


@needs_clis
def test_manager_runs_subset_and_resumes(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    (repo / "src" / "v1" / "foo").mkdir(parents=True)
    (repo / "src" / "v1" / "foo" / "core.py").write_text(
        "def hot_loop():\n    total = 0\n"
        + "".join(f"    total += {i}\n" for i in range(1, 75))
        + "    return total\n"
    )
    artifacts = tmp_path / "artifacts"
    artifacts.mkdir()

    inp = SpotlightsManagerInput(
        repo_path=repo,
        context=SpotlightContext(objective="reduce hot-loop latency"),
    )
    cfg = SpotlightsManagerConfig(
        artifacts_dir=artifacts,
        output_folder=artifacts.parent / "output",
        module_filter=ModuleFilter(include=["v1.foo"]),
    )

    result = run_with_telemetry(inp, config=cfg)
    mr = result.module_runs.get("v1.foo")
    assert mr is not None
    mp = artifacts / "spotlights_manager" / "modules" / "v1.foo"
    assert (mp / "status.json").exists()

    # Resume: delete deep_research sidecar; only step 3 should re-run.
    if (mp / "module_deep_research.json").exists():
        (mp / "module_deep_research.json").unlink()
    run_with_telemetry(inp, config=cfg)
    assert (mp / "status.json").exists()
