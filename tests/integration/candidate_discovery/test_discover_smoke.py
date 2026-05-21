"""Opt-in smoke test: runs `discover()` against real Claude / Codex CLIs.

Skipped unless `STAGE1_NEEDS_CLIS=1` and both `claude` and `codex` are on
PATH. The goal isn't to validate model outputs (they are non-deterministic)
— it's to confirm the argv lines actually launch each CLI and both runners
produce a `last_message.json` the orchestrator can parse.
"""

from __future__ import annotations

import os
import shutil
from pathlib import Path

import pytest

needs_clis = pytest.mark.skipif(
    os.environ.get("STAGE1_NEEDS_CLIS") != "1"
    or shutil.which("claude") is None
    or shutil.which("codex") is None,
    reason="set STAGE1_NEEDS_CLIS=1 and install `claude` + `codex` to run",
)


@needs_clis
def test_discover_against_real_clis(tmp_path: Path) -> None:
    from spotlights_engine.candidate_discovery import (
        DiscoveryConfig,
        DiscoveryResult,
        discover,
    )
    from spotlights_engine.schemas.common import SpotlightContext
    from spotlights_engine.schemas.pipeline import CandidateDiscoveryInput
    from spotlights_engine.schemas.project import (
        File,
        Module,
        ProjectTree,
        Repository,
    )

    repo = tmp_path / "repo"
    (repo / "src" / "foo").mkdir(parents=True)
    (repo / "src" / "foo" / "core.py").write_text(
        "def hot_loop():\n    total = 0\n"
        + "".join(f"    total += {i}\n" for i in range(1, 75))
        + "    return total\n"
    )
    artifacts = tmp_path / "artifacts"
    artifacts.mkdir()

    project_tree = ProjectTree(
        repository=Repository(name="demo", summary="single-file hot-loop demo"),
        modules=[
            Module(
                name="v1",
                path="src/v1",
                description="v1 namespace",
                submodules=[
                    Module(
                        name="foo",
                        path="src/foo",
                        description="single-file hot loop fixture.",
                        main_files=[File(path="src/foo/core.py", role="entry")],
                    ),
                ],
            )
        ],
    )
    inp = CandidateDiscoveryInput(
        project_tree=project_tree,
        module_qualified_name="v1/foo",
        context=SpotlightContext(objective="reduce hot-loop latency"),
    )
    cfg = DiscoveryConfig(
        repo_path=repo,
        artifacts_dir=artifacts,
        num_review_iterations=1,
        claude_max_turns=4,
        per_iteration_wallclock_s=180,
    )
    result = discover(inp, config=cfg)
    assert isinstance(result, DiscoveryResult)
    assert (artifacts / "candidates.json").exists()
