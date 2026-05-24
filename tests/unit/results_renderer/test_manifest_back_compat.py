"""Renderer should handle pre-existing run dirs that lack `manifest.context`."""

from __future__ import annotations

from pathlib import Path

from spotlights_engine.results_renderer.api import (
    RendererInput,
    render,
)

from tests.unit.results_renderer._fixtures import (
    make_full_run,
)


def test_render_without_manifest_context(tmp_path: Path) -> None:
    artifacts = tmp_path / "artifacts"
    output = tmp_path / "output"
    artifacts.mkdir()
    paths, _ = make_full_run(artifacts)
    # Strip context from the manifest to simulate a pre-existing run.
    import json
    manifest = json.loads(paths.manifest_path.read_text())
    manifest.pop("context", None)
    paths.manifest_path.write_text(json.dumps(manifest))

    result = render(RendererInput(artifacts_dir=artifacts, output_folder=output))

    assert result.index_path.exists()
    text = result.index_path.read_text()
    assert "context unavailable for pre-existing run" in text
    assert any("context" in w for w in result.warnings)
