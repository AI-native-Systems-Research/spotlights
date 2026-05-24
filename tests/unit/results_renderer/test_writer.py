"""Writer tests: emission of index.md and per-module pages."""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from spotlights_engine.results_renderer.api import (
    RendererConfig,
    RendererInput,
    render,
)
from spotlights_engine.results_renderer.errors import RendererSetupError

from tests.unit.results_renderer._fixtures import make_full_run


def test_render_writes_index_and_module_pages(tmp_path: Path) -> None:
    artifacts = tmp_path / "artifacts"
    output = tmp_path / "output"
    artifacts.mkdir()
    make_full_run(artifacts)

    result = render(RendererInput(artifacts_dir=artifacts, output_folder=output))

    assert result.index_path == output / "index.md"
    assert result.index_path.exists()
    assert (output / "modules" / "v1.kv_offload.md").exists()
    assert (output / "modules" / "kernels.md").exists()

    text = result.index_path.read_text()
    assert "# Spotlights Run — demo" in text
    assert "[v1.kv_offload](modules/v1.kv_offload.md)" in text
    assert "[kernels](modules/kernels.md)" in text
    assert "reduce latency" in text  # context objective


def test_module_page_contents(tmp_path: Path) -> None:
    artifacts = tmp_path / "artifacts"
    output = tmp_path / "output"
    artifacts.mkdir()
    make_full_run(artifacts)

    render(RendererInput(artifacts_dir=artifacts, output_folder=output))

    page = (output / "modules" / "v1.kv_offload.md").read_text()
    assert "# v1.kv_offload" in page
    assert "Candidate: `hot_1` (`cand-0001`)" in page
    assert "#### Description" in page
    assert "#### Current approach" in page
    assert "#### Estimated impact explanation" in page
    assert "#### Evolve rationale" in page
    assert "#### Deep research proposals" in page
    assert "#### Agent proposals" in page
    # File link present (relative repo link by default).
    assert "[`src/v1/kv_offload/core.py`](src/v1/kv_offload/core.py)" in page

    # Deep research proposal cross-links the finding (title + URL when
    # the finding is in this module's set).
    assert "`find-0001` — *Finding 1*" in page
    assert "<https://example.com/f/1>" in page
    # Dangling finding ID renders without a title/URL — no crash.
    assert "`find-0099`" in page

    # Candidate without proposals -> placeholder.
    assert "_No proposals._" in page

    kernels_page = (output / "modules" / "kernels.md").read_text()
    assert "## Issues" in kernels_page
    assert "rate limited briefly" in kernels_page


def test_overwrite_false_on_nonempty_raises(tmp_path: Path) -> None:
    artifacts = tmp_path / "artifacts"
    output = tmp_path / "output"
    artifacts.mkdir()
    output.mkdir()
    (output / "junk.txt").write_text("x")
    make_full_run(artifacts)

    with pytest.raises(RendererSetupError):
        render(
            RendererInput(artifacts_dir=artifacts, output_folder=output),
            config=RendererConfig(overwrite=False),
        )


def test_rerender_replaces_modules_dir(tmp_path: Path) -> None:
    artifacts = tmp_path / "artifacts"
    output = tmp_path / "output"
    artifacts.mkdir()
    make_full_run(artifacts)

    render(RendererInput(artifacts_dir=artifacts, output_folder=output))
    # Plant an orphan page that should NOT survive the second render.
    orphan = output / "modules" / "ghost.md"
    orphan.write_text("ghost")
    assert orphan.exists()

    render(RendererInput(artifacts_dir=artifacts, output_folder=output))
    assert not orphan.exists()


def test_atomic_index_write(monkeypatch, tmp_path: Path) -> None:
    """If `os.replace` fails after the temp file is materialized, no partial
    `index.md` should remain on disk."""
    artifacts = tmp_path / "artifacts"
    output = tmp_path / "output"
    artifacts.mkdir()
    make_full_run(artifacts)

    real_replace = os.replace

    def boom(src, dst):  # type: ignore[no-untyped-def]
        if str(dst).endswith("index.md"):
            raise OSError("simulated crash")
        return real_replace(src, dst)

    monkeypatch.setattr(
        "spotlights_engine.results_renderer.writer.os.replace", boom
    )
    with pytest.raises(OSError):
        render(RendererInput(artifacts_dir=artifacts, output_folder=output))

    assert not (output / "index.md").exists()
    # Module pages still got written before index.
    assert (output / "modules" / "v1.kv_offload.md").exists()
