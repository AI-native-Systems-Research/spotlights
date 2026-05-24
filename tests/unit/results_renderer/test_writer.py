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

    result = render(RendererInput(artifacts_dir=artifacts, output_folder=output))

    page = (output / "modules" / "v1.kv_offload.md").read_text()
    assert "# v1.kv_offload" in page
    assert "[← All modules](../index.md)" in page

    # Candidates table header + rows.
    assert "| Candidate | Impact | Deep research proposals |" in page
    # Symbol cell links to candidate page; impact column shows enum values.
    assert "[`hot_1`](v1.kv_offload/hot_1__cand-0001.md)" in page
    assert "[`hot_2`](v1.kv_offload/hot_2__cand-0002.md)" in page
    assert "[`hot_3`](v1.kv_offload/hot_3__cand-0003.md)" in page
    # Sort: most deep_research_proposals first. cand-0001 has 2; others 0.
    table_start = page.index("| Candidate | Impact | Deep research proposals |")
    table_block = page[table_start:]
    pos_h1 = table_block.index("hot_1")
    pos_h2 = table_block.index("hot_2")
    pos_h3 = table_block.index("hot_3")
    assert pos_h1 < pos_h2 < pos_h3

    # Per-candidate pages exist for every candidate.
    cand_dir = output / "modules" / "v1.kv_offload"
    assert (cand_dir / "hot_1__cand-0001.md").exists()
    assert (cand_dir / "hot_2__cand-0002.md").exists()
    assert (cand_dir / "hot_3__cand-0003.md").exists()

    # candidate_pages on RendererResult is populated.
    cps = result.candidate_pages["v1.kv_offload"]
    assert cps["cand-0001"] == cand_dir / "hot_1__cand-0001.md"
    assert cps["cand-0002"] == cand_dir / "hot_2__cand-0002.md"

    # Candidate page contents — H1 + breadcrumb + sections.
    cpage = (cand_dir / "hot_1__cand-0001.md").read_text()
    assert "# hot_1" in cpage
    assert "[← v1.kv_offload](../v1.kv_offload.md)" in cpage
    assert "## Description" in cpage
    assert "## Current approach" in cpage
    assert "## Estimated impact explanation" in cpage
    assert "## Evolve rationale" in cpage
    assert "## Deep research proposals" in cpage
    assert "## Agent proposals" in cpage
    # File link on the candidate page (not the module page).
    assert "[`src/v1/kv_offload/core.py`](src/v1/kv_offload/core.py)" in cpage
    # Cross-link to a finding in the candidate page.
    assert "`find-0001` — *Finding 1*" in cpage
    assert "<https://example.com/f/1>" in cpage
    # Dangling finding id still rendered without a title/URL.
    assert "`find-0099`" in cpage

    # Module page no longer carries the per-candidate detail.
    assert "## Description" not in page
    assert "Detailed description" not in page

    # Candidate without proposals -> placeholder shows on its candidate page.
    cpage2 = (cand_dir / "hot_2__cand-0002.md").read_text()
    assert "_No proposals._" in cpage2

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
