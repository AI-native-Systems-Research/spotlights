"""Writer tests: emission of index.md and per-module pages."""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from spotlights_engine.costing.manifest import RunManifestCost
from spotlights_engine.results_renderer.api import (
    RendererConfig,
    RendererInput,
    render,
)
from spotlights_engine.results_renderer.errors import RendererSetupError
from tests.unit.results_renderer._fixtures import (
    make_full_run,
    make_run_manifest,
    write_run_manifest,
)


def test_render_writes_index_and_module_pages(tmp_path: Path) -> None:
    artifacts = tmp_path / "artifacts"
    output = tmp_path / "output"
    artifacts.mkdir()
    make_full_run(artifacts)

    result = render(RendererInput(artifacts_dir=artifacts, output_folder=output))

    assert result.index_path == output / "index.md"
    assert result.index_path.exists()
    assert (output / "modules" / "v1_kv_offload.md").exists()
    assert (output / "modules" / "kernels.md").exists()

    text = result.index_path.read_text()
    assert "# Spotlights Run — demo" in text
    assert "[v1/kv_offload](modules/v1_kv_offload.md)" in text
    assert "[kernels](modules/kernels.md)" in text
    assert "reduce latency" in text  # context objective


def test_module_page_contents(tmp_path: Path) -> None:
    artifacts = tmp_path / "artifacts"
    output = tmp_path / "output"
    artifacts.mkdir()
    make_full_run(artifacts)

    result = render(RendererInput(artifacts_dir=artifacts, output_folder=output))

    page = (output / "modules" / "v1_kv_offload.md").read_text()
    assert "# v1/kv_offload" in page
    assert "[← All modules](../index.md)" in page

    # Candidates table header + rows.
    assert "| Candidate | Impact | Deep research proposals |" in page
    # Symbol cell links to candidate page; impact column shows enum values.
    assert "[`hot_1`](v1_kv_offload/hot_1__cand-mod-0001.md)" in page
    assert "[`hot_2`](v1_kv_offload/hot_2__cand-mod-0002.md)" in page
    assert "[`hot_3`](v1_kv_offload/hot_3__cand-mod-0003.md)" in page
    # Sort: most deep_research_proposals first. cand-mod-0001 has 2; others 0.
    table_start = page.index("| Candidate | Impact | Deep research proposals |")
    table_block = page[table_start:]
    pos_h1 = table_block.index("hot_1")
    pos_h2 = table_block.index("hot_2")
    pos_h3 = table_block.index("hot_3")
    assert pos_h1 < pos_h2 < pos_h3

    # Per-candidate pages exist for every candidate.
    cand_dir = output / "modules" / "v1_kv_offload"
    assert (cand_dir / "hot_1__cand-mod-0001.md").exists()
    assert (cand_dir / "hot_2__cand-mod-0002.md").exists()
    assert (cand_dir / "hot_3__cand-mod-0003.md").exists()

    # candidate_pages on RendererResult is populated.
    cps = result.candidate_pages["v1/kv_offload"]
    assert cps["cand-mod-0001"] == cand_dir / "hot_1__cand-mod-0001.md"
    assert cps["cand-mod-0002"] == cand_dir / "hot_2__cand-mod-0002.md"

    # Candidate page contents — H1 + breadcrumb + sections.
    cpage = (cand_dir / "hot_1__cand-mod-0001.md").read_text()
    assert "# hot_1" in cpage
    assert "[← v1/kv_offload](../v1_kv_offload.md)" in cpage
    assert "## Description" in cpage
    assert "## Current approach" in cpage
    assert "## Estimated impact explanation" in cpage
    assert "## Evolve rationale" in cpage
    assert "## Deep research proposals" in cpage
    assert "## Agent proposals" in cpage
    # File link on the candidate page (not the module page).
    assert "[`src/v1/kv_offload/core.py`](src/v1/kv_offload/core.py)" in cpage
    # Cross-link to a finding in the candidate page.
    assert "`find-mod-0001` — *Finding 1*" in cpage
    assert "<https://example.com/f/1>" in cpage
    # Dangling finding id still rendered without a title/URL.
    assert "`find-mod-0099`" in cpage

    # Module page no longer carries the per-candidate detail.
    assert "## Description" not in page
    assert "Detailed description" not in page

    # Candidate without proposals -> placeholder shows on its candidate page.
    cpage2 = (cand_dir / "hot_2__cand-mod-0002.md").read_text()
    assert "_No proposals._" in cpage2

    kernels_page = (output / "modules" / "kernels.md").read_text()
    assert "## Issues" in kernels_page
    assert "rate limited briefly" in kernels_page


def test_run_manifest_section_rendered(tmp_path: Path) -> None:
    artifacts = tmp_path / "artifacts"
    output = tmp_path / "output"
    artifacts.mkdir()
    paths, _ = make_full_run(artifacts)
    write_run_manifest(
        paths,
        make_run_manifest(
            notes="unpriced models excluded from cost: litellm:litellm/gemma-open"
        ),
    )

    result = render(RendererInput(artifacts_dir=artifacts, output_folder=output))
    text = result.index_path.read_text()

    assert "## Run manifest" in text
    assert "- **Run id:** `run-abc123`" in text
    assert "https://github.com/acme/demo" in text
    assert "`deadbeef`" in text  # target commit
    assert "`cafef00d`" in text  # spotlights commit
    assert "- **Candidates:** 4" in text
    assert "SUCCEEDED: 1, DEGRADED: 1, FAILED: 0, SKIPPED: 0" in text

    # Cost & usage: 4-decimal cost, tokens, both model rows.
    assert "$0.1234" in text
    assert "**Total tokens:** 4650" in text
    assert "| claude-opus-4-8 | anthropic | deep_research | 1000 | 2000 | 300 | 50 |" in text
    assert "| gpt-5-codex | openai | agent_proposals | 500 | 800 | 0 | 0 |" in text

    # Notes travel with the number.
    assert "### Notes" in text
    assert "unpriced models excluded from cost: litellm:litellm/gemma-open" in text

    # Section sits before Renderer warnings (none here, so just after Modules).
    assert text.index("## Modules") < text.index("## Run manifest")


def test_run_manifest_timing_fields_distinct(tmp_path: Path) -> None:
    """Resume-after-crash: wall_clock and accumulated differ; both must show."""
    artifacts = tmp_path / "artifacts"
    output = tmp_path / "output"
    artifacts.mkdir()
    paths, _ = make_full_run(artifacts)
    write_run_manifest(
        paths, make_run_manifest(wall_clock_s=12.0, accumulated_duration_s=340.0)
    )

    text = render(
        RendererInput(artifacts_dir=artifacts, output_folder=output)
    ).index_path.read_text()

    assert "- **Wall clock (s):** 12.0" in text
    assert "- **Accumulated duration (s):** 340.0" in text


def test_run_manifest_missing_shows_placeholder_no_warning(tmp_path: Path) -> None:
    artifacts = tmp_path / "artifacts"
    output = tmp_path / "output"
    artifacts.mkdir()
    make_full_run(artifacts)  # no run_manifest.json written

    result = render(RendererInput(artifacts_dir=artifacts, output_folder=output))
    text = result.index_path.read_text()

    assert "## Run manifest" in text
    assert "_(run manifest unavailable for this run)_" in text
    assert not any("run_manifest" in w for w in result.warnings)


def test_run_manifest_malformed_records_warning(tmp_path: Path) -> None:
    artifacts = tmp_path / "artifacts"
    output = tmp_path / "output"
    artifacts.mkdir()
    paths, _ = make_full_run(artifacts)
    paths.run_manifest_path.write_text("{not valid json", encoding="utf-8")

    result = render(RendererInput(artifacts_dir=artifacts, output_folder=output))
    text = result.index_path.read_text()

    assert "_(run manifest unavailable for this run)_" in text
    assert any("run_manifest.json did not validate" in w for w in result.warnings)


def test_run_manifest_no_models_placeholder(tmp_path: Path) -> None:
    artifacts = tmp_path / "artifacts"
    output = tmp_path / "output"
    artifacts.mkdir()
    paths, _ = make_full_run(artifacts)
    write_run_manifest(paths, make_run_manifest(with_models=False))

    text = render(
        RendererInput(artifacts_dir=artifacts, output_folder=output)
    ).index_path.read_text()

    assert "_(no model usage captured)_" in text


def test_run_manifest_rate_note_and_notes_both_shown(tmp_path: Path) -> None:
    artifacts = tmp_path / "artifacts"
    output = tmp_path / "output"
    artifacts.mkdir()
    paths, _ = make_full_run(artifacts)
    write_run_manifest(
        paths,
        make_run_manifest(
            rate_note="partial pricing: 1 of 2 models priced",
            notes="degraded usage capture for one runner",
        ),
    )

    text = render(
        RendererInput(artifacts_dir=artifacts, output_folder=output)
    ).index_path.read_text()

    assert "partial pricing: 1 of 2 models priced" in text
    assert "degraded usage capture for one runner" in text


def test_run_manifest_external_cost_rendered(tmp_path: Path) -> None:
    artifacts = tmp_path / "artifacts"
    output = tmp_path / "output"
    artifacts.mkdir()
    paths, _ = make_full_run(artifacts)
    write_run_manifest(
        paths,
        make_run_manifest(
            external_cost=RunManifestCost(
                amount_usd=0.5678,
                source="public-api-rate-table",
                rate_note="external: rates applied: anthropic:aws/claude-opus-4-8",
            )
        ),
    )

    text = render(
        RendererInput(artifacts_dir=artifacts, output_folder=output)
    ).index_path.read_text()

    # Both the contracted and external cost lines appear.
    assert "**Total cost (USD):** $0.1234" in text
    assert "**External cost (USD):** $0.5678" in text
    assert "_(source: public-api-rate-table)_" in text
    # External rate note surfaces in Notes.
    assert "external: rates applied: anthropic:aws/claude-opus-4-8" in text
    # JSON/page agreement: rendered value matches the persisted manifest.
    assert "0.5678" in paths.run_manifest_path.read_text()


def test_run_manifest_external_cost_omitted_when_none(tmp_path: Path) -> None:
    artifacts = tmp_path / "artifacts"
    output = tmp_path / "output"
    artifacts.mkdir()
    paths, _ = make_full_run(artifacts)
    write_run_manifest(paths, make_run_manifest())  # external_cost defaults None

    text = render(
        RendererInput(artifacts_dir=artifacts, output_folder=output)
    ).index_path.read_text()

    assert "**Total cost (USD):**" in text
    assert "**External cost (USD):**" not in text


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
    assert (output / "modules" / "v1_kv_offload.md").exists()
