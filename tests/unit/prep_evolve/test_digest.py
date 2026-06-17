"""Digest renderer tests (plan §10.3)."""

from __future__ import annotations

from pathlib import Path

from spotlights_engine.prep_evolve.digest import render_digest
from spotlights_engine.prep_evolve.extract import build_spec
from spotlights_engine.prep_evolve.resolve import (
    load_result,
    resolve_candidate,
    resolve_candidates,
    resolve_findings,
    resolve_module,
    resolve_module_run,
)
from spotlights_engine.prep_evolve.spec import SourceRevision
from spotlights_engine.prep_evolve.validate_target import ValidatedCandidate

from . import _fixtures as fx


def _spec(tmp_path: Path, scope="candidate"):
    loaded = load_result(fx.write_result(tmp_path))
    run = resolve_module_run(loaded, "v1.attention")
    candidates = resolve_candidates(run, "v1.attention")
    candidate = resolve_candidate(candidates, "cand-0002")
    module = resolve_module(loaded.project_tree, "v1.attention")
    return build_spec(
        loaded=loaded,
        module=module,
        dot_qn="v1.attention",
        candidate=candidate,
        findings=resolve_findings(run, "v1.attention"),
        repo_path="/tmp/repo",
        validated=ValidatedCandidate(fx.CAND_START, fx.CAND_END, "x"),
        revision=SourceRevision(git_commit=None, dirty=None, captured_at="t"),
        scope=scope,
        direction="minimize",
    )


def test_digest_sections_present(tmp_path: Path) -> None:
    md = render_digest(_spec(tmp_path))
    assert "## Optimization goal" in md
    assert "## Target" in md
    assert "## Research findings (2)" in md
    assert "## Existing proposals (2)" in md


def test_digest_proposal_finding_cross_reference(tmp_path: Path) -> None:
    md = render_digest(_spec(tmp_path))
    # find-0001 (POD-Attention) is candidate-linked -> listed as finding #1.
    assert "from finding #1: POD-Attention" in md


def test_digest_agent_proposal_no_source(tmp_path: Path) -> None:
    md = render_digest(_spec(tmp_path))
    assert "agent proposal, no source finding" in md


def test_digest_target_shows_lines_and_impact(tmp_path: Path) -> None:
    md = render_digest(_spec(tmp_path))
    assert f"{fx.CAND_FILE}:{fx.CAND_START}–{fx.CAND_END}" in md
    assert "impact: high" in md


def test_digest_whole_file_scope_label(tmp_path: Path) -> None:
    md = render_digest(_spec(tmp_path, scope="module-main-files"))
    assert "(whole file" in md
