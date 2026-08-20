"""FIX-NOTES.md content: oracles verbatim, base commit, no-verification statement."""

from __future__ import annotations

from pathlib import Path

import pytest

from spotlights_engine.one_shot_fix.notes import render_fix_notes
from spotlights_engine.prep_evolve.extract import build_spec, infer_direction
from spotlights_engine.prep_evolve.resolve import (
    find_candidate,
    load_result,
    resolve_findings,
    resolve_module,
)
from spotlights_engine.prep_evolve.spec import EvolveSpec, SourceRevision
from spotlights_engine.prep_evolve.validate_target import validate_candidate_target
from tests.unit.prep_evolve._fixtures import (
    CAND_END,
    CAND_FILE,
    CAND_START,
    make_repo,
    write_result,
)

BASE_SHA = "0123456789abcdef0123456789abcdef01234567"
CAND_ID = "cand-v1_attention-0002"


@pytest.fixture
def spec_and_repo(tmp_path: Path) -> tuple[EvolveSpec, Path]:
    result_path = write_result(tmp_path)
    repo = make_repo(tmp_path)
    loaded = load_result(result_path)
    sel = find_candidate(loaded, CAND_ID)
    spec = build_spec(
        loaded=loaded,
        module=resolve_module(loaded.project_tree, sel.qn),
        qn=sel.qn,
        candidate=sel.candidate,
        findings=resolve_findings(sel.run, sel.qn),
        repo_path=str(repo),
        validated=validate_candidate_target(repo, sel.candidate),
        revision=SourceRevision(
            git_commit=BASE_SHA, dirty=False, captured_at="2026-08-20T00:00:00+00:00"
        ),
        scope="candidate",
        direction=infer_direction(loaded.context.objective),
    )
    return spec, repo


def _notes(spec_and_repo, **overrides) -> str:
    spec, repo = spec_and_repo
    kwargs = {
        "spec": spec,
        "candidate_id": CAND_ID,
        "module_qn": "v1/attention",
        "base_sha": BASE_SHA,
        "repo": repo,
        "change_summary": "Replaced the heuristic with a lookup table.",
        "patch_produced": True,
        "agent_error": None,
    }
    kwargs.update(overrides)
    return render_fix_notes(**kwargs)


def test_records_identity_and_base_commit(spec_and_repo) -> None:
    notes = _notes(spec_and_repo)
    assert CAND_ID in notes
    assert "v1/attention" in notes
    assert BASE_SHA in notes


def test_records_in_scope_files_with_line_ranges(spec_and_repo) -> None:
    notes = _notes(spec_and_repo)
    assert f"{CAND_FILE}:{CAND_START}-{CAND_END}" in notes


def test_carries_the_oracles_verbatim(spec_and_repo) -> None:
    notes = _notes(spec_and_repo)
    assert "pytest tests/kernels/test_tile.py" in notes
    assert "TPOT" in notes


def test_states_that_nothing_was_verified(spec_and_repo) -> None:
    notes = _notes(spec_and_repo)
    assert "Nothing in this directory was verified" in notes


def test_includes_the_apply_and_verify_recipe(spec_and_repo) -> None:
    spec, repo = spec_and_repo
    notes = _notes(spec_and_repo)
    assert f"git -C {repo} checkout {BASE_SHA}" in notes
    assert f"git -C {repo} apply --check fix.patch" in notes
    assert f"git -C {repo} apply fix.patch" in notes
    assert f"git -C {repo} apply -3" in notes
    assert "patch -p1 < fix.patch" in notes
    assert "repo root" in notes


def test_apply_recipe_is_not_location_dependent(spec_and_repo) -> None:
    """Regression guard: every `git apply` invocation must carry `-C {repo}`.

    A bare `git apply --check fix.patch` (no `-C`) either fails with "not a
    git repository" when run from wherever FIX-NOTES.md was saved, or —
    worse — silently applies against whatever unrelated git repo happens to
    contain that directory. See the finding this test encodes.
    """
    notes = _notes(spec_and_repo)
    assert "git apply --check fix.patch" not in notes
    assert "git apply fix.patch" not in notes


def test_includes_findings_with_urls(spec_and_repo) -> None:
    notes = _notes(spec_and_repo)
    assert "https://arxiv.org/abs/2410.18038" in notes


def test_agent_summary_is_folded_in(spec_and_repo) -> None:
    notes = _notes(spec_and_repo)
    assert "Replaced the heuristic with a lookup table." in notes


def test_no_patch_case_is_stated_explicitly(spec_and_repo) -> None:
    notes = _notes(
        spec_and_repo,
        patch_produced=False,
        change_summary="The proposal needs a change in a file outside scope.",
    )
    assert "No patch was produced" in notes
    assert "outside scope" in notes
    assert "git apply" not in notes


def test_agent_error_is_recorded(spec_and_repo) -> None:
    notes = _notes(
        spec_and_repo, patch_produced=False, change_summary=None, agent_error="claude exit=3"
    )
    assert "claude exit=3" in notes
