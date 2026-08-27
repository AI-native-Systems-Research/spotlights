"""The apply prompt: oracles verbatim, base SHA present, scope confined."""

from __future__ import annotations

from pathlib import Path

import pytest

from spotlights_engine.one_shot_apply.prompts import (
    CHANGE_SUMMARY_NAME,
    build_apply_prompt,
)
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


@pytest.fixture
def spec(tmp_path: Path) -> EvolveSpec:
    result_path = write_result(tmp_path)
    repo = make_repo(tmp_path)
    loaded = load_result(result_path)
    sel = find_candidate(loaded, "cand-v1_attention-0002")
    validated = validate_candidate_target(repo, sel.candidate)
    return build_spec(
        loaded=loaded,
        module=resolve_module(loaded.project_tree, sel.qn),
        qn=sel.qn,
        candidate=sel.candidate,
        findings=resolve_findings(sel.run, sel.qn),
        repo_path=str(repo),
        validated=validated,
        revision=SourceRevision(
            git_commit=BASE_SHA, dirty=False, captured_at="2026-08-20T00:00:00+00:00"
        ),
        scope="candidate",
        direction=infer_direction(loaded.context.objective),
    )


def test_base_sha_is_present(spec: EvolveSpec, tmp_path: Path) -> None:
    prompt = build_apply_prompt(spec=spec, worktree=tmp_path / "wt")
    assert BASE_SHA in prompt


def test_oracles_appear_verbatim(spec: EvolveSpec, tmp_path: Path) -> None:
    prompt = build_apply_prompt(spec=spec, worktree=tmp_path / "wt")
    # The fixture rationale yields this correctness oracle and these metrics.
    assert "pytest tests/kernels/test_tile.py" in prompt
    assert "TPOT" in prompt and "TTFT" in prompt


def test_scope_is_confined_to_the_candidate_file_and_range(
    spec: EvolveSpec, tmp_path: Path
) -> None:
    prompt = build_apply_prompt(spec=spec, worktree=tmp_path / "wt")
    assert f"{CAND_FILE}:{CAND_START}-{CAND_END}" in prompt
    # The other module main file is NOT in scope for --scope candidate.
    assert "pkg/attn/launch.py" not in prompt


def test_findings_with_urls_and_proposals_are_included(
    spec: EvolveSpec, tmp_path: Path
) -> None:
    prompt = build_apply_prompt(spec=spec, worktree=tmp_path / "wt")
    assert "POD-Attention" in prompt
    assert "https://arxiv.org/abs/2410.18038" in prompt
    assert "Make tile size GQA-aware" in prompt


def test_prompt_forbids_running_tests_and_names_the_worktree(
    spec: EvolveSpec, tmp_path: Path
) -> None:
    worktree = tmp_path / "wt"
    prompt = build_apply_prompt(spec=spec, worktree=worktree)
    assert str(worktree) in prompt
    assert "Do not run tests" in prompt
    assert CHANGE_SUMMARY_NAME in prompt
