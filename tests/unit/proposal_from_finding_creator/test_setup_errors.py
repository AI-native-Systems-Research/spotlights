"""Setup-time validation surfaces explicit errors before any Claude session runs."""

from __future__ import annotations

from pathlib import Path

import pytest

from spotlights_engine.proposal_from_finding_creator import (
    ProposalFromFindingConfig,
    ProposalFromFindingSetupError,
    ProposalFromFindingValidationError,
    create_proposals,
)
from tests.unit.proposal_from_finding_creator._fakes import (
    fake_runner_factory,
    make_input,
)


@pytest.fixture
def repo(tmp_path: Path) -> Path:
    r = tmp_path / "repo"
    r.mkdir()
    return r


@pytest.fixture
def artifacts(tmp_path: Path) -> Path:
    a = tmp_path / "artifacts"
    a.mkdir()
    return a


def test_missing_repo_path_raises(artifacts: Path) -> None:
    cfg = ProposalFromFindingConfig(artifacts_dir=artifacts)
    with pytest.raises(ProposalFromFindingSetupError):
        create_proposals(
            make_input(),
            config=cfg,
            runner=fake_runner_factory(),
        )


def test_missing_artifacts_dir_raises(repo: Path) -> None:
    cfg = ProposalFromFindingConfig(repo_path=repo)
    with pytest.raises(ProposalFromFindingSetupError):
        create_proposals(
            make_input(),
            config=cfg,
            runner=fake_runner_factory(),
        )


def test_repo_path_must_exist(tmp_path: Path, artifacts: Path) -> None:
    cfg = ProposalFromFindingConfig(
        repo_path=tmp_path / "nope",
        artifacts_dir=artifacts,
    )
    with pytest.raises(ProposalFromFindingSetupError):
        create_proposals(
            make_input(),
            config=cfg,
            runner=fake_runner_factory(),
        )


def test_candidates_must_be_in_discovered_state(
    repo: Path, artifacts: Path
) -> None:
    inp = make_input()
    bad = inp.candidates.candidates[0].model_copy(
        update={"state": "FINDING_PROPOSALS_CREATED"}
    )
    inp = inp.model_copy(
        update={
            "candidates": inp.candidates.model_copy(update={"candidates": [bad]})
        }
    )
    cfg = ProposalFromFindingConfig(repo_path=repo, artifacts_dir=artifacts)
    with pytest.raises(ProposalFromFindingValidationError):
        create_proposals(inp, config=cfg, runner=fake_runner_factory())
