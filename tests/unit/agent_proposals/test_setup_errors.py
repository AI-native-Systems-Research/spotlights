"""Setup/validation error paths for `create_agent_proposals_with_telemetry`."""

from __future__ import annotations

from pathlib import Path

import pytest

from spotlights_engine.agent_proposals import (
    AgentProposalsConfig,
    AgentProposalsSetupError,
    AgentProposalsValidationError,
    create_agent_proposals_with_telemetry,
)
from spotlights_engine.schemas.candidate import Candidates
from spotlights_engine.schemas.common import SpotlightContext
from spotlights_engine.schemas.pipeline import AgentProposalsInput
from tests.unit.agent_proposals._fakes import (
    fake_claude_runner_factory,
    fake_codex_runner_factory,
    make_candidate,
    make_input,
    make_project_tree,
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


def test_missing_repo_path_raises_setup_error(artifacts: Path) -> None:
    inp = make_input(n_candidates=1)
    cfg = AgentProposalsConfig(repo_path=None, artifacts_dir=artifacts)
    with pytest.raises(AgentProposalsSetupError):
        create_agent_proposals_with_telemetry(
            inp,
            config=cfg,
            claude_runner=fake_claude_runner_factory(),
            codex_runner=fake_codex_runner_factory(),
        )


def test_missing_artifacts_dir_raises_setup_error(repo: Path) -> None:
    inp = make_input(n_candidates=1)
    cfg = AgentProposalsConfig(repo_path=repo, artifacts_dir=None)
    with pytest.raises(AgentProposalsSetupError):
        create_agent_proposals_with_telemetry(
            inp,
            config=cfg,
            claude_runner=fake_claude_runner_factory(),
            codex_runner=fake_codex_runner_factory(),
        )


def test_repo_path_missing_on_disk_raises_setup_error(
    tmp_path: Path, artifacts: Path
) -> None:
    inp = make_input(n_candidates=1)
    missing = tmp_path / "does-not-exist"
    cfg = AgentProposalsConfig(repo_path=missing, artifacts_dir=artifacts)
    with pytest.raises(AgentProposalsSetupError):
        create_agent_proposals_with_telemetry(
            inp,
            config=cfg,
            claude_runner=fake_claude_runner_factory(),
            codex_runner=fake_codex_runner_factory(),
        )


def test_candidate_in_wrong_state_raises_validation_error(
    repo: Path, artifacts: Path
) -> None:
    bad = make_candidate(0).model_copy(update={"state": "DISCOVERED"})
    inp = AgentProposalsInput(
        project_tree=make_project_tree(),
        candidates=Candidates(
            module_qualified_name="v1/kv_offload",
            candidates=[bad],
        ),
        context=SpotlightContext(objective="reduce latency"),
    )
    cfg = AgentProposalsConfig(repo_path=repo, artifacts_dir=artifacts)
    with pytest.raises(AgentProposalsValidationError):
        create_agent_proposals_with_telemetry(
            inp,
            config=cfg,
            claude_runner=fake_claude_runner_factory(),
            codex_runner=fake_codex_runner_factory(),
        )
