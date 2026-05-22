"""`debug_first_n_pairs` truncates pair scheduling but preserves all candidates."""

from __future__ import annotations

import logging
from pathlib import Path

import pytest

from spotlights_engine.proposal_from_finding_creator import (
    ProposalFromFindingConfig,
    create_proposals,
)
from tests.unit.proposal_from_finding_creator._fakes import (
    fake_runner_factory,
    make_input,
    make_proposal_payload,
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


def test_debug_first_n_pairs_caps_runner_invocations(
    repo: Path, artifacts: Path, caplog
) -> None:
    inp = make_input(n_candidates=2, n_findings=2)  # 4 pairs total
    invoked: list[str] = []

    def _runner(
        *, pair_key, prompt, schema_text, repo_path, max_turns, wallclock_s
    ):
        invoked.append(pair_key)
        finding_id = pair_key.split("__")[1]
        from spotlights_engine.proposal_from_finding_creator.claude_exec import (
            PairRunResult,
        )

        return PairRunResult(
            pair_key=pair_key,
            duration_s=0.0,
            structured_output=make_proposal_payload(finding_id=finding_id),
        )

    cfg = ProposalFromFindingConfig(
        repo_path=repo,
        artifacts_dir=artifacts,
        debug_first_n_pairs=1,
    )
    with caplog.at_level(logging.WARNING):
        out = create_proposals(inp, config=cfg, runner=_runner)
    assert len(invoked) == 1
    # All input candidates remain in the output, in input order.
    ids = [c.id for c in out.candidates.candidates]
    assert ids == ["cand-0001", "cand-0002"]
    # Only the first scheduled pair's candidate gets a proposal; the rest get [].
    proposals_by_id = {c.id: c.deep_research_proposals for c in out.candidates.candidates}
    assert len(proposals_by_id["cand-0001"]) == 1
    assert proposals_by_id["cand-0002"] == []
    # Truncation logs a warning so the user knows debug mode is active.
    assert any("debug_first_n_pairs" in r.message for r in caplog.records)


def test_debug_first_n_pairs_above_total_runs_all_pairs(
    repo: Path, artifacts: Path
) -> None:
    inp = make_input(n_candidates=1, n_findings=2)  # 2 pairs
    invoked: list[str] = []

    def _runner(
        *, pair_key, prompt, schema_text, repo_path, max_turns, wallclock_s
    ):
        invoked.append(pair_key)
        finding_id = pair_key.split("__")[1]
        from spotlights_engine.proposal_from_finding_creator.claude_exec import (
            PairRunResult,
        )

        return PairRunResult(
            pair_key=pair_key,
            duration_s=0.0,
            structured_output=make_proposal_payload(finding_id=finding_id),
        )

    cfg = ProposalFromFindingConfig(
        repo_path=repo,
        artifacts_dir=artifacts,
        debug_first_n_pairs=10,
    )
    create_proposals(inp, config=cfg, runner=_runner)
    assert len(invoked) == 2
