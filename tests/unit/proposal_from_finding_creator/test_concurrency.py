"""`max_parallel_pairs` bounds simultaneous Claude sessions."""

from __future__ import annotations

import threading
import time
from pathlib import Path

import pytest

from spotlights_engine.proposal_from_finding_creator import (
    ProposalFromFindingConfig,
    create_proposals,
)
from spotlights_engine.proposal_from_finding_creator.claude_exec import (
    PairRunResult,
)
from tests.unit.proposal_from_finding_creator._fakes import (
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


def test_max_parallel_pairs_caps_in_flight_runners(
    repo: Path, artifacts: Path
) -> None:
    inp = make_input(n_candidates=3, n_findings=2)  # 6 pairs

    in_flight = 0
    peak = 0
    lock = threading.Lock()

    def _runner(
        *, pair_key, prompt, schema_text, repo_path, max_turns, wallclock_s
    ) -> PairRunResult:
        nonlocal in_flight, peak
        with lock:
            in_flight += 1
            peak = max(peak, in_flight)
        time.sleep(0.05)
        with lock:
            in_flight -= 1
        finding_id = pair_key.split("__")[1]
        return PairRunResult(
            pair_key=pair_key,
            duration_s=0.05,
            structured_output=make_proposal_payload(finding_id=finding_id),
        )

    cfg = ProposalFromFindingConfig(
        repo_path=repo,
        artifacts_dir=artifacts,
        max_parallel_pairs=2,
    )
    out = create_proposals(inp, config=cfg, runner=_runner)
    assert len(out.candidates.candidates) == 3
    assert peak <= 2
    assert peak >= 1
