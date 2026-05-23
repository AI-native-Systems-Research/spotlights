"""Persistence layer: round-trip, atomic writes, crash recovery."""

from __future__ import annotations

import json
from pathlib import Path

from spotlights_engine.agent_proposals import AgentProposalsConfig
from spotlights_engine.candidate_discovery.api import DiscoveryConfig
from spotlights_engine.module_deep_research.codex_exec import CodexExecOptions
from spotlights_engine.modules_extractor import ExtractorConfig
from spotlights_engine.proposal_from_finding_creator import (
    ProposalFromFindingConfig,
)
from spotlights_engine.spotlights_manager import persistence as P
from spotlights_engine.spotlights_manager.persistence import (
    ManagerPaths,
    ModuleCheckpoint,
)
from tests.unit.spotlights_manager._fakes import (
    make_candidates,
    make_research_output,
)


def test_module_checkpoint_round_trip(tmp_path: Path) -> None:
    paths = ManagerPaths(tmp_path)
    paths.modules_root.mkdir(parents=True)
    mp = paths.for_module("v1.kv_offload")
    cp = ModuleCheckpoint(
        module_qualified_name="v1.kv_offload",
        status="DISCOVERED",
        last_step="candidate_discovery",
        started_at="2026-05-21T00:00:00+00:00",
        updated_at="2026-05-21T00:00:00+00:00",
    )
    P.write_checkpoint(mp, cp)
    state = P.read_module_state(mp)
    assert state.checkpoint == cp


def test_atomic_write_recovers_from_orphan_tmp(tmp_path: Path) -> None:
    """A leftover `.tmp` from a previous crashed write must not block the next write."""
    paths = ManagerPaths(tmp_path)
    mp = paths.for_module("v1.kv_offload")
    mp.dir.mkdir(parents=True)

    # Plant a leftover .tmp file (sim'd crash mid-write).
    (mp.status_path.with_suffix(mp.status_path.suffix + ".tmp")).write_text(
        "garbage", encoding="utf-8"
    )

    cp = ModuleCheckpoint(
        module_qualified_name="v1.kv_offload",
        status="PENDING",
        last_step=None,
        started_at="2026-05-21T00:00:00+00:00",
        updated_at="2026-05-21T00:00:00+00:00",
    )
    P.write_checkpoint(mp, cp)
    payload = json.loads(mp.status_path.read_text(encoding="utf-8"))
    assert payload["status"] == "PENDING"


def test_candidates_and_deep_research_round_trip(tmp_path: Path) -> None:
    paths = ManagerPaths(tmp_path)
    mp = paths.for_module("v1.kv_offload")
    mp.dir.mkdir(parents=True)

    cands = make_candidates("v1.kv_offload", n=2)
    P.write_candidates(mp, cands)

    output = make_research_output(n_findings=3)
    P.write_deep_research(mp, output, duration_s=4.2)

    state = P.read_module_state(mp)
    assert state.candidates is not None
    assert len(state.candidates.candidates) == 2
    assert state.deep_research is not None
    assert len(state.deep_research.findings) == 3
    assert state.deep_research_duration_s == 4.2


def test_proposal_from_finding_round_trip(tmp_path: Path) -> None:
    from spotlights_engine.schemas.candidate import Candidates
    from spotlights_engine.schemas.pipeline import (
        ProposalFromFindingCreatorOutput,
    )

    paths = ManagerPaths(tmp_path)
    mp = paths.for_module("v1.kv_offload")
    mp.dir.mkdir(parents=True)

    cands = make_candidates("v1.kv_offload", n=2)
    advanced = Candidates(
        module_qualified_name=cands.module_qualified_name,
        candidates=[
            c.model_copy(update={"state": "FINDING_PROPOSALS_CREATED"})
            for c in cands.candidates
        ],
    )
    output = ProposalFromFindingCreatorOutput(candidates=advanced, issues=[])
    P.write_proposal_from_finding(
        mp,
        output,
        duration_s=2.5,
        per_pair_durations_s={"cand-0001__find-0001": 1.0},
    )

    state = P.read_module_state(mp)
    assert state.proposal_from_finding is not None
    assert len(state.proposal_from_finding.candidates.candidates) == 2
    assert state.proposal_from_finding_duration_s == 2.5
    assert state.proposal_from_finding_per_pair_durations_s == {
        "cand-0001__find-0001": 1.0
    }


def test_clear_proposal_from_finding_artifacts_removes_sidecar_and_dir(
    tmp_path: Path,
) -> None:
    from spotlights_engine.schemas.candidate import Candidates
    from spotlights_engine.schemas.pipeline import (
        ProposalFromFindingCreatorOutput,
    )

    paths = ManagerPaths(tmp_path)
    mp = paths.for_module("v1.kv_offload")
    mp.dir.mkdir(parents=True)

    cands = make_candidates("v1.kv_offload")
    advanced = Candidates(
        module_qualified_name=cands.module_qualified_name,
        candidates=[
            c.model_copy(update={"state": "FINDING_PROPOSALS_CREATED"})
            for c in cands.candidates
        ],
    )
    P.write_proposal_from_finding(
        mp,
        ProposalFromFindingCreatorOutput(candidates=advanced, issues=[]),
        duration_s=0.0,
        per_pair_durations_s={},
    )
    mp.proposal_from_finding_last_message_dir.mkdir()
    (mp.proposal_from_finding_last_message_dir / "cand-0001__find-0001.json").write_text(
        "{}", encoding="utf-8"
    )

    P.clear_proposal_from_finding_artifacts(mp)
    assert not mp.proposal_from_finding_path.exists()
    assert not mp.proposal_from_finding_last_message_dir.exists()


def test_clear_helpers_remove_artifacts(tmp_path: Path) -> None:
    paths = ManagerPaths(tmp_path)
    mp = paths.for_module("v1.kv_offload")
    mp.dir.mkdir(parents=True)
    mp.discovery_run_dir.mkdir()
    (mp.discovery_run_dir / "marker").write_text("x")
    P.write_candidates(mp, make_candidates("v1.kv_offload"))
    P.write_deep_research(mp, make_research_output(), duration_s=1.0)
    mp.deep_research_last_message_path.write_text("hello", encoding="utf-8")

    P.clear_discovery_artifacts(mp)
    assert not mp.discovery_run_dir.exists()
    assert not mp.candidates_path.exists()

    P.clear_deep_research_artifacts(mp)
    assert not mp.deep_research_path.exists()
    assert not mp.deep_research_last_message_path.exists()


def test_init_manifest_creates_tree(tmp_path: Path) -> None:
    paths = ManagerPaths(tmp_path)
    manifest = P.init_manifest(
        paths,
        input_fingerprint={"a": 1},
        config_fingerprint={"b": 2},
    )
    assert paths.manifest_path.exists()
    assert paths.modules_root.exists()
    assert manifest["status"] == "RUNNING"
    re_read = P.read_manifest(paths)
    assert re_read is not None
    assert re_read["input_fingerprint"] == {"a": 1}


def test_slug_for_replaces_unsafe_chars() -> None:
    assert P.slug_for("v1.kv_offload") == "v1.kv_offload"
    assert P.slug_for("v1/kv_offload") == "v1_kv_offload"
    assert P.slug_for("a.b c") == "a.b_c"


def test_config_fingerprint_treats_none_as_effective_defaults() -> None:
    base = P.build_config_fingerprint(
        module_filter=None,
        extractor_cfg=ExtractorConfig(),
        discovery_cfg=None,
        deep_research_cfg=None,
        proposal_from_finding_cfg=None,
        agent_proposals_cfg=None,
    )
    explicit = P.build_config_fingerprint(
        module_filter=None,
        extractor_cfg=ExtractorConfig(),
        discovery_cfg=DiscoveryConfig(),
        deep_research_cfg=CodexExecOptions(),
        proposal_from_finding_cfg=ProposalFromFindingConfig(),
        agent_proposals_cfg=AgentProposalsConfig(),
    )
    assert base == explicit
