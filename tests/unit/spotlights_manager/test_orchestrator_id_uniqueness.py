"""Repo-wide id uniqueness across parallel modules (decision D3).

Runs a multi-module manager end-to-end with fake agents that attach proposals,
and asserts `cand-`, `find-`, and `prop-` ids are each globally unique across
the whole run (now by slug construction — every id embeds its module's slug),
that proposal `finding_ref_id`s resolve to real findings, and that ids are
byte-identical across a resume (the module-prefix is idempotent). A separate
case covers the multi-session sub-segment grammar.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

from spotlights_engine.agent_proposals.api import AgentProposalsResult
from spotlights_engine.proposal_from_finding_creator.api import (
    ProposalFromFindingCreatorResult,
)
from spotlights_engine.schemas.candidate import Candidates
from spotlights_engine.schemas.pipeline import (
    AgentProposalsOutput,
    ProposalFromFindingCreatorOutput,
)
from spotlights_engine.schemas.proposal import Proposal
from spotlights_engine.spotlights_manager import (
    ModuleFilter,
    SpotlightsManagerConfig,
    run_with_telemetry,
)
from spotlights_engine.spotlights_manager import orchestrator as orch
from spotlights_engine.utils.schema_compat import mint_proposal_ids
from tests.unit.spotlights_manager._fakes import (
    make_discovery_result,
    make_extractor_result,
    make_input,
    make_research_output,
    make_tree,
)

_TWO_MODULES = ["v1/kv_offload", "v1/attention/paged_kv"]


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


def _patch_extractor(monkeypatch, tree) -> None:
    monkeypatch.setattr(
        orch,
        "extract_with_telemetry",
        lambda inp, *, config=None: make_extractor_result(tree),
    )


def _fake_step4(
    inp,
    *,
    config,
    runner=None,
    candidate_states=None,
    proposal_id_start=1,
    segment="seg",
):
    """Attach one research proposal per (candidate, finding) pair, minting
    module-prefixed ids from the supplied counter — mirroring the real step-4
    rebuild."""
    next_id = proposal_id_start
    rebuilt = []
    for c in inp.candidates.candidates:
        ids = mint_proposal_ids(next_id, len(inp.findings), segment=segment)
        next_id += len(inp.findings)
        proposals = list(c.proposals) + [
            Proposal(
                id=pid,
                source="research_finding",
                finding_ref_id=f.finding_id,
                author="creator",
                title="t",
                description="d",
                rationale="r",
            )
            for pid, f in zip(ids, inp.findings, strict=True)
        ]
        rebuilt.append(c.model_copy(update={"proposals": proposals}))
    output = ProposalFromFindingCreatorOutput(
        candidates=Candidates(
            module_qualified_name=inp.candidates.module_qualified_name,
            candidates=rebuilt,
        ),
        issues=[],
    )
    return ProposalFromFindingCreatorResult(
        output=output,
        per_pair_durations_s={},
        total_duration_s=0.0,
        next_proposal_id=next_id,
    )


def _fake_step5(
    inp,
    *,
    config,
    claude_runner=None,
    codex_runner=None,
    candidate_states=None,
    proposal_id_start=1,
    segment="seg",
):
    """Attach one agent proposal per candidate, continuing the module counter."""
    next_id = proposal_id_start
    rebuilt = []
    for c in inp.candidates.candidates:
        [pid] = mint_proposal_ids(next_id, 1, segment=segment)
        next_id += 1
        proposals = list(c.proposals) + [
            Proposal(
                id=pid,
                source="agent_knowledge",
                author="agent",
                title="t",
                description="d",
                rationale="r",
            )
        ]
        rebuilt.append(c.model_copy(update={"proposals": proposals}))
    output = AgentProposalsOutput(
        candidates=Candidates(
            module_qualified_name=inp.candidates.module_qualified_name,
            candidates=rebuilt,
        ),
        issues=[],
    )
    return AgentProposalsResult(
        output=output,
        per_candidate_durations_s={},
        total_duration_s=0.0,
    )


def _wire(monkeypatch, tree) -> None:
    _patch_extractor(monkeypatch, tree)
    monkeypatch.setattr(
        orch,
        "discover",
        lambda inp, *, config: make_discovery_result(
            inp.module_qualified_name, n_candidates=2
        ),
    )
    monkeypatch.setattr(
        orch,
        "research_module",
        # Mirror the real prefixing: findings are renumbered + prefixed with the
        # module segment the manager passes in.
        lambda inp, options=None, *, segment, **_kw: make_research_output(
            n_findings=2, segment=segment
        ),
    )
    monkeypatch.setattr(orch, "create_proposals_with_telemetry", _fake_step4)
    monkeypatch.setattr(orch, "create_agent_proposals_with_telemetry", _fake_step5)


def _collect_ids(result):
    cand_ids: list[str] = []
    find_ids: list[str] = []
    prop_ids: list[str] = []
    finding_refs: list[str] = []
    for mr in result.module_runs.values():
        for f in mr.findings:
            find_ids.append(f.finding_id)
        if mr.candidates is None:
            continue
        for c in mr.candidates.candidates:
            cand_ids.append(c.id)
            for p in c.proposals:
                prop_ids.append(p.id)
                if p.finding_ref_id is not None:
                    finding_refs.append(p.finding_ref_id)
    return cand_ids, find_ids, prop_ids, finding_refs


def _cfg(artifacts: Path) -> SpotlightsManagerConfig:
    return SpotlightsManagerConfig(
        artifacts_dir=artifacts,
        output_folder=artifacts.parent / "output",
        max_parallel_sessions=2,
        module_filter=ModuleFilter(include=list(_TWO_MODULES)),
    )


def test_ids_globally_unique_across_modules(
    monkeypatch, repo: Path, artifacts: Path
) -> None:
    tree = make_tree()
    _wire(monkeypatch, tree)

    result = run_with_telemetry(make_input(repo), config=_cfg(artifacts))

    assert {mr.status for mr in result.module_runs.values()} == {"SUCCEEDED"}

    cand_ids, find_ids, prop_ids, finding_refs = _collect_ids(result)

    # 2 modules x 2 candidates; 2 modules x 2 findings.
    assert len(cand_ids) == 4
    assert len(find_ids) == 4
    # Each candidate: 2 research proposals (one per finding) + 1 agent proposal.
    assert len(prop_ids) == 4 * 3

    # All three id spaces are globally unique (the SpotlightReport flattens them).
    assert len(cand_ids) == len(set(cand_ids))
    assert len(find_ids) == len(set(find_ids))
    assert len(prop_ids) == len(set(prop_ids))

    # Every finding_ref resolves to a real finding in the run (no dangling refs).
    assert set(finding_refs) <= set(find_ids)

    # All ids match the module-prefixed patterns (`<type>-<segment>-NNNN`).
    assert all(re.fullmatch(r"cand-[A-Za-z0-9_.-]+-\d{4}", x) for x in cand_ids)
    assert all(re.fullmatch(r"find-[A-Za-z0-9_.-]+-\d{4}", x) for x in find_ids)
    assert all(re.fullmatch(r"prop-[A-Za-z0-9_.-]+-\d{4}", x) for x in prop_ids)

    # Uniqueness is by slug construction: every id embeds its module's slug.
    slugs = {"v1_kv_offload", "v1_attention_paged_kv"}
    assert {x.rsplit("-", 1)[0].split("-", 1)[1] for x in cand_ids} == slugs


def test_ids_identical_across_resume(
    monkeypatch, repo: Path, artifacts: Path
) -> None:
    """Re-running with resume=True must yield byte-for-byte identical ids,
    confirming the per-module rebase is applied exactly once and is re-derived
    deterministically from `ordered_qns`."""
    tree = make_tree()
    _wire(monkeypatch, tree)

    first = run_with_telemetry(make_input(repo), config=_cfg(artifacts))
    first_ids = _collect_ids(first)

    # Resume against the same artifacts dir: every module is already terminal,
    # so the manager reloads global ids from disk and must not re-offset them.
    cfg = _cfg(artifacts).model_copy(update={"resume": True})
    second = run_with_telemetry(make_input(repo), config=cfg)
    second_ids = _collect_ids(second)

    assert first_ids == second_ids


def test_two_sessions_of_one_module_produce_disjoint_ids() -> None:
    """Multi-session model (D3): two sessions of the same module must yield
    disjoint ids — the first uses the bare slug, the second the `.s2`
    sub-segment — while each session's local counter still starts at 0001.

    No run trigger currently creates a second session, so this exercises the
    segment grammar directly (the manager always passes session 1 today)."""
    from spotlights_engine.utils.id_helpers import module_segment
    from spotlights_engine.utils.schema_compat import mint_proposal_ids

    slug = "v1_kv_offload"
    s1 = module_segment(slug, 1)
    s2 = module_segment(slug, 2)
    assert s1 == "v1_kv_offload"
    assert s2 == "v1_kv_offload.s2"

    ids_s1 = mint_proposal_ids(1, 3, segment=s1)
    ids_s2 = mint_proposal_ids(1, 3, segment=s2)

    # Each session's local counter restarts at 0001.
    assert ids_s1[0].endswith("-0001")
    assert ids_s2[0].endswith("-0001")
    # But the segments make the two sessions' ids globally disjoint.
    assert set(ids_s1).isdisjoint(ids_s2)
