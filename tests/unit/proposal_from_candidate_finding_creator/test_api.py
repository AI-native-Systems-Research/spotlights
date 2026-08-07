"""Candidate-mode step 4: grouping by `candidate_id` and the new fields."""

from __future__ import annotations

import logging
from pathlib import Path

import pytest

from spotlights_engine.proposal_from_candidate_finding_creator import (
    ProposalFromCandidateFindingConfig,
    create_proposals,
    create_proposals_with_telemetry,
)
from spotlights_engine.proposal_from_candidate_finding_creator.api import (
    _build_pair_keys,
)
from spotlights_engine.utils.schema_compat import proposals_from
from tests.unit.proposal_from_candidate_finding_creator._fakes import (
    CREATED_BY,
    candidate_id,
    fake_runner_factory,
    make_finding,
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


def _cfg(repo: Path, artifacts: Path, **kwargs) -> ProposalFromCandidateFindingConfig:
    return ProposalFromCandidateFindingConfig(repo_path=repo, artifacts_dir=artifacts, **kwargs)


# grouping --------------------------------------------------------------------


def test_pairs_are_grouped_not_cartesian(repo: Path, artifacts: Path) -> None:
    # 2 candidates x 2 findings each: module mode would run 8 pairs, this runs 4.
    findings = [make_finding(c, f) for c in range(2) for f in range(2)]
    inp = make_input(n_candidates=2, findings=findings)
    invoked: list[str] = []

    create_proposals(
        inp,
        config=_cfg(repo, artifacts),
        runner=fake_runner_factory(invoked=invoked, default_payload=True),
    )

    assert sorted(invoked) == [
        f"{candidate_id(0)}__find-v1_kv_offload-0001-0001",
        f"{candidate_id(0)}__find-v1_kv_offload-0001-0002",
        f"{candidate_id(1)}__find-v1_kv_offload-0002-0001",
        f"{candidate_id(1)}__find-v1_kv_offload-0002-0002",
    ]


def test_candidate_with_no_findings_yields_no_proposals(repo: Path, artifacts: Path) -> None:
    inp = make_input(n_candidates=2, findings=[make_finding(0)])

    out = create_proposals(
        inp,
        config=_cfg(repo, artifacts),
        runner=fake_runner_factory(default_payload=True),
    )

    by_id = {c.id: proposals_from(c, "research_finding") for c in out.candidates.candidates}
    assert len(by_id[candidate_id(0)]) == 1
    assert by_id[candidate_id(1)] == []


def test_untagged_finding_is_dropped_with_a_warning() -> None:
    inp = make_input(n_candidates=1)
    untagged = make_finding(0, 1).model_copy(update={"candidate_id": None})

    pairs, issues = _build_pair_keys(list(inp.candidates.candidates), [*inp.findings, untagged])

    assert len(pairs) == 1
    assert len(issues) == 1
    assert issues[0].severity == "warning"
    assert issues[0].step == "proposal_from_finding_creator"
    assert "carries no candidate_id" in issues[0].message


def test_finding_referencing_a_foreign_candidate_is_dropped_with_a_warning() -> None:
    inp = make_input(n_candidates=1)
    foreign = make_finding(0, 1, candidate_id_override="cand-other_module-0001")

    pairs, issues = _build_pair_keys(list(inp.candidates.candidates), [*inp.findings, foreign])

    assert len(pairs) == 1
    assert len(issues) == 1
    assert issues[0].severity == "warning"
    assert "not in this step's input" in issues[0].message


def test_a_duplicate_candidate_id_is_paired_once_not_twice(repo: Path, artifacts: Path) -> None:
    # A repeated candidate id produces the same `<cand>__<find>` pair key twice,
    # which would run the pair twice and attach two proposal sets to one code
    # site (and burn double the Claude budget).
    inp = make_input(n_candidates=1)
    duplicated = inp.model_copy(
        update={
            "candidates": inp.candidates.model_copy(
                update={
                    "candidates": [
                        *inp.candidates.candidates,
                        inp.candidates.candidates[0].model_copy(deep=True),
                    ]
                }
            )
        }
    )
    invoked: list[str] = []

    out = create_proposals(
        duplicated,
        config=_cfg(repo, artifacts),
        runner=fake_runner_factory(invoked=invoked, default_payload=True),
    )

    assert invoked == [f"{candidate_id(0)}__find-v1_kv_offload-0001-0001"]
    assert any("duplicate candidate id" in i.message for i in out.issues)


def test_grouping_issues_reach_the_step_output(repo: Path, artifacts: Path) -> None:
    untagged = make_finding(0, 1).model_copy(update={"candidate_id": None})
    inp = make_input(n_candidates=1, findings=[make_finding(0), untagged])

    out = create_proposals(
        inp,
        config=_cfg(repo, artifacts),
        runner=fake_runner_factory(default_payload=True),
    )

    assert any("carries no candidate_id" in i.message for i in out.issues)


def test_zero_surviving_pairs_skips_the_claude_probe(
    repo: Path, artifacts: Path, monkeypatch
) -> None:
    # Non-empty findings are not enough: all of them may be foreign, and then no
    # CLI is ever spawned so `ensure_claude_available` must not be called.
    def _boom() -> None:
        raise AssertionError("ensure_claude_available must not be called")

    monkeypatch.setattr(
        "spotlights_engine.proposal_from_candidate_finding_creator.api.ensure_claude_available",
        _boom,
    )
    inp = make_input(
        n_candidates=1,
        findings=[make_finding(0, candidate_id_override="cand-other_module-0001")],
    )

    result = create_proposals_with_telemetry(inp, config=_cfg(repo, artifacts))

    assert result.per_pair_durations_s == {}
    assert proposals_from(result.output.candidates.candidates[0], "research_finding") == []


# the four new fields ---------------------------------------------------------


def test_new_structured_fields_flow_into_the_unified_proposal(repo: Path, artifacts: Path) -> None:
    inp = make_input(n_candidates=1)
    pair_key = f"{candidate_id(0)}__find-v1_kv_offload-0001-0001"

    out = create_proposals(
        inp,
        config=_cfg(repo, artifacts),
        runner=fake_runner_factory(
            payloads={pair_key: make_proposal_payload(finding_id="find-v1_kv_offload-0001-0001")}
        ),
    )

    p = proposals_from(out.candidates.candidates[0], "research_finding")[0]
    assert p.mechanism == "Swap the loop for a pool"
    assert p.required_changes == "Rewrite core.py hot_1"
    assert p.expected_effect == "2x on the hot path"
    assert p.evaluation_metric == "p99 latency"
    assert p.author == CREATED_BY
    assert p.id == "prop-v1_kv_offload-0001"


def test_created_by_default_names_the_candidate_mode_step() -> None:
    assert (
        ProposalFromCandidateFindingConfig().created_by == "proposal_from_candidate_finding_creator"
    )


# debug truncation ------------------------------------------------------------


def test_debug_first_n_pairs_reports_the_grouped_pair_count(
    repo: Path, artifacts: Path, caplog
) -> None:
    # 2 candidates x 2 findings each => 4 grouped pairs (cartesian would be 8).
    findings = [make_finding(c, f) for c in range(2) for f in range(2)]
    inp = make_input(n_candidates=2, findings=findings)
    invoked: list[str] = []

    with caplog.at_level(logging.WARNING):
        create_proposals(
            inp,
            config=_cfg(repo, artifacts, debug_first_n_pairs=1),
            runner=fake_runner_factory(invoked=invoked, default_payload=True),
        )

    assert len(invoked) == 1
    messages = [r.getMessage() for r in caplog.records]
    assert any("running 1 of 4 pairs" in m for m in messages)


# artifacts -------------------------------------------------------------------


def test_debug_payloads_land_in_the_package_specific_directory(repo: Path, artifacts: Path) -> None:
    inp = make_input(n_candidates=1)

    create_proposals(
        inp,
        config=_cfg(repo, artifacts),
        runner=fake_runner_factory(default_payload=True),
    )

    out_dir = artifacts / "proposal_from_candidate_finding_creator.last_messages"
    assert out_dir.is_dir()
    assert [p.name for p in out_dir.iterdir()] == [
        f"{candidate_id(0)}__find-v1_kv_offload-0001-0001.json"
    ]
