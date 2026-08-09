"""CLI binding for `--deep-research-mode` and the K-run consensus knobs."""

from __future__ import annotations

import pytest

from spotlights_engine import cli
from spotlights_engine.proposal_from_candidate_finding_creator import (
    ProposalFromCandidateFindingConfig,
)
from spotlights_engine.proposal_from_finding_creator import ProposalFromFindingConfig


def _parse(*argv: str):
    return cli._build_argparser().parse_args(["--repo", "/tmp/repo", *argv])


def test_deep_research_mode_defaults_to_module() -> None:
    args = _parse()

    assert args.deep_research_mode == "module"
    assert args.num_search_runs is None
    assert args.search_consensus_threshold is None
    assert cli._build_input(args).deep_research_mode == "module"


def test_deep_research_mode_candidate_binds_to_the_input() -> None:
    args = _parse("--deep-research-mode", "candidate")

    assert cli._build_input(args).deep_research_mode == "candidate"


def test_unknown_mode_is_rejected_by_argparse() -> None:
    with pytest.raises(SystemExit):
        _parse("--deep-research-mode", "hybrid")


def test_consensus_knobs_fall_back_to_the_schema_defaults() -> None:
    # Omitting the flags must not pin the values, so the schema defaults govern.
    inp = cli._build_input(_parse())
    assert inp.num_search_runs == 3
    assert inp.search_consensus_threshold is None


def test_consensus_knobs_bind_when_supplied() -> None:
    args = _parse("--num-search-runs", "5", "--search-consensus-threshold", "2")
    inp = cli._build_input(args)

    assert inp.num_search_runs == 5
    assert inp.search_consensus_threshold == 2


def test_old_max_findings_flags_are_gone() -> None:
    with pytest.raises(SystemExit):
        _parse("--max-findings-per-module", "5")
    with pytest.raises(SystemExit):
        _parse("--max-findings-per-candidate", "5")


# step-4 pair knobs route to the slot the selected mode reads -----------------


def test_module_mode_routes_pair_knobs_to_the_module_slot() -> None:
    args = _parse("--max-parallel-pairs", "3", "--debug-first-n-pairs", "2")

    config = cli._build_config(args)

    assert isinstance(config.proposal_from_finding, ProposalFromFindingConfig)
    assert config.proposal_from_finding.max_parallel_pairs == 3
    assert config.proposal_from_finding.debug_first_n_pairs == 2
    assert config.proposal_from_candidate_finding is None


def test_candidate_mode_routes_pair_knobs_to_the_candidate_slot() -> None:
    args = _parse(
        "--deep-research-mode",
        "candidate",
        "--max-parallel-pairs",
        "3",
        "--debug-first-n-pairs",
        "2",
    )

    config = cli._build_config(args)

    assert config.proposal_from_finding is None
    assert isinstance(config.proposal_from_candidate_finding, ProposalFromCandidateFindingConfig)
    assert config.proposal_from_candidate_finding.max_parallel_pairs == 3
    assert config.proposal_from_candidate_finding.debug_first_n_pairs == 2


def test_neither_slot_is_populated_without_the_pair_knobs() -> None:
    config = cli._build_config(_parse("--deep-research-mode", "candidate"))

    assert config.proposal_from_finding is None
    assert config.proposal_from_candidate_finding is None
