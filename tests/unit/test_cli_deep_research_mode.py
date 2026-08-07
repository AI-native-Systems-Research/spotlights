"""CLI binding for `--deep-research-mode` / `--max-findings-per-candidate`."""

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
    assert args.max_findings_per_candidate is None
    assert cli._build_input(args).deep_research_mode == "module"


def test_deep_research_mode_candidate_binds_to_the_input() -> None:
    args = _parse("--deep-research-mode", "candidate")

    assert cli._build_input(args).deep_research_mode == "candidate"


def test_unknown_mode_is_rejected_by_argparse() -> None:
    with pytest.raises(SystemExit):
        _parse("--deep-research-mode", "hybrid")


def test_max_findings_per_candidate_falls_back_to_the_schema_default() -> None:
    # Omitting the flag must not pin the value, so the schema default governs.
    assert cli._build_input(_parse()).max_findings_per_candidate == 10


def test_max_findings_per_candidate_binds_when_supplied() -> None:
    args = _parse("--deep-research-mode", "candidate", "--max-findings-per-candidate", "4")

    assert cli._build_input(args).max_findings_per_candidate == 4


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
