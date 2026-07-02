"""CLI coverage for the `--no-agent-proposals` step-5 disable flag."""

from __future__ import annotations

from spotlights_engine import cli


def _parse(argv: list[str]):
    return cli._build_argparser().parse_args(argv)


def test_agent_proposals_enabled_by_default() -> None:
    args = _parse([])
    cfg = cli._build_config(args)

    assert args.agent_proposals_enabled is True
    assert cfg.skip_agent_proposals is False


def test_no_agent_proposals_sets_skip_on_config() -> None:
    args = _parse(["--no-agent-proposals"])
    cfg = cli._build_config(args)

    assert args.agent_proposals_enabled is False
    assert cfg.skip_agent_proposals is True
