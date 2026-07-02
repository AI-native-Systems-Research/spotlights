"""CLI coverage for the `--paper-link` / `--paper-title` step-3 filter args."""

from __future__ import annotations

import pytest

from spotlights_engine import cli


def _parse(argv: list[str]):
    return cli._build_argparser().parse_args(argv)


def test_paper_link_builds_paper_filter_on_input() -> None:
    args = _parse(["--paper-link", "https://arxiv.org/abs/2504.19874"])
    inp = cli._build_input(args)

    assert inp.paper_filter is not None
    assert inp.paper_filter.url == "https://arxiv.org/abs/2504.19874"
    assert inp.paper_filter.title is None


def test_paper_link_with_title() -> None:
    args = _parse(
        ["--paper-link", "https://arxiv.org/abs/2504.19874", "--paper-title", "Preble"]
    )
    inp = cli._build_input(args)

    assert inp.paper_filter is not None
    assert inp.paper_filter.title == "Preble"


def test_no_paper_link_leaves_filter_unset() -> None:
    inp = cli._build_input(_parse([]))

    assert inp.paper_filter is None


def test_paper_title_without_link_is_usage_error() -> None:
    with pytest.raises(SystemExit) as exc:
        cli.main(["--paper-title", "Preble", "--repo", "/tmp/x"])

    assert exc.value.code == 2
