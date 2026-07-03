"""CLI coverage for the arXiv-search runner flags."""

from __future__ import annotations

from spotlights_engine import cli


def _parse(argv: list[str]):
    return cli._build_argparser().parse_args(argv)


def test_arxiv_search_enabled_by_default() -> None:
    cfg = cli._build_config(_parse([]))
    assert cfg.arxiv_search is not None
    assert cfg.arxiv_search.query_planner == "claude"
    assert cfg.arxiv_search.mailto is None


def test_arxiv_mailto_flows_through() -> None:
    cfg = cli._build_config(_parse(["--arxiv-mailto", "me@example.com"]))
    assert cfg.arxiv_search is not None
    assert cfg.arxiv_search.mailto == "me@example.com"
    assert cfg.arxiv_search.query_planner == "claude"


def test_arxiv_query_planner_template_flag() -> None:
    cfg = cli._build_config(_parse(["--arxiv-query-planner", "template"]))
    assert cfg.arxiv_search is not None
    assert cfg.arxiv_search.query_planner == "template"


def test_no_arxiv_search_disables_runner() -> None:
    cfg = cli._build_config(_parse(["--no-arxiv-search"]))
    assert cfg.arxiv_search is None
