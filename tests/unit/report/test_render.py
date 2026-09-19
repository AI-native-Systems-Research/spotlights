from __future__ import annotations

import re

from spotlights_engine.report.render import load, render

from ._fixtures import write_run


def test_page_is_self_contained(tmp_path):
    """Nothing to fetch: the page has to open from a file:// URL offline."""
    page = render(load(write_run(tmp_path)))

    assert page.startswith("<!DOCTYPE html>")
    assert "<style>" in page
    assert "<script>" in page
    assert re.search(r"<script[^>]*\ssrc=", page) is None
    assert re.search(r"<link[^>]*stylesheet", page) is None
    assert re.search(r"<img[^>]*\ssrc=", page) is None


def test_candidate_text_is_escaped(tmp_path):
    """A candidate description is agent-authored text, never trusted markup."""
    page = render(load(write_run(tmp_path)))

    assert "&lt;script&gt;alert(1)&lt;/script&gt;" in page
    assert "<script>alert(1)" not in page


def test_renders_without_a_ranking_overlay(tmp_path):
    """Rendering must not depend on sorted/ having been produced."""
    page = render(load(write_run(tmp_path)))

    assert "run-min-0001" in page
    assert "cut median TTFT" in page
    assert len(page) > 10_000


def test_ranking_overlay_reaches_the_page(tmp_path):
    page = render(load(write_run(tmp_path, ranked=True)))

    assert "0.91" in page
    assert "weighted-impact-v1" in page
