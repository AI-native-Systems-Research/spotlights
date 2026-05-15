"""Unit tests for `spotlights_engine.candidate_discovery.prompts`."""

from __future__ import annotations

import re

import pytest

from spotlights_engine.candidate_discovery import prompts
from spotlights_engine.candidate_discovery.prompts import (
    STRICT_RETRY_REMINDER,
    _format_depends_on,
    _format_main_files,
    _format_submodule_names,
    _substitute,
    render_bootstrap,
    render_review,
    wrap,
)
from spotlights_engine.schemas.modules import File, Module

_PLACEHOLDER = re.compile(r"\{(\w+)\}")


def _module(**overrides) -> Module:
    payload = {
        "name": "core",
        "path": "src/engine/core",
        "description": "Hot path scheduling.",
        "depends_on": ["v1/engine/scheduler"],
        "main_files": [File(path="src/engine/core/sched.py", role="entrypoint")],
        "submodules": [],
    }
    payload.update(overrides)
    return Module.model_validate(payload)


def test_strict_retry_reminder_loads():
    assert STRICT_RETRY_REMINDER
    assert "JSON" in STRICT_RETRY_REMINDER


def test_format_helpers_none_fallbacks():
    assert _format_main_files([]) == "(none)"
    assert _format_submodule_names([]) == "(none)"
    assert _format_depends_on([]) == "(none)"


def test_format_main_files_uses_role():
    files = [File(path="a.py", role="entry"), File(path="b.py", role="hot_loop")]
    out = _format_main_files(files)
    assert out == "- a.py (entry)\n- b.py (hot_loop)"


def test_substitute_rejects_unknown_placeholder():
    with pytest.raises(KeyError, match="unknown placeholder"):
        _substitute("hello {who}", {})


def test_substitute_leaves_json_braces_alone():
    template = '{\n  "key": "{name}"\n}'
    rendered = _substitute(template, {"name": "value"})
    assert rendered == '{\n  "key": "value"\n}'


def test_bootstrap_renders_with_no_submodules_or_deps():
    m = _module(depends_on=[], submodules=[])
    out = render_bootstrap("v1/engine/core", m)
    assert "(none)" in out
    # Two `(none)` substitutions are expected: depends_on and submodule_names.
    assert out.count("(none)") >= 2


def test_bootstrap_no_unsubstituted_placeholders_remain():
    m = _module()
    out = render_bootstrap("v1/engine/core", m)
    leftovers = _PLACEHOLDER.findall(out)
    assert leftovers == [], f"leftover placeholders: {leftovers}"


def test_review_no_unsubstituted_placeholders_remain():
    m = _module()
    prev_json = '{"module_qualified_name": "v1/engine/core", "candidates": []}'
    out = render_review("v1/engine/core", m, prev_json, "cand-0007")
    leftovers = _PLACEHOLDER.findall(out)
    assert leftovers == [], f"leftover placeholders: {leftovers}"


def test_review_includes_prev_json_and_max_seen_id():
    m = _module()
    prev_json = '{"module_qualified_name": "v1/engine/core", "candidates": []}'
    out = render_review("v1/engine/core", m, prev_json, "cand-0042")
    assert prev_json in out
    assert "cand-0042" in out


def test_review_preserves_literal_json_braces():
    m = _module()
    out = render_review("v1/engine/core", m, "{}", "cand-0001")
    # The literal JSON-object example at the bottom of the review template
    # must survive substitution intact.
    assert '"candidates": [ ... ]' in out


def test_wrap_prefixes_preamble():
    body = "<body>"
    out = wrap(body)
    assert out.endswith("<body>")
    assert out.startswith(prompts._PREAMBLE)
    assert out.startswith("<preamble>")
    assert "</preamble>\n\n<body>" in out


def test_substitute_rejects_unknown_placeholder_in_template_fixture():
    template = "see {whatever}"
    with pytest.raises(KeyError):
        _substitute(template, {"module_path": "x"})
