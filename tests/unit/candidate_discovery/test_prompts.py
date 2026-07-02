"""Unit tests for `spotlights_engine.candidate_discovery.prompts`."""

from __future__ import annotations

import re
from pathlib import Path

import pytest
from pydantic import ValidationError

from spotlights_engine.candidate_discovery import prompts
from spotlights_engine.candidate_discovery.api import DiscoveryConfig
from spotlights_engine.candidate_discovery.prompts import (
    STRICT_RETRY_REMINDER,
    _format_depends_on,
    _format_main_files,
    _format_repo_context,
    _format_spotlight_context,
    _format_submodule_names,
    _format_submodule_paths,
    _REPO_CONTEXT_DEFAULT,
    _SPOTLIGHT_CONTEXT_DEFAULT,
    _substitute,
    render_bootstrap,
    render_review,
    wrap,
)
from spotlights_engine.schemas.common import SpotlightContext
from spotlights_engine.schemas.project import File, Module

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
    assert _format_submodule_paths([]) == "(none)"
    assert _format_depends_on([]) == "(none)"


def test_format_submodule_paths_lists_paths():
    subs = [
        Module.model_validate({"name": "sub", "path": "src/engine/core/sub"}),
    ]
    assert _format_submodule_paths(subs) == "- src/engine/core/sub"


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


def test_review_renders_removed_section_default():
    m = _module()
    out = render_review("v1/engine/core", m, "{}", "cand-0001")
    assert "## Previously removed candidates" in out
    assert "_(none)_" in out
    leftovers = _PLACEHOLDER.findall(out)
    assert leftovers == [], f"leftover placeholders: {leftovers}"


def test_review_includes_removed_candidates_when_supplied():
    m = _module()
    removed = '{"module_qualified_name": "v1/engine/core", "candidates": [{"id": "cand-0004"}]}'
    out = render_review(
        "v1/engine/core",
        m,
        "{}",
        "cand-0007",
        removed_candidates_json=removed,
    )
    assert removed in out
    leftovers = _PLACEHOLDER.findall(out)
    assert leftovers == [], f"leftover placeholders: {leftovers}"


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


# ----- §7 repo_context tests -----------------------------------------------


def test_format_repo_context_default_when_none():
    assert _format_repo_context(None) == _REPO_CONTEXT_DEFAULT == "_(none provided)_"


def test_format_repo_context_passthrough_verbatim():
    md = "## Tests\n\n`pytest -q`\n"
    assert _format_repo_context(md) == md


def test_bootstrap_renders_repo_context_default_when_none():
    m = _module()
    out = render_bootstrap("v1/engine/core", m)
    assert "## Repository context" in out
    assert "_(none provided)_" in out


def test_review_renders_repo_context_default_when_none():
    m = _module()
    out = render_review("v1/engine/core", m, "{}", "cand-0001")
    assert "## Repository context" in out
    assert "_(none provided)_" in out


def test_bootstrap_repo_context_payload_round_trips_verbatim():
    m = _module()
    payload = "## Tests\n\n`pytest -q`\n"
    out = render_bootstrap("v1/engine/core", m, repo_context_markdown=payload)
    assert payload in out


def test_review_repo_context_payload_round_trips_verbatim():
    m = _module()
    payload = "## Tests\n\n`pytest -q`\n"
    out = render_review(
        "v1/engine/core",
        m,
        "{}",
        "cand-0001",
        repo_context_markdown=payload,
    )
    assert payload in out


def test_bootstrap_no_placeholders_remain_with_repo_context():
    m = _module()
    out = render_bootstrap(
        "v1/engine/core",
        m,
        repo_context_markdown="run pytest -q",
    )
    leftovers = _PLACEHOLDER.findall(out)
    assert leftovers == [], f"leftover placeholders: {leftovers}"


def test_review_no_placeholders_remain_with_repo_context():
    m = _module()
    out = render_review(
        "v1/engine/core",
        m,
        '{"module_qualified_name": "v1/engine/core", "candidates": []}',
        "cand-0007",
        repo_context_markdown="run pytest -q",
    )
    leftovers = _PLACEHOLDER.findall(out)
    assert leftovers == [], f"leftover placeholders: {leftovers}"


def test_repo_context_preserves_literal_brace_payload():
    """User-supplied `{json}`-shaped braces must round-trip — the regex
    scan runs once against the template, never against substituted values,
    so braces inside the payload cannot trigger spurious substitutions."""
    m = _module()
    payload = 'see {json_blob}: { "x": "y" }'
    boot = render_bootstrap("v1/engine/core", m, repo_context_markdown=payload)
    review = render_review(
        "v1/engine/core",
        m,
        "{}",
        "cand-0001",
        repo_context_markdown=payload,
    )
    assert payload in boot
    assert payload in review


def _base_config_kwargs(tmp_path: Path) -> dict:
    repo = tmp_path / "repo"
    repo.mkdir()
    artifacts = tmp_path / "artifacts"
    artifacts.mkdir()
    return {
        "repo_path": repo,
        "module_qualified_name": "v1/foo",
        "module": Module(
            name="foo",
            path="src/foo",
            description="x",
            main_files=[File(path="src/foo/x.py", role="entry")],
        ),
        "artifacts_dir": artifacts,
    }


def test_discovery_config_rejects_empty_repo_context(tmp_path):
    with pytest.raises(ValidationError):
        DiscoveryConfig(**_base_config_kwargs(tmp_path), repo_context_markdown="")


def test_discovery_config_rejects_oversized_repo_context(tmp_path):
    with pytest.raises(ValidationError):
        DiscoveryConfig(
            **_base_config_kwargs(tmp_path),
            repo_context_markdown="x" * 20_001,
        )


def test_discovery_config_accepts_none_repo_context(tmp_path):
    cfg = DiscoveryConfig(**_base_config_kwargs(tmp_path), repo_context_markdown=None)
    assert cfg.repo_context_markdown is None


def test_discovery_config_default_repo_context_is_none(tmp_path):
    cfg = DiscoveryConfig(**_base_config_kwargs(tmp_path))
    assert cfg.repo_context_markdown is None


# ----- Spotlight context plumbing tests ------------------------------------


def test_format_spotlight_context_default_when_none():
    assert _format_spotlight_context(None) == _SPOTLIGHT_CONTEXT_DEFAULT == "_(none provided)_"


def test_format_spotlight_context_renders_objective_and_lists():
    ctx = SpotlightContext(
        objective="reduce decode latency",
        workload_hints=["bs=1-8"],
        validation_plan=["bench tokens/sec"],
    )
    out = _format_spotlight_context(ctx)
    assert "reduce decode latency" in out
    assert "bs=1-8" in out
    assert "bench tokens/sec" in out


def test_bootstrap_includes_spotlight_context_section():
    m = _module()
    out = render_bootstrap(
        "v1/engine/core",
        m,
        spotlight_context=SpotlightContext(
            objective="reduce decode latency",
            workload_hints=["bs=1-8"],
        ),
    )
    assert "## Spotlight context" in out
    assert "reduce decode latency" in out
    assert "bs=1-8" in out


def test_review_includes_spotlight_context_section():
    m = _module()
    out = render_review(
        "v1/engine/core",
        m,
        '{"module_qualified_name": "v1/engine/core", "candidates": []}',
        "cand-0007",
        spotlight_context=SpotlightContext(objective="reduce decode latency"),
    )
    assert "## Spotlight context" in out
    assert "reduce decode latency" in out


def test_bootstrap_spotlight_context_default_when_none():
    m = _module()
    out = render_bootstrap("v1/engine/core", m)
    assert "## Spotlight context" in out
    assert _SPOTLIGHT_CONTEXT_DEFAULT in out


def test_review_spotlight_context_default_when_none():
    m = _module()
    out = render_review("v1/engine/core", m, "{}", "cand-0001")
    assert "## Spotlight context" in out
    assert _SPOTLIGHT_CONTEXT_DEFAULT in out
