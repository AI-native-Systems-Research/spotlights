from __future__ import annotations

from spotlights_engine.modules_extractor.prompts import (
    EXTRACTION_PROMPT,
    render_enrich_prompt,
    render_enrich_shard_prompt,
)


def test_extraction_prompt_forbids_conceptual_submodule_paths() -> None:
    assert "Do not split a directory into conceptual children" in EXTRACTION_PROMPT
    assert (
        "Submodule `path` must be a real nested path below its parent"
        in EXTRACTION_PROMPT
    )


# ── Sharded enrichment scope rules ────────────────────────────────────────
#
# The SCOPE block is the only thing standing between a shard and a fragment
# the deterministic merge cannot use. Each clause below corresponds to a
# failure the pure-Python merge + repair-less Stage 5 cannot recover from.


_REPO = {
    "name": "demo",
    "summary": "A demo repo.",
    "source_root": "",
    "external_dependencies": ["torch"],
}
_SUBTREE = {
    "source_root": "",
    "nodes": [],
    "ignored": [],
    "excluded": [],
    "organizational_only": [],
    "skipped_symlinks": [],
    "inventory_fingerprint": "fp",
}


def _scope(**over):
    base = {
        "key": "pkg__spine",
        "root_path": "pkg",
        "owns_root": True,
        "is_subshard": True,
        "parent_key": "pkg",
        "depth": 1,
        "promoted_children": [],
    }
    base.update(over)
    return base


def _render(scope) -> str:
    return render_enrich_shard_prompt(
        repository=_REPO, subtree=_SUBTREE, scope=scope, top_level_qns=["pkg", "csrc"]
    )


def test_shard_prompt_confines_the_model_to_its_subtree() -> None:
    prompt = _render(_scope())
    assert "Emit or fold **only** paths at or under `pkg`" in prompt
    assert "Return exactly ONE top-level module, whose `path` is `pkg`" in prompt
    # The dependency vocabulary is global, not the shard's own slice.
    assert '"csrc"' in prompt and '"pkg"' in prompt


def test_spine_prompt_fences_off_the_promoted_children() -> None:
    """A spine cannot see the promoted subtrees, so the prompt is the only
    thing that stops it emitting/folding them or claiming a `main_file` inside
    one — the latter passes the spine's own validation and then fails
    unrecoverably after the merge."""
    prompt = _render(_scope(promoted_children=["pkg/a", "pkg/b"]))
    assert "`pkg/a`, `pkg/b`" in prompt
    assert "Do **not** emit them and do **not** fold them." in prompt
    assert "Do **not** choose any `main_files` entry that lives under them" in prompt
    assert "`source_child_count` counts those removed children" in prompt
    # The base template's hard "never exactly one child" rule would otherwise
    # contradict the Rule-4 deferral the spine's root is granted.
    assert "does **not** apply to `pkg` itself" in prompt


def test_child_subshard_prompt_forbids_depends_on() -> None:
    """A child sub-shard's top module is demoted to an `EnrichedSubmodule` at
    merge, and that model has no `depends_on` field at all."""
    prompt = _render(
        _scope(key="pkg__a", root_path="pkg/a", owns_root=False, depth=1)
    )
    assert "Do **not** declare `depends_on`." in prompt
    assert "Do **not** emit any ancestor of `pkg/a`" in prompt


def test_depth_two_spine_also_forbids_depends_on() -> None:
    """`owns_root` alone is not the predicate: a depth-2 spine emits a module
    that becomes a submodule after the merge."""
    prompt = _render(
        _scope(key="pkg__a__spine", root_path="pkg/a", depth=2,
               promoted_children=["pkg/a/g1", "pkg/a/g2"])
    )
    assert "Do **not** declare `depends_on`." in prompt


def test_branch_spine_may_still_declare_depends_on() -> None:
    prompt = _render(_scope(promoted_children=["pkg/a", "pkg/b"]))
    assert "Do **not** declare `depends_on`." not in prompt


_PLACEHOLDERS = (
    "{repository_json}",
    "{skeleton_json}",
    "{scope_json}",
    "{scope_rules}",
    "{top_level_qns_json}",
)


def test_every_placeholder_is_substituted_in_both_renderings() -> None:
    """The template's literal JSON-shape braces mean substitution is explicit
    `str.replace`, not `str.format` — so a missed token fails silently."""
    for prompt in (
        render_enrich_prompt(
            repository=_REPO, skeleton=_SUBTREE, top_level_qns=["pkg"]
        ),
        _render(_scope(promoted_children=["pkg/a"])),
    ):
        for token in _PLACEHOLDERS:
            assert token not in prompt, token
    assert "entire repository" in render_enrich_prompt(
        repository=_REPO, skeleton=_SUBTREE, top_level_qns=["pkg"]
    )
