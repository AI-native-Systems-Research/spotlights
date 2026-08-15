from __future__ import annotations

from spotlights_engine.modules_extractor.prompts import (
    ENRICH_SKELETON_PROMPT,
    EXTRACTION_PROMPT,
    render_assignment_prompt,
    render_assignment_shard_prompt,
    render_enrich_prompt,
    render_enrich_shard_prompt,
    render_metadata_prompt,
)


def test_extraction_prompt_forbids_conceptual_submodule_paths() -> None:
    assert "Do not split a directory into conceptual children" in EXTRACTION_PROMPT
    assert (
        "Submodule `path` must be a real nested path below its parent"
        in EXTRACTION_PROMPT
    )


def test_enrich_prompt_has_no_fold_receipt_coupling() -> None:
    """Level 1 decoupled fold evidence from `main_files`: no statement may put
    an evidence file into the target's documentation slots, while non-empty
    `evidence_files` guidance stays."""
    assert "MUST also appear" not in ENRICH_SKELETON_PROMPT
    assert "evidence file in the target's `main_files`" not in ENRICH_SKELETON_PROMPT
    assert "evidence file in the parent's `main_files`" not in ENRICH_SKELETON_PROMPT
    assert "evidence file goes in" not in ENRICH_SKELETON_PROMPT
    # ≥1 real evidence file per fold is still required…
    assert "evidence_files" in ENRICH_SKELETON_PROMPT
    assert "≥1 real source file under `path`" in ENRICH_SKELETON_PROMPT
    # …and sibling folds are explicitly free of the five-slot budget.
    assert "consume no `main_files` slots" in ENRICH_SKELETON_PROMPT


# ── Sharded enrichment scope rules ────────────────────────────────────────
#
# The SCOPE block is the only thing standing between a shard and a fragment
# the deterministic merge cannot use. Each clause below corresponds to a
# failure the pure-Python merge + repair-less Stage 5 cannot recover from.


_REPO = {
    "name": "demo",
    "summary": "A demo repo.",
    "source_root": "",
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
        repository=_REPO, subtree=_SUBTREE, scope=scope
    )


def test_shard_prompt_confines_the_model_to_its_subtree() -> None:
    prompt = _render(_scope())
    assert "Emit or fold **only** paths at or under `pkg`" in prompt
    assert "Return exactly ONE top-level module, whose `path` is `pkg`" in prompt


def test_spine_prompt_fences_off_the_promoted_children() -> None:
    """A spine cannot see the promoted subtrees, so the prompt is the only
    thing that stops it emitting/folding them or claiming a `main_file` inside
    one — the latter passes the spine's own validation and then fails
    unrecoverably after the merge."""
    prompt = _render(_scope(promoted_children=["pkg/a", "pkg/b"]))
    assert "`pkg/a`, `pkg/b`" in prompt
    assert "Do **not** emit them and do **not** fold them." in prompt
    assert "Do **not** choose any `main_files` entry that lives under them" in prompt
    assert "`pkg`'s `source_child_count` and" in prompt
    assert "count those removed subtrees" in prompt
    # The base template's hard "never exactly one child" rule would otherwise
    # contradict the Rule-4 deferral the spine's root is granted.
    assert "does **not** apply to `pkg` itself" in prompt


def test_chain_split_prompt_addresses_the_promotion_parent_not_the_root() -> None:
    """When derivation walks down a one-child chain to split, the node whose
    children were removed is *inside* the subtree, not the root. Aiming the
    Rule-4 deferral at the root instead would exempt the wrong module and let
    through the one shape the merge cannot repair."""
    prompt = _render(
        _scope(
            root_path="rust",
            promoted_children=["rust/src/chat", "rust/src/server"],
            promotion_parent="rust/src",
        )
    )
    assert "They are children of `rust/src`." in prompt
    assert "does **not** apply to `rust/src` itself" in prompt
    assert "does **not** apply to `rust` itself" not in prompt
    # The chain node above it is held to Rule 4 in full, and the promotion
    # parent itself must survive as a module for the merge to attach to.
    assert "You **must** emit `rust/src` as a module of its own" in prompt
    assert "Every directory between `rust` and `rust/src`" in prompt


def test_root_split_prompt_says_nothing_about_a_chain() -> None:
    """The common case is unchanged: promotion parent == root, no extra rules."""
    prompt = _render(_scope(promoted_children=["pkg/a", "pkg/b"]))
    assert "They are children of `pkg`." in prompt
    assert "must** emit" not in prompt
    assert "Every directory between" not in prompt


def test_child_subshard_prompt_confines_to_its_subtree() -> None:
    """A child sub-shard's top module is demoted to an `EnrichedSubmodule` at
    merge; the prompt must confine it to its own subtree."""
    prompt = _render(
        _scope(key="pkg__a", root_path="pkg/a", owns_root=False, depth=1)
    )
    assert "Do **not** emit any ancestor of `pkg/a`" in prompt


_PLACEHOLDERS = (
    "{repository_json}",
    "{skeleton_json}",
    "{scope_json}",
    "{scope_rules}",
)


def test_every_placeholder_is_substituted_in_both_renderings() -> None:
    """The template's literal JSON-shape braces mean substitution is explicit
    `str.replace`, not `str.format` — so a missed token fails silently."""
    for prompt in (
        render_enrich_prompt(repository=_REPO, skeleton=_SUBTREE),
        _render(_scope(promoted_children=["pkg/a"])),
    ):
        for token in _PLACEHOLDERS:
            assert token not in prompt, token
    assert "entire repository" in render_enrich_prompt(
        repository=_REPO, skeleton=_SUBTREE
    )


# ── Assignment-contract prompts (Stage 3A / 3B) ───────────────────────────


def _assignment_scope(**over):
    base = {
        "key": "pkg__spine",
        "root_path": "pkg",
        "is_branch_root": True,
        "is_subshard": True,
        "parent_key": "pkg",
        "depth": 1,
        "promoted_children": [],
        "promotion_parent": "pkg",
        "forced_part_paths": [],
    }
    base.update(over)
    return base


def _render_assignment(scope) -> str:
    return render_assignment_shard_prompt(
        repository=_REPO, subtree=_SUBTREE, scope=scope, merge_threshold=15
    )


_ASSIGNMENT_PLACEHOLDERS = (
    "{repository_json}",
    "{skeleton_json}",
    "{scope_json}",
    "{scope_rules}",
    "{MERGE_THRESHOLD}",
)


def test_assignment_prompt_substitutes_every_token() -> None:
    whole = render_assignment_prompt(
        repository=_REPO,
        skeleton=_SUBTREE,
        scope={"whole_repository": True, "root_path": "", "forced_part_paths": []},
        merge_threshold=15,
    )
    shard = _render_assignment(_assignment_scope())
    for prompt in (whole, shard):
        for token in _ASSIGNMENT_PLACEHOLDERS:
            assert token not in prompt, token
    assert "subtree_source_file_count > 15" in whole
    assert "entire repository" in whole


def test_assignment_prompt_threshold_is_interpolated() -> None:
    prompt = render_assignment_prompt(
        repository=_REPO,
        skeleton=_SUBTREE,
        scope={"whole_repository": True, "root_path": "", "forced_part_paths": []},
        merge_threshold=12,
    )
    assert "subtree_source_file_count > 12" in prompt


def test_assignment_substitution_is_one_pass() -> None:
    """Untrusted data containing a later placeholder must survive verbatim —
    sequential replace would expand it."""
    poisoned_repo = {**_REPO, "summary": "evil {MERGE_THRESHOLD} {scope_rules}"}
    prompt = render_assignment_prompt(
        repository=poisoned_repo,
        skeleton=_SUBTREE,
        scope={"whole_repository": True, "root_path": "", "forced_part_paths": []},
        merge_threshold=15,
    )
    assert "evil {MERGE_THRESHOLD} {scope_rules}" in prompt


def test_branch_root_shard_is_told_it_is_an_anchor() -> None:
    prompt = _render_assignment(_assignment_scope(is_branch_root=True))
    assert "structural anchor" in prompt
    assert "Label it `MODULE`." in prompt


def test_nested_shard_root_may_be_part() -> None:
    prompt = _render_assignment(
        _assignment_scope(
            key="pkg__sub", root_path="pkg/sub", is_branch_root=False
        )
    )
    assert "NOT a top-level path" in prompt
    assert "Labeling it `PART` is legal" in prompt


def test_assignment_shard_prompt_fences_promoted_children() -> None:
    prompt = _render_assignment(
        _assignment_scope(
            promoted_children=["pkg/a", "pkg/b"], promotion_parent="pkg"
        )
    )
    assert "`pkg/a`, `pkg/b`" in prompt
    assert "Do NOT label them or anything under them." in prompt
    assert "raw count is what the size rule uses" in prompt


def test_forced_part_paths_are_listed() -> None:
    prompt = _render_assignment(
        _assignment_scope(forced_part_paths=["pkg/foo_bar"])
    )
    assert "`pkg/foo_bar`" in prompt
    assert "collision precedence" in prompt


def test_metadata_prompt_interpolates_the_cap_and_scopes() -> None:
    scopes = {
        "key": "batch_00",
        "requested_modules": ["pkg/a"],
        "modules": [{"path": "pkg/a", "origin": "size"}],
    }
    prompt = render_metadata_prompt(repository=_REPO, scopes=scopes)
    assert "{MAX_MAIN_FILES}" not in prompt
    assert "{scopes_json}" not in prompt
    assert "up to 5 unique files" in prompt
    assert '"key": "batch_00"' in prompt
    # Labels are fixed; the metadata pass cannot change them.
    assert "cannot change labels" in prompt or "you cannot change labels" in prompt
