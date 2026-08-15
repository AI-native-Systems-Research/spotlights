from __future__ import annotations

from spotlights_engine.modules_extractor.prompts import (
    EXTRACTION_PROMPT,
    render_assignment_prompt,
    render_assignment_shard_prompt,
    render_metadata_prompt,
)


def test_extraction_prompt_forbids_conceptual_submodule_paths() -> None:
    assert "Do not split a directory into conceptual children" in EXTRACTION_PROMPT
    assert (
        "Submodule `path` must be a real nested path below its parent"
        in EXTRACTION_PROMPT
    )


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


# ── Stage 3A / 3B prompts ─────────────────────────────────────────────────


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
