"""Prompt loading and rendering for the modules extractor.

Prompt templates live in `prompts_data/*.md` as package data and are loaded
once via `importlib.resources` so behavior does not depend on cwd.

The legacy `extraction.md` is parameter-free. The two-phase prompts embed
untrusted repository/inventory data in clearly delimited JSON blocks:
`render_enrich_prompt` substitutes those blocks into the loaded templates.
Substitution uses explicit `{token}` replacement (not `str.format`) so literal
braces in the template body (the JSON shape examples) are left untouched.
"""

from __future__ import annotations

import json
from importlib import resources
from typing import Any


def _load(name: str) -> str:
    return (
        resources.files("spotlights_engine.modules_extractor.prompts_data")
        .joinpath(name)
        .read_text(encoding="utf-8")
    )


EXTRACTION_PROMPT: str = _load("extraction.md")
IDENTIFY_SOURCE_ROOT_PROMPT: str = _load("identify_source_root.md")
ENRICH_SKELETON_PROMPT: str = _load("enrich_skeleton.md")


def _as_json_block(data: Any) -> str:
    """Deterministic, human-readable JSON for embedding in a prompt."""
    return json.dumps(data, indent=2, sort_keys=True)


def render_identify_source_root_prompt() -> str:
    """Stage-1 prompt is parameter-free (repo delivered via subprocess cwd)."""
    return IDENTIFY_SOURCE_ROOT_PROMPT


_WHOLE_REPO_SCOPE_RULES = (
    "Your scope is the **entire repository** below the source root: every node "
    "in the SKELETON above is yours. Emit or fold every `required` path."
)


def _shard_scope_rules(scope: dict[str, Any]) -> str:
    """The hard scope rules for one enrichment shard.

    A sub-shard sees only its own slice of the skeleton, so these rules — not
    the data — are what keep its fragment mergeable: emitting outside the scope
    would break the disjoint-union invariant, and a spine `main_file` reaching
    into a promoted child is a merge-time hard failure with no possible repair.
    """
    root = scope.get("root_path", "")
    lines = [
        f"Your scope is the subtree rooted at `{root}` and nothing else.",
        "",
        f"- Emit or fold **only** paths at or under `{root}`. Never emit or fold "
        "a path outside it, even if you read files there for context.",
        f"- Return exactly ONE top-level module, whose `path` is `{root}`. Every "
        "other module you emit is nested inside it.",
    ]
    owns_root = scope.get("owns_root", True)
    if not owns_root:
        lines.append(
            f"- Do **not** emit any ancestor of `{root}`; `{root}` is your "
            "top-level object here."
        )
    promoted = list(scope.get("promoted_children") or [])
    if promoted:
        listed = ", ".join(f"`{p}`" for p in promoted)
        lines += [
            "",
            f"These subtrees are owned by other shards and have been removed "
            f"from your SKELETON: {listed}. Therefore:",
            "",
            "- Do **not** emit them and do **not** fold them.",
            "- Do **not** choose any `main_files` entry that lives under them — "
            "those files belong to modules another shard emits, and claiming one "
            "invalidates the whole result.",
            f"- `{root}`'s `source_child_count` counts those removed children, so "
            "it will be larger than the `children` actually present above. That "
            "is expected.",
            f"- The zero-or-≥2-children rule below does **not** apply to `{root}` "
            "itself: those removed subtrees are re-attached as its children "
            f"afterwards, so `{root}` may legitimately carry exactly one child "
            "of its own here. The rule still applies to every module nested "
            "inside it.",
        ]
    return "\n".join(lines)


def render_enrich_prompt(*, repository: Any, skeleton: Any) -> str:
    """Stage-3 enrichment prompt for the **whole repository** (single mode).

    `repository` and `skeleton` are JSON-serializable (dicts or the model
    `.model_dump()` output). Returns the rendered prompt string.
    """
    scope = {"root_path": skeleton.get("source_root", "")
             if isinstance(skeleton, dict) else "",
             "whole_repository": True}
    return (
        ENRICH_SKELETON_PROMPT
        .replace("{repository_json}", _as_json_block(repository))
        .replace("{skeleton_json}", _as_json_block(skeleton))
        .replace("{scope_json}", _as_json_block(scope))
        .replace("{scope_rules}", _WHOLE_REPO_SCOPE_RULES)
    )


def render_enrich_shard_prompt(*, repository: Any, subtree: Any, scope: Any) -> str:
    """Stage-3 enrichment prompt scoped to one shard's subtree.

    The `SKELETON` block carries the shard's subtree slice, not the whole
    skeleton — this is the substitution that shrinks the embedded blob from
    "whole monorepo" to "one branch".
    """
    return (
        ENRICH_SKELETON_PROMPT
        .replace("{repository_json}", _as_json_block(repository))
        .replace("{skeleton_json}", _as_json_block(subtree))
        .replace("{scope_json}", _as_json_block(scope))
        .replace("{scope_rules}", _shard_scope_rules(dict(scope)))
    )


__all__ = [
    "ENRICH_SKELETON_PROMPT",
    "EXTRACTION_PROMPT",
    "IDENTIFY_SOURCE_ROOT_PROMPT",
    "render_enrich_prompt",
    "render_enrich_shard_prompt",
    "render_identify_source_root_prompt",
]
