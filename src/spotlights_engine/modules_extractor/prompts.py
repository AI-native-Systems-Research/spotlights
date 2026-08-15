"""Prompt loading and rendering for the modules extractor.

Prompt templates live in `prompts_data/*.md` as package data and are loaded
once via `importlib.resources` so behavior does not depend on cwd.

Four prompt families:

- the legacy parameter-free `extraction.md` (single-shot path);
- Stage-1 `identify_source_root.md` (parameter-free);
- the tree-contract Stage-3 `enrich_skeleton.md` (whole-repo and per-shard
  renderers, preserved byte-for-byte during the contract A/B); and
- the assignment-contract `enrich_assignments.md` (Stage 3A) and
  `enrich_module_metadata.md` (Stage 3B).

The two-phase prompts embed untrusted repository/inventory data in clearly
delimited JSON blocks. Substitution uses explicit `{token}` replacement (not
`str.format`) so literal braces in the template body (the JSON shape
examples) are left untouched. The assignment-contract renderers additionally
use **one-pass** exact-token substitution: with sequential `str.replace`,
untrusted repository text containing a later placeholder would be expanded
when the next replacement runs.
"""

from __future__ import annotations

import json
import re
from importlib import resources
from typing import Any

from spotlights_engine.modules_extractor.constants import MAX_MAIN_FILES


def _load(name: str) -> str:
    return (
        resources.files("spotlights_engine.modules_extractor.prompts_data")
        .joinpath(name)
        .read_text(encoding="utf-8")
    )


EXTRACTION_PROMPT: str = _load("extraction.md")
IDENTIFY_SOURCE_ROOT_PROMPT: str = _load("identify_source_root.md")
ENRICH_SKELETON_PROMPT: str = _load("enrich_skeleton.md")
ENRICH_ASSIGNMENTS_PROMPT: str = _load("enrich_assignments.md")
ENRICH_MODULE_METADATA_PROMPT: str = _load("enrich_module_metadata.md")


def _as_json_block(data: Any) -> str:
    """Deterministic, human-readable JSON for embedding in a prompt."""
    return json.dumps(data, indent=2, sort_keys=True)


def _substitute_once(template: str, mapping: dict[str, str]) -> str:
    """Replace every `{token}` in `template` in a single scan.

    Replacement values are never re-scanned, so a placeholder-shaped string
    inside embedded untrusted data cannot be expanded by a later replacement.
    """
    if not mapping:
        return template
    pattern = re.compile("|".join(re.escape(token) for token in mapping))
    return pattern.sub(lambda m: mapping[m.group(0)], template)


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
        # The removed subtrees are children of the promotion parent, which is
        # the shard root in the common case and a descendant of it when
        # derivation walked down a one-child chain to find a splittable node.
        # Every rule below is about *that* node, so addressing them to the root
        # would be silently wrong for a chain-split spine — including the Rule-4
        # deferral, the one rule where being wrong is unrecoverable.
        parent = str(scope.get("promotion_parent") or root)
        listed = ", ".join(f"`{p}`" for p in promoted)
        lines += [
            "",
            f"These subtrees are owned by other shards and have been removed "
            f"from your SKELETON: {listed}. They are children of `{parent}`. "
            "Therefore:",
            "",
            "- Do **not** emit them and do **not** fold them.",
            "- Do **not** choose any `main_files` entry that lives under them — "
            "those files belong to modules another shard emits, and claiming one "
            "invalidates the whole result.",
            f"- If that leaves `{parent}` with no file of its own to cite (it is "
            "a pure container of sub-directories), give it `\"main_files\": []`. "
            "That is correct here — do not reach into a removed subtree, and do "
            "not fold anything just to have something to cite.",
            f"- `{parent}`'s `source_child_count` and "
            "`subtree_source_file_count` count those removed subtrees, so both "
            "will be larger than the `children` actually present above suggest. "
            "That is expected — do not use its `subtree_source_file_count` as a "
            "small-subtree collapse signal.",
            f"- The zero-or-≥2-children rule below does **not** apply to "
            f"`{parent}` itself: those removed subtrees are re-attached as its "
            f"children afterwards, so `{parent}` may legitimately carry exactly "
            "one child of its own here. The rule still applies to every other "
            "module in your subtree.",
        ]
        if parent != root:
            lines += [
                f"- You **must** emit `{parent}` as a module of its own — never "
                "fold it into a parent. The removed subtrees come back as its "
                "children, so it has to exist for them to attach to.",
                f"- Every directory between `{root}` and `{parent}` keeps all of "
                "its children in your SKELETON, so the zero-or-≥2-children rule "
                f"applies to each of them in full. `{parent}` counts as one "
                "child of its own parent — give that parent its other children "
                "too rather than folding them away.",
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


# ── Assignment-contract renderers (Stage 3A / 3B) ─────────────────────────

_ASSIGNMENT_WHOLE_REPO_SCOPE_RULES = (
    "Your scope is the **entire repository** below the source root: every "
    "node in the SKELETON above is yours. Label every one of them. Every "
    "top-level path of the skeleton is a structural anchor: label each "
    "`MODULE`."
)


def _assignment_scope_rules(scope: dict[str, Any]) -> str:
    """The hard scope rules for one assignment call (whole-repo or shard).

    A nested shard's displayed root is not "top-level" merely because its
    slice contains one root node — only a true branch root is a structural
    anchor. The full-skeleton-derived `forced_part_paths` ride along in every
    scope so collision precedence is identical in every partition.
    """
    forced = sorted(scope.get("forced_part_paths") or [])
    if forced:
        listed = ", ".join(f"`{p}`" for p in forced)
        forced_text = (
            f"\n\n`forced_part_paths` in your scope: {listed}. Label each "
            "`PART` regardless of size (public-name collision precedence)."
        )
    else:
        forced_text = "\n\n`forced_part_paths` in your scope: none."

    if scope.get("whole_repository"):
        return _ASSIGNMENT_WHOLE_REPO_SCOPE_RULES + forced_text

    root = scope.get("root_path", "")
    lines = [
        f"Your scope is the subtree rooted at `{root}` and nothing else.",
        "",
        f"- Label **only** the paths that appear in your SKELETON (all at or "
        f"under `{root}`). Never label a path outside it, even if you read "
        "files there for context.",
    ]
    if scope.get("is_branch_root"):
        lines.append(
            f"- `{root}` is a top-level branch of the full repository "
            "skeleton — a structural anchor. Label it `MODULE`."
        )
    else:
        lines.append(
            f"- `{root}` is NOT a top-level path of the full repository — it "
            "is a nested slice handed to you for scale. Do NOT treat it as an "
            "anchor: label it `MODULE` or `PART` on its own merits (size "
            "rule, then independence). Labeling it `PART` is legal even "
            "though its owning module is outside your view; code resolves "
            "the owner after all slices are united."
        )
    promoted = list(scope.get("promoted_children") or [])
    if promoted:
        parent = str(scope.get("promotion_parent") or root)
        listed = ", ".join(f"`{p}`" for p in promoted)
        lines += [
            "",
            f"These subtrees are owned by other labeling calls and have been "
            f"removed from your SKELETON: {listed}. They are children of "
            f"`{parent}`. Do NOT label them or anything under them. "
            f"`{parent}`'s raw `subtree_source_file_count` still counts "
            "them, and that raw count is what the size rule uses.",
        ]
    return "\n".join(lines) + forced_text


def render_assignment_prompt(
    *, repository: Any, skeleton: Any, scope: Any, merge_threshold: int
) -> str:
    """Stage-3A assignment prompt for the whole repository (single mode)."""
    return _substitute_once(
        ENRICH_ASSIGNMENTS_PROMPT,
        {
            "{repository_json}": _as_json_block(repository),
            "{skeleton_json}": _as_json_block(skeleton),
            "{scope_json}": _as_json_block(scope),
            "{scope_rules}": _assignment_scope_rules(dict(scope)),
            "{MERGE_THRESHOLD}": str(merge_threshold),
        },
    )


def render_assignment_shard_prompt(
    *, repository: Any, subtree: Any, scope: Any, merge_threshold: int
) -> str:
    """Stage-3A assignment prompt scoped to one shard's subtree slice."""
    return _substitute_once(
        ENRICH_ASSIGNMENTS_PROMPT,
        {
            "{repository_json}": _as_json_block(repository),
            "{skeleton_json}": _as_json_block(subtree),
            "{scope_json}": _as_json_block(scope),
            "{scope_rules}": _assignment_scope_rules(dict(scope)),
            "{MERGE_THRESHOLD}": str(merge_threshold),
        },
    )


def render_metadata_prompt(*, repository: Any, scopes: Any) -> str:
    """Stage-3B metadata prompt for one batch of final module territories."""
    return _substitute_once(
        ENRICH_MODULE_METADATA_PROMPT,
        {
            "{repository_json}": _as_json_block(repository),
            "{scopes_json}": _as_json_block(scopes),
            "{MAX_MAIN_FILES}": str(MAX_MAIN_FILES),
        },
    )


__all__ = [
    "ENRICH_ASSIGNMENTS_PROMPT",
    "ENRICH_MODULE_METADATA_PROMPT",
    "ENRICH_SKELETON_PROMPT",
    "EXTRACTION_PROMPT",
    "IDENTIFY_SOURCE_ROOT_PROMPT",
    "render_assignment_prompt",
    "render_assignment_shard_prompt",
    "render_enrich_prompt",
    "render_enrich_shard_prompt",
    "render_identify_source_root_prompt",
    "render_metadata_prompt",
]
