"""Prompt loading and rendering for the modules extractor.

The independent single-shot path uses ``extraction.md``. The standard pipeline
uses ``identify_source_root.md``, then assignment and module-metadata prompts.
Exact-token one-pass substitution prevents placeholder-shaped repository text
from being expanded as another template token.
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
ENRICH_ASSIGNMENTS_PROMPT: str = _load("enrich_assignments.md")
ENRICH_MODULE_METADATA_PROMPT: str = _load("enrich_module_metadata.md")


def _as_json_block(data: Any) -> str:
    return json.dumps(data, indent=2, sort_keys=True)


def _substitute_once(template: str, mapping: dict[str, str]) -> str:
    if not mapping:
        return template
    pattern = re.compile("|".join(re.escape(token) for token in mapping))
    return pattern.sub(lambda match: mapping[match.group(0)], template)


def render_identify_source_root_prompt() -> str:
    return IDENTIFY_SOURCE_ROOT_PROMPT


_ASSIGNMENT_WHOLE_REPO_SCOPE_RULES = (
    "Your scope is the **entire repository** below the source root: every "
    "node in the SKELETON above is yours. Label every one of them. Every "
    "top-level path of the skeleton is a structural anchor: label each "
    "`MODULE`."
)


def _assignment_scope_rules(scope: dict[str, Any]) -> str:
    forced = sorted(scope.get("forced_part_paths") or [])
    if forced:
        listed = ", ".join(f"`{path}`" for path in forced)
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
        listed = ", ".join(f"`{path}`" for path in promoted)
        lines += [
            "",
            "These subtrees are owned by other labeling calls and have been "
            f"removed from your SKELETON: {listed}. They are children of "
            f"`{parent}`. Do NOT label them or anything under them. "
            f"`{parent}`'s raw `subtree_source_file_count` still counts "
            "them, and that raw count is what the size rule uses.",
        ]
    return "\n".join(lines) + forced_text


def render_assignment_prompt(
    *, repository: Any, skeleton: Any, scope: Any, merge_threshold: int
) -> str:
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
    "EXTRACTION_PROMPT",
    "IDENTIFY_SOURCE_ROOT_PROMPT",
    "render_assignment_prompt",
    "render_assignment_shard_prompt",
    "render_identify_source_root_prompt",
    "render_metadata_prompt",
]
