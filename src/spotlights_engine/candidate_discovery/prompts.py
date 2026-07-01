"""Prompt assembly for Stage 1.

Templates live in `prompts_data/` as package data and are loaded once via
`importlib.resources` so behavior does not depend on cwd. Substitution is
regex-driven (`\\{word\\}`) rather than `str.format`, because the templates
contain literal JSON object braces that `str.format` would choke on.

The `review.md` template substitutes `{prev_candidates_json}` where spec §9
shows a `@candidates_prev.json` reference. The spec's `@`-reference is a
Claude file-attach convention that does not apply in non-interactive
subprocess mode (no file is actually written), so the orchestrator inlines
the previous iteration's JSON directly via this placeholder. Intent matches
spec §5 ("the orchestrator additionally inlines the prior iteration's
candidates.json content into the review prompt").
"""

from __future__ import annotations

import re
from importlib import resources

from spotlights_engine.schemas.common import SpotlightContext
from spotlights_engine.schemas.project import File, Module

_PLACEHOLDER = re.compile(r"\{(\w+)\}")


def _load(name: str) -> str:
    return (
        resources.files("spotlights_engine.candidate_discovery.prompts_data")
        .joinpath(name)
        .read_text(encoding="utf-8")
    )


_PREAMBLE = _load("preamble.md")
_BOOTSTRAP = _load("bootstrap.md")
_REVIEW = _load("review.md")
STRICT_RETRY_REMINDER: str = _load("strict_retry.md").rstrip("\n")

_REPO_CONTEXT_DEFAULT = "_(none provided)_"
_SPOTLIGHT_CONTEXT_DEFAULT = "_(none provided)_"
_REMOVED_DEFAULT = "_(none)_"


def _substitute(template: str, values: dict[str, str]) -> str:
    def repl(m: re.Match[str]) -> str:
        key = m.group(1)
        if key not in values:
            raise KeyError(f"unknown placeholder {{{key}}} in template")
        return values[key]

    return _PLACEHOLDER.sub(repl, template)


def _format_main_files(files: list[File]) -> str:
    if not files:
        return "(none)"
    return "\n".join(f"- {f.path} ({f.role})" for f in files)


def _format_submodule_names(submodules: list[Module]) -> str:
    return ", ".join(m.name for m in submodules) or "(none)"


def _format_submodule_paths(submodules: list[Module]) -> str:
    if not submodules:
        return "(none)"
    return "\n".join(f"- {m.path}" for m in submodules)


def _format_depends_on(deps: list[str]) -> str:
    return ", ".join(deps) or "(none)"


def _format_repo_context(md: str | None) -> str:
    if md is None:
        return _REPO_CONTEXT_DEFAULT
    return md


def _format_bullet_list(items: list[str]) -> str:
    if not items:
        return "(none)"
    return "\n".join(f"- {item}" for item in items)


def _format_spotlight_context(context: SpotlightContext | None) -> str:
    """Render the run's `SpotlightContext` into a markdown block.

    Steps 2/3/4/5/6 in the architecture all consume `SpotlightContext`. For
    candidate discovery the agent uses `objective` + `workload_hints` to bias
    candidate selection; `validation_plan` is shown for completeness so the
    agent can prefer locations whose evidence is testable here.
    """
    if context is None:
        return _SPOTLIGHT_CONTEXT_DEFAULT
    return (
        f"- **objective:** {context.objective}\n"
        f"- **workload_hints:**\n{_indent(_format_bullet_list(context.workload_hints))}\n"
        f"- **validation_plan:**\n{_indent(_format_bullet_list(context.validation_plan))}"
    )


def _indent(text: str, prefix: str = "  ") -> str:
    return "\n".join(prefix + line if line else line for line in text.split("\n"))


def wrap(prompt: str) -> str:
    """Prepend the §5 preamble to a substituted prompt body."""
    return _PREAMBLE + "\n" + prompt


def render_bootstrap(
    module_qualified_name: str,
    module: Module,
    *,
    repo_context_markdown: str | None = None,
    spotlight_context: SpotlightContext | None = None,
) -> str:
    values = {
        "module_qualified_name": module_qualified_name,
        "module_name": module.name,
        "module_path": module.path,
        "module_description": module.description or "(none)",
        "depends_on": _format_depends_on(module.depends_on),
        "main_files": _format_main_files(module.main_files),
        "submodule_names": _format_submodule_names(module.submodules),
        "submodule_paths": _format_submodule_paths(module.submodules),
        "repo_context": _format_repo_context(repo_context_markdown),
        "spotlight_context": _format_spotlight_context(spotlight_context),
    }
    return wrap(_substitute(_BOOTSTRAP, values))


def render_review(
    module_qualified_name: str,
    module: Module,
    prev_candidates_json: str,
    max_seen_candidate_id: str,
    *,
    removed_candidates_json: str | None = None,
    repo_context_markdown: str | None = None,
    spotlight_context: SpotlightContext | None = None,
) -> str:
    values = {
        "module_qualified_name": module_qualified_name,
        "module_path": module.path,
        "submodule_paths": _format_submodule_paths(module.submodules),
        "prev_candidates_json": prev_candidates_json,
        "max_seen_candidate_id": max_seen_candidate_id,
        "removed_candidates": removed_candidates_json or _REMOVED_DEFAULT,
        "repo_context": _format_repo_context(repo_context_markdown),
        "spotlight_context": _format_spotlight_context(spotlight_context),
    }
    return wrap(_substitute(_REVIEW, values))


__all__ = [
    "STRICT_RETRY_REMINDER",
    "render_bootstrap",
    "render_review",
    "wrap",
]
