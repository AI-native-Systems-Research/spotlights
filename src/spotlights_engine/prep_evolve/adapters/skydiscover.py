"""skydiscover adapter (single-file): config.yaml + seed.<ext> + evaluator.py.

skydiscover evolves the region between literal `# EVOLVE-BLOCK-START/END`
markers in a single seed file and MAXIMIZES `evaluate()`'s `combined_score`.
"""

from __future__ import annotations

from pathlib import Path

from spotlights_engine.prep_evolve.adapters.base import GeneratedFile
from spotlights_engine.prep_evolve.digest import render_digest
from spotlights_engine.prep_evolve.spec import EvolveSpec, Target
from spotlights_engine.prep_evolve.templates import render_template
from spotlights_engine.prep_evolve.yaml_emit import dump_yaml

_DEFAULT_MODEL = "anthropic/claude-sonnet-4-6"

# File suffix -> skydiscover `language`. Only `#`-comment languages are safe for
# the literal marker convention.
_LANG_BY_SUFFIX = {
    ".py": "python",
    ".rb": "ruby",
    ".sh": "bash",
    ".pl": "perl",
    ".r": "r",
    ".jl": "julia",
    ".yaml": "yaml",
    ".yml": "yaml",
    ".toml": "toml",
}

# Suffixes where a leading `#` line is a valid comment (marker-safe).
_HASH_COMMENT_SUFFIXES = set(_LANG_BY_SUFFIX)


def _candidate_target(spec: EvolveSpec) -> Target | None:
    cands = [t for t in spec.targets if t.scope_kind == "candidate"]
    return cands[0] if len(cands) == 1 else None


class SkydiscoverAdapter:
    name = "skydiscover"

    def __init__(self, model: str | None = None) -> None:
        self._model = model or _DEFAULT_MODEL

    def supports(self, spec: EvolveSpec) -> tuple[bool, str]:
        cand_targets = [t for t in spec.targets if t.scope_kind == "candidate"]
        if len(cand_targets) != 1:
            return (
                False,
                "skydiscover needs exactly one candidate target "
                f"(got {len(cand_targets)}); use --scope candidate",
            )
        scope_only = [t for t in spec.targets if t.scope_kind != "candidate"]
        if scope_only:
            return (
                False,
                "skydiscover is single-file only; module-main-files scope is "
                "not supported (route to coral/nous)",
            )
        t = cand_targets[0]
        if t.line_start is None or t.line_end is None:
            return (False, "candidate target has no line range to mark")
        suffix = Path(t.file).suffix.lower()
        if suffix not in _HASH_COMMENT_SUFFIXES:
            return (
                False,
                f"file suffix {suffix!r} is not safe for literal '#' "
                "EVOLVE-BLOCK markers",
            )
        return (True, "")

    def render(self, spec: EvolveSpec) -> list[GeneratedFile]:
        t = _candidate_target(spec)
        if t is None or t.line_start is None or t.line_end is None:
            raise ValueError("skydiscover.render called on an unsupported spec")

        digest = render_digest(spec)
        seed_text, prefix, suffix = self._build_seed(spec, t)
        suffix_ext = Path(t.file).suffix.lower() or ".py"
        language = _LANG_BY_SUFFIX.get(suffix_ext, "python")

        files = [
            GeneratedFile(
                path=f"seed{suffix_ext}",
                text=seed_text,
                overwrite="always",
            ),
            GeneratedFile(
                path="config.yaml",
                text=self._build_config(spec, t, digest, language),
                overwrite="always",
            ),
            GeneratedFile(
                path="evaluator.py",
                text=self._build_evaluator(spec, t, prefix, suffix),
                overwrite="preserve_if_modified",
            ),
        ]
        return files

    # --- seed ---

    def _build_seed(
        self, spec: EvolveSpec, t: Target
    ) -> tuple[str, str, str]:
        """Insert EVOLVE-BLOCK markers around the 1-indexed inclusive range.

        Returns (seed_text, prefix, suffix) where prefix/suffix are the text
        outside the block (markers excluded) — exactly what the evaluator
        compares against to reject out-of-scope edits.
        """
        repo = Path(spec.run.repo_path)
        src = (repo / t.file).read_text(encoding="utf-8")
        lines = src.splitlines(keepends=True)
        start, end = t.line_start, t.line_end  # type: ignore[assignment]

        prefix = "".join(lines[: start - 1])
        block = "".join(lines[start - 1 : end])
        rest = "".join(lines[end:])

        # The evaluator splits on the marker strings; prefix/suffix are what
        # surrounds the markers. We keep the block body unchanged.
        marker_start = "# EVOLVE-BLOCK-START\n"
        marker_end = "# EVOLVE-BLOCK-END\n"
        # Ensure the block body ends with a newline so the end marker sits on
        # its own line.
        if block and not block.endswith("\n"):
            block += "\n"
        seed_text = prefix + marker_start + block + marker_end + rest

        # What the evaluator sees after stripping markers: prefix and
        # (block + rest). Split the same way the evaluator does — on the marker
        # strings — so its byte comparison matches.
        eval_prefix = prefix
        eval_suffix = rest
        return seed_text, eval_prefix, eval_suffix

    # --- config ---

    def _build_config(
        self, spec: EvolveSpec, t: Target, digest: str, language: str
    ) -> str:
        system_message = self._system_message(spec, t, digest)
        config = {
            "language": language,
            "max_iterations": 50,
            "checkpoint_interval": 10,
            "llm": {
                "models": [
                    {
                        "name": self._model,
                        "weight": 1.0,
                        "max_tokens": 32000,
                        "timeout": 600,
                    }
                ]
            },
            "prompt": {"system_message": system_message},
            "search": {"type": "adaevolve"},
            "evaluator": {"timeout": 600},
            "monitor": {"enabled": True},
        }
        header = (
            "# Generated by spotlights-engine prep-evolve.\n"
            "# Run: skydiscover-run seed"
            f"{Path(t.file).suffix.lower() or '.py'} evaluator.py -c config.yaml\n"
        )
        return header + dump_yaml(config)

    def _system_message(self, spec: EvolveSpec, t: Target, digest: str) -> str:
        parts = [
            f"You are optimizing `{t.symbol or t.file}` in repository "
            f"{spec.run.repo_name}.",
            "",
            f"Objective: {spec.objective.goal} "
            f"(direction: {spec.objective.direction}).",
            "",
        ]
        if t.description:
            parts.append(f"What this code does: {t.description}")
        if t.current_approach:
            parts.append(f"Current approach: {t.current_approach}")
        if t.evolve_rationale:
            parts.append(f"Why evolve it: {t.evolve_rationale}")
        parts.append("")
        parts.append(
            "Only the region between # EVOLVE-BLOCK-START and # EVOLVE-BLOCK-END "
            "may change; everything else is fixed scaffolding."
        )
        parts.append("")
        parts.append(digest)
        # Keep it well over 256 chars and multi-line so skydiscover treats it
        # as an inline string, not a file path.
        return "\n".join(parts)

    # --- evaluator ---

    def _build_evaluator(
        self, spec: EvolveSpec, t: Target, prefix: str, suffix: str
    ) -> str:
        correctness = t.oracles.correctness
        return render_template(
            "skydiscover_evaluator.py.tmpl",
            repo_name=spec.run.repo_name,
            module_qn=spec.module.qualified_name,
            candidate_id=t.candidate_id or "(unknown)",
            correctness_oracle=", ".join(correctness) or "(none parsed — add one)",
            performance_oracle=t.oracles.performance or "(none parsed — see objective)",
            direction=spec.objective.direction,
            target_file_repr=repr(t.file),
            repo_path_repr=repr(spec.run.repo_path),
            correctness_commands_repr=repr(correctness),
            seed_prefix_repr=repr(prefix),
            seed_suffix_repr=repr(suffix),
        )


__all__ = ["SkydiscoverAdapter"]
