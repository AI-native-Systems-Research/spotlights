"""Live-repo validation for prep-evolve (the staleness gate).

This is the only module that reads the target repo. It validates the selected
candidate (and any scope-only main-file targets) against the live tree before
any bundle is written, and captures the repo git revision. Keeping this
separate from `extract.py` lets the resolver/spec tests stay unit-level while
making the staleness gate explicit (plan §4).
"""

from __future__ import annotations

import ast
import hashlib
import re
import subprocess
from dataclasses import dataclass
from pathlib import Path

from spotlights_engine.prep_evolve.errors import (
    RepoResolutionError,
    StalenessError,
)
from spotlights_engine.prep_evolve.spec import SourceRevision
from spotlights_engine.schemas.candidate import Candidate
from spotlights_engine.utils.schema_compat import primary_file, primary_span

# How far above/below the recorded range to look for the recorded symbol when
# checking staleness. result.json does not carry the original excerpt, so the
# check is intentionally heuristic.
_SYMBOL_WINDOW = 5

# Recorded symbols are rarely clean qualified names. Discovery labels region
# candidates with prose annotations ("STRInference.do_predict (ROI extraction
# and detection-reuse policy)", "AngleDetectorManager.__init__ self.detectors
# registry", "region: DSIZE / RotatePadFitTransform / rotate_image"), and may
# record a Python private method in its *mangled* form
# ("Detector._Detector__combine_channels"), which never appears in source. The
# recorded string therefore never matches verbatim; the gate matches
# identifier-shaped tokens extracted from it, demangling private names.
_IDENTIFIER = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")

# Structural separators of a qualified name ("." "/" "#" "::" and whitespace),
# used to split the symbol's leading qualified name into container components.
_SYMBOL_SEPARATORS = re.compile(r"[./#\s]+|::")

# A mangled private-attribute reference (`_ClassName__member`); source spells
# it `__member`.
_MANGLED = re.compile(r"_[A-Za-z][A-Za-z0-9_]*?(__[A-Za-z0-9_]+)")

# Tokens too common in code or in annotation prose to serve as evidence that
# the recorded range is the described code ("for" is a substring of every
# loop). Closed-class words only, not an attempt to enumerate English:
# annotation nouns ("policy", "loop") stay, they just rarely match anything.
_GENERIC_TOKENS = frozenset(
    {
        "and",
        "the",
        "for",
        "with",
        "from",
        "into",
        "over",
        "self",
        "cls",
        "not",
        "all",
        "any",
        "per",
        "via",
        "non",
        "new",
        "old",
    }
)


def _demangled_forms(token: str) -> tuple[str, ...]:
    """The token itself plus, for a mangled private name, its in-source form."""
    mangled = _MANGLED.fullmatch(token)
    return (token, mangled.group(1)) if mangled else (token,)


def _found(token: str, text: str) -> bool:
    return any(form in text for form in _demangled_forms(token))


def _evidence_tokens(symbol: str) -> list[str]:
    """Identifier tokens of the symbol usable as evidence near the range.

    Prefers specific tokens (length >= 3, not a generic word); falls back to
    all identifier tokens when the filter would leave nothing to check.
    """
    tokens = _IDENTIFIER.findall(symbol)
    specific = [t for t in tokens if len(t) >= 3 and t.lower() not in _GENERIC_TOKENS]
    return specific or tokens


def _container_tokens(symbol: str) -> list[str]:
    """Container components of the symbol's leading qualified name.

    For "AngleDetectorManager.__init__ self.detectors registry" the leading
    whitespace-chunk is "AngleDetectorManager.__init__" and the containers are
    ["AngleDetectorManager"]: a container (class/type) is expected somewhere in
    the file — at its declaration — even when the recorded range covers only a
    method body. Annotation text after the first whitespace never produces
    containers, and non-identifier fragments (e.g. a "region:" prefix) are
    dropped rather than required.
    """
    chunks = symbol.split()
    if not chunks:
        return []
    parts = [p for p in _SYMBOL_SEPARATORS.split(chunks[0]) if p]
    return [p for p in parts[:-1] if _IDENTIFIER.fullmatch(p)]


def _python_symbol_contains_range(
    source: str, symbol: str, start: int, end: int, kind: str
) -> bool:
    """Recognize a selected range inside its enclosing Python definition."""
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return False
    names = {
        node.name
        for node in ast.walk(tree)
        if isinstance(node, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef))
        and node.lineno <= end
        and getattr(node, "end_lineno", node.lineno) >= start
    }
    components = [part for part in _SYMBOL_SEPARATORS.split(symbol) if _IDENTIFIER.fullmatch(part)]
    if kind == "region" and len(components) > 1:
        # A region label may end with a descriptive name rather than a Python
        # definition, while all preceding qualified definitions remain real.
        return all(component in names for component in components[:-1])
    return bool(components) and all(component in names for component in components)


@dataclass
class ValidatedCandidate:
    """The validated, live-repo-confirmed view of the candidate target."""

    line_start: int
    line_end: int
    source_excerpt_sha256: str


def _resolve_inside(repo_path: Path, rel_file: str) -> Path:
    """Resolve `rel_file` inside `repo_path`, rejecting path escapes."""
    rel_path = Path(rel_file)
    if rel_path.is_absolute():
        raise StalenessError(f"target file {rel_file!r} must be repo-relative, not absolute")

    repo_root = repo_path.resolve()
    resolved = (repo_root / rel_path).resolve()
    if not resolved.is_relative_to(repo_root):
        raise StalenessError(
            f"target file {rel_file!r} resolves outside the repo ({resolved} not under {repo_root})"
        )
    return resolved


def validate_scope_file(repo_path: Path, rel_file: str) -> None:
    """Validate a scope-only (whole-file) target: containment + existence."""
    resolved = _resolve_inside(repo_path, rel_file)
    if not resolved.is_file():
        raise StalenessError(f"scope target file does not exist in repo: {rel_file}")


def validate_candidate_target(
    repo_path: Path,
    candidate: Candidate,
) -> ValidatedCandidate:
    """Validate the selected candidate against the live repo.

    Checks path containment, file existence, line bounds, and a heuristic
    staleness gate (the recorded symbol must appear in/around the recorded
    range). Returns the validated range and the excerpt hash.
    """
    cand_file = primary_file(candidate)
    span = primary_span(candidate)
    resolved = _resolve_inside(repo_path, cand_file)
    if not resolved.is_file():
        raise StalenessError(f"candidate file does not exist in repo: {cand_file}")

    lines = resolved.read_text(encoding="utf-8", errors="replace").splitlines()
    n = len(lines)
    start, end = span.line_start, span.line_end
    if start < 1 or end < start or end > n:
        raise StalenessError(
            f"candidate line range [{start}, {end}] is out of bounds for "
            f"{cand_file} ({n} lines). result.json is stale relative to "
            f"the repo; re-run spotlights or correct the selected result."
        )

    # 1-indexed inclusive slice.
    excerpt = "\n".join(lines[start - 1 : end])
    digest = hashlib.sha256(excerpt.encode("utf-8")).hexdigest()

    # Staleness heuristic, two independent checks (see the token helpers
    # above for why the symbol string is never matched verbatim):
    #  1. at least one identifier token of the recorded symbol — annotation
    #     words included, private names demangled — must appear within
    #     `_SYMBOL_WINDOW` lines of the recorded range, so we don't wrap an
    #     EVOLVE-BLOCK around the wrong code;
    #  2. the containers of the leading qualified name (the class in
    #     `Class.method`) must appear somewhere in the file — a container may
    #     legitimately sit far above the range, but one that is gone entirely
    #     means the file no longer holds the recorded symbol.
    win_start = max(0, start - 1 - _SYMBOL_WINDOW)
    win_end = min(n, end + _SYMBOL_WINDOW)
    window_text = "\n".join(lines[win_start:win_end])
    file_text = "\n".join(lines)
    problems: list[str] = []
    if span.symbol:
        evidence = _evidence_tokens(span.symbol)
        if evidence and not any(_found(t, window_text) for t in evidence):
            problems.append(
                f"no component of the symbol ({', '.join(evidence)}) appears "
                f"within {_SYMBOL_WINDOW} lines of the range"
            )
        missing = [c for c in _container_tokens(span.symbol) if not _found(c, file_text)]
        if missing:
            problems.append(
                f"container component(s) {', '.join(missing)} no longer "
                f"appear anywhere in the file"
            )
    structural_match = resolved.suffix == ".py" and _python_symbol_contains_range(
        file_text, span.symbol or "", start, end, span.kind
    )
    if problems and not structural_match:
        raise StalenessError(
            f"recorded symbol {span.symbol!r} not found near lines "
            f"[{start}, {end}] of {cand_file} ({'; '.join(problems)}). "
            f"result.json is stale relative to the repo; re-run spotlights or "
            f"correct the selected result."
        )

    return ValidatedCandidate(
        line_start=start,
        line_end=end,
        source_excerpt_sha256=digest,
    )


def capture_revision(repo_path: Path, captured_at: str) -> SourceRevision:
    """Best-effort git revision capture. `git_commit`/`dirty` are `None` when
    `repo_path` is not a git checkout or git is unavailable."""
    commit: str | None = None
    dirty: bool | None = None
    try:
        head = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=repo_path,
            capture_output=True,
            text=True,
            timeout=10,
        )
        if head.returncode == 0:
            commit = head.stdout.strip() or None
            status = subprocess.run(
                ["git", "status", "--porcelain"],
                cwd=repo_path,
                capture_output=True,
                text=True,
                timeout=10,
            )
            if status.returncode == 0:
                dirty = bool(status.stdout.strip())
    except (OSError, subprocess.SubprocessError):
        commit, dirty = None, None

    return SourceRevision(git_commit=commit, dirty=dirty, captured_at=captured_at)


def ensure_repo_dir(repo_path: Path) -> None:
    """Re-assert that `repo_path` is an existing directory (defensive)."""
    if not repo_path.is_dir():
        raise RepoResolutionError(f"repo path is not a directory: {repo_path}")


__all__ = [
    "ValidatedCandidate",
    "capture_revision",
    "ensure_repo_dir",
    "validate_candidate_target",
    "validate_scope_file",
]
