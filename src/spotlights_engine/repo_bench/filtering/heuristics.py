"""Low-level heuristic primitives used by `filtering/rules.py`.

Single source of truth for: bot detection, revert detection, the strict
perf-claim regex, and the largest-% extractor used by the ranker.

These are pure functions over plain strings / dicts. Bumping any regex
here is a behavior change for the filter — bump `version` in the
corresponding `Rule.to_spec()` so `view_id` invalidates.
"""

from __future__ import annotations

import re

# ── Bot detection ─────────────────────────────────────────────────────

_BOT_SUFFIX_RE = re.compile(r"\[bot\]$|-bot$|^bot-", re.IGNORECASE)
_KNOWN_BOTS = frozenset({
    "dependabot[bot]",
    "pre-commit-ci[bot]",
    "github-actions[bot]",
    "renovate[bot]",
    "mergify[bot]",
    "copilot",
    "vllm-bot",
})

BOT_HEURISTIC_VERSION = "v1"


def is_bot(author: str) -> bool:
    """True iff the author looks like a bot.

    Rule: in the explicit known-bot set, OR matches the suffix regex.
    """
    if not author:
        return False
    if author in _KNOWN_BOTS:
        return True
    return bool(_BOT_SUFFIX_RE.search(author))


def known_bot_authors() -> frozenset[str]:
    return _KNOWN_BOTS


def bot_suffix_pattern() -> str:
    return _BOT_SUFFIX_RE.pattern


# ── Revert detection ──────────────────────────────────────────────────

_REVERT_TITLE_RE = re.compile(r"^\s*\[?revert\b", re.IGNORECASE)

REVERT_HEURISTIC_VERSION = "v1"


def is_revert_title(title: str) -> bool:
    """True iff the title starts with `Revert` / `[Revert ...]`."""
    if not title:
        return False
    return bool(_REVERT_TITLE_RE.search(title))


def revert_pattern() -> str:
    return _REVERT_TITLE_RE.pattern


# ── Non-perf chore detection ──────────────────────────────────────────
# vLLM tags PRs in title brackets fairly consistently. Anything tagged
# as a bugfix, CI, test, doc, refactor, or chore is unlikely to be a
# perf opportunity in the sense the discovery pipeline targets. The
# rule callers may still want to *override* this drop when the body
# carries a clear perf claim — that composition lives in `rules.py`.
#
# Pattern: title starts with `[Tag]` or `tag:` or `Tag:` (case
# insensitive) for any of the chore categories. Anchored at start so
# `[Perf] Fix slow attention by 30%` is NOT matched.

def _build_non_perf_chore_re(tags_alt: str) -> "re.Pattern[str]":
    """Build the chore-tag regex from a `tag1|tag2|...` alternation.

    The alternation is sourced from config (baseline tags + repo-specific
    extras). Two anchored variants: `[Tag]` form and `tag:`/`tag-` prefix.
    """
    return re.compile(
        rf"^\s*(?:\[\s*(?:{tags_alt})\s*\]|(?:{tags_alt})\s*[:\-])",
        re.IGNORECASE | re.VERBOSE,
    )


_NON_PERF_CHORE_RE = _build_non_perf_chore_re(
    "bug\\s*fix|bugfix|fix|ci|ci/?build|build|test|tests|doc|docs|"
    "chore|refactor|cleanup|style|lint|typo|nit|misc|deps|dep"
)

NON_PERF_CHORE_HEURISTIC_VERSION = "v1"


def is_non_perf_chore_title(title: str) -> bool:
    """True iff the title is tagged as a non-perf chore (bugfix/CI/test/etc).

    Anchored at start; tolerant of `[Tag]`, `Tag:`, and minor whitespace.
    """
    if not title:
        return False
    return bool(_NON_PERF_CHORE_RE.search(title))


def non_perf_chore_pattern() -> str:
    return _NON_PERF_CHORE_RE.pattern


def perf_strict_pattern_re() -> "re.Pattern[str]":
    """Used by tests to introspect the active strict-perf regex object."""
    return _PERF_STRICT_RE


# ── Author-tagged perf signal ────────────────────────────────────────
# Some perf work merges without a numeric claim — vLLM authors mark
# such PRs with `[Perf]` / `[Performance]` / `[Optimize]` / `perf:` in
# the title. This pattern detects that signal at the title start.
# Composed by `AnyPerfSignal` together with the strict-perf-claim
# regex so a PR passes if it has *either* a number or an author tag.

_PERF_TAG_RE = re.compile(
    r"(?ix)"
    r"^\s*"
    r"(?:"
    r"  \[\s*(?:perf(?:ormance)?|optimiz(?:e|ation)|speed[- ]?up)\b[^\]]*\]"
    r"  | (?:perf|performance)\s*[:\-]"
    r")"
)

PERF_TAG_HEURISTIC_VERSION = "v1"


def has_perf_tag_title(title: str) -> bool:
    """True iff title carries a `[Perf]` / `[Performance]` / `[Optimize]` / `perf:` tag at the start."""
    if not title:
        return False
    return bool(_PERF_TAG_RE.search(title))


def perf_tag_pattern() -> str:
    return _PERF_TAG_RE.pattern


# ── Perf-claim regex (strict) ─────────────────────────────────────────
# Two patterns layered:
#   _PERF_LOOSE_RE — high recall, includes incidental mentions and many
#     benchmark-dump false positives.
#   _PERF_STRICT_RE — number bound to a perf noun (optionally with a
#     bridge word like "improvement"), or "Nx speedup". The filter
#     uses strict only.
#
# `_FALSE_POSITIVE_RE` strips common "cache usage: NN%" / "utilization:
# NN%" stat-dump fragments before matching, so a benchmark-output dump
# in the body doesn't trigger a false positive.

_BRIDGE = (
    r"improvement|improvements|gain|gains|increase|decrease|reduction|"
    r"drop|boost|regression|faster|slower|lower|higher|reduced|increased"
)


def _build_perf_loose_re(perf_noun_alt: str) -> "re.Pattern[str]":
    return re.compile(
        rf"(?is)("
        rf"  \d+(?:\.\d+)?\s*%[^\n]{{0,80}}\b(?:{perf_noun_alt})\b"
        rf"  | \b(?:{perf_noun_alt})\b[^\n]{{0,80}}\d+(?:\.\d+)?\s*%"
        rf"  | \d+(?:\.\d+)?\s*[x×]\s*(?:speed[- ]?up|speedup|faster|slower)"
        rf")",
        re.VERBOSE,
    )


def _build_perf_strict_re(perf_noun_alt: str) -> "re.Pattern[str]":
    return re.compile(
        rf"(?ix)"
        rf"(?:"
        rf"  \b\d+(?:\.\d+)?\s*%\s*(?:e2e\s+|end[- ]to[- ]end\s+)?"
        rf"  (?:{_BRIDGE}\s+(?:in\s+|to\s+|of\s+)?)?"
        rf"  (?:in\s+|to\s+|on\s+)?(?:{perf_noun_alt})\b"
        rf"  | \b(?:{perf_noun_alt})\b\s+(?:{_BRIDGE})?\s*(?:by|of|from)?\s*"
        rf"     (?:up\s+to\s+)?\d+(?:\.\d+)?\s*%"
        rf"  | \b\d+(?:\.\d+)?\s*[x×]\s+(?:speed[- ]?up|speedup|faster|slower)\b"
        rf")",
    )


# Module-level pattern objects pre-built from the default (vllm) config.
# `set_active_config()` swaps them when a different config is loaded.
_DEFAULT_PERF_NOUN = (
    r"throughput|tput|tps|qps|rps|latency|ttft|tpot|itl|"
    r"speed[- ]?up|speedup|memory|mem|vram|footprint|"
    r"perf(?:ormance)?"
)
_PERF_LOOSE_RE = _build_perf_loose_re(_DEFAULT_PERF_NOUN)
_PERF_STRICT_RE = _build_perf_strict_re(_DEFAULT_PERF_NOUN)
_PERF_LABELS: frozenset[str] = frozenset()


def set_active_config(filter_patterns: "Any") -> None:
    """Replace module-level regex objects from compiled-config patterns.

    `filter_patterns` is a `CompiledFilterPatterns` (perf_nouns +
    chore_tags + perf_labels). We rebuild the regex objects in place so
    existing function calls (`strict_perf_match`, etc.) pick up the new
    patterns without needing parameter threading.

    Call exactly once per process, before any filter rules run.
    """
    global _PERF_LOOSE_RE, _PERF_STRICT_RE, _NON_PERF_CHORE_RE, _PERF_LABELS
    _PERF_LOOSE_RE = _build_perf_loose_re(filter_patterns.perf_nouns)
    _PERF_STRICT_RE = _build_perf_strict_re(filter_patterns.perf_nouns)
    _NON_PERF_CHORE_RE = _build_non_perf_chore_re(filter_patterns.chore_tags)
    _PERF_LABELS = frozenset(getattr(filter_patterns, "perf_labels", ()))


PERF_LABEL_HEURISTIC_VERSION = "v1"


def has_perf_label(labels: list[str] | tuple[str, ...] | None) -> bool:
    """True iff any label is in the active config's perf-label whitelist."""
    if not labels or not _PERF_LABELS:
        return False
    return any(lbl.lower() in _PERF_LABELS for lbl in labels)


def perf_labels() -> tuple[str, ...]:
    return tuple(sorted(_PERF_LABELS))

_FALSE_POSITIVE_RE = re.compile(
    r"(?i)(?:cache\s+usage|kv\s+cache\s+usage|utilization|util\.?)\s*:\s*\d+(?:\.\d+)?\s*%"
)

_PCT_NUM_RE = re.compile(r"(\d+(?:\.\d+)?)\s*%")

PERF_CLAIM_HEURISTIC_VERSION = "v1"


def strict_perf_match(text: str) -> "re.Match[str] | None":
    """Return the first strict-perf match in `text`, or None.

    Strips known false-positive fragments (cache-usage stat dumps)
    before matching.
    """
    if not text:
        return None
    cleaned = _FALSE_POSITIVE_RE.sub("", text)
    return _PERF_STRICT_RE.search(cleaned)


def loose_perf_match(text: str) -> "re.Match[str] | None":
    if not text:
        return None
    cleaned = _FALSE_POSITIVE_RE.sub("", text)
    return _PERF_LOOSE_RE.search(cleaned)


def max_perf_pct(text: str) -> float | None:
    """Largest `%` value across strict-perf matches in `text`. None if none.

    Skips `Nx speedup` matches — they don't have a `%`, and converting
    `x` to `%` would require a fake mapping.
    """
    if not text:
        return None
    cleaned = _FALSE_POSITIVE_RE.sub("", text)
    best: float | None = None
    for m in _PERF_STRICT_RE.finditer(cleaned):
        for num in _PCT_NUM_RE.findall(m.group(0)):
            try:
                v = float(num)
            except ValueError:
                continue
            if best is None or v > best:
                best = v
    return best


def perf_strict_pattern() -> str:
    return _PERF_STRICT_RE.pattern


def perf_false_positive_pattern() -> str:
    return _FALSE_POSITIVE_RE.pattern


__all__ = [
    "BOT_HEURISTIC_VERSION",
    "NON_PERF_CHORE_HEURISTIC_VERSION",
    "PERF_TAG_HEURISTIC_VERSION",
    "REVERT_HEURISTIC_VERSION",
    "PERF_CLAIM_HEURISTIC_VERSION",
    "has_perf_tag_title",
    "is_bot",
    "is_non_perf_chore_title",
    "is_revert_title",
    "strict_perf_match",
    "loose_perf_match",
    "max_perf_pct",
    "known_bot_authors",
    "bot_suffix_pattern",
    "non_perf_chore_pattern",
    "perf_tag_pattern",
    "revert_pattern",
    "perf_strict_pattern",
    "perf_false_positive_pattern",
]
