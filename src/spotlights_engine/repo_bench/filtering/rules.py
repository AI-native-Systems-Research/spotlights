"""Filter rules — composable predicates and rankers over `RawPR`.

Each rule has a `to_spec()` method that returns a `RuleSpec`; the spec
is what `view_id` is hashed against. Two rules with the same spec
produce identical filter behavior.

A predicate decides keep/drop. A ranker assigns a numeric score
(higher = more relevant). A view can have at most one ranker.
Predicates are commutative; rankers are positional. See `derive.py`
for canonicalization.

All pattern logic lives in `heuristics.py` — this file just wraps
those primitives in the Rule protocol.
"""

from __future__ import annotations

from typing import Any, Protocol, runtime_checkable

from spotlights_engine.repo_bench.filtering import heuristics
from spotlights_engine.repo_bench.schemas import RawPR, RuleSpec


# ── Protocols ─────────────────────────────────────────────────────────


@runtime_checkable
class Rule(Protocol):
    """Common surface — every rule serializes to a RuleSpec."""

    def to_spec(self) -> RuleSpec: ...


@runtime_checkable
class RulePredicate(Rule, Protocol):
    """Boolean keep/drop. Composable as logical AND."""

    def keep(self, pr: RawPR) -> bool: ...


@runtime_checkable
class RuleRanker(Rule, Protocol):
    """Numeric score; higher = ranked higher in the view."""

    def score(self, pr: RawPR) -> float: ...


# ── Predicates ────────────────────────────────────────────────────────


class NotBot:
    """Drop bot authors."""

    NAME = "not-bot"

    def keep(self, pr: RawPR) -> bool:
        return not heuristics.is_bot(pr.author)

    def to_spec(self) -> RuleSpec:
        return RuleSpec(
            name=self.NAME,
            version=heuristics.BOT_HEURISTIC_VERSION,
            kind="predicate",
            params={
                "known_bots": sorted(heuristics.known_bot_authors()),
                "suffix_pattern": heuristics.bot_suffix_pattern(),
            },
        )


class NotRevert:
    """Drop PRs whose title starts with `Revert` / `[Revert ...]`."""

    NAME = "not-revert"

    def keep(self, pr: RawPR) -> bool:
        return not heuristics.is_revert_title(pr.title)

    def to_spec(self) -> RuleSpec:
        return RuleSpec(
            name=self.NAME,
            version=heuristics.REVERT_HEURISTIC_VERSION,
            kind="predicate",
            params={"pattern": heuristics.revert_pattern()},
        )


class NotChore:
    """Drop PRs tagged as non-perf chores (bugfix / CI / test / doc / refactor / etc).

    Override: if the title or body carries a strict perf claim (number
    bound to a perf noun), keep the PR. This preserves the rare
    `[Bugfix] fix slow attention path, 12% throughput regression`
    where the chore tag is incidental to a real perf signal.
    """

    NAME = "not-chore"

    def keep(self, pr: RawPR) -> bool:
        if not heuristics.is_non_perf_chore_title(pr.title):
            return True
        # Override: strict perf claim anywhere → keep.
        return (
            heuristics.strict_perf_match(pr.title) is not None
            or heuristics.strict_perf_match(pr.body) is not None
        )

    def to_spec(self) -> RuleSpec:
        return RuleSpec(
            name=self.NAME,
            version=heuristics.NON_PERF_CHORE_HEURISTIC_VERSION,
            kind="predicate",
            params={
                "pattern": heuristics.non_perf_chore_pattern(),
                "perf_override": "strict_perf_match(title|body)",
            },
        )


class TitleStrictPerfClaim:
    """Keep PRs whose **title** matches the strict perf-claim regex.

    Title-only on purpose: when an author puts `13.9% throughput
    improvement` in the title, it's almost always a deliberate perf
    advertisement; the body alone is too noisy (benchmark-output dumps
    leak through).
    """

    NAME = "title-strict-perf-claim"

    def keep(self, pr: RawPR) -> bool:
        return heuristics.strict_perf_match(pr.title) is not None

    def to_spec(self) -> RuleSpec:
        return RuleSpec(
            name=self.NAME,
            version=heuristics.PERF_CLAIM_HEURISTIC_VERSION,
            kind="predicate",
            params={
                "pattern": heuristics.perf_strict_pattern(),
                "false_positive_strip": heuristics.perf_false_positive_pattern(),
                "scope": "title-only",
            },
        )


class BodyStrictPerfClaim:
    """Keep PRs whose **body** matches the strict perf-claim regex.

    Companion to `TitleStrictPerfClaim` for measurement experiments —
    most real perf PRs put the headline number in the body table, not
    the title. The strict regex + false-positive strip already filters
    out cache-usage stat dumps, so body-text matching is workable.
    """

    NAME = "body-strict-perf-claim"

    def keep(self, pr: RawPR) -> bool:
        return heuristics.strict_perf_match(pr.body) is not None

    def to_spec(self) -> RuleSpec:
        return RuleSpec(
            name=self.NAME,
            version=heuristics.PERF_CLAIM_HEURISTIC_VERSION,
            kind="predicate",
            params={
                "pattern": heuristics.perf_strict_pattern(),
                "false_positive_strip": heuristics.perf_false_positive_pattern(),
                "scope": "body-only",
            },
        )


class AnyStrictPerfClaim:
    """Keep PRs whose **title or body** matches the strict perf-claim regex."""

    NAME = "any-strict-perf-claim"

    def keep(self, pr: RawPR) -> bool:
        return (
            heuristics.strict_perf_match(pr.title) is not None
            or heuristics.strict_perf_match(pr.body) is not None
        )

    def to_spec(self) -> RuleSpec:
        return RuleSpec(
            name=self.NAME,
            version=heuristics.PERF_CLAIM_HEURISTIC_VERSION,
            kind="predicate",
            params={
                "pattern": heuristics.perf_strict_pattern(),
                "false_positive_strip": heuristics.perf_false_positive_pattern(),
                "scope": "title-or-body",
            },
        )


class AnyPerfSignal:
    """Keep PRs that carry **either** a strict numeric perf claim
    (title or body) **or** an author perf tag (`[Perf]`, `[Performance]`,
    `[Optimize]`, `perf:` at title start).

    This is the broader "perf work" signal — useful when the goal is
    to benchmark discovery quality across all perf-relevant work, not
    just the subset where authors quantified the win.
    """

    NAME = "any-perf-signal"

    def keep(self, pr: RawPR) -> bool:
        return (
            heuristics.strict_perf_match(pr.title) is not None
            or heuristics.strict_perf_match(pr.body) is not None
            or heuristics.has_perf_tag_title(pr.title)
        )

    def to_spec(self) -> RuleSpec:
        return RuleSpec(
            name=self.NAME,
            # Bump when either composing heuristic bumps.
            version=(
                heuristics.PERF_CLAIM_HEURISTIC_VERSION + "+"
                + heuristics.PERF_TAG_HEURISTIC_VERSION
            ),
            kind="predicate",
            params={
                "strict_pattern": heuristics.perf_strict_pattern(),
                "false_positive_strip": heuristics.perf_false_positive_pattern(),
                "tag_pattern": heuristics.perf_tag_pattern(),
                "scope": "title-or-body for strict; title-only for tag",
            },
        )


class AnyLoosePerfClaim:
    """Keep PRs whose **title or body** matches the loose perf-claim regex.

    Loose = same number+perf-noun shape but allows up to 80 chars
    between them; tolerates phrasings the strict regex rejects.
    """

    NAME = "any-loose-perf-claim"

    def keep(self, pr: RawPR) -> bool:
        return (
            heuristics.loose_perf_match(pr.title) is not None
            or heuristics.loose_perf_match(pr.body) is not None
        )

    def to_spec(self) -> RuleSpec:
        return RuleSpec(
            name=self.NAME,
            version=heuristics.PERF_CLAIM_HEURISTIC_VERSION,
            kind="predicate",
            params={
                # Loose regex isn't currently exposed on heuristics; embed
                # the structural difference instead so the spec captures
                # the rule semantics, not just a pattern string.
                "scope": "title-or-body",
                "matcher": "loose",
                "false_positive_strip": heuristics.perf_false_positive_pattern(),
            },
        )


# ── Rankers ───────────────────────────────────────────────────────────


class RankBySpecificityAndMagnitude:
    """Rank by `claimed_pct × (1 / files_changed)`.

    Combines two axes: magnitude (the largest `%` in title or body via
    the strict regex) and specificity (`1/files_changed`, so a 1-file
    PR scores 1.0 on the specificity axis vs. 0.125 for an 8-file PR).

    A PR with no extractable `%` scores 0 — it stays in the view (the
    predicate accepted it) but ranks at the bottom.

    Magnitude prefers the title — authors put the headline number in the
    title. Falls back to title+body if no title number is found.
    """

    NAME = "rank-spec-mag"

    def score(self, pr: RawPR) -> float:
        pct = heuristics.max_perf_pct(pr.title)
        if pct is None:
            pct = heuristics.max_perf_pct(pr.title + "\n" + pr.body)
        if pct is None:
            return 0.0
        files_n = len(pr.files_changed) or 1
        return pct * (1.0 / files_n)

    def rank_inputs(self, pr: RawPR) -> dict[str, Any]:
        """Inputs to the score, surfaced in the view file for debugging."""
        pct = heuristics.max_perf_pct(pr.title)
        if pct is None:
            pct = heuristics.max_perf_pct(pr.title + "\n" + pr.body)
        files_n = len(pr.files_changed) or 1
        loc = (pr.additions_total or 0) + (pr.deletions_total or 0)
        return {"pct": pct, "files": files_n, "loc": loc}

    def to_spec(self) -> RuleSpec:
        return RuleSpec(
            name=self.NAME,
            version="v1",
            kind="ranker",
            params={
                "formula": "pct * (1 / files_changed)",
                "magnitude_source": "title-then-body",
                "magnitude_pattern": heuristics.perf_strict_pattern(),
            },
        )


__all__ = [
    "AnyLoosePerfClaim",
    "AnyPerfSignal",
    "AnyStrictPerfClaim",
    "BodyStrictPerfClaim",
    "NotBot",
    "NotChore",
    "NotRevert",
    "RankBySpecificityAndMagnitude",
    "Rule",
    "RulePredicate",
    "RuleRanker",
    "TitleStrictPerfClaim",
]
