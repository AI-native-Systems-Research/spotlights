"""Filter raw PRs into a ranked view (phase 3a).

Public surface:
    derive(window_id, rules) -> ViewHandle    # apply rules, write view file
    NotBot, NotRevert, TitleStrictPerfClaim   # predicates
    RankBySpecificityAndMagnitude             # ranker
    Rule, RulePredicate, RuleRanker           # protocol types

See docs/repo-bench/README.md.
"""

from spotlights_engine.repo_bench.filtering.derive import (
    ViewEntry,
    ViewHandle,
    ViewManifest,
    derive,
    view_id_for,
)
from spotlights_engine.repo_bench.filtering.rules import (
    AnyLoosePerfClaim,
    AnyPerfSignal,
    AnyStrictPerfClaim,
    BodyStrictPerfClaim,
    NotBot,
    NotChore,
    NotRevert,
    RankBySpecificityAndMagnitude,
    Rule,
    RulePredicate,
    RuleRanker,
    TitleStrictPerfClaim,
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
    "ViewEntry",
    "ViewHandle",
    "ViewManifest",
    "derive",
    "view_id_for",
]
