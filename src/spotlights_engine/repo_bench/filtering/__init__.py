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
from spotlights_engine.repo_bench.filtering.report import (
    FilterReport,
    RuleStats,
    format_report,
    format_report_md,
    generate_filter_report,
    report_to_dict,
)
from spotlights_engine.repo_bench.filtering.rules import (
    AnyLoosePerfClaim,
    AnyPerfSignal,
    AnyPerfSignalOrLabel,
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
    "AnyPerfSignalOrLabel",
    "AnyStrictPerfClaim",
    "BodyStrictPerfClaim",
    "FilterReport",
    "NotBot",
    "NotChore",
    "NotRevert",
    "RankBySpecificityAndMagnitude",
    "Rule",
    "RulePredicate",
    "RuleRanker",
    "RuleStats",
    "TitleStrictPerfClaim",
    "ViewEntry",
    "ViewHandle",
    "ViewManifest",
    "derive",
    "format_report",
    "format_report_md",
    "generate_filter_report",
    "report_to_dict",
    "view_id_for",
]
