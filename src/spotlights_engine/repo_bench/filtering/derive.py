"""Apply rules to raw PRs and write the ranked view into a run dir.

`derive(window_id, rules, run_dir)` reads
`raw/<window_id>/prs.jsonl` (cached input under data_root), applies
predicates (logical AND), then optionally a single ranker, then
writes:

    <run_dir>/view/prs.jsonl     — ranked subset, one row per kept PR
    <run_dir>/view/manifest.json — rule specs, counts, derived_at

`view_id` is canonical: predicates are sorted by (name, version)
before hashing; rankers stay positional. So `[NotBot(), NotRevert()]`
and `[NotRevert(), NotBot()]` produce the same `view_id`. The id is
recorded in the manifest and surfaces in the run-dir name; it's no
longer used as a cache key on disk.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from spotlights_engine.repo_bench.filtering.rules import (
    Rule,
    RulePredicate,
    RuleRanker,
)
from spotlights_engine.repo_bench.schemas import (
    SCHEMA_VERSION,
    RawPR,
    RuleSpec,
)
from spotlights_engine.repo_bench.storage import (
    raw_dir,
    read_jsonl_lenient,
    write_json,
    write_jsonl,
)
from spotlights_engine.signal_pipeline.layout import artifact_hash

VIEW_VERSION = "0.2.0"
VIEW_ID_PREFIX_LEN = 12


# ── Result types ──────────────────────────────────────────────────────


@dataclass(frozen=True)
class ViewHandle:
    """What `derive()` returns. Where the view lives + its id."""

    view_id: str
    path: Path
    n_kept: int
    n_dropped: int


class ViewEntry(BaseModel):
    """One row written to `<run_dir>/view/prs.jsonl`.

    Keeps only the data needed to re-find the raw row (`raw_pr_ref`)
    plus what's useful for at-a-glance inspection. Downstream stages
    join against `raw/<window_id>/prs.jsonl` for the full PR data.
    """

    model_config = ConfigDict(extra="forbid")

    pr_number: int
    title: str
    score: float
    rank_inputs: dict = Field(default_factory=dict)
    raw_pr_ref: dict


class ViewManifest(BaseModel):
    """Sibling of view's prs.jsonl. Records exactly what was applied."""

    model_config = ConfigDict(extra="forbid")

    schema_version: str = SCHEMA_VERSION
    view_version: str = VIEW_VERSION
    view_id: str
    window_id: str
    rules: list[RuleSpec]
    n_input: int
    n_kept: int
    n_dropped: int
    derived_at: datetime


# ── view_id canonicalization ──────────────────────────────────────────


def _canonical_specs(rules: list[Rule]) -> list[RuleSpec]:
    """Predicates sorted by (name, version); rankers positional."""
    specs = [r.to_spec() for r in rules]
    predicates = sorted(
        (s for s in specs if s.kind == "predicate"),
        key=lambda s: (s.name, s.version),
    )
    rankers = [s for s in specs if s.kind == "ranker"]
    return [*predicates, *rankers]


def view_id_for(rules: list[Rule]) -> str:
    """Hash the canonical spec list. Same logical rule set → same id."""
    return artifact_hash(_canonical_specs(rules))[:VIEW_ID_PREFIX_LEN]


# ── derive ────────────────────────────────────────────────────────────


def _split_rules(
    rules: list[Rule],
) -> tuple[list[RulePredicate], RuleRanker | None]:
    predicates: list[RulePredicate] = []
    rankers: list[RuleRanker] = []
    for r in rules:
        if isinstance(r, RuleRanker):
            rankers.append(r)
        elif isinstance(r, RulePredicate):
            predicates.append(r)
        else:
            raise TypeError(
                f"rule {r!r} is neither RulePredicate nor RuleRanker; "
                f"each rule must implement `keep` or `score` plus `to_spec`"
            )
    if len(rankers) > 1:
        raise ValueError(
            f"a view may have at most one ranker; got {len(rankers)}: "
            f"{[r.to_spec().name for r in rankers]}"
        )
    return predicates, rankers[0] if rankers else None


def _load_raw(window_id: str, *, data_root_override: Path | None = None) -> list[RawPR]:
    raw_path = raw_dir(window_id, root=data_root_override) / "prs.jsonl"
    if not raw_path.exists():
        raise FileNotFoundError(
            f"no raw scrape at {raw_path}; run `repo-bench aggregate` first"
        )
    rows: list[RawPR] = []
    bad = 0
    for d in read_jsonl_lenient(raw_path):
        try:
            rows.append(RawPR.model_validate(d))
        except ValidationError:
            bad += 1
    if bad:
        print(f"# warning: {bad} raw rows failed RawPR validation; ignored")
    return rows


def derive(
    window_id: str,
    rules: list[Rule],
    *,
    run_dir: Path,
    data_root_override: Path | None = None,
) -> ViewHandle:
    """Apply `rules` to raw PRs; write `<run_dir>/view/`.

    `view_id` is derived from the canonical rule-spec list and recorded
    in the manifest. The view is written into the run dir, not into a
    shared cache — re-running with the same rules in a new run dir
    re-derives (regex over raw is cheap; raw is the input cache).
    """
    if not rules:
        raise ValueError("at least one rule required")

    predicates, ranker = _split_rules(rules)
    vid = view_id_for(rules)

    raw_rows = _load_raw(window_id, data_root_override=data_root_override)
    n_input = len(raw_rows)

    # Apply predicates (AND).
    kept: list[RawPR] = [
        pr for pr in raw_rows if all(p.keep(pr) for p in predicates)
    ]

    # Score + sort. Tie-break: score desc, pct desc, files asc, pr_number asc.
    if ranker is not None:
        scored: list[tuple[RawPR, float, dict]] = []
        for pr in kept:
            ri = (
                ranker.rank_inputs(pr)
                if hasattr(ranker, "rank_inputs")
                else {}
            )
            scored.append((pr, ranker.score(pr), ri))
        scored.sort(
            key=lambda t: (
                -t[1],
                -(t[2].get("pct") or 0.0),
                t[2].get("files") or 0,
                t[0].pr_number,
            )
        )
        ordered: list[tuple[RawPR, float, dict]] = scored
    else:
        # No ranker: stable order by pr_number.
        ordered = [(pr, 0.0, {}) for pr in sorted(kept, key=lambda p: p.pr_number)]

    view_rows: list[ViewEntry] = [
        ViewEntry(
            pr_number=pr.pr_number,
            title=pr.title,
            score=score,
            rank_inputs=ri,
            raw_pr_ref={"window_id": window_id, "pr_number": pr.pr_number},
        )
        for (pr, score, ri) in ordered
    ]

    out_dir = run_dir / "view"
    out_dir.mkdir(parents=True, exist_ok=True)

    prs_path = out_dir / "prs.jsonl"
    write_jsonl(prs_path, view_rows)

    manifest = ViewManifest(
        view_id=vid,
        window_id=window_id,
        rules=_canonical_specs(rules),
        n_input=n_input,
        n_kept=len(view_rows),
        n_dropped=n_input - len(view_rows),
        derived_at=datetime.now(timezone.utc),
    )
    write_json(out_dir / "manifest.json", manifest)

    return ViewHandle(
        view_id=vid,
        path=out_dir,
        n_kept=len(view_rows),
        n_dropped=n_input - len(view_rows),
    )


__all__ = [
    "VIEW_VERSION",
    "ViewEntry",
    "ViewHandle",
    "ViewManifest",
    "derive",
    "view_id_for",
]
