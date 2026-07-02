#!/usr/bin/env python3
"""match_papers.py — match PR-cited papers against the engine's findings.

The second, orthogonal ground-truth signal: did the engine's deep research
independently surface the same paper the PR author cited? This is a *recall*
signal on its own axis, independent of whether candidates landed on the right
lines.

Matching mirrors the engine's **own** finding-dedup keys exactly (design
"Normalizer reuse"): we import `_finding_keys` / `_normalize_url` /
`_normalize_text` from `module_deep_research/orchestration.py` so cited papers
and engine findings are compared in one coordinate frame. A finding matches a
cited paper when their normalized-URL keys are equal **or** (when the cited
paper carries a title) their normalized-title keys are equal. Title-OR-URL is
deliberate: the engine frequently surfaces a paper at a *different* URL than the
PR cites (arxiv abs vs. conference-proceedings PDF vs. DOI), so URL-only matching
would under-count real hits. The report surfaces `matched_on: "url"|"title"` so
a title-only match can be eyeballed.

`via_candidate_ids` reverse-links each matched `finding_id` through
`candidate.proposals[].finding_ref_id`, so the report can say which candidate the
matched paper rode in on (or that it sits in findings unattached).

Emits `paper_match.json`:
    {
      "paper_cited": bool,
      "paper_hit": bool,
      "matched": [
        {"finding_id","title","url","source_type",
         "matched_on": "url"|"title", "via_candidate_ids": [...]}
      ],
      "cited_papers": [...],     # echoed from cited_papers.json
      "num_findings": int
    }

Deterministic. If `cited_papers.json` has no papers, `paper_cited` is false and
nothing else runs.

Usage:
    match_papers.py --cited cited_papers.json --result result.json -o paper_match.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# Reuse the engine's dedup-key helpers verbatim (design: do NOT hand-roll).
_REPO_ROOT = Path(__file__).resolve().parents[2]
_SRC = _REPO_ROOT / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from spotlights_engine.module_deep_research.orchestration import (  # noqa: E402
    _finding_keys,
    _normalize_text,
    _normalize_url,
)


def _findings_of(result: dict) -> list[dict]:
    """Engine findings, tolerant of the `report` wrapper and older flat dumps."""
    report = result.get("report")
    if isinstance(report, dict) and isinstance(report.get("findings"), list):
        return list(report["findings"])
    if isinstance(result.get("findings"), list):
        return list(result["findings"])
    # Last resort: concatenate per-module findings.
    out: list[dict] = []
    module_runs = result.get("module_runs")
    if isinstance(module_runs, dict):
        for run in module_runs.values():
            if isinstance(run, dict) and isinstance(run.get("findings"), list):
                out.extend(run["findings"])
    return out


def _candidates_of(result: dict) -> list[dict]:
    """Engine candidates (for reverse-linking finding_ref_id), tolerant of shape."""
    report = result.get("report")
    if isinstance(report, dict) and isinstance(report.get("candidates"), list):
        return list(report["candidates"])
    if isinstance(result.get("candidates"), list):
        return list(result["candidates"])
    out: list[dict] = []
    module_runs = result.get("module_runs")
    if isinstance(module_runs, dict):
        for run in module_runs.values():
            if not isinstance(run, dict):
                continue
            wrapper = run.get("candidates")
            if isinstance(wrapper, dict) and isinstance(wrapper.get("candidates"), list):
                out.extend(wrapper["candidates"])
    return out


def _finding_to_candidates(candidates: list[dict]) -> dict[str, list[str]]:
    """Map finding_id -> [candidate id, ...] via candidate.proposals[].finding_ref_id."""
    mapping: dict[str, list[str]] = {}
    for c in candidates:
        cid = c.get("id")
        for p in c.get("proposals") or []:
            if not isinstance(p, dict):
                continue
            fref = p.get("finding_ref_id")
            if fref:
                mapping.setdefault(fref, [])
                if cid is not None and cid not in mapping[fref]:
                    mapping[fref].append(cid)
    return mapping


def _cited_keys(paper: dict) -> tuple[str | None, str | None]:
    """Return (url_key, title_key) for a cited paper, in the engine's key space.

    The url_key reuses the engine's normalizer; cited_papers.json already stores
    `normalized` from the same normalizer, but we recompute defensively from the
    raw url so this script is self-contained. The title_key is built only when a
    human title was captured.
    """
    raw_url = paper.get("raw_url") or ""
    normalized = paper.get("normalized") or _normalize_url(raw_url)
    url_key = f"url:{normalized}" if normalized else None
    title = paper.get("title")
    title_key = f"title:{_normalize_text(title)}" if title else None
    return url_key, title_key


def match(cited_papers: list[dict], findings: list[dict], candidates: list[dict]) -> list[dict]:
    finding_to_cands = _finding_to_candidates(candidates)
    matched: list[dict] = []
    matched_finding_ids: set[str] = set()

    for f in findings:
        fid = f.get("finding_id")
        if fid in matched_finding_ids:
            continue
        fkeys = _finding_keys(f.get("title", ""), f.get("url", ""))
        for paper in cited_papers:
            url_key, title_key = _cited_keys(paper)
            matched_on = None
            if url_key and url_key in fkeys:
                matched_on = "url"
            elif title_key and title_key in fkeys:
                matched_on = "title"
            if matched_on:
                matched.append({
                    "finding_id": fid,
                    "title": f.get("title"),
                    "url": f.get("url"),
                    "source_type": f.get("source_type"),
                    "matched_on": matched_on,
                    "matched_cited_url": paper.get("raw_url"),
                    "via_candidate_ids": finding_to_cands.get(fid, []),
                })
                if fid is not None:
                    matched_finding_ids.add(fid)
                break  # one finding matches at most one cited paper entry
    return matched


def main() -> None:
    ap = argparse.ArgumentParser(description="Match PR-cited papers against engine findings.")
    ap.add_argument("--cited", required=True, help="cited_papers.json (from extract_pr_papers.py)")
    ap.add_argument("--result", required=True, help="engine result.json")
    ap.add_argument("-o", "--output", default=None, help="Write JSON here (default: stdout)")
    args = ap.parse_args()

    cited = json.loads(Path(args.cited).read_text(encoding="utf-8"))
    cited_papers = cited.get("papers", []) if isinstance(cited, dict) else []

    result = json.loads(Path(args.result).read_text(encoding="utf-8"))
    findings = _findings_of(result)
    candidates = _candidates_of(result)

    if not cited_papers:
        out = {
            "paper_cited": False,
            "paper_hit": False,
            "matched": [],
            "cited_papers": [],
            "num_findings": len(findings),
        }
    else:
        matched = match(cited_papers, findings, candidates)
        out = {
            "paper_cited": True,
            "paper_hit": bool(matched),
            "matched": matched,
            "cited_papers": cited_papers,
            "num_findings": len(findings),
        }

    text = json.dumps(out, indent=2) + "\n"
    if args.output:
        Path(args.output).write_text(text, encoding="utf-8")
    else:
        sys.stdout.write(text)


if __name__ == "__main__":
    main()
