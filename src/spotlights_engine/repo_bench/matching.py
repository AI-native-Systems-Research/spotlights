"""Match agent findings to PRs by inspecting actual diffs.

For each finding, find PRs in the filtered view whose diff plausibly
addresses the same issue, then ask an LLM to judge each candidate.

Pipeline (per finding):
  1. Tier 1 candidates: PRs whose diff has a hunk inside the
     finding's symbol (function/method/class).
  2. Tier 2 candidates: PRs touching the same file but not Tier 1.
  3. Slice each candidate's diff to only hunks in the finding's file.
  4. Single LLM call per finding: judge produces verdict +
     hunk_citation per candidate.

Verdicts:
  same_idea     — diff makes essentially the same change as the finding
  related       — same code touched, related but distinct change
  neighborhood  — same file, unrelated code
  no_match      — diff has nothing to do with the finding

Output: <run_dir>/match_report.json with per-finding entries +
top-line summary.
"""

from __future__ import annotations

import hashlib
import json
import logging
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Literal, Protocol

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from spotlights_engine.agent_proposals.claude_exec import (
    CandidateAgentRunResult,
    run_candidate_claude as _run_claude_default,
)
from spotlights_engine.repo_bench.schemas import RawPR
from spotlights_engine.repo_bench.storage import (
    atomic_write_text,
    data_root,
    raw_dir,
    read_jsonl_lenient,
)

log = logging.getLogger(__name__)

PROMPT_VERSION = "v1"
DEFAULT_WALLCLOCK_S = 600
DEFAULT_MAX_TURNS = 15
TIER2_CAP_PER_FINDING = 8
DIFF_MAX_KB_PER_PR = 60


# ── Diff parsing ──────────────────────────────────────────────────────


_DIFF_FILE_HEADER = re.compile(r"^diff --git a/(.+?) b/(.+?)$", re.MULTILINE)
_HUNK_HEADER = re.compile(r"^@@ -\d+(?:,\d+)? \+\d+(?:,\d+)? @@(.*)$", re.MULTILINE)


def _norm_path(p: str) -> str:
    s = p.replace("\\", "/")
    while s.startswith("./"):
        s = s[2:]
    return s.lstrip("/")


@dataclass(frozen=True)
class FileSlice:
    path: str
    text: str
    hunks: tuple[str, ...]


def parse_diff_by_file(diff_text: str) -> dict[str, FileSlice]:
    """Split a unified diff into per-file slices."""
    out: dict[str, FileSlice] = {}
    if not diff_text:
        return out
    file_starts: list[tuple[int, str, str]] = []
    for m in _DIFF_FILE_HEADER.finditer(diff_text):
        file_starts.append((m.start(), m.group(1), m.group(2)))
    if not file_starts:
        return out
    for i, (start, _a, b) in enumerate(file_starts):
        end = file_starts[i + 1][0] if i + 1 < len(file_starts) else len(diff_text)
        text = diff_text[start:end]
        path = _norm_path(b)
        hunks = tuple(m.group(1).strip() for m in _HUNK_HEADER.finditer(text))
        out[path] = FileSlice(path=path, text=text, hunks=hunks)
    return out


def hunk_touches_symbol(hunks: tuple[str, ...], symbol: str) -> bool:
    """True iff any hunk header context line names the finding's symbol."""
    if not symbol:
        return False
    short = symbol.split(".")[-1]
    klass = symbol.split(".")[0] if "." in symbol else ""
    needles = [n for n in (short, klass) if n and len(n) >= 3]
    if not needles:
        return False
    for h in hunks:
        for n in needles:
            if re.search(rf"\b{re.escape(n)}\b", h):
                return True
    return False


# ── Schemas ───────────────────────────────────────────────────────────


Verdict = Literal["same_idea", "related", "neighborhood", "no_match"]


class CandidateMatch(BaseModel):
    model_config = ConfigDict(extra="forbid")
    pr_number: int
    tier: int = Field(ge=1, le=2)
    verdict: Verdict
    hunk_citation: str = ""


class MatchOutput(BaseModel):
    model_config = ConfigDict(extra="forbid")
    matches: list[CandidateMatch] = Field(default_factory=list)


class _RunnerFn(Protocol):
    def __call__(
        self,
        *,
        candidate_id: str,
        prompt: str,
        schema_text: str,
        repo_path: Path,
        max_turns: int,
        wallclock_s: int,
    ) -> CandidateAgentRunResult: ...


# ── Prompt ────────────────────────────────────────────────────────────


PROMPT_TEMPLATE = """\
You are evaluating whether any of the candidate PRs addressed the same
performance issue as a discovery finding.

# The finding

```json
{finding_json}
```

# Candidate PRs

For each candidate, you have its title, body excerpt, and the unified
diff slice covering only the file the finding targets.

{candidates_block}

# Your task

For each candidate PR (by `pr_number`), produce a verdict:

- `same_idea`    — the diff makes essentially the same change the
                   finding proposes (or a clear superset).
- `related`      — diff touches the same code with a related but
                   distinct change.
- `neighborhood` — diff touches the same file but unrelated code.
- `no_match`     — diff has nothing to do with the finding.

For `same_idea` and `related`, include a one-line `hunk_citation`
(max ~120 chars). Otherwise leave empty.

Tier hint: tier 1 = diff hunk inside the finding's symbol;
tier 2 = same file, different function.

Return a single JSON object:

```json
{{
  "matches": [
    {{"pr_number": <int>, "tier": <1|2>, "verdict": "<same_idea|related|neighborhood|no_match>", "hunk_citation": "<string>"}},
    ...
  ]
}}
```

One entry per candidate PR shown above. Don't invent PR numbers.
"""


def _format_candidates(candidates: list[dict]) -> str:
    parts: list[str] = []
    for c in candidates:
        parts.append(f"## PR #{c['pr_number']}  (tier {c['tier']})\n")
        parts.append(f"**Title:** {c['title']}\n")
        if c.get("body_excerpt"):
            parts.append(f"**Body excerpt:**\n```\n{c['body_excerpt']}\n```\n")
        parts.append(f"**Diff (file-filtered):**\n```diff\n{c['file_diff']}\n```\n")
    return "\n".join(parts)


def _truncate_diff(text: str, max_kb: int) -> str:
    max_chars = max_kb * 1024
    if len(text) <= max_chars:
        return text
    return text[:max_chars] + f"\n... [truncated, original {len(text)//1024} KB]\n"


def _build_finding_view(finding: dict) -> dict:
    cand = finding["candidate"]
    chg = finding.get("change") or {}
    return {
        "id": cand["id"],
        "file": _norm_path(cand.get("file", "")),
        "symbol": cand.get("symbol", ""),
        "description": cand.get("description", ""),
        "current_approach": cand.get("current_approach", ""),
        "evolve_rationale": cand.get("evolve_rationale", ""),
        "change_mechanism": chg.get("mechanism", ""),
        "expected_effect": chg.get("expected_effect", ""),
    }


def prompt_sha256() -> str:
    payload = {
        "template": PROMPT_TEMPLATE,
        "schema": MatchOutput.model_json_schema(),
        "version": PROMPT_VERSION,
    }
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, ensure_ascii=False, separators=(",", ":")).encode()
    ).hexdigest()


# ── Orchestration ────────────────────────────────────────────────────


@dataclass(frozen=True)
class MatchHandle:
    out_dir: Path
    report_path: Path
    n_findings: int
    n_judged: int
    same_idea: int
    related: int
    neighborhood: int
    no_match: int
    tier_2_yield: int


def run_matching(
    *,
    findings_path: Path,
    bench_run_dir: Path,
    experiment_id: str,
    judge_model: str = "sonnet",
    runner: _RunnerFn | None = None,
    repo_path: Path | None = None,
    data_root_override: Path | None = None,
    wallclock_s: int = DEFAULT_WALLCLOCK_S,
    max_turns: int = DEFAULT_MAX_TURNS,
) -> MatchHandle:
    """Run matching against a bench run's filtered view.

    `bench_run_dir` is the run dir produced by `repo-bench
    run` (contains `view/prs.jsonl`, `snapshot.json`).
    """
    runner = runner or _run_claude_default
    repo_path = repo_path or Path.cwd()
    cache_root = data_root_override or data_root()

    findings = json.loads(findings_path.read_text(encoding="utf-8"))

    view_path = bench_run_dir / "view" / "prs.jsonl"
    snapshot_path = bench_run_dir / "snapshot.json"
    if not view_path.exists():
        raise FileNotFoundError(f"view not found: {view_path}")
    if not snapshot_path.exists():
        raise FileNotFoundError(f"snapshot not found: {snapshot_path}")

    snapshot = json.loads(snapshot_path.read_text(encoding="utf-8"))
    window_id = snapshot["window_id"]

    raw_path = raw_dir(window_id, root=cache_root) / "prs.jsonl"
    diffs_dir = raw_dir(window_id, root=cache_root) / "diffs"
    if not raw_path.exists():
        raise FileNotFoundError(f"raw scrape not found: {raw_path}")

    view_pr_numbers: set[int] = set()
    for line_ in view_path.open(encoding="utf-8"):
        line = line_.strip()
        if not line:
            continue
        view_pr_numbers.add(int(json.loads(line)["pr_number"]))

    raw_by_n: dict[int, RawPR] = {}
    for d in read_jsonl_lenient(raw_path):
        try:
            pr = RawPR.model_validate(d)
        except ValidationError:
            continue
        if pr.pr_number in view_pr_numbers:
            raw_by_n[pr.pr_number] = pr

    prs_by_file: dict[str, list[RawPR]] = {}
    for pr in raw_by_n.values():
        for fc in pr.files_changed:
            prs_by_file.setdefault(_norm_path(fc.path), []).append(pr)

    out_dir = bench_run_dir / "matching" / experiment_id
    out_dir.mkdir(parents=True, exist_ok=True)

    schema_text = json.dumps(
        MatchOutput.model_json_schema(),
        sort_keys=True, ensure_ascii=False, separators=(",", ":"),
    )
    p_sha = prompt_sha256()

    log.info(
        "match: findings=%d view=%d judge=%s prompt_sha=%s",
        len(findings), len(view_pr_numbers), judge_model, p_sha[:12],
    )

    per_finding: list[dict] = []
    n_judged = 0
    same_idea_total = 0
    related_total = 0
    neighborhood_total = 0
    no_match_total = 0
    tier_2_yield = 0

    for f in findings:
        cand = f["candidate"]
        cid = cand["id"]
        file = _norm_path(cand.get("file", ""))
        symbol = cand.get("symbol", "")

        same_file_prs = prs_by_file.get(file, [])
        candidates_with_diff: list[dict] = []
        for pr in same_file_prs:
            diff_path = diffs_dir / f"{pr.pr_number}.diff"
            if not diff_path.exists():
                continue
            slices = parse_diff_by_file(diff_path.read_text(encoding="utf-8", errors="replace"))
            slc = slices.get(file)
            if slc is None:
                continue
            tier = 1 if hunk_touches_symbol(slc.hunks, symbol) else 2
            candidates_with_diff.append({
                "pr_number": pr.pr_number,
                "tier": tier,
                "title": pr.title,
                "body_excerpt": (pr.body or "")[:500],
                "file_diff": _truncate_diff(slc.text, DIFF_MAX_KB_PER_PR),
            })

        tier1 = [c for c in candidates_with_diff if c["tier"] == 1]
        tier2 = [c for c in candidates_with_diff if c["tier"] == 2][:TIER2_CAP_PER_FINDING]
        candidates = tier1 + tier2

        if not candidates:
            per_finding.append({
                "finding_id": cid,
                "file": file,
                "symbol": symbol,
                "n_tier1_candidates": 0,
                "n_tier2_candidates": 0,
                "matches": [],
                "skip_reason": "no_file_match",
            })
            no_match_total += 1
            continue

        prompt = PROMPT_TEMPLATE.format(
            finding_json=json.dumps(_build_finding_view(f), indent=2),
            candidates_block=_format_candidates(candidates),
        )
        log.info(
            "match: judging finding=%s candidates=%d (tier1=%d tier2=%d)",
            cid, len(candidates), len(tier1), len(tier2),
        )
        result = runner(
            candidate_id=f"match-{cid}",
            prompt=prompt,
            schema_text=schema_text,
            repo_path=repo_path,
            max_turns=max_turns,
            wallclock_s=wallclock_s,
        )
        if result.error is not None or result.structured_output is None:
            log.warning(
                "match: finding=%s judge failed: %s",
                cid, result.error or "no structured output",
            )
            per_finding.append({
                "finding_id": cid, "file": file, "symbol": symbol,
                "n_tier1_candidates": len(tier1),
                "n_tier2_candidates": len(tier2),
                "matches": [],
                "skip_reason": f"judge_error: {result.error or 'no structured output'}",
            })
            continue
        try:
            output = MatchOutput.model_validate(result.structured_output)
        except ValidationError as e:
            log.warning("match: finding=%s invalid output: %s", cid, e)
            continue
        n_judged += 1

        verdicts_by_tier: dict[int, list[Verdict]] = {1: [], 2: []}
        for m in output.matches:
            verdicts_by_tier[m.tier].append(m.verdict)
            if m.verdict == "same_idea":
                same_idea_total += 1
            elif m.verdict == "related":
                related_total += 1
            elif m.verdict == "neighborhood":
                neighborhood_total += 1
            else:
                no_match_total += 1

        t1_strong = any(v in ("same_idea", "related") for v in verdicts_by_tier[1])
        t2_strong = any(v in ("same_idea", "related") for v in verdicts_by_tier[2])
        if t2_strong and not t1_strong:
            tier_2_yield += 1

        per_finding.append({
            "finding_id": cid,
            "file": file,
            "symbol": symbol,
            "n_tier1_candidates": len(tier1),
            "n_tier2_candidates": len(tier2),
            "matches": [m.model_dump() for m in output.matches],
        })

    hit_strict = sum(
        1 for f in per_finding
        if any(m.get("verdict") == "same_idea" for m in f.get("matches", []))
    )
    hit_loose = sum(
        1 for f in per_finding
        if any(m.get("verdict") in ("same_idea", "related") for m in f.get("matches", []))
    )

    summary = {
        "n_findings": len(findings),
        "n_judged": n_judged,
        "verdicts": {
            "same_idea": same_idea_total,
            "related": related_total,
            "neighborhood": neighborhood_total,
            "no_match": no_match_total,
        },
        "per_finding_hit": {
            "strict_same_idea": hit_strict,
            "strict_share": hit_strict / len(findings) if findings else 0.0,
            "loose_same_idea_or_related": hit_loose,
            "loose_share": hit_loose / len(findings) if findings else 0.0,
        },
        "tier_2_yield": tier_2_yield,
        "tier_2_yield_share": tier_2_yield / len(findings) if findings else 0.0,
    }

    report = {
        "experiment_id": experiment_id,
        "window_id": window_id,
        "judge_model": judge_model,
        "prompt_version": PROMPT_VERSION,
        "prompt_sha256": p_sha,
        "findings_path": str(findings_path),
        "scored_at": datetime.now(timezone.utc).isoformat(),
        "summary": summary,
        "per_finding": per_finding,
    }
    report_path = out_dir / "match_report.json"
    atomic_write_text(report_path, json.dumps(report, indent=2) + "\n")

    log.info(
        "match: wrote %s — strict=%d/%d loose=%d/%d tier2_yield=%d judged=%d",
        report_path,
        hit_strict, len(findings), hit_loose, len(findings),
        tier_2_yield, n_judged,
    )

    return MatchHandle(
        out_dir=out_dir,
        report_path=report_path,
        n_findings=len(findings),
        n_judged=n_judged,
        same_idea=same_idea_total,
        related=related_total,
        neighborhood=neighborhood_total,
        no_match=no_match_total,
        tier_2_yield=tier_2_yield,
    )


__all__ = [
    "MatchHandle",
    "MatchOutput",
    "PROMPT_VERSION",
    "Verdict",
    "parse_diff_by_file",
    "hunk_touches_symbol",
    "prompt_sha256",
    "run_matching",
]
