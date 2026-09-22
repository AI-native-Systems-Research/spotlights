"""Offline determinism analysis over `--deep-research-repeat N` sidecars.

`--deep-research-repeat N` runs step 3 N times per module and writes each pass
to `module_deep_research.json` (run 0, canonical) + `module_deep_research.{i}.json`
(i=1..N-1). This module compares those passes to quantify run-to-run drift.

Paper matching is fuzzy, done by Claude (`claude_matcher`): the same paper can
surface with a reformatted title or an arXiv-vs-published URL across runs, and a
raw `(title, url)` string diff over-counts those cosmetic differences as
disagreements. The matcher is injectable so unit tests pass a deterministic fake
instead of spawning a real agent.

Two views come out (see `DeterminismReport`):
  * voting histogram — non-normalized: for each distinct paper, how many of the
    N runs surfaced it. `votes == N` = every run agreed (stable); `votes == 1` =
    a paper only one run ever found (flaky).
  * all-pairs confusion matrix — aggregated over every ordered run pair (i, j):
    TP  = paper present in both i and j
    FP  = present in j, absent in i  }  equal by symmetry when summed over all
    FN  = present in i, absent in j  }  ordered pairs — reported separately anyway
    TN  = 0 (we never enumerate the universe of papers correctly absent from both)
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

from spotlights_engine.schemas.finding import Finding
from spotlights_engine.schemas.pipeline import ModuleDeepResearchOutput

# A cluster is the set of (run_index, finding_index) refs the matcher judged to
# be the same paper. A Matcher maps per-run finding lists to those clusters.
PaperRef = tuple[int, int]
Cluster = list[PaperRef]
Matcher = Callable[[list[list[Finding]]], list[Cluster]]


def load_repeat_runs(module_dir: Path) -> list[ModuleDeepResearchOutput]:
    """Load run 0 (`module_deep_research.json`) + every indexed repeat sidecar
    (`module_deep_research.{i}.json`) in ascending `i`, as a list indexed by run.

    Raises FileNotFoundError if the canonical run-0 sidecar is missing.
    """
    canonical = module_dir / "module_deep_research.json"
    if not canonical.exists():
        raise FileNotFoundError(f"no canonical deep-research sidecar: {canonical}")

    def _load(p: Path) -> ModuleDeepResearchOutput:
        payload = json.loads(p.read_text(encoding="utf-8"))
        return ModuleDeepResearchOutput.model_validate(payload["output"])

    runs = [_load(canonical)]
    indexed = sorted(
        module_dir.glob("module_deep_research.[0-9]*.json"),
        key=lambda p: int(p.stem.split(".")[-1]),
    )
    runs.extend(_load(p) for p in indexed)
    return runs


# --------------------------------------------------------------------------- #
# Report
# --------------------------------------------------------------------------- #


@dataclass
class PaperVote:
    """One distinct paper (a matcher cluster) and which runs surfaced it."""

    title: str
    url: str
    votes: int  # number of distinct runs containing this paper (1..N)
    runs: list[int]  # sorted distinct run indices


@dataclass
class ConfusionMatrix:
    tp: int = 0
    fp: int = 0
    fn: int = 0
    tn: int = 0  # always 0 — no negative universe

    @property
    def precision(self) -> float:
        d = self.tp + self.fp
        return self.tp / d if d else 1.0

    @property
    def recall(self) -> float:
        d = self.tp + self.fn
        return self.tp / d if d else 1.0

    @property
    def determinism(self) -> float:
        """Jaccard-style agreement: TP / (TP + FP + FN). 1.0 = fully stable."""
        d = self.tp + self.fp + self.fn
        return self.tp / d if d else 1.0


@dataclass
class DeterminismReport:
    n_runs: int
    n_papers: int  # distinct papers across all runs (cluster count)
    histogram: dict[int, int]  # votes -> #papers with exactly that many votes
    papers: list[PaperVote]  # flakiest first (fewest votes first)
    confusion: ConfusionMatrix
    issues: list[str] = field(default_factory=list)

    def render(self) -> str:
        lines = [
            f"Determinism over {self.n_runs} runs — {self.n_papers} distinct papers",
            "",
            "Voting histogram (papers appearing in exactly k runs):",
        ]
        for k in range(self.n_runs, 0, -1):
            bar = "#" * self.histogram.get(k, 0)
            lines.append(f"  {k:>2}/{self.n_runs} runs: {self.histogram.get(k, 0):>3} {bar}")
        c = self.confusion
        lines += [
            "",
            "All-pairs confusion matrix (ordered run pairs):",
            f"  TP={c.tp}  FP={c.fp}  FN={c.fn}  TN={c.tn}",
            f"  precision={c.precision:.3f}  recall={c.recall:.3f}  "
            f"determinism={c.determinism:.3f}",
        ]
        flaky = [p for p in self.papers if p.votes < self.n_runs]
        if flaky:
            lines += ["", f"Flaky papers ({len(flaky)}, not in every run):"]
            for p in flaky:
                lines.append(f"  [{p.votes}/{self.n_runs}] {p.title}  ({p.url})")
        for issue in self.issues:
            lines.append(f"  ! {issue}")
        return "\n".join(lines)


def build_report(
    runs: list[ModuleDeepResearchOutput],
    *,
    matcher: Matcher,
) -> DeterminismReport:
    """Compute the voting histogram + all-pairs confusion matrix for N runs.

    `matcher` clusters findings across runs into distinct papers (fuzzy match).
    """
    n = len(runs)
    findings_by_run = [list(r.findings) for r in runs]
    total_findings = sum(len(f) for f in findings_by_run)

    issues: list[str] = []
    clusters = matcher(findings_by_run) if total_findings else []
    clusters, cover_issues = _validate_and_complete_clusters(clusters, findings_by_run)
    issues.extend(cover_issues)

    papers: list[PaperVote] = []
    histogram: Counter[int] = Counter()
    confusion = ConfusionMatrix()
    for cluster in clusters:
        run_idxs = sorted({r for (r, _f) in cluster})
        votes = len(run_idxs)
        histogram[votes] += 1
        rep = _representative(cluster, findings_by_run)
        papers.append(PaperVote(title=rep.title, url=rep.url, votes=votes, runs=run_idxs))
        # Ordered-pair contributions for this paper present in `votes` of n runs.
        m = votes
        confusion.tp += m * (m - 1)  # both present
        confusion.fp += (n - m) * m  # absent in i, present in j
        confusion.fn += m * (n - m)  # present in i, absent in j

    papers.sort(key=lambda p: (p.votes, p.title))
    return DeterminismReport(
        n_runs=n,
        n_papers=len(clusters),
        histogram=dict(histogram),
        papers=papers,
        confusion=confusion,
        issues=issues,
    )


def _representative(cluster: Cluster, findings_by_run: list[list[Finding]]) -> Finding:
    r, f = cluster[0]
    return findings_by_run[r][f]


def _validate_and_complete_clusters(
    clusters: list[Cluster],
    findings_by_run: list[list[Finding]],
) -> tuple[list[Cluster], list[str]]:
    """Drop out-of-range refs, dedupe, and add every unmatched finding as its own
    singleton cluster so the report always covers 100% of findings even if the
    matcher omits some."""
    issues: list[str] = []
    seen: set[PaperRef] = set()
    clean: list[Cluster] = []
    for cluster in clusters:
        members: Cluster = []
        for ref in cluster:
            r, f = ref
            if not (0 <= r < len(findings_by_run) and 0 <= f < len(findings_by_run[r])):
                issues.append(f"matcher returned out-of-range ref {ref}")
                continue
            if ref in seen:
                issues.append(f"matcher returned duplicate ref {ref}")
                continue
            seen.add(ref)
            members.append(ref)
        if members:
            clean.append(members)

    for r, findings in enumerate(findings_by_run):
        for f in range(len(findings)):
            if (r, f) not in seen:
                seen.add((r, f))
                clean.append([(r, f)])
                issues.append(f"finding ({r},{f}) unmatched — treated as singleton")
    return clean, issues


# --------------------------------------------------------------------------- #
# Claude matcher (default)
# --------------------------------------------------------------------------- #

_CLUSTER_SCHEMA = json.dumps(
    {
        "type": "object",
        "properties": {
            "clusters": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "members": {"type": "array", "items": {"type": "string"}}
                    },
                    "required": ["members"],
                    "additionalProperties": False,
                },
            }
        },
        "required": ["clusters"],
        "additionalProperties": False,
    }
)


def _render_matcher_prompt(findings_by_run: list[list[Finding]]) -> str:
    lines = [
        "You are given lists of research papers found across several independent "
        "runs of the same search. Group entries that refer to the SAME paper, even "
        "when the title is reformatted or the URL differs (arXiv vs published, "
        "http vs https, trailing slash, DOI vs landing page).",
        "",
        "Each entry has a stable id `rR_fF` (run R, finding F). Return clusters; "
        "each cluster lists the ids of all entries that are the same paper. Every "
        "id must appear in exactly one cluster. A paper unique to one run is a "
        "singleton cluster.",
        "",
    ]
    for r, findings in enumerate(findings_by_run):
        lines.append(f"## Run {r}")
        if not findings:
            lines.append("  (no findings)")
        for f, finding in enumerate(findings):
            lines.append(f"  - r{r}_f{f}: {finding.title!r} | {finding.url}")
        lines.append("")
    return "\n".join(lines)


def _parse_cluster_members(members: list[str]) -> Cluster:
    refs: Cluster = []
    for m in members:
        m = m.strip()
        if not (m.startswith("r") and "_f" in m):
            continue
        run_part, fin_part = m[1:].split("_f", 1)
        try:
            refs.append((int(run_part), int(fin_part)))
        except ValueError:
            continue
    return refs


def _clean_env() -> dict[str, str]:
    env = os.environ.copy()
    for key in list(env):
        # NOTE: do NOT strip ANTHROPIC_BASE_URL — the IBM-LiteLLM wiring points
        # claude at the proxy via that var; dropping it makes claude fall back to
        # the default endpoint with no key → 401 "Not logged in" (exit 1).
        if key.startswith(("SPOTLIGHTS_", "VSCODE_")) or key in {
            "HTTP_PROXY",
            "HTTPS_PROXY",
            "ALL_PROXY",
        }:
            env.pop(key)
    return env


def claude_matcher(
    findings_by_run: list[list[Finding]],
    *,
    claude_model: str | None = None,
    wallclock_s: int = 300,
    max_turns: int = 3,
) -> list[Cluster]:
    """Default `Matcher`: one `claude -p --json-schema` session clusters the
    papers across runs. Kept subprocess-shaped like the step-4 runner so it works
    under the same IBM-LiteLLM env wiring."""
    claude = shutil.which("claude") or "claude"
    argv = [
        claude,
        "-p",
        "--output-format",
        "stream-json",
        "--verbose",
        "--json-schema",
        _CLUSTER_SCHEMA,
        "--permission-mode",
        "plan",
        "--max-turns",
        str(max_turns),
    ]
    if claude_model:
        argv += ["--model", claude_model]
    completed = subprocess.run(
        argv,
        input=_render_matcher_prompt(findings_by_run).encode("utf-8"),
        capture_output=True,
        env=_clean_env(),
        timeout=wallclock_s,
        check=False,
    )
    if completed.returncode != 0:
        tail = (completed.stderr or b"")[-500:].decode("utf-8", "replace")
        raise RuntimeError(f"claude matcher exit={completed.returncode}: {tail!r}")

    payload = _extract_structured(completed.stdout)
    return [_parse_cluster_members(c.get("members", [])) for c in payload.get("clusters", [])]


def _extract_structured(stdout: bytes) -> dict:
    last: dict | None = None
    for line in stdout.splitlines():
        line = line.strip()
        if not line:
            continue
        obj = json.loads(line)
        if isinstance(obj, dict):
            last = obj
    if not last or last.get("type") != "result":
        raise RuntimeError("claude matcher: no terminal result event")
    structured = last.get("structured_output")
    if isinstance(structured, dict):
        return structured
    fallback = last.get("result")
    if isinstance(fallback, str) and fallback.strip():
        return json.loads(fallback)
    raise RuntimeError("claude matcher: no structured_output")


__all__ = [
    "PaperRef",
    "Cluster",
    "Matcher",
    "PaperVote",
    "ConfusionMatrix",
    "DeterminismReport",
    "load_repeat_runs",
    "build_report",
    "claude_matcher",
]
