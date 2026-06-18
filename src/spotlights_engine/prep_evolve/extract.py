"""Stage A: build an `EvolveSpec` from a resolved selection.

`build_spec` is pure over `(result, module, candidate, findings, repo_path,
validated, revision, scope, direction)`. The only live-repo reader is
`validate_target.py`, whose output (validated line range, excerpt hash, git
revision) is folded in here.
"""

from __future__ import annotations

import re
from typing import Literal

from spotlights_engine.prep_evolve.resolve import LoadedResult
from spotlights_engine.prep_evolve.spec import (
    EvolveSpec,
    FindingRef,
    MainFile,
    ModuleInfo,
    Objective,
    Oracles,
    ProposalRef,
    RepoInfo,
    SourceRevision,
    Target,
)
from spotlights_engine.prep_evolve.validate_target import ValidatedCandidate
from spotlights_engine.schemas.candidate import Candidate
from spotlights_engine.schemas.common import SpotlightContext
from spotlights_engine.schemas.finding import Finding
from spotlights_engine.schemas.project import Module

Scope = Literal["candidate", "module-main-files"]
Direction = Literal["minimize", "maximize"]

# --- Oracle + direction parsing (plan §7.1) -------------------------------

# Test-file / test-command mentions in the rationale.
_TEST_PATH_RE = re.compile(
    r"(?:tests?/[\w./-]+?\.py|[\w./-]+?_test\.py|test_[\w./-]+?\.py)",
)
_PYTEST_RE = re.compile(r"\bpytest\b[^\n`]*", re.IGNORECASE)
_SENTENCE_BOUNDARY_RE = re.compile(r"(?<=[.!?])\s+(?=[A-Z][A-Za-z])")

# Performance metric phrases worth surfacing. Deliberately specific (the plan
# names TTFT/TPOT/latency/throughput) so noisy words like "memory locality" in
# a rationale don't masquerade as the performance oracle.
_METRIC_RE = re.compile(
    r"\b(TTFT|TPOT|latency|throughput|tokens?/s(?:ec)?|QPS|p\d{2})\b",
    re.IGNORECASE,
)

_MIN_VERBS = ("reduce", "minimi", "lower", "decrease", "shrink", "cut", "less")
_MAX_VERBS = ("increase", "maximi", "raise", "improve throughput", "boost", "higher")


def parse_correctness_oracles(rationale: str) -> list[str]:
    """Best-effort extraction of runnable correctness oracle commands."""
    found: list[str] = []
    seen: set[str] = set()
    for match in _PYTEST_RE.finditer(rationale):
        # The rationale often continues with prose on the same line
        # ("pytest tests/foo.py -q. Performance oracle is ..."). Keep the
        # command-shaped first sentence, not the whole paragraph.
        token = _SENTENCE_BOUNDARY_RE.split(match.group(0).strip(), maxsplit=1)[0].rstrip(".,;:)")
        if token and token not in seen:
            seen.add(token)
            found.append(token)
    for match in _TEST_PATH_RE.finditer(rationale):
        path = match.group(0).strip().rstrip(".,;:)")
        if any(path in command for command in found):
            continue
        token = f"pytest {path}"
        if token not in seen:
            seen.add(token)
            found.append(token)
    return found


def parse_performance_oracle(
    rationale: str,
    objective: str,
    estimated_impact_explanation: str,
) -> str | None:
    """Best-effort performance-metric hint from rationale, then objective."""
    for source in (rationale, objective, estimated_impact_explanation):
        if source and _METRIC_RE.search(source):
            metrics: list[str] = []
            seen: set[str] = set()
            for m in _METRIC_RE.finditer(source):
                tok = m.group(0)
                key = tok.lower()
                if key not in seen:
                    seen.add(key)
                    metrics.append(tok)
            return ", ".join(metrics)
    return None


def infer_direction(objective: str) -> Direction:
    """Infer optimization direction from objective verbs; default `minimize`."""
    text = objective.lower()
    has_min = any(v in text for v in _MIN_VERBS)
    has_max = any(v in text for v in _MAX_VERBS)
    if has_min and not has_max:
        return "minimize"
    if has_max and not has_min:
        return "maximize"
    # Ambiguous (both or neither) -> default minimize (README notes this).
    return "minimize"


# --- Spec construction ----------------------------------------------------


def _build_oracles(candidate: Candidate, context: SpotlightContext) -> Oracles:
    return Oracles(
        correctness=parse_correctness_oracles(candidate.evolve_rationale),
        performance=parse_performance_oracle(
            candidate.evolve_rationale,
            context.objective,
            candidate.estimated_impact_explanation,
        ),
    )


def _candidate_target(
    candidate: Candidate,
    validated: ValidatedCandidate,
    oracles: Oracles,
) -> Target:
    return Target(
        scope_kind="candidate",
        candidate_id=candidate.id,
        file=candidate.file,
        symbol=candidate.symbol,
        kind=candidate.kind,
        line_start=validated.line_start,
        line_end=validated.line_end,
        source_excerpt_sha256=validated.source_excerpt_sha256,
        description=candidate.description,
        current_approach=candidate.current_approach,
        evolve_rationale=candidate.evolve_rationale,
        estimated_impact=candidate.estimated_impact,
        estimated_impact_explanation=candidate.estimated_impact_explanation,
        oracles=oracles,
    )


def _main_file_targets(module: Module, exclude_file: str) -> list[Target]:
    """De-duplicated whole-file scope targets from `module.main_files`.

    The candidate's own file is excluded (it is already the candidate target).
    """
    targets: list[Target] = []
    seen: set[str] = {exclude_file}
    for mf in module.main_files:
        if mf.path in seen:
            continue
        seen.add(mf.path)
        targets.append(
            Target(
                scope_kind="module_main_file",
                file=mf.path,
                role=mf.role,
                kind="region",
            )
        )
    return targets


def _ordered_findings(candidate: Candidate, findings: list[Finding]) -> list[FindingRef]:
    """Only findings linked to this candidate's deep-research proposals.

    Findings are attached at the module level, so the same list is shared by
    every candidate in the module. Including all of them buries the few that
    actually motivated this candidate under unrelated module-wide research, so
    we keep only the ones referenced by `candidate.deep_research_proposals`,
    in proposal order.
    """
    linked_ids = [p.finding_id for p in candidate.deep_research_proposals]
    by_id = {f.finding_id: f for f in findings}

    ordered: list[Finding] = []
    seen: set[str] = set()
    for fid in linked_ids:
        f = by_id.get(fid)
        if f is not None and f.finding_id not in seen:
            seen.add(f.finding_id)
            ordered.append(f)

    return [
        FindingRef(
            finding_id=f.finding_id,
            title=f.title,
            url=f.url,
            source_type=f.source_type,
            technique_summary=f.technique_summary,
            supporting_evidence=f.supporting_evidence,
        )
        for f in ordered
    ]


def _proposals(candidate: Candidate) -> list[ProposalRef]:
    refs: list[ProposalRef] = []
    for drp in candidate.deep_research_proposals:
        refs.append(
            ProposalRef(
                origin="deep_research",
                agent=drp.created_by,
                title=drp.title,
                detailed_description=drp.detailed_description,
                finding_id=drp.finding_id,
                rationale=drp.proposal_rationale,
            )
        )
    for ap in candidate.agent_proposals:
        refs.append(
            ProposalRef(
                origin="agent",
                agent=ap.agent_name,
                title=ap.title,
                detailed_description=ap.detailed_description,
                finding_id=None,
                rationale=ap.novelty_rationale,
            )
        )
    return refs


def build_spec(
    *,
    loaded: LoadedResult,
    module: Module,
    qn: str,
    candidate: Candidate,
    findings: list[Finding],
    repo_path: str,
    validated: ValidatedCandidate,
    revision: SourceRevision,
    scope: Scope,
    direction: Direction,
) -> EvolveSpec:
    """Assemble the agnostic `EvolveSpec` (pure)."""
    context = loaded.context
    repo = loaded.project_tree.repository

    oracles = _build_oracles(candidate, context)
    targets = [_candidate_target(candidate, validated, oracles)]
    if scope == "module-main-files":
        targets.extend(_main_file_targets(module, exclude_file=candidate.file))

    return EvolveSpec(
        run=RepoInfo(
            repo_name=repo.name,
            repo_path=repo_path,
            repo_summary=repo.summary,
        ),
        source_revision=revision,
        objective=Objective(
            goal=context.objective,
            workload_hints=list(context.workload_hints),
            direction=direction,
        ),
        module=ModuleInfo(
            qualified_name=qn,
            name=module.name,
            path=module.path,
            description=module.description,
            main_files=[MainFile(path=mf.path, role=mf.role) for mf in module.main_files],
        ),
        targets=targets,
        findings=_ordered_findings(candidate, findings),
        proposals=_proposals(candidate),
    )


__all__ = [
    "Direction",
    "Scope",
    "build_spec",
    "infer_direction",
    "parse_correctness_oracles",
    "parse_performance_oracle",
]
