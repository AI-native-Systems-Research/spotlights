"""Build a `SpotlightReport` from signal-pipeline run-dir artifacts.

Per `docs/specs/spotlight_report.md` section 5 PR 2: the signal pipeline
emits a unified `SpotlightReport` directly when the runner finishes,
alongside the existing per-stage artifacts (`01_signals.json`,
`03_candidates.json`, `04_changes/<id>.json`, etc.).

This module is the boundary translation: it walks the parsed
`Signals.anomalies` + `list[CandidateDraft]` from stage 03 +
`dict[id, Change]` from stage 04 and produces:

- `Anomaly` records (closed shape, `anomaly_id` / `type` / `description` /
  `confidence` / `evidence_pointer` / `magnitude` / `estimated_severity`,
  with the upstream descriptive id renumbered to `anom-signal-NNNN` per
  spec §10).
- `Candidate` records, one per draft, wrapping the draft's flat location
  fields into a single `CodeLocation` containing one `CodeSpan`. `origin`
  is hard-coded to `"telemetry_anomaly"`. `module_qualified_name` is
  resolved by deepest-prefix match against `ProjectTree.walk()`.
- `Proposal` records, one per change, with `source="telemetry_anomaly"`,
  `anomaly_ref_ids` populated from the upstream `CandidateDraft.anomaly_refs`,
  and the structured fields (`mechanism` / `required_changes` / etc.) lifted
  from the `Change`.

IDs are renumbered globally in deterministic walk order: candidates take
the order they appear in the stage 03 artifact; proposals follow each
candidate. ID prefixes are `cand-signal-NNNN` / `prop-signal-NNNN`
(4-digit zero-padded; `signal` is the fixed pipeline segment per #28).

The legacy `Change.change_type` field is intentionally NOT carried into
the unified report (per spec section 4.10 -- one-pipeline + domain-locked
vocabulary). It survives on the legacy `04_changes/<id>.json` artifact for
anyone reading those directly.
"""

from __future__ import annotations

import json
from typing import Iterable

from spotlights_engine.schemas.anomaly import Anomaly
from spotlights_engine.schemas.candidate import (
    Candidate,
    CodeLocation,
    CodeSpan,
)
from spotlights_engine.schemas.common import SpotlightContext
from spotlights_engine.schemas.pipeline import RunInfo, SpotlightReport
from spotlights_engine.schemas.project import ProjectTree
from spotlights_engine.schemas.proposal import Proposal
from spotlights_engine.signal_pipeline.layout import RunDirLayout
from spotlights_engine.signal_pipeline.schemas import (
    AnomalyLite,
    CandidateDraft,
    Change,
    Signals,
)


__all__ = ["build_spotlight_report", "emit_spotlight_report"]


# Fixed pipeline segment for the unified id pattern (`<type>-signal-NNNN`),
# per the resolution in #28.
_SEGMENT = "signal"


# ---- Module assignment ------------------------------------------------------


def _resolve_module(file_path: str, project_tree: ProjectTree) -> str | None:
    """Return the `module_qualified_name` whose source path is the deepest
    prefix of `file_path`, or `None` if no module matches.

    Uses `ProjectTree.walk()` so the qualified-name encoding (slash-separated,
    source_root-relative per Ophir's `qualified-name change`) stays in lockstep
    with the rest of the codebase. The `file_path` is repo-relative as written
    by stage 03; we strip leading `./` and slashes for stable comparison.
    """
    norm = file_path.strip().lstrip("./").lstrip("/")
    if not norm:
        return None

    best_qn: str | None = None
    best_depth = -1
    for qn, module in project_tree.walk():
        module_path = module.path.strip().lstrip("./").lstrip("/")
        if not module_path:
            continue
        # Match if file lives strictly under the module path (or equals it,
        # though candidates live in files, not in the module dir itself).
        if norm == module_path or norm.startswith(module_path + "/"):
            depth = module_path.count("/")
            if depth > best_depth:
                best_depth = depth
                best_qn = qn
    return best_qn


# ---- Translation primitives -------------------------------------------------


_VALID_SEVERITIES = ("high", "medium", "low")


def _normalize_estimated_severity(value: object) -> str | None:
    """Coerce an agent-emitted severity label to a valid `AnomalySeverity`.

    Accepts the upstream prompt's preferred field name `estimated_severity`
    AND the legacy `severity` key (older prompts / hand-written fixtures
    still use it). Lower-cases the value so case differences don't
    disqualify a valid label, and drops anything outside the closed set
    so the boundary stays robust against agent off-script values.
    """
    if value is None:
        return None
    if not isinstance(value, str):
        return None
    norm = value.strip().lower()
    return norm if norm in _VALID_SEVERITIES else None


def _clamp_confidence(value: object) -> float | None:
    """Coerce confidence to a float in [0, 1] or None.

    Production telemetry sometimes emits string confidence (`"0.6"`); coerce
    anything that looks numeric, drop NaN / out-of-range values to None so the
    closed `Anomaly.confidence: float | None` stays valid.
    """
    if value is None:
        return None
    try:
        f = float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None
    if f != f or f < 0.0 or f > 1.0:  # NaN or OOR
        return None
    return f


def _anomaly_from_lite(lite: AnomalyLite, *, new_id: str) -> Anomaly:
    """Translate a stage-01 `AnomalyLite` (extra="allow") into the closed
    `Anomaly` shape. Pulls fields the production prompt + design doc spec
    (see `runs/max1-smoke/01_signals.json` for the real shape), defaulting
    to empty strings for the optional text fields.

    `new_id` is the renumbered `anom-signal-NNNN` id minted at the
    report-build boundary; the upstream descriptive id from `lite` is
    discarded so the unified `Anomaly.anomaly_id` pattern (spec §10)
    is honored without leaking pipeline-internal naming.
    """
    extras = lite.model_dump()
    return Anomaly(
        anomaly_id=new_id,
        type=lite.type,
        description=lite.description or "",
        estimated_severity=_normalize_estimated_severity(
            extras.get("estimated_severity") or extras.get("severity")
        ),
        confidence=_clamp_confidence(extras.get("confidence")),
        evidence_pointer=str(extras.get("evidence_pointer") or ""),
        magnitude=str(extras.get("magnitude") or ""),
    )


def _candidate_from_draft(
    draft: CandidateDraft,
    *,
    new_id: str,
    project_tree: ProjectTree,
    proposals: list[Proposal],
) -> Candidate:
    span = CodeSpan(
        line_start=draft.line_start,
        line_end=draft.line_end,
        symbol=draft.symbol,
        kind=draft.kind,
    )
    location = CodeLocation(file=draft.file, spans=[span])
    return Candidate(
        id=new_id,
        module_qualified_name=_resolve_module(draft.file, project_tree),
        origin="telemetry_anomaly",
        locations=[location],
        description=draft.description,
        current_approach=draft.current_approach,
        evolve_rationale=draft.evolve_rationale,
        estimated_impact=draft.estimated_impact,
        estimated_impact_explanation=draft.estimated_impact_explanation,
        proposals=proposals,
    )


def _proposal_from_change(
    *,
    new_id: str,
    candidate_id: str,
    draft: CandidateDraft,
    change: Change,
    anomaly_id_remap: dict[str, str],
) -> Proposal:
    """Build a `Proposal` from a stage-04 `Change` + its source draft.

    `title` is synthesized from the candidate symbol (the legacy
    `Change.change_type` is dropped from the unified shape per spec
    section 4.10; consumers wanting the categorical label read the legacy
    `04_changes/<id>.json` artifact directly).
    """
    title = f"Change in {draft.symbol}".strip() or f"Change for {candidate_id}"
    description = change.mechanism
    if change.expected_effect:
        description = f"{description}\n\nExpected: {change.expected_effect}"
    rationale = draft.evolve_rationale
    remapped_refs: list[str] | None
    if draft.anomaly_refs:
        remapped_refs = [
            anomaly_id_remap[ref] for ref in draft.anomaly_refs if ref in anomaly_id_remap
        ] or None
    else:
        remapped_refs = None
    return Proposal(
        id=new_id,
        source="telemetry_anomaly",
        anomaly_ref_ids=remapped_refs,
        finding_ref_id=None,
        author=None,
        title=title,
        description=description,
        rationale=rationale,
        mechanism=change.mechanism,
        required_changes=change.required_changes,
        expected_effect=change.expected_effect,
        evaluation_metric=change.evaluation_metric,
    )


# ---- Top-level builder ------------------------------------------------------


def build_spotlight_report(
    *,
    signals: Signals,
    project_tree: ProjectTree,
    drafts: Iterable[CandidateDraft],
    changes: dict[str, Change],
    context: SpotlightContext,
    run_info: RunInfo,
) -> SpotlightReport:
    """Assemble a `SpotlightReport` from in-memory parsed artifacts.

    Walk order is deterministic: drafts iterate in the order they appear
    in stage 03's artifact; each draft's stage-04 `Change` (if present)
    becomes its single proposal. Both ID streams (`cand-signal-NNNN`,
    `prop-signal-NNNN`) are renumbered globally so the report can be
    flat-iterated without cross-module collisions.
    """
    # Renumber upstream anomaly ids (e.g. `A1-queue-dominated-latency`) to
    # the unified `anom-signal-NNNN` shape per spec §10. `anomaly_id_remap`
    # is then used to translate each draft's `anomaly_refs` so proposals'
    # `anomaly_ref_ids` continue to join cleanly to the renumbered anomalies.
    anomaly_id_remap: dict[str, str] = {
        lite.anomaly_id: f"anom-{_SEGMENT}-{idx:04d}"
        for idx, lite in enumerate(signals.anomalies, start=1)
    }
    anomalies = [
        _anomaly_from_lite(lite, new_id=anomaly_id_remap[lite.anomaly_id])
        for lite in signals.anomalies
    ]

    drafts_list = list(drafts)
    cand_idx = 0
    prop_idx = 0
    candidates: list[Candidate] = []
    for draft in drafts_list:
        cand_idx += 1
        new_cand_id = f"cand-{_SEGMENT}-{cand_idx:04d}"
        proposals: list[Proposal] = []
        change = changes.get(draft.id)
        if change is not None:
            prop_idx += 1
            new_prop_id = f"prop-{_SEGMENT}-{prop_idx:04d}"
            proposals.append(
                _proposal_from_change(
                    new_id=new_prop_id,
                    candidate_id=new_cand_id,
                    draft=draft,
                    change=change,
                    anomaly_id_remap=anomaly_id_remap,
                )
            )
        candidates.append(
            _candidate_from_draft(
                draft,
                new_id=new_cand_id,
                project_tree=project_tree,
                proposals=proposals,
            )
        )

    return SpotlightReport(
        project_tree=project_tree,
        context=context,
        candidates=candidates,
        findings=[],
        anomalies=anomalies,
        run=run_info,
        issues=[],
    )


# ---- Disk I/O ---------------------------------------------------------------


def _load_signals(layout: RunDirLayout) -> Signals | None:
    path = layout.stage_artifact("01", shape="single")
    if not path.exists():
        return None
    return Signals.model_validate_json(path.read_text(encoding="utf-8"))


def _load_project_tree(layout: RunDirLayout) -> ProjectTree | None:
    path = layout.stage_artifact("02", shape="single")
    if not path.exists():
        return None
    return ProjectTree.model_validate_json(path.read_text(encoding="utf-8"))


def _load_drafts(layout: RunDirLayout) -> list[CandidateDraft] | None:
    path = layout.stage_artifact("03", shape="single")
    if not path.exists():
        return None
    raw = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw, list):
        return None
    return [CandidateDraft.model_validate(item) for item in raw]


def _load_changes(layout: RunDirLayout) -> dict[str, Change]:
    """Load every stage-04 change spec on disk, keyed by candidate id.

    Missing per-id files are silently skipped -- a partial run that ended
    before stage 04 finished still produces a valid report (proposals are
    omitted for the candidates without a change spec).
    """
    fanout_dir = layout.stage_artifact("04", shape="fanout")
    if not fanout_dir.exists():
        return {}
    out: dict[str, Change] = {}
    for entry in sorted(fanout_dir.iterdir(), key=lambda p: p.name):
        if entry.name == "_manifest.json":
            continue
        if entry.suffix != ".json" or not entry.is_file():
            continue
        try:
            raw = json.loads(entry.read_text(encoding="utf-8"))
            change = Change.model_validate(raw)
        except Exception:
            continue
        out[entry.stem] = change
    return out


_DEFAULT_SIGNAL_OBJECTIVE = "Address performance issues surfaced by telemetry signals"


def _default_context(signals: Signals) -> SpotlightContext:
    """Fallback `SpotlightContext` when the caller didn't supply one.

    Used by `emit_spotlight_report(layout, ..., context=None)` and by
    `run_pipeline` when `SignalPipelineInput.context is None` (today's
    `signal-pipeline` CLI default — the standalone CLI doesn't expose
    `--objective` and never sets the field). Programmatic callers
    (notably the unified runner) thread an explicit
    `SpotlightContext` so signal- and DR-side reports agree on
    `objective` for clean merging — when they do, this fallback is
    not used.

    The unified `SpotlightReport` schema requires
    `SpotlightContext.objective` to be a (possibly empty) string;
    using a stable, descriptive default here keeps the standalone
    signal output self-describing.
    """
    return SpotlightContext(
        objective=_DEFAULT_SIGNAL_OBJECTIVE,
        workload_hints=[signals.workload.workload_id],
        validation_plan=[],
    )


def emit_spotlight_report(
    layout: RunDirLayout,
    *,
    run_id: str,
    started_at: str,
    finished_at: str | None = None,
    cost_usd: float | None = None,
    context: SpotlightContext | None = None,
) -> SpotlightReport | None:
    """Best-effort: build + write `<run_dir>/spotlight_report.json`.

    Returns the built report on success, or `None` if a precondition
    artifact (signals, project tree, or candidates) is missing on disk
    -- e.g. the run was scoped to stage 01 only.

    Idempotent: every call rebuilds and overwrites. Errors propagate to
    the caller (the runner catches them as best-effort issues, mirroring
    `emit_findings`).
    """
    signals = _load_signals(layout)
    project_tree = _load_project_tree(layout)
    drafts = _load_drafts(layout)
    if signals is None or project_tree is None or drafts is None:
        return None

    changes = _load_changes(layout)

    run_info = RunInfo(
        pipeline="signal",
        run_id=run_id,
        started_at=started_at,
        finished_at=finished_at,
        cost_usd=cost_usd,
    )
    spot_context = context if context is not None else _default_context(signals)

    report = build_spotlight_report(
        signals=signals,
        project_tree=project_tree,
        drafts=drafts,
        changes=changes,
        context=spot_context,
        run_info=run_info,
    )

    target = layout.root / "spotlight_report.json"
    target.write_text(
        report.model_dump_json(indent=2, by_alias=True) + "\n",
        encoding="utf-8",
    )
    return report
