"""Checkpoint layout, atomic writes, and resume-state inspection for the manager.

See plan §7 for the on-disk layout. The orchestrator never touches the
filesystem directly — it always goes through this module so write protocol
(payload before status; atomic temp-file swaps) is enforced in one place.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
import os
import shutil
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from spotlights_engine.agent_proposals import AgentProposalsConfig
from spotlights_engine.candidate_discovery.api import (
    DiscoveryConfig,
    IterationTelemetry,
)
from spotlights_engine.costing.records import UsageRecord, UsageStep
from spotlights_engine.module_deep_research.codex_exec import CodexExecOptions
from spotlights_engine.module_deep_research.search_log import render_search_log_markdown
from spotlights_engine.modules_extractor.agent import ExtractionInvocation
from spotlights_engine.proposal_from_finding_creator import (
    ProposalFromFindingConfig,
)
from spotlights_engine.schemas.candidate import Candidates
from spotlights_engine.schemas.common import PipelineStep, SpotlightContext, StepIssue
from spotlights_engine.schemas.pipeline import (
    AgentProposalsOutput,
    ModuleDeepResearchOutput,
    ProposalFromFindingCreatorOutput,
)
from spotlights_engine.schemas.project import ProjectTree
from spotlights_engine.utils.id_helpers import slug_for

# Bumped to 2 when qualified names became source-root-relative (the package
# prefix is now retained, e.g. `spotlights_engine/modules_extractor`). Run dirs
# written under the old name-chain layout carry version 1 and are not
# resume-compatible — their per-module slugs and `module_runs` keys differ.
# Bumped to 3 for the new `Candidate`/`Proposal` shapes: candidates now nest
# their location under `locations[].spans[]`, drop the per-candidate `state`,
# and merge `deep_research_proposals`/`agent_proposals` into one
# `proposals: list[Proposal]`. On-disk `candidates.json` /
# `proposal_from_finding_creator.json` / `agent_proposals.json` from version-2
# runs fail `model_validate`, so a hard cutover (re-run from scratch) is the
# back-compat strategy. `_ensure_resume_compatible` reads and compares this;
# pre-change dirs are rejected cleanly rather than silently reloaded.
# Bumped to 4 for the module-name-prefix id scheme: ids are now
# `<type>-<slug>[.s<k>]-NNNN` instead of bare `cand-0001`/`find-0001`/`prop-0001`.
# Old run dirs hold bare ids that fail the widened schema patterns, so they ride
# the same hard cutover.
# Bumped to 5 for per-candidate deep research (steps 3 & 4 rework): step-3
# findings now carry `candidate_id` and per-candidate finding segments, step-4
# groups findings by candidate instead of the old `|C|×|F|` cartesian, the
# `include_candidate_hotspots` fingerprint key is gone, and step 3 writes one
# last-message file per candidate under `module_deep_research.last_messages/`.
# Old (version-4) run dirs hold candidate-less findings that cannot be grouped
# correctly, so they ride the same hard cutover.
SCHEMA_VERSION = 5


CheckpointStatus = Literal[
    "PENDING",
    "DISCOVERED",
    "DEEP_RESEARCHED",
    "FINDING_PROPOSALS_CREATED",
    "AGENT_PROPOSALS_CREATED",
    "SUCCEEDED",
    "DEGRADED",
    "SKIPPED",
    "FAILED",
]


def _now_iso() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


def _atomic_write_text(path: Path, text: str) -> None:
    """Write `text` to `path` atomically via a sibling `.tmp` file + replace."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(text, encoding="utf-8")
    os.replace(tmp, path)


def _atomic_write_json(path: Path, payload: Any) -> None:
    _atomic_write_text(path, json.dumps(payload, indent=2, sort_keys=True) + "\n")


class ModuleCheckpoint(BaseModel):
    """Persisted per-module run state (`modules/<slug>/status.json`).

    `last_step` is the most recent step that completed successfully.
    `failed_step` is the step to retry when `status == FAILED` and
    `retryable` is True.

    `session_index` is the per-module discovery/run session counter used to
    build the id segment (decision D3): the first/only session uses `1` (or
    `None`), which maps to the bare-slug segment, and a genuinely new session of
    an already-present module increments it so its ids carry a distinct `.s<k>`
    sub-segment. It is persisted so resume re-derives the same segment and the
    prefix stays idempotent. (Today no run trigger creates a second session, so
    this stays `1`; the field reserves the hook — see the manager docstring.)
    """

    model_config = ConfigDict(extra="forbid")

    module_qualified_name: str = Field(min_length=1)
    status: CheckpointStatus
    last_step: PipelineStep | None = None
    failed_step: PipelineStep | None = None
    error: str | None = None
    retryable: bool = False
    issues: list[StepIssue] = Field(default_factory=list)
    session_index: int = Field(default=1, ge=1)
    started_at: str
    updated_at: str


@dataclass
class ManagerPaths:
    """Path resolver for the manager checkpoint tree."""

    artifacts_dir: Path

    @property
    def root(self) -> Path:
        return self.artifacts_dir / "spotlights_manager"

    @property
    def manifest_path(self) -> Path:
        return self.root / "manifest.json"

    @property
    def run_manifest_path(self) -> Path:
        """The public per-run manifest (provenance + usage + cost) — distinct
        from the internal resume `manifest.json`."""
        return self.root / "run_manifest.json"

    @property
    def project_tree_path(self) -> Path:
        return self.root / "project_tree.json"

    @property
    def extractor_invocation_path(self) -> Path:
        return self.root / "extractor_invocation.json"

    @property
    def extractor_run_dir(self) -> Path:
        return self.root / "modules_extractor"

    @property
    def modules_root(self) -> Path:
        return self.root / "modules"

    def for_module(self, qualified_name: str) -> ModulePaths:
        return ModulePaths(self.modules_root / slug_for(qualified_name))


@dataclass
class ModulePaths:
    """Path resolver for a single module subtree."""

    dir: Path

    @property
    def status_path(self) -> Path:
        return self.dir / "status.json"

    @property
    def discovery_run_dir(self) -> Path:
        return self.dir / "candidate_discovery"

    @property
    def candidates_path(self) -> Path:
        return self.dir / "candidates.json"

    @property
    def discovery_telemetry_path(self) -> Path:
        return self.dir / "candidate_discovery.telemetry.json"

    @property
    def deep_research_path(self) -> Path:
        return self.dir / "module_deep_research.json"

    @property
    def deep_research_last_message_dir(self) -> Path:
        """Per-candidate codex last-message directory (decision D8).

        Step 3 now runs one survey per candidate, and the codex runner reads its
        `--output-last-message` file back as the response it parses. A single
        shared file would let one candidate parse another's JSON, so each
        candidate writes `<dir>/<candidate_id>.md`, mirroring the step-4/step-5
        `*.last_messages` layout."""
        return self.dir / "module_deep_research.last_messages"

    @property
    def deep_research_search_log_path(self) -> Path:
        return self.dir / "module_deep_research.search_log.md"

    @property
    def proposal_from_finding_path(self) -> Path:
        return self.dir / "proposal_from_finding_creator.json"

    @property
    def proposal_from_finding_last_message_dir(self) -> Path:
        return self.dir / "proposal_from_finding_creator.last_messages"

    @property
    def agent_proposals_path(self) -> Path:
        return self.dir / "agent_proposals.json"

    @property
    def agent_proposals_last_message_dir(self) -> Path:
        return self.dir / "agent_proposals.last_messages"

    # Per-step usage-record directories. One atomically-written JSON file per
    # CLI invocation, named from the idempotency key (`s<k>.i<NNNN>.<cli>.json`).

    @property
    def candidate_discovery_usage_dir(self) -> Path:
        return self.dir / "candidate_discovery.usage"

    @property
    def deep_research_usage_dir(self) -> Path:
        return self.dir / "module_deep_research.usage"

    @property
    def proposal_from_finding_usage_dir(self) -> Path:
        return self.dir / "proposal_from_finding_creator.usage"

    @property
    def agent_proposals_usage_dir(self) -> Path:
        return self.dir / "agent_proposals.usage"

    def usage_dir(self, step: UsageStep) -> Path:
        return self.dir / f"{step}.usage"


@dataclass
class LoadedModuleState:
    """In-memory view assembled by `read_module_state`.

    Kept distinct from `ModuleCheckpoint` (which is the persisted JSON model)
    because the orchestrator wants the sidecar payloads alongside the
    checkpoint flags.
    """

    checkpoint: ModuleCheckpoint | None
    candidates: Candidates | None
    discovery_telemetry: list[IterationTelemetry]
    discovery_total_duration_s: float | None
    discovery_total_cost_usd: float | None
    deep_research: ModuleDeepResearchOutput | None
    deep_research_duration_s: float | None
    proposal_from_finding: ProposalFromFindingCreatorOutput | None
    proposal_from_finding_duration_s: float | None
    proposal_from_finding_per_pair_durations_s: dict[str, float]
    agent_proposals: AgentProposalsOutput | None
    agent_proposals_duration_s: float | None
    agent_proposals_per_candidate_durations_s: dict[str, dict[str, float]]


# ---------------------------------------------------------------------------
# Manifest
# ---------------------------------------------------------------------------


def _stable_hash(payload: Any) -> str:
    """Hash a JSON-serializable payload (sorted keys) into a sha256 hex."""
    text = json.dumps(payload, sort_keys=True, default=str)
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def hash_pydantic_excluding(model: BaseModel | None, *, exclude: set[str]) -> str:
    """Stable hash of a pydantic model with selected fields excluded.

    Used for `config_fingerprint` so that changing manager-owned path fields
    (`artifacts_dir`, `repo_path`) doesn't trigger a spurious resume mismatch.
    """
    if model is None:
        return _stable_hash(None)
    payload = model.model_dump(mode="json", exclude=exclude)
    return _stable_hash(payload)


def build_input_fingerprint(
    *,
    repo_path: Path,
    context: BaseModel,
    max_findings_per_candidate: int,
    continue_on_module_failure: bool,
    enable_claude_search: bool = False,
) -> dict[str, Any]:
    return {
        "repo_path": str(repo_path),
        "context_hash": _stable_hash(context.model_dump(mode="json")),
        "max_findings_per_candidate": max_findings_per_candidate,
        "continue_on_module_failure": continue_on_module_failure,
        "enable_claude_search": enable_claude_search,
    }


def build_config_fingerprint(
    *,
    module_filter: BaseModel | None,
    extractor_cfg: BaseModel,
    discovery_cfg: BaseModel | None,
    deep_research_cfg: BaseModel | None,
    proposal_from_finding_cfg: BaseModel | None,
    agent_proposals_cfg: BaseModel | None,
) -> dict[str, Any]:
    effective_discovery_cfg = discovery_cfg or DiscoveryConfig()
    effective_deep_research_cfg = deep_research_cfg or CodexExecOptions()
    effective_proposal_cfg = proposal_from_finding_cfg or ProposalFromFindingConfig()
    effective_agent_proposals_cfg = agent_proposals_cfg or AgentProposalsConfig()
    return {
        "module_filter": (
            module_filter.model_dump(mode="json") if module_filter is not None else None
        ),
        "extractor_hash": hash_pydantic_excluding(
            extractor_cfg, exclude={"artifacts_dir"}
        ),
        "discovery_hash": hash_pydantic_excluding(
            effective_discovery_cfg, exclude={"artifacts_dir", "repo_path"}
        ),
        "deep_research_hash": hash_pydantic_excluding(
            effective_deep_research_cfg, exclude={"cwd", "output_last_message"}
        ),
        "proposal_from_finding_hash": hash_pydantic_excluding(
            effective_proposal_cfg, exclude={"artifacts_dir", "repo_path"}
        ),
        "agent_proposals_hash": hash_pydantic_excluding(
            effective_agent_proposals_cfg, exclude={"artifacts_dir", "repo_path"}
        ),
    }


def default_agent_proposals_hash() -> str:
    """Stable hash of the default `AgentProposalsConfig`. Used by the one-shot
    pre-step-5 manifest forward migration; see orchestrator
    `_ensure_resume_compatible`."""
    return hash_pydantic_excluding(
        AgentProposalsConfig(), exclude={"artifacts_dir", "repo_path"}
    )


def read_manifest(paths: ManagerPaths) -> dict[str, Any] | None:
    if not paths.manifest_path.exists():
        return None
    return json.loads(paths.manifest_path.read_text(encoding="utf-8"))


def write_manifest(paths: ManagerPaths, manifest: dict[str, Any]) -> None:
    manifest = dict(manifest)
    manifest["updated_at"] = _now_iso()
    _atomic_write_json(paths.manifest_path, manifest)


def write_run_manifest(paths: ManagerPaths, manifest: Any) -> None:
    payload = (
        manifest.model_dump(mode="json")
        if hasattr(manifest, "model_dump")
        else manifest
    )
    _atomic_write_json(paths.run_manifest_path, payload)


def init_manifest(
    paths: ManagerPaths,
    *,
    input_fingerprint: dict[str, Any],
    config_fingerprint: dict[str, Any],
    context: SpotlightContext,
    provenance: dict[str, Any] | None = None,
) -> dict[str, Any]:
    now = _now_iso()
    manifest: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "created_at": now,
        "updated_at": now,
        "status": "RUNNING",
        "input_fingerprint": input_fingerprint,
        "config_fingerprint": config_fingerprint,
        "context": context.model_dump(mode="json"),
        "extractor": {"completed": False, "duration_s": None},
        "modules": {},
        # Pinned once at first run; resume reads it back, never recomputes it,
        # so a resume from a different working tree can't rewrite provenance.
        "provenance": dict(provenance or {}),
    }
    paths.root.mkdir(parents=True, exist_ok=True)
    paths.modules_root.mkdir(parents=True, exist_ok=True)
    write_manifest(paths, manifest)
    return manifest


# ---------------------------------------------------------------------------
# Per-module read / write
# ---------------------------------------------------------------------------


def read_module_state(module_paths: ModulePaths) -> LoadedModuleState:
    checkpoint: ModuleCheckpoint | None = None
    if module_paths.status_path.exists():
        checkpoint = ModuleCheckpoint.model_validate_json(
            module_paths.status_path.read_text(encoding="utf-8")
        )

    candidates: Candidates | None = None
    if module_paths.candidates_path.exists():
        candidates = Candidates.model_validate_json(
            module_paths.candidates_path.read_text(encoding="utf-8")
        )

    discovery_telemetry: list[IterationTelemetry] = []
    discovery_total_duration_s: float | None = None
    discovery_total_cost_usd: float | None = None
    if module_paths.discovery_telemetry_path.exists():
        payload = json.loads(
            module_paths.discovery_telemetry_path.read_text(encoding="utf-8")
        )
        discovery_telemetry = [
            IterationTelemetry.model_validate(it)
            for it in payload.get("iterations", [])
        ]
        discovery_total_duration_s = payload.get("total_duration_s")
        discovery_total_cost_usd = payload.get("total_cost_usd")

    deep_research: ModuleDeepResearchOutput | None = None
    deep_research_duration_s: float | None = None
    if module_paths.deep_research_path.exists():
        payload = json.loads(
            module_paths.deep_research_path.read_text(encoding="utf-8")
        )
        # Sidecar wraps the output with telemetry — the contract object lives
        # under "output", everything else is runtime metadata.
        if "output" in payload:
            deep_research = ModuleDeepResearchOutput.model_validate(payload["output"])
            deep_research_duration_s = payload.get("duration_s")
        else:
            # Backwards-compatible fallback: file may be the bare output.
            deep_research = ModuleDeepResearchOutput.model_validate(payload)

    proposal_from_finding: ProposalFromFindingCreatorOutput | None = None
    proposal_from_finding_duration_s: float | None = None
    proposal_from_finding_per_pair_durations_s: dict[str, float] = {}
    if module_paths.proposal_from_finding_path.exists():
        payload = json.loads(
            module_paths.proposal_from_finding_path.read_text(encoding="utf-8")
        )
        if "output" in payload:
            proposal_from_finding = ProposalFromFindingCreatorOutput.model_validate(
                payload["output"]
            )
            proposal_from_finding_duration_s = payload.get("duration_s")
            per_pair = payload.get("per_pair_durations_s") or {}
            if isinstance(per_pair, dict):
                proposal_from_finding_per_pair_durations_s = {
                    str(k): float(v) for k, v in per_pair.items()
                }
        else:
            proposal_from_finding = ProposalFromFindingCreatorOutput.model_validate(
                payload
            )

    agent_proposals: AgentProposalsOutput | None = None
    agent_proposals_duration_s: float | None = None
    agent_proposals_per_candidate_durations_s: dict[str, dict[str, float]] = {}
    if module_paths.agent_proposals_path.exists():
        payload = json.loads(
            module_paths.agent_proposals_path.read_text(encoding="utf-8")
        )
        if "output" in payload:
            agent_proposals = AgentProposalsOutput.model_validate(payload["output"])
            agent_proposals_duration_s = payload.get("duration_s")
            per_cand = payload.get("per_candidate_durations_s") or {}
            if isinstance(per_cand, dict):
                normalized: dict[str, dict[str, float]] = {}
                for cand_id, durations in per_cand.items():
                    if isinstance(durations, dict):
                        normalized[str(cand_id)] = {
                            str(k): float(v) for k, v in durations.items()
                        }
                agent_proposals_per_candidate_durations_s = normalized
        else:
            agent_proposals = AgentProposalsOutput.model_validate(payload)

    return LoadedModuleState(
        checkpoint=checkpoint,
        candidates=candidates,
        discovery_telemetry=discovery_telemetry,
        discovery_total_duration_s=discovery_total_duration_s,
        discovery_total_cost_usd=discovery_total_cost_usd,
        deep_research=deep_research,
        deep_research_duration_s=deep_research_duration_s,
        proposal_from_finding=proposal_from_finding,
        proposal_from_finding_duration_s=proposal_from_finding_duration_s,
        proposal_from_finding_per_pair_durations_s=proposal_from_finding_per_pair_durations_s,
        agent_proposals=agent_proposals,
        agent_proposals_duration_s=agent_proposals_duration_s,
        agent_proposals_per_candidate_durations_s=agent_proposals_per_candidate_durations_s,
    )


def write_checkpoint(module_paths: ModulePaths, checkpoint: ModuleCheckpoint) -> None:
    module_paths.dir.mkdir(parents=True, exist_ok=True)
    _atomic_write_text(
        module_paths.status_path, checkpoint.model_dump_json(indent=2) + "\n"
    )


def write_candidates(module_paths: ModulePaths, candidates: Candidates) -> None:
    module_paths.dir.mkdir(parents=True, exist_ok=True)
    _atomic_write_text(
        module_paths.candidates_path, candidates.model_dump_json(indent=2) + "\n"
    )


def write_discovery_telemetry(
    module_paths: ModulePaths,
    iterations: list[IterationTelemetry],
    total_duration_s: float,
    total_cost_usd: float | None,
) -> None:
    payload = {
        "iterations": [it.model_dump(mode="json") for it in iterations],
        "total_duration_s": total_duration_s,
        "total_cost_usd": total_cost_usd,
    }
    _atomic_write_json(module_paths.discovery_telemetry_path, payload)


def write_deep_research(
    module_paths: ModulePaths,
    output: ModuleDeepResearchOutput,
    duration_s: float,
) -> None:
    payload = {
        "output": output.model_dump(mode="json"),
        "duration_s": duration_s,
    }
    _atomic_write_json(module_paths.deep_research_path, payload)


def write_deep_research_search_log(
    module_paths: ModulePaths,
    output: ModuleDeepResearchOutput,
    qn: str,
) -> None:
    """Render and persist the human-diffable search-query log markdown.

    `qn` (the module qualified name) is threaded in for the document header
    because `ModulePaths` stores only `dir` and `ModuleDeepResearchOutput` has
    no module-name field. Idempotent: safe to call on both the fresh-run and
    resume/skip paths."""
    module_paths.dir.mkdir(parents=True, exist_ok=True)
    _atomic_write_text(
        module_paths.deep_research_search_log_path,
        render_search_log_markdown(output, qn=qn),
    )


# ---------------------------------------------------------------------------
# Usage records (see costing.records)
# ---------------------------------------------------------------------------


def write_usage_record(module_paths: ModulePaths, record: UsageRecord) -> Path:
    """Atomically persist one per-invocation usage record.

    The filename encodes the idempotency key, so rewriting the same
    invocation (after a session-scoped clear on resume) replaces rather than
    duplicates it.
    """
    path = module_paths.usage_dir(record.step) / record.filename
    _atomic_write_text(path, record.model_dump_json(indent=2) + "\n")
    return path


def read_usage_records(
    module_paths: ModulePaths, step: UsageStep | None = None
) -> tuple[list[UsageRecord], list[str]]:
    """Read this module's usage records from disk, tolerating partial writes.

    Returns `(records, notes)`: `*.tmp` leftovers from interrupted atomic
    writes and unparseable files are skipped and reported as notes instead of
    failing the aggregation.
    """
    steps: tuple[UsageStep, ...] = (
        (step,)
        if step is not None
        else (
            "candidate_discovery",
            "module_deep_research",
            "proposal_from_finding_creator",
            "agent_proposals",
        )
    )
    records: list[UsageRecord] = []
    notes: list[str] = []
    for s in steps:
        usage_dir = module_paths.usage_dir(s)
        if not usage_dir.is_dir():
            continue
        for path in sorted(usage_dir.iterdir()):
            if path.name.endswith(".tmp"):
                notes.append(f"ignored partial usage file: {path.name}")
                continue
            if path.suffix != ".json":
                continue
            try:
                records.append(
                    UsageRecord.model_validate_json(
                        path.read_text(encoding="utf-8")
                    )
                )
            except (OSError, ValueError) as exc:
                notes.append(f"ignored unreadable usage file {path.name}: {exc}")
    return records, notes


def clear_usage_records(
    module_paths: ModulePaths,
    step: UsageStep,
    session_index: int | None = None,
) -> None:
    """Drop a step's usage records so a re-run can't double count.

    Clearing is session-scoped when `session_index` is given: only files
    carrying that session's `s<k>.` prefix are removed, because records from
    earlier sessions must survive (the manifest sums across sessions to
    reflect the run's real spend). `None` clears the whole step directory.
    """
    usage_dir = module_paths.usage_dir(step)
    if not usage_dir.is_dir():
        return
    if session_index is None:
        shutil.rmtree(usage_dir)
        return
    prefix = f"s{session_index}."
    for path in usage_dir.iterdir():
        if path.name.startswith(prefix):
            path.unlink()


def clear_discovery_artifacts(
    module_paths: ModulePaths, *, session_index: int | None = None
) -> None:
    """Remove everything step 2 produced so the per-step guard is happy."""
    if module_paths.discovery_run_dir.exists():
        shutil.rmtree(module_paths.discovery_run_dir)
    for p in (module_paths.candidates_path, module_paths.discovery_telemetry_path):
        if p.exists():
            p.unlink()
    clear_usage_records(module_paths, "candidate_discovery", session_index)


def clear_deep_research_artifacts(
    module_paths: ModulePaths, *, session_index: int | None = None
) -> None:
    for p in (
        module_paths.deep_research_path,
        module_paths.deep_research_search_log_path,
    ):
        if p.exists():
            p.unlink()
    # The last-message path is now a per-candidate directory (D8); rmtree it so a
    # `--redo` of step 3 doesn't leak a previous session's per-candidate messages
    # (an `unlink()` would silently no-op on a directory).
    if module_paths.deep_research_last_message_dir.exists():
        shutil.rmtree(module_paths.deep_research_last_message_dir)
    clear_usage_records(module_paths, "module_deep_research", session_index)


def write_proposal_from_finding(
    module_paths: ModulePaths,
    output: ProposalFromFindingCreatorOutput,
    duration_s: float,
    per_pair_durations_s: dict[str, float],
) -> None:
    payload = {
        "output": output.model_dump(mode="json"),
        "duration_s": duration_s,
        "per_pair_durations_s": dict(per_pair_durations_s),
    }
    _atomic_write_json(module_paths.proposal_from_finding_path, payload)


def clear_proposal_from_finding_artifacts(
    module_paths: ModulePaths, *, session_index: int | None = None
) -> None:
    if module_paths.proposal_from_finding_last_message_dir.exists():
        shutil.rmtree(module_paths.proposal_from_finding_last_message_dir)
    if module_paths.proposal_from_finding_path.exists():
        module_paths.proposal_from_finding_path.unlink()
    clear_usage_records(module_paths, "proposal_from_finding_creator", session_index)


def write_agent_proposals(
    module_paths: ModulePaths,
    output: AgentProposalsOutput,
    duration_s: float,
    per_candidate_durations_s: dict[str, dict[str, float]],
) -> None:
    payload = {
        "output": output.model_dump(mode="json"),
        "duration_s": duration_s,
        "per_candidate_durations_s": {
            str(cand_id): {str(k): float(v) for k, v in durations.items()}
            for cand_id, durations in per_candidate_durations_s.items()
        },
    }
    _atomic_write_json(module_paths.agent_proposals_path, payload)


def clear_agent_proposals_artifacts(
    module_paths: ModulePaths, *, session_index: int | None = None
) -> None:
    if module_paths.agent_proposals_last_message_dir.exists():
        shutil.rmtree(module_paths.agent_proposals_last_message_dir)
    if module_paths.agent_proposals_path.exists():
        module_paths.agent_proposals_path.unlink()
    clear_usage_records(module_paths, "agent_proposals", session_index)


def write_extractor_outputs(
    paths: ManagerPaths,
    project_tree: ProjectTree,
    invocation: ExtractionInvocation,
) -> None:
    paths.root.mkdir(parents=True, exist_ok=True)
    _atomic_write_text(
        paths.project_tree_path,
        project_tree.model_dump_json(indent=2) + "\n",
    )
    _atomic_write_json(
        paths.extractor_invocation_path, dataclasses.asdict(invocation)
    )


def read_extractor_outputs(
    paths: ManagerPaths,
) -> tuple[ProjectTree | None, ExtractionInvocation | None]:
    tree = (
        ProjectTree.from_json(paths.project_tree_path)
        if paths.project_tree_path.exists()
        else None
    )
    invocation: ExtractionInvocation | None = None
    if paths.extractor_invocation_path.exists():
        payload = json.loads(
            paths.extractor_invocation_path.read_text(encoding="utf-8")
        )
        invocation = ExtractionInvocation(**payload)
    return tree, invocation


def clear_extractor_artifacts(paths: ManagerPaths) -> None:
    if paths.extractor_run_dir.exists():
        shutil.rmtree(paths.extractor_run_dir)
    for p in (paths.project_tree_path, paths.extractor_invocation_path):
        if p.exists():
            p.unlink()


__all__ = [
    "SCHEMA_VERSION",
    "CheckpointStatus",
    "LoadedModuleState",
    "ManagerPaths",
    "ModuleCheckpoint",
    "ModulePaths",
    "build_config_fingerprint",
    "build_input_fingerprint",
    "clear_agent_proposals_artifacts",
    "clear_deep_research_artifacts",
    "clear_discovery_artifacts",
    "clear_extractor_artifacts",
    "clear_proposal_from_finding_artifacts",
    "clear_usage_records",
    "default_agent_proposals_hash",
    "init_manifest",
    "read_extractor_outputs",
    "read_manifest",
    "read_module_state",
    "read_usage_records",
    "slug_for",
    "write_agent_proposals",
    "write_candidates",
    "write_checkpoint",
    "write_deep_research",
    "write_deep_research_search_log",
    "write_discovery_telemetry",
    "write_extractor_outputs",
    "write_manifest",
    "write_proposal_from_finding",
    "write_run_manifest",
    "write_usage_record",
]
