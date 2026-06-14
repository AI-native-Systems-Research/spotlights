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
import re
import shutil
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from spotlights_engine.agent_proposals import AgentProposalsConfig
from spotlights_engine.candidate_discovery.api import (
    DiscoveryConfig,
    IterationTelemetry,
)
from spotlights_engine.module_deep_research.codex_exec import CodexExecOptions
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


_SLUG_SAFE = re.compile(r"[^A-Za-z0-9._-]")


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
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _atomic_write_text(path: Path, text: str) -> None:
    """Write `text` to `path` atomically via a sibling `.tmp` file + replace."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(text, encoding="utf-8")
    os.replace(tmp, path)


def _atomic_write_json(path: Path, payload: Any) -> None:
    _atomic_write_text(path, json.dumps(payload, indent=2, sort_keys=True) + "\n")


def slug_for(qualified_name: str) -> str:
    """Slug used as the per-module directory name. Dot-form name with any
    char outside `[A-Za-z0-9._-]` replaced by `_`. Dot-form is unique by
    construction so collisions are theoretical, but the orchestrator still
    checks at filter-resolution time."""
    return _SLUG_SAFE.sub("_", qualified_name)


class ModuleCheckpoint(BaseModel):
    """Persisted per-module run state (`modules/<slug>/status.json`).

    `last_step` is the most recent step that completed successfully.
    `failed_step` is the step to retry when `status == FAILED` and
    `retryable` is True.
    """

    model_config = ConfigDict(extra="forbid")

    module_qualified_name: str = Field(min_length=1)
    status: CheckpointStatus
    last_step: PipelineStep | None = None
    failed_step: PipelineStep | None = None
    error: str | None = None
    retryable: bool = False
    issues: list[StepIssue] = Field(default_factory=list)
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

    def for_module(self, qualified_name: str) -> "ModulePaths":
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
    def deep_research_last_message_path(self) -> Path:
        return self.dir / "module_deep_research.last_message.md"

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

    @property
    def archived_records_path(self) -> Path:
        return self.dir / "archived_records.json"


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
    archived_record_ids: list[str] = dataclasses.field(default_factory=list)


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
    max_findings_per_module: int,
    continue_on_module_failure: bool,
) -> dict[str, Any]:
    return {
        "repo_path": str(repo_path),
        "context_hash": _stable_hash(context.model_dump(mode="json")),
        "max_findings_per_module": max_findings_per_module,
        "continue_on_module_failure": continue_on_module_failure,
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


def init_manifest(
    paths: ManagerPaths,
    *,
    input_fingerprint: dict[str, Any],
    config_fingerprint: dict[str, Any],
    context: SpotlightContext,
) -> dict[str, Any]:
    now = _now_iso()
    manifest: dict[str, Any] = {
        "schema_version": 1,
        "created_at": now,
        "updated_at": now,
        "status": "RUNNING",
        "input_fingerprint": input_fingerprint,
        "config_fingerprint": config_fingerprint,
        "context": context.model_dump(mode="json"),
        "extractor": {"completed": False, "duration_s": None},
        "modules": {},
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

    archived_record_ids: list[str] = []
    if module_paths.archived_records_path.exists():
        try:
            payload = json.loads(
                module_paths.archived_records_path.read_text(encoding="utf-8")
            )
            if isinstance(payload, list):
                archived_record_ids = [str(rid) for rid in payload if rid]
        except (json.JSONDecodeError, OSError):
            pass

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
        archived_record_ids=archived_record_ids,
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


def clear_discovery_artifacts(module_paths: ModulePaths) -> None:
    """Remove everything step 2 produced so the per-step guard is happy."""
    if module_paths.discovery_run_dir.exists():
        shutil.rmtree(module_paths.discovery_run_dir)
    for p in (module_paths.candidates_path, module_paths.discovery_telemetry_path):
        if p.exists():
            p.unlink()


def clear_deep_research_artifacts(module_paths: ModulePaths) -> None:
    for p in (
        module_paths.deep_research_path,
        module_paths.deep_research_last_message_path,
    ):
        if p.exists():
            p.unlink()


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


def clear_proposal_from_finding_artifacts(module_paths: ModulePaths) -> None:
    if module_paths.proposal_from_finding_last_message_dir.exists():
        shutil.rmtree(module_paths.proposal_from_finding_last_message_dir)
    if module_paths.proposal_from_finding_path.exists():
        module_paths.proposal_from_finding_path.unlink()


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


def clear_agent_proposals_artifacts(module_paths: ModulePaths) -> None:
    if module_paths.agent_proposals_last_message_dir.exists():
        shutil.rmtree(module_paths.agent_proposals_last_message_dir)
    if module_paths.agent_proposals_path.exists():
        module_paths.agent_proposals_path.unlink()


def write_archived_records(module_paths: ModulePaths, record_ids: list[str]) -> None:
    """Overwrite the module's archived_records.json with the current run's ids."""
    module_paths.dir.mkdir(parents=True, exist_ok=True)
    _atomic_write_json(module_paths.archived_records_path, record_ids)


def clear_archived_records(module_paths: ModulePaths) -> None:
    if module_paths.archived_records_path.exists():
        module_paths.archived_records_path.unlink()


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
    "CheckpointStatus",
    "LoadedModuleState",
    "ManagerPaths",
    "ModuleCheckpoint",
    "ModulePaths",
    "build_config_fingerprint",
    "build_input_fingerprint",
    "clear_agent_proposals_artifacts",
    "clear_archived_records",
    "clear_deep_research_artifacts",
    "clear_discovery_artifacts",
    "clear_extractor_artifacts",
    "clear_proposal_from_finding_artifacts",
    "default_agent_proposals_hash",
    "init_manifest",
    "read_extractor_outputs",
    "read_manifest",
    "read_module_state",
    "slug_for",
    "write_agent_proposals",
    "write_archived_records",
    "write_candidates",
    "write_checkpoint",
    "write_deep_research",
    "write_discovery_telemetry",
    "write_extractor_outputs",
    "write_manifest",
    "write_proposal_from_finding",
]
