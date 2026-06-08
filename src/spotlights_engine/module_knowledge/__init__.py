"""Module knowledge archive/retrieve/wiki APIs.

This module is the public Spotlights memory layer: archive evidence-bearing
records, retrieve ranked local records, and render a generated wiki. It is
intentionally phase-free and dependency-light.
"""

from spotlights_engine.module_knowledge.adapters import (
    record_from_agent_proposal,
    record_from_candidate,
    record_from_deep_research_proposal,
    record_from_finding,
    record_from_module,
    records_from_candidate_proposals,
    records_from_candidates,
    records_from_findings,
    records_from_module_deep_research,
    records_from_module_run,
    records_from_project_tree,
    records_from_spotlights_result,
)
from spotlights_engine.module_knowledge.archive import KnowledgeBase
from spotlights_engine.module_knowledge.layout import KnowledgeLayout
from spotlights_engine.module_knowledge.retrieve import retrieve_records
from spotlights_engine.module_knowledge.schemas import (
    SCHEMA_VERSION,
    ArchiveResult,
    KnowledgeRecord,
    KnowledgeSourceType,
    Provenance,
    RetrievedItem,
    RetrieveRequest,
    SourceRef,
    TrustTier,
    WikiRenderResult,
    WikiVerificationIssue,
    WikiVerificationReport,
)
from spotlights_engine.module_knowledge.store import read_records, write_records
from spotlights_engine.module_knowledge.wiki import WikiRenderer, verify_wiki

__all__ = [
    "SCHEMA_VERSION",
    "ArchiveResult",
    "KnowledgeBase",
    "KnowledgeLayout",
    "KnowledgeRecord",
    "KnowledgeSourceType",
    "Provenance",
    "RetrieveRequest",
    "RetrievedItem",
    "SourceRef",
    "TrustTier",
    "WikiRenderResult",
    "WikiRenderer",
    "WikiVerificationIssue",
    "WikiVerificationReport",
    "record_from_agent_proposal",
    "record_from_candidate",
    "record_from_deep_research_proposal",
    "records_from_candidate_proposals",
    "records_from_candidates",
    "records_from_findings",
    "records_from_module_run",
    "records_from_spotlights_result",
    "record_from_finding",
    "record_from_module",
    "records_from_module_deep_research",
    "records_from_project_tree",
    "read_records",
    "retrieve_records",
    "verify_wiki",
    "write_records",
]
