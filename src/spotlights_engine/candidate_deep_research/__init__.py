"""Candidate deep-research step for Spotlights (candidate mode step 3).

Sibling of `module_deep_research`: one survey per candidate rather than one per
module. The runner clients, wire schema, and merge helpers are reused from
`module_deep_research`; only the prompt and the per-candidate orchestration are
new.
"""

from spotlights_engine.candidate_deep_research.api import (
    CandidateCliUsage,
    CandidateDeepResearchResult,
    research_candidates,
    research_candidates_with_telemetry,
)
from spotlights_engine.candidate_deep_research.prompts import (
    render_candidate_deep_research_prompt,
)
from spotlights_engine.module_deep_research.agent_exec import ModuleResearchRunner

__all__ = [
    "CandidateCliUsage",
    "CandidateDeepResearchResult",
    "ModuleResearchRunner",
    "render_candidate_deep_research_prompt",
    "research_candidates",
    "research_candidates_with_telemetry",
]
