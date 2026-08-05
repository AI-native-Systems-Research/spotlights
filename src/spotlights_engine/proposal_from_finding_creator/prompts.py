"""Per-pair prompt builder for step 4 (`proposal_from_finding_creator`)."""

from __future__ import annotations

from spotlights_engine.schemas.candidate import Candidate
from spotlights_engine.schemas.common import SpotlightContext
from spotlights_engine.schemas.finding import Finding
from spotlights_engine.utils.schema_compat import primary_file, primary_span


def _format_list(values: list[str]) -> str:
    if not values:
        return "(none)"
    return "\n".join(f"- {value}" for value in values)


def build_prompt(
    *,
    candidate: Candidate,
    finding: Finding,
    context: SpotlightContext,
    module_qualified_name: str,
    created_by: str,
) -> str:
    """Render the per-pair prompt.

    The prompt asks the agent to either emit one `DeepResearchProposal` (in the
    `proposals` array) when the finding meaningfully supports a change to the
    candidate, or to emit an empty array otherwise. The `--json-schema`
    machinery enforces the wrapper shape and pins `finding_id` / `created_by`.
    """
    return f"""You are running the Spotlights proposal_from_finding_creator pipeline step.
Do not modify files. Do not ask questions. You may read repository files
under the current working directory to ground your reasoning.

Goal:
Decide whether this finding is really relevant to THIS candidate — the specific
code site described below — and meaningfully supports a change to it. This
finding was surveyed for this candidate, but you must still judge it on its
merits: if it is only topically adjacent, restates what the candidate already
does, or does not carry a concrete idea this candidate could adopt, emit an
empty array and nothing else. If it is genuinely relevant, emit exactly one
DeepResearchProposal describing the change. Be conservative: only emit a
proposal when the finding contributes a concrete, transferable idea that can
plausibly improve this specific candidate.

Target module: {module_qualified_name}

Candidate:
- id: {candidate.id}
- file: {primary_file(candidate)}
- lines: {primary_span(candidate).line_start}-{primary_span(candidate).line_end}
- symbol: {primary_span(candidate).symbol}
- kind: {primary_span(candidate).kind}
- description: {candidate.description}
- current_approach: {candidate.current_approach}
- evolve_rationale: {candidate.evolve_rationale}
- estimated_impact: {candidate.estimated_impact}
- estimated_impact_explanation: {candidate.estimated_impact_explanation}

Finding:
- finding_id: {finding.finding_id}
- title: {finding.title}
- url: {finding.url}
- source_type: {finding.source_type}
- technique_summary: {finding.technique_summary}
- supporting_evidence: {finding.supporting_evidence or "(none)"}

Caller context:
Objective: {context.objective}
Workload hints:
{_format_list(context.workload_hints)}
Validation plan:
{_format_list(context.validation_plan)}

Output rules:
- Return a JSON object with a single property `proposals` whose value is a
  JSON array of length 0 or 1.
- An empty array (`[]`) means: this finding is not really relevant to this
  candidate, or does not meaningfully apply to it. Prefer emptiness when the
  finding is only topically adjacent or restates the candidate's current
  approach.
- A 1-element array means: emit one DeepResearchProposal. When you emit a
  proposal you must fill every field below (all are required):
  - title: short, action-oriented (1 line).
  - detailed_description: concrete description of the change applied to
    this candidate, grounded in the finding. May reference the candidate's
    file/lines/symbol; do not invent unrelated locations.
  - finding_id: must be exactly "{finding.finding_id}".
  - proposal_rationale: why this finding plausibly improves this specific
    candidate, including which gap or constraint it addresses.
  - created_by: must be exactly "{created_by}".
  - mechanism: the concrete technique/algorithm/design from the finding and
    HOW it works when applied to this candidate (not just its name).
  - required_changes: the concrete edits this candidate's code would need —
    which functions/regions change and roughly how; do not invent unrelated
    locations.
  - expected_effect: the expected improvement (e.g. lower latency, less
    memory, higher throughput) and, where possible, in what direction/scale,
    tied to the caller objective.
  - evaluation_metric: how you would measure whether the change worked — the
    concrete metric(s) and comparison to validate against.
- Do not wrap the object in Markdown. Do not include explanatory prose.
""".strip()


__all__ = ["build_prompt"]
