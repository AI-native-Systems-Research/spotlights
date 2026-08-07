"""Per-pair prompt builder for step 4 in candidate mode.

Fork of `proposal_from_finding_creator.prompts` with two differences: the
relevance gate is stronger (the finding was already surveyed *for* this
candidate, so a proposal must justify itself against this code site rather than
against the module), and the output rules cover the four structured fields the
candidate-mode schema requires.
"""

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
    machinery enforces the wrapper shape, pins `finding_id` / `created_by`, and
    requires the four structured fields when a proposal is emitted.
    """
    return f"""You are running the Spotlights proposal_from_candidate_finding_creator
pipeline step. Do not modify files. Do not ask questions. You may read
repository files under the current working directory to ground your reasoning.

Goal:
Decide whether the supplied finding meaningfully supports a change to the
supplied candidate. If yes, emit exactly one DeepResearchProposal describing
the change. If not, emit nothing. Be conservative: only emit a proposal when
the finding contributes a concrete, transferable idea that can plausibly
improve this specific candidate.

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

Relevance gate (apply before writing anything):
- This finding was surveyed while researching THIS candidate, so topical
  overlap is expected and is NOT evidence of relevance. Ask instead: does the
  source carry a technique that would change what this candidate's code does?
- Emit `[]` when any of these hold:
  1. The idea does not touch the candidate's own responsibility (it belongs
     to a caller, a neighbouring module, or the surrounding infrastructure).
  2. The candidate's `current_approach` already implements the idea, and the
     source does not meaningfully extend or contrast with it.
  3. You cannot name the concrete code change in one sentence.
  4. You cannot name an observable effect that the change would produce.
- Emitting `[]` is a correct, expected outcome. Do not stretch a weak match
  into a proposal.

Output rules:
- Return a JSON object with a single property `proposals` whose value is a
  JSON array of length 0 or 1.
- An empty array (`[]`) means: this finding does not meaningfully apply to
  this candidate.
- A 1-element array means: emit one DeepResearchProposal with all of:
  - title: short, action-oriented (1 line).
  - detailed_description: concrete description of the change applied to
    this candidate, grounded in the finding. May reference the candidate's
    file/lines/symbol; do not invent unrelated locations.
  - finding_id: must be exactly "{finding.finding_id}".
  - proposal_rationale: why this finding plausibly improves this specific
    candidate, including which gap or constraint it addresses.
  - created_by: must be exactly "{created_by}".
  - mechanism: the technique from the finding and *how* it produces the
    improvement in this candidate — the causal chain, not a restatement of
    the title. 1-3 sentences.
  - required_changes: the concrete code changes needed at this candidate's
    site, named against the real symbols/files you read. Say what is added,
    replaced, or removed. Do not invent locations.
  - expected_effect: the observable effect if the change lands, stated in the
    direction that matters for the caller objective, with a magnitude or
    bound when the finding supports one. Say "unquantified" rather than
    inventing a number.
  - evaluation_metric: how to verify the effect — the specific measurement,
    benchmark, or check, ideally consistent with the validation plan above.
- Do not wrap the object in Markdown. Do not include explanatory prose.
""".strip()


__all__ = ["build_prompt"]
