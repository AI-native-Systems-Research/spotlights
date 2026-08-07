"""Per-pair JSON schema handed to Claude `--json-schema` (candidate mode).

Fork of `proposal_from_finding_creator.agent_schema` that additionally requires
the four structured proposal fields (`mechanism`, `required_changes`,
`expected_effect`, `evaluation_metric`) whenever a proposal is emitted. The
wrapper contract is unchanged and load-bearing: the top level is an object with
a single `proposals` array, which `claude_exec._unwrap_proposals` strips before
`parse_pair_payload` validates a bare list.
"""

from __future__ import annotations

import json


def build_per_pair_schema_text(*, finding_id: str, created_by: str) -> str:
    """Return a JSON schema text suitable for Claude's `--json-schema`.

    The schema is a top-level object with one property, `proposals`: an array of
    length 0 or 1. When non-empty the single entry is a `DeepResearchProposal`
    with `finding_id` and `created_by` constrained to the supplied values and
    the four structured fields required and non-empty.
    """
    schema = {
        "$schema": "http://json-schema.org/draft-07/schema#",
        "title": "ProposalFromCandidateFindingPairOutput",
        "type": "object",
        "additionalProperties": False,
        "required": ["proposals"],
        "properties": {
            "proposals": {
                "type": "array",
                "minItems": 0,
                "maxItems": 1,
                "items": {
                    "type": "object",
                    "additionalProperties": False,
                    "required": [
                        "title",
                        "detailed_description",
                        "finding_id",
                        "proposal_rationale",
                        "created_by",
                        "mechanism",
                        "required_changes",
                        "expected_effect",
                        "evaluation_metric",
                    ],
                    "properties": {
                        "title": {"type": "string", "minLength": 1},
                        "detailed_description": {"type": "string", "minLength": 1},
                        "finding_id": {
                            "type": "string",
                            "const": finding_id,
                        },
                        "proposal_rationale": {"type": "string", "minLength": 1},
                        "created_by": {
                            "type": "string",
                            "const": created_by,
                        },
                        "mechanism": {"type": "string", "minLength": 1},
                        "required_changes": {"type": "string", "minLength": 1},
                        "expected_effect": {"type": "string", "minLength": 1},
                        "evaluation_metric": {"type": "string", "minLength": 1},
                    },
                },
            },
        },
    }
    return json.dumps(schema)


__all__ = ["build_per_pair_schema_text"]
