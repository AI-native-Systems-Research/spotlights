"""Per-pair JSON schema handed to Claude `--json-schema`.

Step 4 asks the agent for a 0..1-length `list[DeepResearchProposal]` for one
(candidate, finding) pair. The schema pins `finding_id` to the input
finding's id and `created_by` to the configured value so the response is
self-attributing and unambiguous to validate.
"""

from __future__ import annotations

import json


def build_per_pair_schema_text(*, finding_id: str, created_by: str) -> str:
    """Return a JSON schema text suitable for Claude's `--json-schema`.

    The schema accepts an array of length 0 or 1, where (when present) the
    single object is a `DeepResearchProposal` with `finding_id` and
    `created_by` constrained to the supplied values.
    """
    schema = {
        "$schema": "http://json-schema.org/draft-07/schema#",
        "title": "ProposalFromFindingPairOutput",
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
            },
        },
    }
    return json.dumps(schema)


__all__ = ["build_per_pair_schema_text"]
