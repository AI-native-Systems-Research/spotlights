"""Per-candidate JSON schema handed to the Claude / Codex agents in step 5.

The schema is the wrapper-object shape `{"proposals": [...]}` because both
Anthropic's tool API and OpenAI structured-output (used by codex
`--output-schema`) require the top-level type to be `"object"`. The inner
list is 0..1 long; when non-empty the single entry is an `AgentProposal`
whose `agent_name` is pinned via `const` to the configured value.
"""

from __future__ import annotations

import json


def build_per_candidate_schema_text(*, agent_name: str) -> str:
    """Return a JSON schema text suitable for `--json-schema` / `--output-schema`."""
    schema = {
        "$schema": "http://json-schema.org/draft-07/schema#",
        "title": "AgentProposalsPerCandidateOutput",
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
                        "agent_name",
                        "novelty_rationale",
                    ],
                    "properties": {
                        "title": {"type": "string", "minLength": 1},
                        "detailed_description": {
                            "type": "string",
                            "minLength": 1,
                        },
                        "agent_name": {
                            "type": "string",
                            "const": agent_name,
                        },
                        "novelty_rationale": {
                            "type": "string",
                            "minLength": 1,
                        },
                    },
                },
            },
        },
    }
    return json.dumps(schema)


__all__ = ["build_per_candidate_schema_text"]
