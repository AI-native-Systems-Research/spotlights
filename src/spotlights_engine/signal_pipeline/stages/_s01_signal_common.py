"""Shared contract for stage 01 signal extraction, across telemetry sources.

Both the file/JSON path (`s01_signal_extraction`) and the SigNoz/SQL path
(`s01_signal_extraction_signoz`) produce the same `Signals` and share this
minimal contract: the stage error type, the agent output schema, and the
`claude -p` budgets. Kept in a leaf module (no imports from either source) so
the two extraction paths stay fully decoupled.
"""

from __future__ import annotations


class SignalExtractionError(RuntimeError):
    pass


# Minimal output schema. Top-level keys required, plus `workload_id` —
# everything else is the agent's call. `additionalProperties` is
# *unspecified* so the agent can surface richer fields without violating
# the constraint.
OUTPUT_JSON_SCHEMA = {
    "type": "object",
    "required": ["workload", "traces", "anomalies"],
    "properties": {
        "workload": {
            "type": "object",
            "required": ["workload_id"],
            "properties": {
                "workload_id": {"type": "string", "minLength": 1},
            },
        },
        "traces": {"type": "array", "items": {"type": "object"}},
        "anomalies": {"type": "array", "items": {"type": "object"}},
    },
}

# Generous budget — agent may iterate to grep, jq/SQL, sample, then summarize.
# Wall-clock is the binding limit in practice: with a large (~76MB)
# metrics.jsonl (file path) or a chatty multi-family run (SigNoz), repeated
# passes plus model latency can blow past 30min before the summarize step.
# 60min gives headroom; turns (60) are rarely the limiter.
MAX_TURNS = 60
TIMEOUT_S = 3600


__all__ = ["SignalExtractionError", "OUTPUT_JSON_SCHEMA", "MAX_TURNS", "TIMEOUT_S"]
