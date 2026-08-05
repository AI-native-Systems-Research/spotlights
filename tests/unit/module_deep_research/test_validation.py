"""Unit tests for module deep-research output validation."""

from __future__ import annotations

import json

from spotlights_engine.module_deep_research.validation import (
    parse_agent_output,
    parse_module_deep_research_output,
)


def _payload_with_one_finding() -> dict:
    return {
        "findings": [
            {
                "finding_id": "find-0001",
                "title": "Paged KV allocation",
                "url": "https://example.com/a",
                "source_type": "paper",
                "technique_summary": "Paged allocation reduces KV fragmentation.",
                "supporting_evidence": "The source reports lower memory waste.",
            }
        ],
        "issues": [],
    }


def test_parse_accepts_fenced_json_and_normalizes_ids_and_cap() -> None:
    text = """Here is the result:
```json
{
  "findings": [
    {
      "finding_id": "find-0099",
      "title": "Paged KV allocation",
      "url": "https://example.com/a",
      "source_type": "paper",
      "technique_summary": "Paged allocation reduces KV fragmentation.",
      "supporting_evidence": "The source reports lower memory waste."
    },
    {
      "finding_id": "find-0100",
      "title": "Chunked prefill",
      "url": "https://example.com/b",
      "source_type": "docs",
      "technique_summary": "Chunking prefill improves latency isolation."
    }
  ],
  "issues": []
}
```
"""

    output = parse_module_deep_research_output(
        text, max_findings_per_candidate=1, segment="kv_offload"
    )

    assert len(output.findings) == 1
    # The agent emits bare `find-NNNN`; the finding id is renumbered and
    # prefixed with the module segment (D3).
    assert output.findings[0].finding_id == "find-kv_offload-0001"
    assert output.findings[0].title == "Paged KV allocation"
    assert output.issues == []


def test_parse_invalid_response_returns_recoverable_issue() -> None:
    output = parse_module_deep_research_output("not json")

    assert output.findings == []
    assert len(output.issues) == 1
    assert output.issues[0].step == "module_deep_research"
    assert output.issues[0].severity == "error"
    assert output.issues[0].recoverable is True


def test_parse_picks_findings_object_after_leading_preamble() -> None:
    """Some runners concatenate a multi-object NDJSON stream into `{...}\\n{...}`;
    the payload both starts with `{` and ends with `}`, which used to make
    `json.loads` raise `Extra data`. The parser now scans the objects and picks
    the one carrying `findings`, even when a smaller preamble object precedes it."""
    preamble = json.dumps({"status": "thinking"})
    payload = _payload_with_one_finding()
    text = f"{preamble}\n{json.dumps(payload)}"

    output = parse_module_deep_research_output(text, segment="kv_offload")

    assert output.issues == []
    assert len(output.findings) == 1
    assert output.findings[0].finding_id == "find-kv_offload-0001"


def test_parse_keeps_findings_when_first_object_carries_them() -> None:
    """When the real payload is first and extra data trails it, the findings are
    kept and the trailing content is ignored rather than failing the parse."""
    payload = _payload_with_one_finding()
    text = f"{json.dumps(payload)}\n{json.dumps({'status': 'done'})}"

    output = parse_module_deep_research_output(text, segment="kv_offload")

    assert output.issues == []
    assert len(output.findings) == 1
    assert output.findings[0].finding_id == "find-kv_offload-0001"


def test_parse_picks_issue_only_payload_after_leading_preamble() -> None:
    """If an empty/failed survey omits `findings`, the parser should still skip
    transport preamble objects and keep the issue payload."""
    preamble = json.dumps({"status": "thinking"})
    payload = {
        "issues": [
            {
                "step": "module_deep_research",
                "severity": "warning",
                "message": "no relevant sources survived filtering",
                "recoverable": True,
            }
        ],
        "search_queries": [],
    }
    text = f"{preamble}\n{json.dumps(payload)}"

    output = parse_module_deep_research_output(text, segment="kv_offload")

    assert output.findings == []
    assert len(output.issues) == 1
    assert output.issues[0].message == "no relevant sources survived filtering"


def test_parse_skips_malformed_leading_brace_before_payload() -> None:
    payload = _payload_with_one_finding()
    text = "{not valid json\n" + json.dumps(payload)

    output = parse_module_deep_research_output(text, segment="kv_offload")

    assert output.issues == []
    assert len(output.findings) == 1
    assert output.findings[0].finding_id == "find-kv_offload-0001"


def test_parse_ignores_non_payload_fence_when_findings_object_follows() -> None:
    payload = _payload_with_one_finding()
    text = f'```json\n{json.dumps({"status": "thinking"})}\n```\n{json.dumps(payload)}'

    output = parse_module_deep_research_output(text, segment="kv_offload")

    assert output.issues == []
    assert len(output.findings) == 1
    assert output.findings[0].finding_id == "find-kv_offload-0001"


def test_search_query_extra_field_and_empty_query_do_not_drop_findings() -> None:
    """HIGH strictness regression: a search result carrying an extra field, and
    an empty query, must still parse and keep ALL findings — the advisory search
    log must never be able to fail validation and discard load-bearing findings.
    """
    payload = {
        "findings": [
            {
                "finding_id": "find-0001",
                "title": "Paged KV allocation",
                "url": "https://example.com/a",
                "source_type": "paper",
                "technique_summary": "Paged allocation reduces KV fragmentation.",
            }
        ],
        "issues": [],
        "search_queries": [
            {
                "query": "paged kv cache allocation",
                "tool": "web_search",
                "results": [
                    {
                        "title": "vLLM",
                        "url": "https://example.com/a",
                        "snippet": "PagedAttention.",
                        "rank": 1,  # extra field must be tolerated (extra="ignore")
                    }
                ],
            },
            {"query": "", "results": []},  # empty query must not fail the payload
        ],
    }

    output = parse_module_deep_research_output(
        json.dumps(payload), max_findings_per_candidate=30, segment="kv_offload"
    )

    assert len(output.findings) == 1
    assert output.findings[0].title == "Paged KV allocation"
    assert output.issues == []
    # Queries carried through and tagged with the standalone synthetic agent.
    assert len(output.search_queries) == 2
    assert output.search_queries[0].query == "paged kv cache allocation"
    assert output.search_queries[0].agent == "agent"
    assert output.search_queries[0].results[0].title == "vLLM"
    assert output.search_queries[1].query == ""


def test_parse_agent_output_captures_search_queries() -> None:
    payload = {
        "findings": [],
        "issues": [],
        "search_queries": [
            {"query": "q1", "tool": "web_search", "results": []},
        ],
    }

    parsed = parse_agent_output(json.dumps(payload))

    assert len(parsed.search_queries) == 1
    assert parsed.search_queries[0].query == "q1"


def test_parse_agent_output_without_search_queries_defaults_to_empty() -> None:
    payload = {"findings": [], "issues": []}

    parsed = parse_agent_output(json.dumps(payload))

    assert parsed.search_queries == []


def test_parse_stamps_candidate_id_on_findings_and_queries() -> None:
    """Decision D2/D7: the promoted findings and search queries carry the
    candidate id the survey ran for, and the finding id uses the composite
    per-candidate segment (D3)."""
    payload = {
        "findings": [
            {
                "finding_id": "find-0001",
                "title": "Paged KV allocation",
                "url": "https://example.com/a",
                "source_type": "paper",
                "technique_summary": "Paged allocation reduces KV fragmentation.",
            }
        ],
        "issues": [],
        "search_queries": [{"query": "paged kv", "tool": "web_search", "results": []}],
    }

    output = parse_module_deep_research_output(
        json.dumps(payload),
        segment="kv_offload-0002",
        candidate_id="cand-kv_offload-0002",
    )

    assert len(output.findings) == 1
    assert output.findings[0].finding_id == "find-kv_offload-0002-0001"
    assert output.findings[0].candidate_id == "cand-kv_offload-0002"
    assert output.search_queries[0].candidate_id == "cand-kv_offload-0002"
