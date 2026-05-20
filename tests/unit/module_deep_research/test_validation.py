"""Unit tests for module deep-research output validation."""

from __future__ import annotations

from spotlights_engine.module_deep_research.validation import parse_module_deep_research_output


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

    output = parse_module_deep_research_output(text, max_findings_per_module=1)

    assert len(output.findings) == 1
    assert output.findings[0].finding_id == "find-0001"
    assert output.findings[0].title == "Paged KV allocation"
    assert output.issues == []


def test_parse_invalid_response_returns_recoverable_issue() -> None:
    output = parse_module_deep_research_output("not json")

    assert output.findings == []
    assert len(output.issues) == 1
    assert output.issues[0].step == "module_deep_research"
    assert output.issues[0].severity == "error"
    assert output.issues[0].recoverable is True
