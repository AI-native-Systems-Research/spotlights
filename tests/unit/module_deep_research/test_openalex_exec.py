"""Unit tests for the OpenAlex deep-research runner.

The runner hits the free OpenAlex REST API; tests fake `urllib.request.urlopen`
so the URL build + response parse are exercised without network.
"""

from __future__ import annotations

import io
import json
import urllib.error
import urllib.parse

import pytest

from spotlights_engine.module_deep_research import openalex_exec
from spotlights_engine.module_deep_research.agent_exec import AgentExecResult
from spotlights_engine.module_deep_research.openalex_exec import (
    OPENALEX_API_KEY_ENV,
    OPENALEX_MAILTO_ENV,
    OpenAlexRunner,
    OpenAlexRunnerOptions,
    _derive_query,
    _merge_works,
    _reconstruct_abstract,
    _sanitize_query,
    _works_to_output_json,
)
from spotlights_engine.module_deep_research.validation import parse_agent_output


class _FakeQueryWriter:
    """Stand-in Codex query-writer; records prompts, returns a canned reply."""

    def __init__(self, final_message: str | None, returncode: int = 0) -> None:
        self.final_message = final_message
        self.returncode = returncode
        self.prompts: list[str] = []

    def run(self, prompt: str, *, check: bool = True) -> AgentExecResult:
        self.prompts.append(prompt)
        return AgentExecResult(
            command=["codex", "exec"],
            returncode=self.returncode,
            stdout="",
            stderr="",
            final_message=self.final_message,
            usage=None,
        )


def _regex_opts(**kw) -> OpenAlexRunnerOptions:
    """Options with deterministic (no-LLM) query mode for hermetic URL tests.

    Extra slices (recency, semantic) default OFF so URL-shape assertions see
    only the keyword slice; individual tests re-enable the slice they exercise.
    """
    kw.setdefault("query_mode", "regex")
    kw.setdefault("recency_results", 0)
    kw.setdefault("semantic_results", 0)
    return OpenAlexRunnerOptions(**kw)


def _work(
    wid="https://openalex.org/W1",
    title="Efficient KV Cache Compression",
    doi="https://doi.org/10.1/x",
    abstract_words=("We", "propose", "a", "method."),
    year=2024,
    cited=42,
    venue="NeurIPS",
    authors=("Ada Lovelace", "Alan Turing"),
):
    aii = {word: [i] for i, word in enumerate(abstract_words)} if abstract_words else None
    return {
        "id": wid,
        "doi": doi,
        "title": title,
        "abstract_inverted_index": aii,
        "publication_year": year,
        "cited_by_count": cited,
        "primary_location": {"source": {"display_name": venue}} if venue else None,
        "authorships": [{"author": {"display_name": n}} for n in authors],
    }


def _install_fake_urlopen(monkeypatch, *, results=None, exc=None, body=None):
    """Patch urllib.request.urlopen; capture the requested URL."""
    calls = {}

    def fake_urlopen(request, timeout=None):
        calls["url"] = request.full_url
        calls.setdefault("urls", []).append(request.full_url)
        calls["headers"] = request.headers
        calls["timeout"] = timeout
        if exc is not None:
            raise exc
        payload = body if body is not None else json.dumps({"results": results or []})
        return io.BytesIO(payload.encode("utf-8"))

    monkeypatch.setattr(openalex_exec.urllib.request, "urlopen", fake_urlopen)
    return calls


def _query_params(url: str) -> dict:
    return dict(urllib.parse.parse_qsl(urllib.parse.urlsplit(url).query))


# --------------------------------------------------------------------------
# API key is optional (free access)
# --------------------------------------------------------------------------


def test_optional_api_key_none_when_unset(monkeypatch):
    monkeypatch.delenv(OPENALEX_API_KEY_ENV, raising=False)
    client = OpenAlexRunner()
    assert client._optional_api_key() is None


def test_optional_api_key_from_option():
    client = OpenAlexRunner(OpenAlexRunnerOptions(api_key="premium"))
    assert client._optional_api_key() == "premium"


def test_url_omits_api_key_when_free(monkeypatch):
    monkeypatch.delenv(OPENALEX_API_KEY_ENV, raising=False)
    monkeypatch.delenv(OPENALEX_MAILTO_ENV, raising=False)
    calls = _install_fake_urlopen(monkeypatch, results=[_work()])
    OpenAlexRunner(_regex_opts()).run("Qualified name: m/x\nObjective: speed\n")
    assert "api_key" not in _query_params(calls["url"])


def test_url_includes_api_key_and_mailto_when_set(monkeypatch):
    calls = _install_fake_urlopen(monkeypatch, results=[_work()])
    client = OpenAlexRunner(
        _regex_opts(api_key="premium", mailto="dev@example.com")
    )
    client.run("Qualified name: m/x\nObjective: speed\n")
    params = _query_params(calls["url"])
    assert params["api_key"] == "premium"
    assert params["mailto"] == "dev@example.com"
    assert "mailto:dev@example.com" in calls["headers"]["User-agent"]


# --------------------------------------------------------------------------
# Query derivation
# --------------------------------------------------------------------------


def test_derive_query_combines_module_and_objective():
    prompt = (
        "Target module:\n"
        "Qualified name: inference/attention-kv\n"
        "Name: attention\n\n"
        "Caller context:\n"
        "Objective: reduce decode latency\n"
    )
    assert _derive_query(prompt) == "inference attention kv reduce decode latency"


def test_derive_query_drops_none_objective():
    assert _derive_query("Qualified name: mod/x\nObjective: (none)\n") == "mod x"


def test_derive_query_empty_when_absent():
    assert _derive_query("no structured fields here") == ""


# --------------------------------------------------------------------------
# Codex query mode (default): LLM query-writer + regex fallback
# --------------------------------------------------------------------------


def test_codex_mode_uses_writer_output_as_search(monkeypatch):
    calls = _install_fake_urlopen(monkeypatch, results=[_work()])
    writer = _FakeQueryWriter("attention kv cache compression")
    client = OpenAlexRunner(
        OpenAlexRunnerOptions(query_mode="codex"), query_writer=writer
    )
    client.run("Qualified name: inference/attention\nObjective: cut latency\n")

    # The writer saw the research prompt; its reply drove the OpenAlex search.
    assert "Qualified name: inference/attention" in writer.prompts[0]
    assert _query_params(calls["urls"][0])["search"] == "attention kv cache compression"


def test_codex_query_brief_strips_execution_sections(monkeypatch):
    # The writer must NOT see Workflow/Output-rules/schema tail — those name
    # wire fields (findings/issues/severity/module_deep_research) that pollute
    # the emitted queries.
    calls = _install_fake_urlopen(monkeypatch, results=[_work()])
    writer = _FakeQueryWriter("fused optimizer update")
    client = OpenAlexRunner(
        OpenAlexRunnerOptions(query_mode="codex"), query_writer=writer
    )
    prompt = (
        "Objective: fuse optimizer kernels\n"
        "Target module: torch/optim\n"
        "\nWorkflow:\n1. read files\n"
        "\nOutput rules:\n- emit findings/issues with severity\n"
        "ModuleDeepResearchOutput JSON schema:\n{...}\n"
    )
    client.run(prompt)

    seen = writer.prompts[0]
    assert "Objective: fuse optimizer kernels" in seen
    assert "Workflow:" not in seen
    assert "findings" not in seen
    assert "severity" not in seen
    assert _query_params(calls["urls"][0])["search"] == "fused optimizer update"


def test_codex_mode_falls_back_to_regex_when_writer_empty(monkeypatch):
    calls = _install_fake_urlopen(monkeypatch, results=[_work()])
    writer = _FakeQueryWriter("")  # no usable query
    client = OpenAlexRunner(
        OpenAlexRunnerOptions(query_mode="codex"), query_writer=writer
    )
    client.run("Qualified name: inference/attention\nObjective: cut latency\n")

    assert _query_params(calls["urls"][0])["search"] == "inference attention cut latency"


def test_codex_mode_falls_back_to_regex_when_writer_fails(monkeypatch):
    calls = _install_fake_urlopen(monkeypatch, results=[_work()])
    writer = _FakeQueryWriter("ignored", returncode=1)
    client = OpenAlexRunner(
        OpenAlexRunnerOptions(query_mode="codex"), query_writer=writer
    )
    client.run("Qualified name: m/x\nObjective: speed\n")

    assert _query_params(calls["urls"][0])["search"] == "m x speed"


def test_codex_mode_issues_one_search_per_query_line(monkeypatch):
    # Each query line hits its own URL; per-query results merge into findings.
    def per_url_results(url):
        if "search=attention" in url:
            return [_work(wid="https://openalex.org/W1", title="Attention Paper", doi="https://doi.org/10.1/a")]
        return [_work(wid="https://openalex.org/W2", title="Quantization Paper", doi="https://doi.org/10.1/b")]

    def fake_urlopen(request, timeout=None):
        results = per_url_results(request.full_url)
        return io.BytesIO(json.dumps({"results": results}).encode("utf-8"))

    monkeypatch.setattr(openalex_exec.urllib.request, "urlopen", fake_urlopen)
    writer = _FakeQueryWriter("attention kv cache\nquantization compression")
    client = OpenAlexRunner(
        OpenAlexRunnerOptions(query_mode="codex", max_queries=4), query_writer=writer
    )
    result = client.run("Qualified name: inference/attention\nObjective: cut latency\n")

    parsed = parse_agent_output(result.final_message)
    # Round-robin interleave: top hit of each query first.
    assert [f.title for f in parsed.findings] == ["Attention Paper", "Quantization Paper"]
    assert [q.query for q in parsed.search_queries] == [
        "attention kv cache",
        "quantization compression",
    ]


def test_codex_mode_dedups_same_work_across_queries(monkeypatch):
    # Both queries return the SAME work (same OpenAlex id) -> one finding.
    def fake_urlopen(request, timeout=None):
        return io.BytesIO(
            json.dumps({"results": [_work(wid="https://openalex.org/W1")]}).encode("utf-8")
        )

    monkeypatch.setattr(openalex_exec.urllib.request, "urlopen", fake_urlopen)
    writer = _FakeQueryWriter("attention\nkv cache")
    client = OpenAlexRunner(
        OpenAlexRunnerOptions(query_mode="codex"), query_writer=writer
    )
    result = client.run("Qualified name: inference/attention\nObjective: cut latency\n")

    parsed = parse_agent_output(result.final_message)
    assert len(parsed.findings) == 1
    assert len(parsed.search_queries) == 2  # both queries still reported


def test_sanitize_query_preserves_operators_takes_last_line():
    raw = 'Here is a query:\n`("KV cache" OR "key-value cache") compression`'
    assert _sanitize_query(raw) == '("KV cache" OR "key-value cache") compression'


def test_sanitize_query_strips_label_and_markdown():
    assert _sanitize_query("Keywords: **attention** quantization.") == (
        "attention quantization"
    )


def test_sanitize_query_drops_unbalanced_quotes():
    assert _sanitize_query('"attention cache') == "attention cache"


def test_sanitize_query_empty_inputs():
    assert _sanitize_query("") == ""
    assert _sanitize_query("\n  \n") == ""


# --------------------------------------------------------------------------
# URL construction
# --------------------------------------------------------------------------


def test_run_builds_relevance_search_url(monkeypatch):
    calls = _install_fake_urlopen(monkeypatch, results=[_work()])
    client = OpenAlexRunner(_regex_opts(api_key="k", max_results=5))
    client.run("Qualified name: inference/attention\nObjective: cut latency\n")

    # calls["urls"][0] is the relevance slice; a recency slice may follow.
    params = _query_params(calls["urls"][0])
    assert params["search"] == "inference attention cut latency"
    assert params["filter"] == "type:article|preprint,primary_topic.field.id:17|26|22|31|18"
    assert params["per_page"] == "5"
    assert "abstract_inverted_index" in params["select"]


def test_run_command_redacts_api_key(monkeypatch):
    _install_fake_urlopen(monkeypatch, results=[_work()])
    client = OpenAlexRunner(_regex_opts(api_key="secret-key"))
    result = client.run("Qualified name: m/x\nObjective: speed\n")
    assert result.command[0] == "GET"
    assert "secret-key" not in result.command[1]
    assert "api_key=***" in result.command[1]


def test_per_page_clamped_to_200(monkeypatch):
    calls = _install_fake_urlopen(monkeypatch, results=[])
    OpenAlexRunner(_regex_opts(max_results=9999)).run("Qualified name: m/x\n")
    # Relevance slice is the first URL; recency slice (if any) uses its own per_page.
    assert _query_params(calls["urls"][0])["per_page"] == "200"


def test_recency_slice_appends_recent_low_cited_work(monkeypatch):
    # Relevance slice returns a high-cited family paper; the recency slice
    # (sort=publication_date:desc) returns a brand-new low-cited preprint the
    # relevance ranking would bury. Both must land in the findings.
    def fake_urlopen(request, timeout=None):
        if "sort=publication_date" in request.full_url:
            work = _work(wid="https://openalex.org/W-NEW", title="Fresh Preprint",
                         doi="https://doi.org/10.1/new", cited=1)
        else:
            work = _work(wid="https://openalex.org/W-OLD", title="Established Paper",
                         doi="https://doi.org/10.1/old", cited=999)
        return io.BytesIO(json.dumps({"results": [work]}).encode("utf-8"))

    monkeypatch.setattr(openalex_exec.urllib.request, "urlopen", fake_urlopen)
    result = OpenAlexRunner(_regex_opts(recency_results=3)).run(
        "Qualified name: m/x\nObjective: speed\n"
    )

    parsed = parse_agent_output(result.final_message)
    titles = [f.title for f in parsed.findings]
    assert titles == ["Established Paper", "Fresh Preprint"]  # relevance first, recency tail


def test_recency_slice_disabled_when_zero(monkeypatch):
    calls = _install_fake_urlopen(monkeypatch, results=[_work()])
    OpenAlexRunner(_regex_opts(recency_results=0)).run("Qualified name: m/x\nObjective: speed\n")
    assert len(calls["urls"]) == 1  # only the relevance slice, no recency fetch
    assert "sort=publication_date" not in calls["urls"][0]


def test_semantic_slice_uses_search_semantic_and_caps_per_page(monkeypatch):
    calls = _install_fake_urlopen(monkeypatch, results=[_work()])
    OpenAlexRunner(_regex_opts(semantic_results=200)).run(
        "Qualified name: m/x\nObjective: speed\n"
    )
    # Two slices: keyword relevance, then the semantic slice.
    assert len(calls["urls"]) == 2
    keyword, semantic = calls["urls"]
    assert "search" in _query_params(keyword)
    params = _query_params(semantic)
    # Semantic slice swaps to search.semantic (no plain `search`) and its
    # per_page is clamped to the endpoint's 50 ceiling even when asked for 200.
    assert params["search.semantic"] == "m x speed"
    assert "search" not in params
    assert params["per_page"] == "50"


def test_semantic_slice_disabled_when_zero(monkeypatch):
    calls = _install_fake_urlopen(monkeypatch, results=[_work()])
    OpenAlexRunner(_regex_opts(semantic_results=0)).run(
        "Qualified name: m/x\nObjective: speed\n"
    )
    assert len(calls["urls"]) == 1
    assert not any("search.semantic" in u for u in calls["urls"])


def test_semantic_slice_rescues_work_missing_from_keyword_pool(monkeypatch):
    # The keyword (relevance) slice buries a fresh low-cited GT; the semantic
    # slice (search.semantic) surfaces it. Both must land in the findings.
    def fake_urlopen(request, timeout=None):
        if "search.semantic" in request.full_url:
            work = _work(wid="https://openalex.org/W-SEM", title="Semantic Rescue",
                         doi="https://doi.org/10.1/sem", cited=1)
        else:
            work = _work(wid="https://openalex.org/W-KW", title="Keyword Hit",
                         doi="https://doi.org/10.1/kw", cited=999)
        return io.BytesIO(json.dumps({"results": [work]}).encode("utf-8"))

    monkeypatch.setattr(openalex_exec.urllib.request, "urlopen", fake_urlopen)
    result = OpenAlexRunner(_regex_opts(semantic_results=50)).run(
        "Qualified name: m/x\nObjective: speed\n"
    )

    parsed = parse_agent_output(result.final_message)
    titles = [f.title for f in parsed.findings]
    assert titles == ["Keyword Hit", "Semantic Rescue"]  # keyword first, semantic tail


# --------------------------------------------------------------------------
# Response parsing -> findings
# --------------------------------------------------------------------------


def test_run_parses_results_into_findings(monkeypatch):
    results = [
        _work(wid="https://openalex.org/W1", title="Paper One"),
        _work(
            wid="https://openalex.org/W2",
            title="Paper Two",
            doi=None,
            abstract_words=None,
            venue="",
            authors=(),
        ),
    ]
    _install_fake_urlopen(monkeypatch, results=results)
    result = OpenAlexRunner(_regex_opts()).run("Qualified name: m/x\nObjective: speed\n")

    assert result.returncode == 0
    parsed = parse_agent_output(result.final_message)
    assert [f.title for f in parsed.findings] == ["Paper One", "Paper Two"]
    assert all(f.source_type == "paper" for f in parsed.findings)
    assert [f.finding_id for f in parsed.findings] == ["find-0001", "find-0002"]
    # W2 has no DOI -> falls back to the OpenAlex id
    assert parsed.findings[1].url == "https://openalex.org/W2"
    # enrichment: venue/year + authors + cited_by land in the text
    assert "NeurIPS" in parsed.findings[0].technique_summary
    assert "Ada Lovelace" in parsed.findings[0].supporting_evidence
    assert "cited_by=42" in parsed.findings[0].supporting_evidence
    assert parsed.search_queries[0].tool == "openalex"


def test_run_no_results_yields_issue(monkeypatch):
    _install_fake_urlopen(monkeypatch, results=[])
    result = OpenAlexRunner(_regex_opts()).run("Qualified name: m/x\nObjective: speed\n")
    parsed = parse_agent_output(result.final_message)
    assert parsed.findings == []
    assert parsed.issues
    assert "no usable works" in parsed.issues[0].message


# --------------------------------------------------------------------------
# Failure handling
# --------------------------------------------------------------------------


def test_run_http_error_raises_when_checked(monkeypatch):
    err = urllib.error.HTTPError("u", 429, "Too Many Requests", {}, None)
    _install_fake_urlopen(monkeypatch, exc=err)
    with pytest.raises(RuntimeError, match="HTTP 429"):
        OpenAlexRunner(_regex_opts()).run("Qualified name: m/x\n", check=True)


def test_run_http_error_returns_code_when_unchecked(monkeypatch):
    err = urllib.error.HTTPError("u", 429, "Too Many Requests", {}, None)
    _install_fake_urlopen(monkeypatch, exc=err)
    result = OpenAlexRunner(_regex_opts()).run("Qualified name: m/x\n", check=False)
    assert result.returncode == 429
    assert result.final_message is None
    assert "429" in result.stderr


def test_run_urlerror_returns_1_when_unchecked(monkeypatch):
    _install_fake_urlopen(monkeypatch, exc=urllib.error.URLError("offline"))
    result = OpenAlexRunner(_regex_opts()).run("Qualified name: m/x\n", check=False)
    assert result.returncode == 1
    assert result.final_message is None
    assert "offline" in result.stderr


def test_run_bad_json_returns_1_when_unchecked(monkeypatch):
    _install_fake_urlopen(monkeypatch, body="not json")
    result = OpenAlexRunner(_regex_opts()).run("Qualified name: m/x\n", check=False)
    assert result.returncode == 1
    assert result.final_message is None


# --------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------


def test_reconstruct_abstract_orders_by_position():
    assert _reconstruct_abstract({"cache": [2], "The": [0], "kv": [1]}) == "The kv cache"


def test_reconstruct_abstract_empty_inputs():
    assert _reconstruct_abstract(None) == ""
    assert _reconstruct_abstract({}) == ""


def test_works_to_output_json_roundtrips_through_parser():
    per_query = [("kv cache", "https://api.openalex.org/works?x=1", [_work()])]
    merged = _merge_works(per_query)
    payload = _works_to_output_json(per_query, merged)
    parsed = parse_agent_output(payload)
    assert len(parsed.findings) == 1
    f = parsed.findings[0]
    assert f.source_type == "paper"
    assert f.technique_summary
    assert "OpenAlex" in f.supporting_evidence
