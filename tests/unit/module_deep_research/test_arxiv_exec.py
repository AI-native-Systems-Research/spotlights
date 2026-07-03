"""Unit tests for the arXiv-search runner (no live network)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from spotlights_engine.module_deep_research import arxiv_exec
from spotlights_engine.module_deep_research.arxiv_exec import (
    ArxivQueryPlan,
    ArxivSearchClient,
    ArxivSearchOptions,
    PlannedArxivQuery,
    _humanize_identifier,
    _looks_like_code_identifier,
    _validate_and_render_plan,
    plan_queries_template,
    render_arxiv_query_planner_prompt,
)
from spotlights_engine.module_deep_research.validation import parse_agent_output
from spotlights_engine.schemas.candidate import Candidate, CodeLocation, CodeSpan
from spotlights_engine.schemas.common import SpotlightContext
from spotlights_engine.schemas.pipeline import ModuleDeepResearchInput
from spotlights_engine.schemas.project import Module, ProjectTree, Repository

# ---------------------------------------------------------------------------
# fixtures / helpers
# ---------------------------------------------------------------------------


def _module() -> Module:
    return Module(
        name="kv_offload",
        path="src/engine/kv_offload",
        description=(
            "Offloads transformer KV cache blocks from GPU to host memory under "
            "memory pressure. An LRU policy picks the victim blocks to evict; "
            "evicted blocks are copied back to GPU memory on demand."
        ),
    )


def _tree() -> ProjectTree:
    return ProjectTree(
        repository=Repository(name="demo", summary="Demo.", source_root="src"),
        modules=[Module(name="engine", path="src/engine", submodules=[_module()])],
    )


def _candidates() -> list[Candidate]:
    return [
        Candidate(
            id="cand-kv_offload-0002",
            origin="code_agent",
            estimated_impact="medium",
            locations=[
                CodeLocation(
                    file="src/engine/kv_offload/transfer.py",
                    spans=[
                        CodeSpan(
                            line_start=1,
                            line_end=2,
                            symbol="_copy_block_to_host",
                            kind="function",
                        )
                    ],
                )
            ],
            description=(
                "Blocks are copied to host memory synchronously on the compute "
                "stream, stalling the forward pass during eviction bursts."
            ),
            current_approach="synchronous copy",
            evolve_rationale=(
                "Overlapping the copies with computation on a side stream could "
                "hide the copy latency entirely."
            ),
            estimated_impact_explanation="x",
        ),
        Candidate(
            id="cand-kv_offload-0001",
            origin="code_agent",
            estimated_impact="high",
            locations=[
                CodeLocation(
                    file="src/engine/kv_offload/policy.py",
                    spans=[
                        CodeSpan(
                            line_start=1,
                            line_end=2,
                            symbol="LRUPolicy.select_victim",
                            kind="method",
                        )
                    ],
                )
            ],
            description=(
                "Victim selection is plain LRU and ignores attention patterns, so "
                "hot blocks get evicted and re-fetched repeatedly."
            ),
            current_approach="lru",
            evolve_rationale=(
                "Attention-aware cache eviction keeps hot blocks resident and "
                "should cut transfer traffic."
            ),
            estimated_impact_explanation="x",
        ),
    ]


def _request(*, hotspots: bool = True, candidates: bool = True) -> ModuleDeepResearchInput:
    return ModuleDeepResearchInput(
        project_tree=_tree(),
        module_qualified_name="engine/kv_offload",
        context=SpotlightContext(
            objective=(
                "Reduce GPU memory usage during long-context inference without "
                "increasing latency."
            ),
            workload_hints=["long context inference", "single GPU serving"],
            validation_plan=["run the latency benchmark"],
        ),
        repo_path=Path("/tmp/example-repo"),
        candidates=_candidates() if candidates else [],
        include_candidate_hotspots=hotspots,
    )


_ATOM_TEMPLATE = """<?xml version="1.0" encoding="UTF-8"?>
<feed xmlns="http://www.w3.org/2005/Atom" xmlns:arxiv="http://arxiv.org/schemas/atom">
{entries}
</feed>"""

_ENTRY_TEMPLATE = """  <entry>
    <id>http://arxiv.org/abs/{id}</id>
    <title>{title}</title>
    <summary>{summary}</summary>
    {doi}
  </entry>"""


def _atom(entries: list[dict]) -> str:
    blocks = []
    for e in entries:
        doi = (
            f"<arxiv:doi>{e['doi']}</arxiv:doi>" if e.get("doi") else ""
        )
        blocks.append(
            _ENTRY_TEMPLATE.format(
                id=e["id"], title=e["title"], summary=e["summary"], doi=doi
            )
        )
    return _ATOM_TEMPLATE.format(entries="\n".join(blocks))


# ---------------------------------------------------------------------------
# deterministic helpers
# ---------------------------------------------------------------------------


def test_humanize_identifier() -> None:
    assert _humanize_identifier("kv_offload") == "kv offload"
    assert _humanize_identifier("LRUPolicy.select_victim") == "lru policy select victim"
    assert _humanize_identifier("_copy_block_to_host") == "copy block to host"


def test_looks_like_code_identifier() -> None:
    assert _looks_like_code_identifier("kv_offload")
    assert _looks_like_code_identifier("LRUPolicy.select_victim")
    assert _looks_like_code_identifier("camelCase")
    assert _looks_like_code_identifier("KVCache")
    assert _looks_like_code_identifier("LRUPolicy")
    assert not _looks_like_code_identifier("kv cache offloading")
    # A decimal / version number is natural-language, not a dotted code path.
    assert not _looks_like_code_identifier("gpt-4.5")
    assert not _looks_like_code_identifier("3.1")


def test_parse_arxiv_id_preserves_old_style_category() -> None:
    from spotlights_engine.module_deep_research.arxiv_exec import _parse_arxiv_id

    assert _parse_arxiv_id("http://arxiv.org/abs/2401.01234v2") == "2401.01234"
    # Old-style ids keep their category prefix (must match orchestration's key).
    assert _parse_arxiv_id("http://arxiv.org/abs/hep-th/9901001") == "hep-th/9901001"
    assert _parse_arxiv_id("http://arxiv.org/abs/math.GT/0309136v1") == "math.GT/0309136"
    assert _parse_arxiv_id("https://arxiv.org/pdf/2401.01234v1.pdf") == "2401.01234"


# ---------------------------------------------------------------------------
# template planner
# ---------------------------------------------------------------------------


def test_template_planner_is_deterministic() -> None:
    req = _request()
    opts = ArxivSearchOptions()
    assert plan_queries_template(req, _module(), opts) == plan_queries_template(
        req, _module(), opts
    )


def test_template_planner_query_shape_and_traps() -> None:
    queries = plan_queries_template(_request(), _module(), ArxivSearchOptions())
    assert queries
    # No raw span symbol / module-name identifier leaks into any query.
    joined = " ".join(queries)
    for leak in ("select_victim", "LRUPolicy", "_copy_block_to_host", "kv_offload"):
        assert leak not in joined
    # Multi-word concepts are quoted phrases (no unquoted prose implicit-OR).
    assert 'all:"' in queries[0]
    # A broad (OR) and a precision (AND) query both appear.
    assert any(" OR " in q for q in queries)
    assert any(" AND " in q for q in queries)
    # All under the 1024-char cap; at most max_queries.
    assert all(len(q) <= 1024 for q in queries)
    assert len(queries) <= 4


def test_template_planner_hints_verbatim_and_no_candidates_when_gated() -> None:
    queries = plan_queries_template(
        _request(hotspots=False), _module(), ArxivSearchOptions()
    )
    joined = " ".join(queries)
    # Workload hints appear verbatim as quoted phrases.
    assert 'all:"long context inference"' in joined
    # No candidate-derived precision terms leak in when hotspots are gated.
    assert "attention" not in joined


def test_template_planner_respects_max_queries_cap() -> None:
    queries = plan_queries_template(
        _request(), _module(), ArxivSearchOptions(max_queries=2)
    )
    assert len(queries) <= 2


# ---------------------------------------------------------------------------
# parse / validate / render (Claude wire schema)
# ---------------------------------------------------------------------------


def test_validate_render_cnf_shape() -> None:
    plan = ArxivQueryPlan(
        queries=[
            PlannedArxivQuery(clauses=[["p1", "p2"], ["p3"]]),
        ]
    )
    _, rendered, _ = _validate_and_render_plan(plan, max_queries=4)
    assert rendered == ['(all:"p1" OR all:"p2") AND (all:"p3")']


def test_validate_drops_code_identifiers_and_long_phrases() -> None:
    plan = ArxivQueryPlan(
        queries=[
            PlannedArxivQuery(
                clauses=[["LRUPolicy.select_victim", "KVCache", "kv cache eviction"]]
            ),
            PlannedArxivQuery(clauses=[["one two three four five six"]]),
        ]
    )
    survived, rendered, rejections = _validate_and_render_plan(plan, max_queries=4)
    reasons = {r.reason for r in rejections}
    assert "code_identifier" in reasons
    assert "too_many_words" in reasons
    # First query survives on the remaining valid phrase; second dies entirely.
    assert rendered == ['(all:"kv cache eviction")']


def test_validate_drops_over_constrained_queries() -> None:
    plan = ArxivQueryPlan(
        queries=[PlannedArxivQuery(clauses=[["a"], ["b"], ["c"], ["d"]])]
    )
    _, rendered, rejections = _validate_and_render_plan(plan, max_queries=4)
    assert rendered == []
    assert any(r.reason == "too_many_clauses" for r in rejections)


def test_validate_dedups_and_truncates() -> None:
    plan = ArxivQueryPlan(
        queries=[
            PlannedArxivQuery(clauses=[["dup"]]),
            PlannedArxivQuery(clauses=[["dup"]]),
            PlannedArxivQuery(clauses=[["one"]]),
            PlannedArxivQuery(clauses=[["two"]]),
        ]
    )
    _, rendered, rejections = _validate_and_render_plan(plan, max_queries=2)
    assert len(rendered) == 2
    assert any(r.reason == "duplicate" for r in rejections)


def test_validate_enforces_length_cap() -> None:
    # A genuinely long single OR clause (distinct phrases) blows past 1024 chars.
    big_clause = [f"phrase number {i:03d}" for i in range(300)]
    plan = ArxivQueryPlan(queries=[PlannedArxivQuery(clauses=[big_clause])])
    _, rendered, rejections = _validate_and_render_plan(plan, max_queries=4)
    assert rendered == []
    assert any(r.reason == "over_length_cap" for r in rejections)


def test_parse_query_plan_handles_fenced_json() -> None:
    from spotlights_engine.module_deep_research.arxiv_exec import _parse_query_plan

    text = 'prefix\n```json\n{"queries": [{"clauses": [["kv cache"]]}]}\n```\nsuffix'
    plan = _parse_query_plan(text)
    assert plan.queries[0].clauses == [["kv cache"]]


# ---------------------------------------------------------------------------
# planner prompt render
# ---------------------------------------------------------------------------


def test_planner_prompt_sorts_candidates_and_omits_validation_plan() -> None:
    prompt = render_arxiv_query_planner_prompt(
        _request(), _module(), ArxivSearchOptions()
    )
    # High-impact candidate is listed before the medium one.
    assert prompt.index("LRUPolicy.select_victim") < prompt.index("_copy_block_to_host")
    # validation_plan is not embedded.
    assert "run the latency benchmark" not in prompt
    # schema + max_queries embedded.
    assert "ArxivQueryPlan" in prompt
    assert "up to 4 arXiv" in prompt


def test_planner_prompt_omits_hotspots_when_gated() -> None:
    prompt = render_arxiv_query_planner_prompt(
        _request(hotspots=False), _module(), ArxivSearchOptions()
    )
    assert "Identified hot spots" not in prompt


# ---------------------------------------------------------------------------
# Claude planner fallback contract
# ---------------------------------------------------------------------------


class _FakeClaude:
    """Stand-in for ClaudeExecClient with a scripted single response."""

    instances: list = []

    def __init__(self, options=None) -> None:  # noqa: ANN001
        self.options = options
        _FakeClaude.instances.append(self)

    # Class-level knobs the tests set before run().
    behavior = "ok"

    def run(self, prompt: str, *, check: bool = True):  # noqa: ANN001, ARG002
        from spotlights_engine.module_deep_research.agent_exec import AgentExecResult

        if _FakeClaude.behavior == "raise":
            raise RuntimeError("boom")
        if _FakeClaude.behavior == "garbage":
            msg = "not json at all"
        elif _FakeClaude.behavior == "empty":
            msg = '{"queries": []}'
        elif _FakeClaude.behavior == "invalid_only":
            msg = '{"queries": [{"clauses": [["LRUPolicy.select_victim"]]}]}'
        elif _FakeClaude.behavior == "nonzero":
            return AgentExecResult(
                command=["claude"], returncode=1, stdout="", stderr="err",
                final_message=None,
            )
        else:  # ok
            msg = '{"queries": [{"clauses": [["kv cache offloading"], ["gpu memory"]]}]}'
        return AgentExecResult(
            command=["claude"], returncode=0, stdout="", stderr="",
            final_message=msg,
        )


@pytest.fixture(autouse=True)
def _reset_fake_claude():
    _FakeClaude.instances = []
    _FakeClaude.behavior = "ok"
    yield


def _client(monkeypatch, options: ArxivSearchOptions, **kw) -> ArxivSearchClient:
    monkeypatch.setattr(arxiv_exec, "ClaudeExecClient", _FakeClaude)
    return ArxivSearchClient(_request(), _module(), options, **kw)


def _run_no_network(monkeypatch, client: ArxivSearchClient):
    """Run the client with HTTP + sleep stubbed to return no entries."""
    monkeypatch.setattr(arxiv_exec, "_sleep", lambda *_: None)
    monkeypatch.setattr(
        arxiv_exec,
        "_get_text",
        lambda url, *, timeout, headers: arxiv_exec._HttpResponse(body=_atom([]), status=200),
    )
    return client.run("ignored prompt")


def test_claude_planner_ok_path(monkeypatch) -> None:
    _FakeClaude.behavior = "ok"
    client = _client(monkeypatch, ArxivSearchOptions())
    planned, rendered, planner_used, reason, _ = client._plan()
    assert planner_used == "claude"
    assert reason is None
    assert planned == [
        PlannedArxivQuery(clauses=[["kv cache offloading"], ["gpu memory"]])
    ]
    assert rendered == ['(all:"kv cache offloading") AND (all:"gpu memory")']


@pytest.mark.parametrize(
    "behavior", ["raise", "garbage", "empty", "invalid_only", "nonzero"]
)
def test_claude_planner_falls_back_to_template(monkeypatch, behavior) -> None:
    _FakeClaude.behavior = behavior
    client = _client(monkeypatch, ArxivSearchOptions())
    planned, rendered, planner_used, reason, _ = client._plan()
    assert planner_used == "template"
    assert reason is not None
    assert planned
    assert rendered  # template produced queries


def test_template_mode_never_constructs_session(monkeypatch) -> None:
    client = _client(monkeypatch, ArxivSearchOptions(query_planner="template"))
    client._plan()
    assert _FakeClaude.instances == []


def test_planner_session_uses_readonly_tools(monkeypatch) -> None:
    _FakeClaude.behavior = "ok"
    client = _client(monkeypatch, ArxivSearchOptions())
    client._plan()
    opts = _FakeClaude.instances[0].options
    assert set(opts.allowed_tools) == {"Read", "Grep", "Glob", "LS"}


def test_run_fallback_emits_single_issue(monkeypatch) -> None:
    _FakeClaude.behavior = "raise"
    client = _client(monkeypatch, ArxivSearchOptions())
    result = _run_no_network(monkeypatch, client)
    parsed = parse_agent_output(result.final_message)
    fallback_issues = [i for i in parsed.issues if "template planner" in i.message]
    assert len(fallback_issues) == 1


# ---------------------------------------------------------------------------
# fetch / normalize
# ---------------------------------------------------------------------------


def test_atom_entry_to_finding_mapping(monkeypatch) -> None:
    entries = [
        {
            "id": "2401.01234v2",
            "title": "KV Cache Offloading for Long-Context Inference",
            "summary": "We offload KV cache blocks to host memory. It works well.",
            "doi": "10.1000/xyz",
        }
    ]
    monkeypatch.setattr(arxiv_exec, "_sleep", lambda *_: None)
    monkeypatch.setattr(
        arxiv_exec,
        "_get_text",
        lambda url, *, timeout, headers: arxiv_exec._HttpResponse(
            body=_atom(entries), status=200
        ),
    )
    client = _client(monkeypatch, ArxivSearchOptions(query_planner="template"))
    result = client.run("ignored")
    parsed = parse_agent_output(result.final_message)
    assert result.returncode == 0
    assert len(parsed.findings) == 1
    f = parsed.findings[0]
    assert f.source_type == "paper"
    # Version-stripped abs URL.
    assert f.url == "https://arxiv.org/abs/2401.01234"
    assert f.technique_summary.startswith("[arxiv]")
    assert "arXiv:2401.01234" in f.supporting_evidence
    assert "DOI:10.1000/xyz" in f.supporting_evidence


def test_run_dedups_versioned_duplicates(monkeypatch) -> None:
    entries = [
        {"id": "2401.01234v1", "title": "Same Paper", "summary": "abc.", "doi": ""},
        {"id": "2401.01234v2", "title": "Same Paper", "summary": "abc.", "doi": ""},
    ]
    monkeypatch.setattr(arxiv_exec, "_sleep", lambda *_: None)
    monkeypatch.setattr(
        arxiv_exec,
        "_get_text",
        lambda url, *, timeout, headers: arxiv_exec._HttpResponse(
            body=_atom(entries), status=200
        ),
    )
    client = _client(monkeypatch, ArxivSearchOptions(query_planner="template"))
    parsed = parse_agent_output(client.run("x").final_message)
    assert len(parsed.findings) == 1


def test_run_http_429_yields_recoverable_issue_no_findings(monkeypatch) -> None:
    monkeypatch.setattr(arxiv_exec, "_sleep", lambda *_: None)
    monkeypatch.setattr(
        arxiv_exec,
        "_get_text",
        lambda url, *, timeout, headers: arxiv_exec._HttpResponse(body="", status=429),
    )
    client = _client(monkeypatch, ArxivSearchOptions(query_planner="template"))
    result = client.run("x")
    parsed = parse_agent_output(result.final_message)
    assert result.returncode == 0
    assert parsed.findings == []
    assert any(i.recoverable for i in parsed.issues)


def test_run_network_error_never_raises(monkeypatch) -> None:
    def boom(url, *, timeout, headers):  # noqa: ANN001, ANN202
        raise OSError("connection refused")

    monkeypatch.setattr(arxiv_exec, "_sleep", lambda *_: None)
    monkeypatch.setattr(arxiv_exec, "_get_text", boom)
    client = _client(monkeypatch, ArxivSearchOptions(query_planner="template"))
    result = client.run("x")
    parsed = parse_agent_output(result.final_message)
    assert result.returncode == 0
    assert parsed.findings == []
    assert parsed.issues


def test_run_respects_max_findings(monkeypatch) -> None:
    entries = [
        {"id": f"2401.{i:05d}", "title": f"Paper {i}", "summary": "s.", "doi": ""}
        for i in range(20)
    ]
    monkeypatch.setattr(arxiv_exec, "_sleep", lambda *_: None)
    monkeypatch.setattr(
        arxiv_exec,
        "_get_text",
        lambda url, *, timeout, headers: arxiv_exec._HttpResponse(
            body=_atom(entries), status=200
        ),
    )
    client = _client(
        monkeypatch, ArxivSearchOptions(query_planner="template", max_findings=3)
    )
    parsed = parse_agent_output(client.run("x").final_message)
    assert len(parsed.findings) == 3


# ---------------------------------------------------------------------------
# debug trace
# ---------------------------------------------------------------------------


def test_debug_trace_claude_path(monkeypatch, tmp_path: Path) -> None:
    _FakeClaude.behavior = "ok"
    client = _client(
        monkeypatch, ArxivSearchOptions(), debug_dir=tmp_path
    )
    _run_no_network(monkeypatch, client)
    plan = json.loads((tmp_path / "arxiv.plan.json").read_text())
    assert plan["planner_used"] == "claude"
    assert plan["fallback_reason"] is None
    assert plan["planned_queries"] == [
        {"clauses": [["kv cache offloading"], ["gpu memory"]]}
    ]
    assert (tmp_path / "arxiv.planner_prompt.md").exists()
    assert (tmp_path / "arxiv.planner_output.md").exists()


def test_debug_trace_fallback_records_attempt(monkeypatch, tmp_path: Path) -> None:
    _FakeClaude.behavior = "garbage"
    client = _client(monkeypatch, ArxivSearchOptions(), debug_dir=tmp_path)
    _run_no_network(monkeypatch, client)
    plan = json.loads((tmp_path / "arxiv.plan.json").read_text())
    assert plan["planner_used"] == "template"
    assert plan["fallback_reason"] is not None
    # The failed attempt still left the prompt + output on disk.
    assert (tmp_path / "arxiv.planner_prompt.md").exists()
    assert (tmp_path / "arxiv.planner_output.md").exists()


def test_debug_trace_records_raised_planner_attempt(
    monkeypatch, tmp_path: Path
) -> None:
    _FakeClaude.behavior = "raise"
    client = _client(monkeypatch, ArxivSearchOptions(), debug_dir=tmp_path)
    _run_no_network(monkeypatch, client)
    plan = json.loads((tmp_path / "arxiv.plan.json").read_text())
    assert plan["planner_used"] == "template"
    assert plan["fallback_reason"] is not None
    assert "RuntimeError: boom" in (
        tmp_path / "arxiv.planner_output.md"
    ).read_text()


def test_debug_trace_template_mode_only_plan(monkeypatch, tmp_path: Path) -> None:
    client = _client(
        monkeypatch,
        ArxivSearchOptions(query_planner="template"),
        debug_dir=tmp_path,
    )
    _run_no_network(monkeypatch, client)
    assert (tmp_path / "arxiv.plan.json").exists()
    assert not (tmp_path / "arxiv.planner_prompt.md").exists()
    assert not (tmp_path / "arxiv.planner_output.md").exists()


def test_debug_none_writes_nothing(monkeypatch, tmp_path: Path) -> None:
    client = _client(
        monkeypatch, ArxivSearchOptions(query_planner="template"), debug_dir=None
    )
    _run_no_network(monkeypatch, client)
    assert list(tmp_path.iterdir()) == []


def test_debug_rejections_recorded(monkeypatch, tmp_path: Path) -> None:
    # A plan carrying an over-constrained (4-AND-clause) query renders nothing
    # for that query and records a `too_many_clauses` rejection.
    _FakeClaude.behavior = "ok"

    def _bad_plan_run(self, prompt, *, check=True):  # noqa: ANN001, ARG001
        from spotlights_engine.module_deep_research.agent_exec import AgentExecResult

        msg = (
            '{"queries": ['
            '{"clauses": [["kv cache"]]},'
            '{"clauses": [["a"],["b"],["c"],["d"]]}'
            "]}"
        )
        return AgentExecResult(
            command=["claude"], returncode=0, stdout="", stderr="", final_message=msg
        )

    monkeypatch.setattr(_FakeClaude, "run", _bad_plan_run)
    client = _client(monkeypatch, ArxivSearchOptions(), debug_dir=tmp_path)
    _run_no_network(monkeypatch, client)
    plan = json.loads((tmp_path / "arxiv.plan.json").read_text())
    assert plan["planner_used"] == "claude"
    reasons = {r["reason"] for r in plan["rejected"]}
    assert "too_many_clauses" in reasons


def test_debug_unwritable_dir_does_not_fail_run(monkeypatch, tmp_path: Path) -> None:
    bad = tmp_path / "file"
    bad.write_text("x")  # a file, not a dir — writes under it will fail
    client = _client(
        monkeypatch, ArxivSearchOptions(query_planner="template"), debug_dir=bad
    )
    # Should not raise.
    result = _run_no_network(monkeypatch, client)
    assert result.returncode == 0


# ---------------------------------------------------------------------------
# opt-in live smoke test (skipped by default)
# ---------------------------------------------------------------------------


@pytest.mark.network
def test_live_arxiv_smoke() -> None:
    """Hits the real arXiv API once. Opt-in: `pytest -m network`."""
    client = ArxivSearchClient(
        _request(hotspots=False, candidates=False),
        _module(),
        ArxivSearchOptions(query_planner="template", max_queries=1, results_per_query=3),
    )
    result = client.run("ignored")
    parsed = parse_agent_output(result.final_message)
    assert result.returncode == 0
    # Either findings came back or a recoverable issue explains why (never raise).
    assert parsed.findings or parsed.issues
