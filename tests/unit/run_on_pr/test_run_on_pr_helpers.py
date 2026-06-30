"""Unit tests for the run-on-pr helper scripts (scripts/run_on_pr/).

Covers the schema translation (`extract_candidates`), the file→module qn
mapper (`map_files_to_modules`), conservative paper extraction
(`extract_pr_papers`), the paper matcher (`match_papers`), and — most
importantly — that the matcher's key logic stays in lockstep with the engine's
own finding-dedup keys (`_finding_keys`). Divergence there would silently
mis-score the paper signal, so it gets an explicit parity assertion.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parents[3]
_SCRIPTS = _REPO_ROOT / "scripts" / "run_on_pr"


def _load(module_name: str):
    spec = importlib.util.spec_from_file_location(
        f"run_on_pr_{module_name}", _SCRIPTS / f"{module_name}.py"
    )
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


extract_candidates = _load("extract_candidates")
map_files_to_modules = _load("map_files_to_modules")
extract_pr_papers = _load("extract_pr_papers")
match_papers = _load("match_papers")

from spotlights_engine.module_deep_research.orchestration import _finding_keys  # noqa: E402
from spotlights_engine.schemas.project import ProjectTree  # noqa: E402

# --- extract_candidates: nested locations[].spans[] → flat records -----------

def test_explode_nested_candidate_multi_span_multi_file():
    candidates = [
        {
            "id": "cand-mod-0001",
            "module_qualified_name": "vllm/v1/kv_offload",
            "origin": "code_agent",
            "estimated_impact": "high",
            "locations": [
                {
                    "file": "a.py",
                    "spans": [
                        {"line_start": 19, "line_end": 22, "symbol": "X", "kind": "plugin_seam"},
                        {"line_start": 40, "line_end": 55, "symbol": "Y", "kind": "function"},
                    ],
                },
                {
                    "file": "b.py",
                    "spans": [{"line_start": 1, "line_end": 3, "symbol": "Z", "kind": "region"}],
                },
            ],
        },
        {
            "id": "cand-mod-0002",
            "module_qualified_name": "vllm/v1/kv_offload",
            "origin": "telemetry_anomaly",
            "estimated_impact": "low",
            "locations": [
                {
                    "file": "c.py",
                    "spans": [{"line_start": 5, "line_end": 9, "symbol": "W", "kind": "loop"}],
                },
            ],
        },
    ]
    flat = extract_candidates.explode(candidates)
    # 2 spans on a.py + 1 on b.py + 1 on c.py = 4 records.
    assert len(flat) == 4
    # All three records of cand-0001 share id and rank=1.
    c1 = [r for r in flat if r["id"] == "cand-mod-0001"]
    assert len(c1) == 3
    assert {r["rank"] for r in c1} == {1}
    assert flat[-1]["rank"] == 2  # cand-0002
    # Range + symbol carried from the span; impact from the parent.
    first = flat[0]
    assert (first["file"], first["line_start"], first["line_end"]) == ("a.py", 19, 22)
    assert first["symbol"] == "X" and first["estimated_impact"] == "high"


def test_iter_candidates_prefers_report_then_flat_then_module_runs():
    nested = {"report": {"candidates": [{"id": "cand-x-0001", "locations": []}]}}
    assert extract_candidates._iter_candidates(nested)[0]["id"] == "cand-x-0001"

    flat = {"candidates": [{"id": "cand-y-0001"}]}
    assert extract_candidates._iter_candidates(flat)[0]["id"] == "cand-y-0001"

    mr = {"module_runs": {"m": {"candidates": {"candidates": [{"id": "cand-z-0001"}]}}}}
    assert extract_candidates._iter_candidates(mr)[0]["id"] == "cand-z-0001"


def test_legacy_flat_candidate_without_locations():
    flat = extract_candidates.explode(
        [{
            "id": "cand-m-0001", "file": "x.py",
            "line_start": 3, "line_end": 7, "estimated_impact": "medium",
        }]
    )
    assert len(flat) == 1
    assert flat[0]["file"] == "x.py" and flat[0]["line_start"] == 3


def test_off_schema_empties_emit_no_junk_records():
    # locations: [] (off-schema) emits nothing — never an all-null legacy row.
    assert extract_candidates.explode([{"id": "cand-m-0001", "locations": []}]) == []
    # A location with empty spans contributes nothing.
    flat = extract_candidates.explode(
        [{"id": "cand-m-0002", "locations": [{"file": "x.py", "spans": []}]}]
    )
    assert flat == []
    # A legacy candidate with no file at all is skipped, not emitted as null.
    assert extract_candidates.explode([{"id": "cand-m-0003"}]) == []


# --- map_files_to_modules: deepest-match, source_root, unmapped --------------

def _tree(data: dict) -> ProjectTree:
    return ProjectTree.model_validate(data)


def test_deepest_module_wins_and_uses_walk_qn():
    tree = _tree({
        "repository": {"name": "vllm", "summary": "s", "source_root": ""},
        "modules": [
            {"name": "v1", "path": "vllm/v1", "submodules": [
                {"name": "kv_offload", "path": "vllm/v1/kv_offload", "submodules": [
                    {"name": "cpu", "path": "vllm/v1/kv_offload/cpu", "submodules": []},
                ]},
            ]},
        ],
    })
    out = map_files_to_modules.map_files(
        tree,
        ["vllm/v1/kv_offload/cpu/manager.py", "vllm/v1/kv_offload/top.py", "vllm/other/x.py"],
    )
    assert out["file_module"]["vllm/v1/kv_offload/cpu/manager.py"] == "vllm/v1/kv_offload/cpu"
    assert out["file_module"]["vllm/v1/kv_offload/top.py"] == "vllm/v1/kv_offload"
    assert out["unmapped_files"] == ["vllm/other/x.py"]
    assert "vllm/v1/kv_offload/cpu" in out["include"]


def test_source_root_stripped_in_qn():
    tree = _tree({
        "repository": {"name": "pkg", "summary": "s", "source_root": "src"},
        "modules": [{"name": "cache", "path": "src/pkg/cache", "submodules": []}],
    })
    out = map_files_to_modules.map_files(tree, ["src/pkg/cache/x.py"])
    assert out["include"] == ["pkg/cache"]


def test_segment_aware_no_prefix_false_match():
    tree = _tree({
        "repository": {"name": "r", "summary": "s", "source_root": ""},
        "modules": [{"name": "cache", "path": "pkg/cache", "submodules": []}],
    })
    # pkg/cache_v2/x.py must NOT match module pkg/cache.
    out = map_files_to_modules.map_files(tree, ["pkg/cache_v2/x.py"])
    assert out["include"] == []
    assert out["unmapped_files"] == ["pkg/cache_v2/x.py"]


# --- extract_pr_papers: conservative, arxiv/DOI/host only --------------------

def test_extract_keeps_papers_drops_repo_and_ci():
    texts = [
        "See [Preble](https://arxiv.org/abs/2504.19874v2) and repo "
        "https://github.com/vllm-project/vllm , CI https://buildkite.com/x , "
        "DOI 10.1145/3600006.3613145 and https://openreview.net/forum?id=abc"
    ]
    papers = extract_pr_papers.extract(texts)
    norms = {p["normalized"] for p in papers}
    assert "arxiv:2504.19874" in norms
    assert "https://doi.org/10.1145/3600006.3613145" in norms
    assert "https://openreview.net/forum?id=abc" in norms
    # No github / buildkite.
    assert not any("github.com" in p["raw_url"] or "buildkite" in p["raw_url"] for p in papers)
    # Markdown link text captured as a title for the arxiv paper.
    arxiv = next(p for p in papers if p["normalized"] == "arxiv:2504.19874")
    assert arxiv.get("title") == "Preble"


def test_extract_empty_when_no_papers():
    assert extract_pr_papers.extract(["just prose, https://github.com/a/b only"]) == []


# --- match_papers: url-OR-title, reverse-link via proposals ------------------

def _result_with(findings, candidates=None):
    return {"report": {"findings": findings, "candidates": candidates or []}}


def test_match_by_title_when_engine_url_differs():
    cited = [{
        "raw_url": "https://arxiv.org/abs/2504.19874",
        "normalized": "arxiv:2504.19874",
        "title": "Preble: Efficient Distributed Prompt Scheduling for LLM Serving",
    }]
    findings = [{
        "finding_id": "find-preble-0001",
        "title": "Preble: Efficient Distributed Prompt Scheduling for LLM Serving",
        "url": "https://proceedings.iclr.cc/x.pdf",
        "source_type": "paper",
    }]
    candidates = [{
        "id": "cand-m-0001",
        "proposals": [{
            "id": "prop-m-0001", "source": "research_finding",
            "finding_ref_id": "find-preble-0001",
        }],
    }]
    matched = match_papers.match(cited, findings, candidates)
    assert len(matched) == 1
    assert matched[0]["matched_on"] == "title"
    assert matched[0]["via_candidate_ids"] == ["cand-m-0001"]


def test_match_by_url_arxiv_version_insensitive():
    cited = [{"raw_url": "https://arxiv.org/pdf/2504.19874", "normalized": "arxiv:2504.19874"}]
    findings = [{
        "finding_id": "find-x-0001",
        "title": "Different Title",
        "url": "https://arxiv.org/abs/2504.19874v3",
        "source_type": "paper",
    }]
    matched = match_papers.match(cited, findings, [])
    assert len(matched) == 1 and matched[0]["matched_on"] == "url"


def test_no_match_returns_empty():
    cited = [{"raw_url": "https://arxiv.org/abs/9999.99999", "normalized": "arxiv:9999.99999"}]
    findings = [{
        "finding_id": "find-x-0001", "title": "Other",
        "url": "https://arxiv.org/abs/1111.11111", "source_type": "paper",
    }]
    assert match_papers.match(cited, findings, []) == []


# --- normalizer parity: the matcher MUST reuse the engine's keys -------------

@pytest.mark.parametrize(
    "title,url",
    [
        ("Preble: X", "https://arxiv.org/abs/2504.19874v2"),
        ("Some Paper", "https://arxiv.org/pdf/2504.19874.pdf"),
        ("DOI Paper", "https://doi.org/10.1145/3600006"),
        ("Plain", "https://example.com/paper/"),
    ],
)
def test_cited_keys_match_engine_finding_keys(title, url):
    """A cited paper carrying (title, url) must produce keys that are a subset
    of the engine's `_finding_keys(title, url)` for the same inputs — i.e. the
    matcher is using the engine's own coordinate frame, not a hand-rolled one."""
    url_key, title_key = match_papers._cited_keys({"raw_url": url, "title": title})
    engine_keys = _finding_keys(title, url)
    assert title_key in engine_keys
    assert url_key in engine_keys
