import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import build_bundle as bb


def test_render_inline_external_autolink_is_clickable():
    out = bb.render_inline("see <https://arxiv.org/abs/2506.06752>")
    assert '<a href="https://arxiv.org/abs/2506.06752" target="_blank" rel="noopener">https://arxiv.org/abs/2506.06752</a>' in out


def test_render_inline_external_markdown_link_is_clickable():
    out = bb.render_inline("[the paper](https://doi.org/10.1145/3445814.3446706)")
    assert '<a href="https://doi.org/10.1145/3445814.3446706" target="_blank" rel="noopener">the paper</a>' in out


def test_render_inline_internal_md_link_becomes_plain_code():
    out = bb.render_inline("[← qiskit/compiler](../qiskit_compiler.md)")
    assert "<a " not in out
    assert "<code>← qiskit/compiler</code>" in out


def test_render_inline_source_py_link_becomes_plain_code():
    out = bb.render_inline("[`qiskit/compiler/transpiler.py`](qiskit/compiler/transpiler.py)")
    assert "<a " not in out
    assert "<code>qiskit/compiler/transpiler.py</code>" in out


def test_render_inline_bold_and_code():
    out = bb.render_inline("**Id:** `cand-x`")
    assert "<strong>Id:</strong>" in out
    assert "<code>cand-x</code>" in out


def test_render_inline_escapes_html():
    out = bb.render_inline("a < b & c > d")
    assert "&lt;" in out and "&amp;" in out and "&gt;" in out


def test_body_renders_headings():
    out = bb.md_to_html_body("# Title\n\n## Section")
    assert "<h1>Title</h1>" in out
    assert "<h2>Section</h2>" in out


def test_body_renders_bullet_list():
    out = bb.md_to_html_body("- one\n- two")
    assert "<ul>" in out and "</ul>" in out
    assert "<li>one</li>" in out and "<li>two</li>" in out


def test_body_renders_hr_and_paragraph():
    out = bb.md_to_html_body("hello world\n\n---\n\nnext para")
    assert "<hr" in out
    assert "<p>hello world</p>" in out
    assert "<p>next para</p>" in out


def test_body_applies_inline_inside_blocks():
    out = bb.md_to_html_body("- see <https://arxiv.org/abs/1>")
    assert '<a href="https://arxiv.org/abs/1"' in out


def test_body_renders_fenced_code_block():
    md = "```python\ndef f(x):  # comment with a | pipe\n    return x\n```"
    out = bb.md_to_html_body(md)
    assert "<pre><code>" in out and "</code></pre>" in out
    assert "def f(x):" in out
    assert "return x" in out
    # newline between the two code lines is preserved
    assert "def f(x):  # comment with a | pipe\n    return x" in out
    # no spurious inline <code> spans opened inside the block
    assert "<code>" not in out.replace("<pre><code>", "")
    # literal fence markers must not leak into the output
    assert "```" not in out


SAMPLE_SORTED = """# Sorted candidates — run-abc (qiskit / circuit-depth)

**Objective:** Reduce circuit depth after transpilation
**Ranked:** 167 candidates · **Method:** listwise sub-agent judge
**Source:** `result.json`

## Ranking summary

| # | Candidate | Module | Symbol | Impact | Score | Rationale |
|---|-----------|--------|--------|--------|-------|-----------|
| 1 | [`cand-a-0001`](../modules/qiskit_compiler/foo__cand-a-0001.md) | qiskit/compiler | `foo` | high | 96 | Broad lever on depth |
| 2 | [`cand-b-0002`](../modules/qiskit_transpiler_passes/bar__cand-b-0002.md) | qiskit/transpiler/passes | `bar` | high | 95 | Layout quality compounds |
| 3 | [`cand-c-0003`](../modules/qiskit_passmanager/baz__cand-c-0003.md) | qiskit/passmanager | `baz` | medium | 80 | Loop controller |
"""


def test_parse_header_title_and_objective():
    h = bb.parse_header(SAMPLE_SORTED)
    assert h["title"] == "Sorted candidates — run-abc (qiskit / circuit-depth)"
    assert h["objective"] == "Reduce circuit depth after transpilation"


def test_parse_ranking_table_top_n():
    rows = bb.parse_ranking_table(SAMPLE_SORTED, top_n=2)
    assert len(rows) == 2
    assert rows[0]["rank"] == "1"
    assert rows[0]["cand_id"] == "cand-a-0001"
    assert rows[0]["rel_link"] == "../modules/qiskit_compiler/foo__cand-a-0001.md"
    assert rows[0]["module"] == "qiskit/compiler"
    assert rows[0]["symbol"] == "foo"
    assert rows[0]["impact"] == "high"
    assert rows[0]["score"] == "96"
    assert rows[0]["rationale"] == "Broad lever on depth"


def test_parse_ranking_table_more_than_available():
    rows = bb.parse_ranking_table(SAMPLE_SORTED, top_n=10)
    assert len(rows) == 3


def test_parse_ranking_table_rationale_with_pipe():
    sample = """## Ranking summary

| # | Candidate | Module | Symbol | Impact | Score | Rationale |
|---|-----------|--------|--------|--------|-------|-----------|
| 1 | [`cand-a-0001`](../modules/qiskit_compiler/foo__cand-a-0001.md) | qiskit/compiler | `foo` | high | 96 | O(|gate_map|) scan is the bottleneck |
"""
    rows = bb.parse_ranking_table(sample, top_n=1)
    assert len(rows) == 1
    assert rows[0]["rationale"] == "O(|gate_map|) scan is the bottleneck"
    assert "|gate_map|" in rows[0]["rationale"]


def test_render_candidate_page_is_standalone_html():
    page = bb.render_candidate_page("# Foo\n\nbody", "Foo", "../../../index.html")
    assert page.startswith("<!DOCTYPE html>")
    assert "<style>" in page  # embedded CSS, no CDN
    assert "http" not in page.split("<style>")[1].split("</style>")[0]  # no CDN url in CSS
    assert 'href="../../../index.html"' in page  # back link
    assert "<h1>Foo</h1>" in page


def test_render_index_lists_cards_with_html_hrefs():
    header = {"title": "T", "objective": "O"}
    rows = [
        {"rank": "1", "cand_id": "cand-a-0001", "module": "qiskit/compiler",
         "symbol": "foo", "impact": "high", "score": "96", "rationale": "R1",
         "html_href": "candidates/modules/qiskit_compiler/foo__cand-a-0001.html"},
    ]
    idx = bb.render_index(header, rows)
    assert idx.startswith("<!DOCTYPE html>")
    assert "T" in idx and "O" in idx
    assert 'href="candidates/modules/qiskit_compiler/foo__cand-a-0001.html"' in idx
    assert "foo" in idx and "96" in idx and "R1" in idx


import tempfile
import zipfile as _zip


def test_build_end_to_end(tmp_root=None):
    root = Path(tmp_root or tempfile.mkdtemp())
    src = root / "sorted"
    src.mkdir(parents=True, exist_ok=True)
    (src / "sorted_candidates.md").write_text(SAMPLE_SORTED, encoding="utf-8")

    # Create the two candidate files referenced by top-2 (relative to src).
    c1 = src / ".." / "modules" / "qiskit_compiler" / "foo__cand-a-0001.md"
    c2 = src / ".." / "modules" / "qiskit_transpiler_passes" / "bar__cand-b-0002.md"
    for c, body in ((c1, "# foo\n\nsee <https://arxiv.org/abs/1>\n"),
                    (c2, "# bar\n\n[`x.py`](x.py)\n")):
        c = c.resolve()
        c.parent.mkdir(parents=True, exist_ok=True)
        c.write_text(body, encoding="utf-8")

    result = bb.build(str(src), top_n=2)

    bundle = Path(result["bundle_dir"])
    assert (bundle / "index.html").exists()
    h1 = bundle / "candidates" / "modules" / "qiskit_compiler" / "foo__cand-a-0001.html"
    m1 = bundle / "candidates" / "modules" / "qiskit_compiler" / "foo__cand-a-0001.md"
    assert h1.exists() and m1.exists()
    assert '<a href="https://arxiv.org/abs/1"' in h1.read_text(encoding="utf-8")

    idx = (bundle / "index.html").read_text(encoding="utf-8")
    assert 'href="candidates/modules/qiskit_compiler/foo__cand-a-0001.html"' in idx

    zpath = Path(result["zip_path"])
    assert zpath.exists()
    with _zip.ZipFile(zpath) as zf:
        names = zf.namelist()
    assert any(n.endswith("index.html") for n in names)
    assert result["count"] == 2
    assert result["skipped"] == []


def test_build_skips_missing_candidate_file():
    root = Path(tempfile.mkdtemp())
    src = root / "sorted"
    src.mkdir(parents=True, exist_ok=True)
    # top-1 points at a file we never create -> should be skipped, not crash.
    (src / "sorted_candidates.md").write_text(SAMPLE_SORTED, encoding="utf-8")
    result = bb.build(str(src), top_n=1)
    assert result["count"] == 0
    assert len(result["skipped"]) == 1


def _run_all():
    fns = [g for n, g in sorted(globals().items()) if n.startswith("test_") and callable(g)]
    failed = 0
    for fn in fns:
        try:
            fn()
            print(f"PASS {fn.__name__}")
        except AssertionError as e:
            failed += 1
            print(f"FAIL {fn.__name__}: {e}")
        except Exception as e:  # noqa: BLE001
            failed += 1
            print(f"ERROR {fn.__name__}: {type(e).__name__}: {e}")
    print(f"\n{len(fns) - failed}/{len(fns)} passed")
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    _run_all()
