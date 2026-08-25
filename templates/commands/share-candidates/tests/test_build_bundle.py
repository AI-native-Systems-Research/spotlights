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


# --- evolve bundles ---------------------------------------------------------


def test_find_evolve_join_and_absent(tmp_root=None):
    root = Path(tmp_root or tempfile.mkdtemp())
    evolve = root / "evolve"
    # cand-<module_slug>-NNNN -> <module_slug>/<cand_id>/<engine>/
    eng_dir = evolve / "qiskit_compiler" / "cand-qiskit_compiler-0001" / "coral"
    eng_dir.mkdir(parents=True)
    (eng_dir / "task.yaml").write_text("goal: x\n", encoding="utf-8")

    found = bb.find_evolve("cand-qiskit_compiler-0001", evolve)
    assert found is not None
    assert list(found) == ["coral"]
    assert found["coral"][0].name == "task.yaml"

    assert bb.find_evolve("cand-qiskit_compiler-9999", evolve) is None


def test_render_engine_coral_has_repo_install_quickstart_download():
    files = [Path("task.yaml")]  # name-only; render_engine reads .md files, not this
    # use a real temp file so stat()/read work for the non-md raw view
    d = Path(tempfile.mkdtemp())
    f = d / "task.yaml"
    f.write_text("goal: reduce latency\n", encoding="utf-8")
    out = bb.render_engine("coral", [f], "foo__cand-a-0001__evolve")
    assert 'https://github.com/Human-Agent-Society/CORAL' in out
    assert "curl -fsSL" in out and "install.sh" in out           # shell installer
    assert "/plugin install coral@coral-marketplace" in out       # plugin install
    assert "You must write" in out                                # coral writes grader+seed
    assert "Quickstart" in out and "&lt;CORAL_BUNDLE_PATH&gt;" in out  # literal placeholder
    assert 'href="foo__cand-a-0001__evolve/coral.zip"' in out     # per-engine download
    assert 'href="foo__cand-a-0001__evolve/coral/task.yaml"' in out   # open-raw link


def test_render_engine_nous_writes_row_names_pass_condition():
    d = Path(tempfile.mkdtemp())
    f = d / "campaign.yaml"
    f.write_text("research_question: q\n", encoding="utf-8")
    out = bb.render_engine("nous", [f], "bar__cand-b-0002__evolve")
    assert "agentic-strategy-evolution.git@reflective" in out     # install command
    assert "You must write" in out                                # the pass_condition gap
    assert "ground_truth.pass_condition" in out
    assert "Quickstart" not in out                                # coral-only


def test_render_index_evolve_badge_and_footer_link():
    header = {"title": "T", "objective": "O"}
    rows = [
        {"rank": "1", "cand_id": "cand-a-0001", "module": "m", "symbol": "foo",
         "impact": "high", "score": "96", "rationale": "R",
         "html_href": "candidates/modules/m/foo__cand-a-0001.html",
         "evolve_count": 2, "evolve_engines": ["coral", "nous"],
         "evolve_href": "candidates/modules/m/foo__cand-a-0001__evolve.html"},
    ]
    idx = bb.render_index(header, rows)
    assert '<span class="badge evolve">evolve · 2</span>' in idx
    assert 'class="evolve-link"' in idx
    assert 'href="candidates/modules/m/foo__cand-a-0001__evolve.html"' in idx
    assert "coral · nous" in idx


def test_render_index_no_evolve_keys_is_plain_card():
    # Regression: rows without evolve_* keys must render a normal card, no badge.
    header = {"title": "T", "objective": "O"}
    rows = [
        {"rank": "1", "cand_id": "cand-a-0001", "module": "m", "symbol": "foo",
         "impact": "high", "score": "96", "rationale": "R",
         "html_href": "candidates/modules/m/foo__cand-a-0001.html"},
    ]
    idx = bb.render_index(header, rows)
    # look only at the body (the CSS names .evolve-link / .badge.evolve)
    body = idx.split("</style>", 1)[1]
    assert '<span class="badge evolve">' not in body
    assert '<div class="card-foot">' not in body
    assert 'href="candidates/modules/m/foo__cand-a-0001.html"' in body


def test_build_end_to_end_with_evolve():
    root = Path(tempfile.mkdtemp())
    src = root / "sorted"
    src.mkdir(parents=True, exist_ok=True)
    (src / "sorted_candidates.md").write_text(SAMPLE_SORTED, encoding="utf-8")

    # top-2 candidate files
    c1 = (src / ".." / "modules" / "qiskit_compiler" / "foo__cand-a-0001.md").resolve()
    c2 = (src / ".." / "modules" / "qiskit_transpiler_passes" / "bar__cand-b-0002.md").resolve()
    for c, body in ((c1, "# foo\n\nbody\n"), (c2, "# bar\n\nbody\n")):
        c.parent.mkdir(parents=True, exist_ok=True)
        c.write_text(body, encoding="utf-8")

    # sibling evolve/ for cand-a-0001 only: coral (README + task.yaml) + nous (campaign.yaml)
    ev = src / ".." / "evolve" / "a" / "cand-a-0001"
    (ev / "coral").mkdir(parents=True)
    (ev / "coral" / "README.md").write_text("# coral\n\nrun me\n", encoding="utf-8")
    (ev / "coral" / "task.yaml").write_text("goal: x\n", encoding="utf-8")
    (ev / "nous").mkdir(parents=True)
    (ev / "nous" / "campaign.yaml").write_text("research_question: q\n", encoding="utf-8")

    result = bb.build(str(src), top_n=2)
    assert result["evolve_bundles"] == 1
    bundle = Path(result["bundle_dir"])

    # evolve page exists and lists both engines
    evolve_html = (bundle / "candidates" / "modules" / "qiskit_compiler"
                   / "foo__cand-a-0001__evolve.html")
    assert evolve_html.exists()
    ev_text = evolve_html.read_text(encoding="utf-8")
    assert "<h3>coral</h3>" in ev_text and "<h3>nous</h3>" in ev_text

    # candidate page gained the Evolve bundles section + link
    cand_html = (bundle / "candidates" / "modules" / "qiskit_compiler"
                 / "foo__cand-a-0001.html").read_text(encoding="utf-8")
    assert "Evolve bundles" in cand_html
    assert 'href="foo__cand-a-0001__evolve.html"' in cand_html

    # index card gained the evolve badge; the candidate without evolve did not
    idx = (bundle / "index.html").read_text(encoding="utf-8")
    idx_body = idx.split("</style>", 1)[1]
    assert '<span class="badge evolve">' in idx_body
    assert idx_body.count('<div class="card-foot">') == 1

    # per-engine zips namespace their entries under <engine>/
    coral_zip = (bundle / "candidates" / "modules" / "qiskit_compiler"
                 / "foo__cand-a-0001__evolve" / "coral.zip")
    with _zip.ZipFile(coral_zip) as zf:
        names = zf.namelist()
    assert "coral/README.md" in names and "coral/task.yaml" in names

    # original files copied verbatim into the bundle
    assert (bundle / "candidates" / "modules" / "qiskit_compiler"
            / "foo__cand-a-0001__evolve" / "nous" / "campaign.yaml").exists()

    # the second candidate has no evolve data and still builds cleanly
    assert (bundle / "candidates" / "modules" / "qiskit_transpiler_passes"
            / "bar__cand-b-0002.html").exists()
    assert not (bundle / "candidates" / "modules" / "qiskit_transpiler_passes"
                / "bar__cand-b-0002__evolve.html").exists()


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
