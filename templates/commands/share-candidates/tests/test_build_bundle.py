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


def test_inline_code_span_escapes_html_metacharacters_exactly_once():
    # A code span containing HTML metacharacters must be escaped exactly once,
    # not double-escaped. Before the fix: `stderr="oops"` rendered as
    # stderr=&amp;quot;oops&amp;quot;, displaying literal &quot; in the browser.
    out_direct = bb._inline_no_links('`stderr="oops"`')
    assert "<code>stderr=&quot;oops&quot;</code>" in out_direct
    assert "&amp;quot;" not in out_direct
    # Also through the public render_inline path (which stashes anchor links and
    # restores them after escape, so the sentinel-preservation matters)
    out_indirect = bb.render_inline('`a < b && c`')
    assert "<code>a &lt; b &amp;&amp; c</code>" in out_indirect
    assert "&amp;lt;" not in out_indirect
    assert "&amp;amp;" not in out_indirect


_ARXIV = "https://arxiv.org/abs/2309.06180"


def _anchor(url, text=None):
    return f'<a href="{url}" target="_blank" rel="noopener">{text or url}</a>'


def test_render_inline_bare_url_in_prose_is_clickable():
    # The engine writes APPLY-NOTES.md references as bare URLs, not autolinks.
    out = bb.render_inline(f"Ref (paper) — {_ARXIV} and more")
    assert _anchor(_ARXIV) in out


def test_render_inline_bare_url_trailing_punctuation_stays_outside():
    for punct in (".", ",", ";", ":"):
        out = bb.render_inline(f"See {_ARXIV}{punct}")
        assert out.endswith(_anchor(_ARXIV) + punct), (punct, out)


def test_render_inline_parenthesised_bare_url_keeps_the_paren_outside():
    out = bb.render_inline(f"(see {_ARXIV})")
    assert _anchor(_ARXIV) + ")" in out
    # ...but a paren that belongs to the URL is kept inside it
    wiki = "https://en.wikipedia.org/wiki/PagedAttention_(vLLM)"
    assert _anchor(wiki) in bb.render_inline(f"see {wiki} here")


def test_render_inline_bare_url_in_code_span_stays_literal():
    # The \x01/\x02 code-span sentinels must shield the URL from linkification.
    out = bb.render_inline(f"run `curl {_ARXIV}` now")
    assert f"<code>curl {_ARXIV}</code>" in out
    assert "<a " not in out


def test_render_inline_existing_markdown_link_is_not_double_linkified():
    out = bb.render_inline(f"[the paper]({_ARXIV})")
    assert out.count("<a ") == 1
    assert _anchor(_ARXIV, "the paper") in out
    # a link whose visible text is itself a URL keeps that text as plain text
    out2 = bb.render_inline(f"[{_ARXIV}](https://doi.org/10.1/2)")
    assert out2.count("<a ") == 1
    assert _anchor("https://doi.org/10.1/2", _ARXIV) in out2


def test_render_inline_existing_autolink_is_not_double_linkified():
    out = bb.render_inline(f"see <{_ARXIV}>")
    assert out.count("<a ") == 1
    assert _anchor(_ARXIV) in out


def test_render_inline_bare_url_query_string_is_escaped_exactly_once():
    # Regression guard: this file has already shipped a double-escaping bug.
    url = "https://docs.example.dev/s?a=1&b=2"
    esc = "https://docs.example.dev/s?a=1&amp;b=2"
    out = bb.render_inline(f"see {url} now")
    assert _anchor(esc) in out
    assert "&amp;amp;" not in out


def test_render_inline_repeated_bare_url_is_linkified_every_time():
    out = bb.render_inline(f"{_ARXIV} and again {_ARXIV}")
    assert out.count(_anchor(_ARXIV)) == 2


def test_render_inline_bare_url_only_http_schemes():
    # No bare www., no email, no scheme-relative.
    out = bb.render_inline("www.example.com me@example.com //example.com/x")
    assert "<a " not in out


def test_md_to_html_body_linkifies_bare_urls_in_every_block():
    # The shared inline path: apply notes, evolve READMEs and candidate pages all
    # reach it, so one block type getting it is not enough.
    md = (f"# H {_ARXIV}\n\n"
          f"para {_ARXIV}\n\n"
          f"- item {_ARXIV}\n\n"
          f"> quote {_ARXIV}\n\n"
          f"| a | b |\n| --- | --- |\n| {_ARXIV} | x |\n")
    out = bb.md_to_html_body(md)
    assert out.count(_anchor(_ARXIV)) == 5
    # ...and a fenced block is still verbatim, never linkified
    fenced = bb.md_to_html_body(f"```\ncurl {_ARXIV}\n```\n")
    assert "<a " not in fenced


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


def test_body_renders_a_pipe_table():
    # Every real APPLY-NOTES.md carries a "Files changed" table. Without table
    # support it rendered as literal pipes in a paragraph.
    md = ("| File | Lines |\n"
          "| --- | --- |\n"
          "| `a.py` | +58/-16 |\n"
          "| `b.py` | +1/-0 |\n")
    out = bb.md_to_html_body(md)
    assert "<table>" in out and "</table>" in out
    assert "<th>File</th>" in out and "<th>Lines</th>" in out
    assert "<td><code>a.py</code></td>" in out
    assert "<td><code>b.py</code></td>" in out
    assert out.count("<tr>") == 3          # header + two body rows
    # the separator row is consumed, never rendered as a data row
    assert "<td>---</td>" not in out
    # and none of it leaks out as literal markdown
    assert "| File |" not in out


def test_body_renders_a_blockquote_as_one_block():
    # The unverified warning is a wrapped multi-line blockquote. The paragraph
    # gatherer used to join its lines with spaces, emitting a stray `&gt;` at
    # every source line break, mid-sentence.
    md = ("> **Nothing here was verified.** No test was\n"
          "> run, no benchmark was measured.\n")
    out = bb.md_to_html_body(md)
    assert out.count("<blockquote>") == 1 and out.count("</blockquote>") == 1
    assert "&gt;" not in out               # THE regression: no stray markers
    assert "<strong>Nothing here was verified.</strong>" in out
    # the wrapped lines join into one sentence
    assert "No test was run, no benchmark was measured." in out
    # a bare `>` inside the quote opens a second paragraph, still one blockquote
    out2 = bb.md_to_html_body("> one\n>\n> two\n")
    assert out2.count("<blockquote>") == 1
    assert out2.count("<p>") == 2
    assert "&gt;" not in out2


def test_body_pipe_row_without_a_separator_degrades_to_a_paragraph():
    # REGRESSION GUARD AGAINST A HANG. If the paragraph loop breaks on any pipe
    # row, a lone `| a | b |` with no `|---|` under it breaks the loop without
    # being consumed by the table branch, so `i` never advances and this call
    # spins forever. The break must be gated on the separator lookahead.
    out = bb.md_to_html_body("| a | b |\n\nnext")
    assert "<table>" not in out
    assert "<p>| a | b |</p>" in out
    assert "<p>next</p>" in out
    # the same row as the very last line of the input (no lookahead available)
    out2 = bb.md_to_html_body("intro\n\n| a | b |")
    assert "<table>" not in out2
    assert "| a | b |" in out2


def test_body_table_cell_escapes_html_metacharacters_exactly_once():
    md = ("| Expr | Note |\n"
          "| --- | --- |\n"
          '| a < b && c | `x="<script>"` |\n')
    out = bb.md_to_html_body(md)
    assert "<td>a &lt; b &amp;&amp; c</td>" in out
    assert "&amp;lt;" not in out and "&amp;amp;" not in out
    # inside a code span too: escaped once, and never live markup
    assert "<code>x=&quot;&lt;script&gt;&quot;</code>" in out
    assert "<script>" not in out


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


import filecmp
import re
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


# --- one-shot apply --------------------------------------------------------


def test_cand_module_slug_strips_prefix_and_sequence():
    assert bb._cand_module_slug("cand-vllm_v1_worker-0001") == "vllm_v1_worker"
    assert bb._cand_module_slug("cand-qiskit_compiler-0042") == "qiskit_compiler"
    # a slug containing digits and underscores must survive intact
    assert bb._cand_module_slug("cand-vllm_v1_kv_offload-0002") == "vllm_v1_kv_offload"


SAMPLE_PATCH = """\
# spotlights one-shot apply
# candidate: cand-vllm_v1_worker-0001
# module:    vllm/v1/worker
# repo:      /Users/someone/checkouts/vllm
# base:      83ad767eed3be3ee7f2df63be693bfaca5c7c922
# apply with (from the directory containing this patch):
#   git -C /Users/someone/checkouts/vllm checkout 83ad767e
#   git -C /Users/someone/checkouts/vllm apply "$PWD/apply.patch"
diff --git a/vllm/v1/worker/gpu_model_runner.py b/vllm/v1/worker/gpu_model_runner.py
index 43f5c45323..225e46f2a7 100644
--- a/vllm/v1/worker/gpu_model_runner.py
+++ b/vllm/v1/worker/gpu_model_runner.py
@@ -2019,7 +2019,9 @@ class GPUModelRunner:
         self.input_batch.block_table.commit_block_table(num_reqs)
-        # Get request indices.
-        req_indices = np.repeat(self.arange_np[:num_reqs], num_scheduled)
+        # OPTIMIZATION: pure-decode fast path <&> escaped
+        pure_decode = total == num_reqs
+        req_indices = self.arange_np[:num_reqs]

diff --git a/vllm/v1/worker/utils.py b/vllm/v1/worker/utils.py
--- a/vllm/v1/worker/utils.py
+++ b/vllm/v1/worker/utils.py
@@ -10,3 +10,4 @@ def helper():
     pass
+    # one added line
"""


def test_parse_patch_header_extracts_all_four_fields():
    h = bb.parse_patch_header(SAMPLE_PATCH)
    assert h["candidate"] == "cand-vllm_v1_worker-0001"
    assert h["module"] == "vllm/v1/worker"
    assert h["repo"] == "/Users/someone/checkouts/vllm"
    assert h["base"] == "83ad767eed3be3ee7f2df63be693bfaca5c7c922"


def test_parse_patch_header_ignores_continuation_and_diff_lines():
    h = bb.parse_patch_header(SAMPLE_PATCH)
    # the "#   git -C … checkout" continuation lines must not become fields
    assert set(h) == {"candidate", "module", "repo", "base"}


def test_parse_patch_header_missing_fields_degrade():
    h = bb.parse_patch_header("diff --git a/x b/x\n--- a/x\n+++ b/x\n")
    assert h == {}
    # a header naming only the base is still usable
    h2 = bb.parse_patch_header("# base:  abc123\ndiff --git a/x b/x\n")
    assert h2 == {"base": "abc123"}


def test_diff_line_kind_classifies_every_case():
    # The file markers must be checked BEFORE the +/- tests, or every marker
    # is miscounted as a changed line. This is the whole point of the helper.
    assert bb._diff_line_kind("+++ b/vllm/x.py") == "marker"
    assert bb._diff_line_kind("--- a/vllm/x.py") == "marker"
    assert bb._diff_line_kind("@@ -1,2 +1,3 @@ def f():") == "hunk"
    assert bb._diff_line_kind("+    added") == "add"
    assert bb._diff_line_kind("-    removed") == "del"
    assert bb._diff_line_kind("     context") == "context"
    assert bb._diff_line_kind("") == "context"
    # a bare +/- is a real added/removed blank line, not a marker
    assert bb._diff_line_kind("+") == "add"
    assert bb._diff_line_kind("-") == "del"
    # two dashes/pluses are NOT the three-character marker
    assert bb._diff_line_kind("--x") == "del"
    assert bb._diff_line_kind("++x") == "add"


def test_diff_line_kind_inside_a_hunk_marker_prefixes_are_content():
    # A file marker only ever appears in a file's header block, before its
    # first @@. Inside a hunk the same prefix is a changed line whose CONTENT
    # begins with --/++: deleting a markdown `---` rule emits `----`, and
    # deleting a `-- flag` doc line emits `--- flag`. Classifying those as
    # markers drops them from the count and under-reports removals.
    assert bb._diff_line_kind("----", in_hunk=True) == "del"
    assert bb._diff_line_kind("--- flag doc", in_hunk=True) == "del"
    assert bb._diff_line_kind("+++x", in_hunk=True) == "add"
    # ...but in the header block they are still markers
    assert bb._diff_line_kind("--- a/x.py", in_hunk=False) == "marker"
    assert bb._diff_line_kind("+++ b/x.py", in_hunk=False) == "marker"
    # These two are the assertions that pin the STATE MACHINE as the mechanism.
    # The three above are all distinguishable by their content alone, so they
    # would also pass a stateless `startswith("--- a/")` heuristic. These cannot:
    # the exact marker strings must classify as changed lines inside a hunk,
    # which is only reachable by consulting `in_hunk`. A patch of a patch file
    # is the real input that produces them.
    assert bb._diff_line_kind("--- a/x.py", in_hunk=True) == "del"
    assert bb._diff_line_kind("+++ b/x.py", in_hunk=True) == "add"


def test_diffstat_counts_per_file_and_totals():
    st = bb.diffstat(SAMPLE_PATCH)
    assert st["files"] == [
        {"path": "vllm/v1/worker/gpu_model_runner.py", "added": 3, "removed": 2},
        {"path": "vllm/v1/worker/utils.py", "added": 1, "removed": 0},
    ]
    assert st["added"] == 4
    assert st["removed"] == 2


def test_diffstat_counts_removed_lines_whose_content_starts_with_dashes():
    # Deleting a markdown `---` rule emits the diff line `----`, and deleting
    # a `-- flag` doc line emits `--- flag`. Both must count as removals.
    # `git diff --numstat` reports 3 removals here; so must we, or the page
    # under-reports a patch a recipient is deciding whether to apply.
    patch = ("diff --git a/README.md b/README.md\n"
             "--- a/README.md\n"
             "+++ b/README.md\n"
             "@@ -1,4 +1,2 @@\n"
             " title\n"
             "----\n"
             "--- flag doc\n"
             "-plain\n"
             "+++added\n")
    st = bb.diffstat(patch)
    assert st["removed"] == 3      # NOT 1 (would mean '----'/'--- ' read as markers)
    assert st["added"] == 1        # the '+++added' content line


def test_diffstat_ignores_lines_before_the_first_diff_git():
    # The `#` header block `spotlights-engine apply` writes above the diff can
    # contain a line starting with +/-; nothing before the first `diff --git`
    # belongs to any file. Without the `cur is None` guard this raises
    # TypeError on a None subscript.
    st = bb.diffstat("# spotlights one-shot apply\n-stray\n+stray\n")
    assert st == {"files": [], "added": 0, "removed": 0}


def test_diffstat_counts_blank_added_and_removed_lines():
    patch = ("diff --git a/x.py b/x.py\n"
             "--- a/x.py\n"
             "+++ b/x.py\n"
             "@@ -1,2 +1,2 @@\n"
             "-\n"
             "+\n"
             " context\n")
    st = bb.diffstat(patch)
    assert st["added"] == 1 and st["removed"] == 1


def test_diffstat_resets_hunk_state_on_a_quoted_diff_git_path():
    # `core.quotePath` defaults to true and `apply.patch` is a plain `git diff`,
    # so any path holding a non-ASCII byte arrives C-quoted. If the boundary is
    # only recognised when the *path* parses, `in_hunk` survives into the next
    # file and its own markers count as a removal and an addition. Real
    # `--numstat` for this input is 2 files, +2/-4.
    patch = ('diff --git a/a.py b/a.py\n'
             '--- a/a.py\n'
             '+++ b/a.py\n'
             '@@ -1,3 +1,2 @@\n'
             ' keep\n'
             '-gone\n'
             '+new\n'
             'diff --git "a/zz-caf\\303\\251.py" "b/zz-caf\\303\\251.py"\n'
             '--- "a/zz-caf\\303\\251.py"\n'
             '+++ "b/zz-caf\\303\\251.py"\n'
             '@@ -1,4 +1,2 @@\n'
             ' keep\n'
             '-one\n'
             '-two\n'
             '-three\n'
             '+one\n')
    st = bb.diffstat(patch)
    assert st["added"] == 2 and st["removed"] == 4
    assert len(st["files"]) == 2                    # the quoted file has its own row
    assert st["files"][0] == {"path": "a.py", "added": 1, "removed": 1}
    assert st["files"][1]["added"] == 1 and st["files"][1]["removed"] == 3


def test_diff_git_path_extracts_the_post_image_path_in_every_header_form():
    # The boundary helper is what resets `in_hunk`, so it must return a path —
    # never None — for every shape a `diff --git` line can take. Git quotes each
    # side independently, so a rename can quote one side and not the other.
    assert bb._diff_git_path("diff --git a/x/y.py b/x/y.py") == "x/y.py"
    # a rename reports the POST-image path, which is the one a reader applies to
    assert bb._diff_git_path("diff --git a/old.py b/new.py") == "new.py"
    assert bb._diff_git_path('diff --git "a/caf\\303\\251.py" "b/caf\\303\\251.py"') \
        == "caf\\303\\251.py"
    # quoted source, bare destination — the whole header remainder must not leak
    assert bb._diff_git_path('diff --git "a/caf\\303\\251.py" b/ascii.py') == "ascii.py"
    assert bb._diff_git_path('diff --git a/ascii.py "b/caf\\303\\251.py"') \
        == "caf\\303\\251.py"
    # and None for anything that is not a boundary, or `diffstat` would start a
    # spurious file and drop the real one's counts
    assert bb._diff_git_path("--- a/x.py") is None
    assert bb._diff_git_path("@@ -1,2 +1,3 @@") is None
    assert bb._diff_git_path("-removed") is None
    assert bb._diff_git_path("# spotlights one-shot apply") is None


def test_diffstat_counts_every_hunk_of_a_multi_hunk_file():
    # `in_hunk` is already True at the second `@@`. A classifier that only
    # recognises a hunk header when not already in one still totals correctly
    # here, but would leave Task 5 colouring every later hunk header as
    # context — so assert the count that keeps both readings honest.
    patch = ("diff --git a/x.py b/x.py\n"
             "--- a/x.py\n"
             "+++ b/x.py\n"
             "@@ -1,3 +1,3 @@\n"
             " a\n"
             "-b\n"
             "+B\n"
             "@@ -20,3 +20,4 @@\n"
             " c\n"
             "+d\n"
             "+e\n")
    st = bb.diffstat(patch)
    assert st["files"] == [{"path": "x.py", "added": 3, "removed": 1}]
    assert bb._diff_line_kind("@@ -20,3 +20,4 @@", in_hunk=True) == "hunk"


def test_diffstat_empty_patch_is_zero():
    st = bb.diffstat("# spotlights one-shot apply\n# base:  abc\n")
    assert st == {"files": [], "added": 0, "removed": 0}


def test_format_diffstat_singular_plural_and_minus_sign():
    assert bb.format_diffstat({"files": [{}], "added": 69, "removed": 26}) == \
        "1 file changed, +69/−26"
    assert bb.format_diffstat({"files": [{}, {}], "added": 70, "removed": 26}) == \
        "2 files changed, +70/−26"
    assert bb.format_diffstat({"files": [], "added": 0, "removed": 0}) == \
        "0 files changed, +0/−0"


def test_format_diffstat_short_is_the_badge_form():
    # The badge form drops the file count and keeps the U+2212 minus sign.
    assert bb.format_diffstat_short({"files": [{}], "added": 69, "removed": 26}) == \
        "+69/−26"
    assert bb.format_diffstat_short({"files": [{}, {}], "added": 70, "removed": 26}) == \
        "+70/−26"
    assert "−" in bb.format_diffstat_short({"files": [{}], "added": 1, "removed": 2})


def _make_apply_tree(root, cand_id="cand-qiskit_compiler-0001", slug="qiskit_compiler",
                     patch=True, notes=True, prompt=True):
    """Create <root>/apply/<slug>/<cand_id>/ with the requested artifacts."""
    d = root / "apply" / slug / cand_id
    d.mkdir(parents=True, exist_ok=True)
    if patch:
        (d / "apply.patch").write_text(SAMPLE_PATCH, encoding="utf-8")
    if notes:
        (d / "APPLY-NOTES.md").write_text(
            "# Apply notes\n\n- **Candidate:** `%s`\n" % cand_id, encoding="utf-8")
    if prompt:
        (d / "apply.prompt.txt").write_text(
            "CANDIDATE: %s\nPROMPT:\nmake the change\n" % cand_id, encoding="utf-8")
    return d


def test_find_apply_joins_slug_path_and_returns_artifacts():
    root = Path(tempfile.mkdtemp())
    d = _make_apply_tree(root)
    fx = bb.find_apply("cand-qiskit_compiler-0001", root / "apply")
    assert fx is not None
    assert fx["patch"] == d / "apply.patch"
    assert fx["notes"] == d / "APPLY-NOTES.md"
    assert fx["prompt"] == d / "apply.prompt.txt"
    assert fx["header"]["base"] == "83ad767eed3be3ee7f2df63be693bfaca5c7c922"
    assert fx["stat"]["added"] == 4


def test_find_apply_absent_candidate_is_none():
    root = Path(tempfile.mkdtemp())
    _make_apply_tree(root)
    assert bb.find_apply("cand-qiskit_compiler-9999", root / "apply") is None
    # an absent apply/ tree entirely
    assert bb.find_apply("cand-qiskit_compiler-0001", root / "nope") is None


def test_find_apply_notes_only_directory_is_skipped():
    # `apply` writes APPLY-NOTES.md and no patch when the change was not made.
    # A share bundle skips those: no page, no badge, no copies.
    root = Path(tempfile.mkdtemp())
    _make_apply_tree(root, patch=False, notes=True, prompt=False)
    assert bb.find_apply("cand-qiskit_compiler-0001", root / "apply") is None


def test_find_apply_patch_without_notes_still_found():
    root = Path(tempfile.mkdtemp())
    _make_apply_tree(root, patch=True, notes=False)
    fx = bb.find_apply("cand-qiskit_compiler-0001", root / "apply")
    assert fx is not None and fx["notes"] is None


def test_find_apply_derives_the_slug_directory_from_the_candidate_id():
    # Two candidates under DIFFERENT slugs, each found only at its own path.
    # Every other test here uses one slug, which an implementation that ignored
    # `_cand_module_slug` and globbed `apply/*/<cand_id>` would also satisfy.
    root = Path(tempfile.mkdtemp())
    _make_apply_tree(root, cand_id="cand-qiskit_compiler-0001", slug="qiskit_compiler")
    _make_apply_tree(root, cand_id="cand-vllm_v1_kv_offload-0002", slug="vllm_v1_kv_offload")
    for cand_id, slug in (("cand-qiskit_compiler-0001", "qiskit_compiler"),
                          ("cand-vllm_v1_kv_offload-0002", "vllm_v1_kv_offload")):
        fx = bb.find_apply(cand_id, root / "apply")
        assert fx is not None, cand_id
        assert fx["patch"] == root / "apply" / slug / cand_id / "apply.patch"


def test_find_apply_reads_a_patch_that_is_not_valid_utf8():
    # The engine collects the diff as raw bytes and writes it with `write_bytes`,
    # so a non-UTF-8 context byte from the target repo reaches this file intact.
    # Strict decoding would raise UnicodeDecodeError, and `build()` gets here only
    # after removing the previous share-bundle/ — so one such candidate would
    # abort the whole run. It must degrade to a replacement character instead.
    root = Path(tempfile.mkdtemp())
    d = root / "apply" / "qiskit_compiler" / "cand-qiskit_compiler-0001"
    d.mkdir(parents=True)
    (d / "apply.patch").write_bytes(
        b"# spotlights one-shot apply\n"
        b"# base:      83ad767eed3be3ee7f2df63be693bfaca5c7c922\n"
        b"diff --git a/x.py b/x.py\n"
        b"--- a/x.py\n"
        b"+++ b/x.py\n"
        b"@@ -1,2 +1,2 @@\n"
        b"-caf\xe9\n"                      # latin-1 e-acute: invalid UTF-8
        b"+cafe\n")
    fx = bb.find_apply("cand-qiskit_compiler-0001", root / "apply")
    assert fx is not None
    assert fx["header"]["base"] == "83ad767eed3be3ee7f2df63be693bfaca5c7c922"
    assert fx["stat"]["added"] == 1 and fx["stat"]["removed"] == 1


def test_repo_placeholder_from_basename_and_fallback():
    assert bb._repo_placeholder("/Users/someone/checkouts/vllm") == "<YOUR_VLLM_CHECKOUT>"
    assert bb._repo_placeholder("/srv/qiskit-terra/") == "<YOUR_QISKIT_TERRA_CHECKOUT>"
    assert bb._repo_placeholder(None) == "<YOUR_REPO_CHECKOUT>"
    assert bb._repo_placeholder("") == "<YOUR_REPO_CHECKOUT>"


def test_render_patch_colorizes_and_escapes():
    out = bb.render_patch(SAMPLE_PATCH, "foo__apply/apply.patch", 7.1)
    # collapsed by default: a <details> with no `open` attribute
    assert '<details class="patch">' in out and "<details open" not in out
    # one file block per changed file, with its own per-file stat
    assert "vllm/v1/worker/gpu_model_runner.py" in out
    assert "vllm/v1/worker/utils.py" in out
    assert "+3" in out and "−2" in out
    # line classes (emitted as `class="l d-add"` etc.)
    assert 'class="l d-add"' in out
    assert 'class="l d-del"' in out
    assert 'class="l d-hunk"' in out
    # a context line gets the base class only
    assert 'class="l"' in out
    # the raw file is linked from the summary
    assert 'href="foo__apply/apply.patch"' in out
    # HTML from the patch body is escaped, never live markup
    assert "&lt;&amp;&gt;" in out
    assert "<&>" not in out


def test_render_patch_colors_match_counts_with_in_hunk_content_dashes():
    # The in_hunk state machine is critical: inside a hunk, a line starting
    # with --- or +++ is real content (e.g., deleting a markdown --- rule),
    # not a file marker. If in_hunk threading is deleted from render_patch,
    # lines like ---- and ++++marker are skipped (treated as markers), so the
    # coloured spans do not render for them — but diffstat counts them anyway,
    # so the file row's +X/−Y disagrees with the sum of the coloured spans.
    patch = (
        "diff --git a/file_<&>.py b/file_<&>.py\n"
        "--- a/file_<&>.py\n"
        "+++ b/file_<&>.py\n"
        "@@ -1,5 +1,4 @@\n"
        " context\n"
        "----\n"
        "--- removed\n"
        "++++added\n"
        " more context\n"
    )
    out = bb.render_patch(patch, "patch.txt", 0.1)
    # The file path must be escaped
    assert "file_&lt;&amp;&gt;.py" in out
    assert "file_<&>.py" not in out
    # Count the coloured spans in the rendered output
    add_spans = out.count('class="l d-add"')
    del_spans = out.count('class="l d-del"')
    # The file row shows the correct counts from diffstat: +1 −2
    # (---- is 1 deletion, --- removed is 1 deletion, ++++added is 1 addition)
    assert "+1" in out and "−2" in out
    # CRITICAL: the row counts and the span count must agree. If in_hunk is not
    # threaded, the ---- and --- lines are skipped and no spans render, but the
    # row still shows the correct counts, creating a disagreement.
    assert add_spans == 1, f"expected 1 d-add span, got {add_spans}"
    assert del_spans == 2, f"expected 2 d-del spans, got {del_spans}"


def test_render_patch_resets_hunk_state_at_every_file_boundary():
    # THE MULTI-FILE INVARIANT. `render_patch` must reset `in_hunk` at each
    # `diff --git`, because the next file opens with its own `--- a/…` /
    # `+++ b/…` markers. Without the reset, `in_hunk` is still True from the
    # previous file's hunk, so those two markers classify as a removed and an
    # added line and render as a spurious red and green diff row — every file
    # after the first. Invisible on a single-file patch, which is why all four
    # example patches missed it.
    stat = bb.diffstat(SAMPLE_PATCH)          # two files: +4 / −2
    assert len(stat["files"]) == 2, "this test needs a multi-file patch"
    out = bb.render_patch(SAMPLE_PATCH, "foo__apply/apply.patch", 7.1)
    # The coloured spans must total exactly what the file rows claim.
    assert out.count('class="l d-add"') == stat["added"]
    assert out.count('class="l d-del"') == stat["removed"]
    # And the second file's markers must not appear as diff content at all.
    assert '<span class="l d-del">--- a/vllm/v1/worker/utils.py' not in out
    assert '<span class="l d-add">+++ b/vllm/v1/worker/utils.py' not in out


def test_render_patch_escapes_paths_containing_html_metacharacters():
    # File paths can contain <, &, > (though git auto-quotes them). If the
    # html.escape on the path is deleted, these metacharacters appear live in
    # the rendered file header, allowing script injection.
    patch = (
        "diff --git a/weird_<script>_file.py b/weird_<script>_file.py\n"
        "--- a/weird_<script>_file.py\n"
        "+++ b/weird_<script>_file.py\n"
        "@@ -1 +1 @@\n"
        " context\n"
    )
    out = bb.render_patch(patch, "patch.txt", 0.1)
    # The path must appear escaped in the output
    assert "weird_&lt;script&gt;_file.py" in out
    # The raw metacharacters must not appear live
    assert "<script>_file" not in out


def test_render_apply_page_has_warning_apply_strip_and_notes():
    fx = {"patch": None, "notes": None,
          "header": {"candidate": "cand-a-0001", "module": "m/n",
                     "repo": "/Users/someone/checkouts/vllm",
                     "base": "83ad767eed3be3ee7f2df63be693bfaca5c7c922"},
          "stat": bb.diffstat(SAMPLE_PATCH),
          "patch_text": SAMPLE_PATCH, "notes_text": "# Apply notes\n\n- **Objective:** speed\n",
          "patch_kb": 7.1}
    out = bb.render_apply_page("cand-a-0001", "foo", fx, "foo__apply", "foo.html")
    assert "<!DOCTYPE html>" in out
    assert 'href="foo.html"' in out                      # back link
    assert "One-shot apply" in out and "foo" in out
    assert "2 files changed, +4/−2" in out       # SAMPLE_PATCH's real stat
    # the unverified warning is first-class content
    assert "nothing" in out.lower() and "verified" in out.lower()
    # apply strip: real SHA, placeholder, both fallbacks, producer-path note
    assert "83ad767eed3be3ee7f2df63be693bfaca5c7c922" in out
    assert "&lt;YOUR_VLLM_CHECKOUT&gt;" in out
    assert "apply -3" in out
    assert "patch -p1" in out
    assert "producer" in out.lower() or "machine that produced" in out.lower()
    # the producer's real path is never rendered as a command to run
    assert "$ REPO=/Users/someone" not in out
    # download button and the inlined notes
    assert 'href="foo__apply/apply.zip"' in out
    assert "Objective" in out


def test_render_apply_page_without_notes_omits_that_section():
    fx = {"header": {"base": "abc123"}, "stat": bb.diffstat(SAMPLE_PATCH),
          "patch_text": SAMPLE_PATCH, "notes_text": None, "patch_kb": 7.1}
    out = bb.render_apply_page("cand-a-0001", "foo", fx, "foo__apply", "foo.html")
    assert "<!DOCTYPE html>" in out          # still a valid page
    assert "The patch" in out                # the diff is still there
    assert "Objective" not in out            # no notes content leaked in
    # a missing base commit degrades to the placeholder, not a crash
    out2 = bb.render_apply_page("cand-a-0001", "foo",
                                {**fx, "header": {}}, "foo__apply", "foo.html")
    assert "&lt;BASE_COMMIT&gt;" in out2
    assert "&lt;YOUR_REPO_CHECKOUT&gt;" in out2


def test_render_apply_page_mentions_the_prompt_only_when_one_ships():
    # The download button is labelled "Download patch", so a recipient has no
    # reason to expect apply.prompt.txt inside the zip. The page names it —
    # but only when the file actually exists, or the note points at nothing.
    fx = {"header": {"base": "abc123"}, "stat": bb.diffstat(SAMPLE_PATCH),
          "patch_text": SAMPLE_PATCH, "notes_text": None, "patch_kb": 7.1}
    assert "apply.prompt.txt" not in bb.render_apply_page(
        "cand-a-0001", "foo", fx, "foo__apply", "foo.html")
    out = bb.render_apply_page("cand-a-0001", "foo",
                               {**fx, "prompt": Path("apply.prompt.txt")},
                               "foo__apply", "foo.html")
    assert "apply.prompt.txt" in out


def test_copy_apply_files_puts_the_prompt_in_the_folder_and_the_zip():
    # The zip is what the "Download patch (.zip)" button hands over, so the
    # prompt has to be a zip member — being copied into the raw folder beside it
    # is not enough for anyone who only clicks the button.
    root = Path(tempfile.mkdtemp())
    d = _make_apply_tree(root, notes=False)          # patch + prompt, no notes
    fx = bb.find_apply("cand-qiskit_compiler-0001", root / "apply")
    assert fx["prompt"] == d / "apply.prompt.txt"

    out = root / "out__apply"
    bb._copy_apply_files(fx, out)
    assert filecmp.cmp(d / "apply.prompt.txt", out / "apply.prompt.txt", shallow=False)
    with _zip.ZipFile(out / "apply.zip") as zf:
        assert zf.namelist() == ["apply/apply.patch", "apply/apply.prompt.txt"]


def test_copy_apply_files_without_a_prompt_ships_no_placeholder():
    # apply.prompt.txt is optional: an older apply/ tree has none, and the zip
    # must not gain an empty entry (nor `shutil.copy2(None, …)` a TypeError).
    root = Path(tempfile.mkdtemp())
    _make_apply_tree(root, prompt=False)
    fx = bb.find_apply("cand-qiskit_compiler-0001", root / "apply")
    assert fx["prompt"] is None

    out = root / "out__apply"
    bb._copy_apply_files(fx, out)
    assert not (out / "apply.prompt.txt").exists()
    with _zip.ZipFile(out / "apply.zip") as zf:
        assert zf.namelist() == ["apply/apply.patch", "apply/APPLY-NOTES.md"]


def test_render_apply_page_says_which_directory_the_apply_commands_assume():
    # The apply commands use "$PWD/{raw_reldir}/apply.patch", so they are
    # silently cwd-dependent. The page must state which directory to cd into.
    # Without this note, running the commands from a different directory
    # (e.g., share-bundle/ or from the unpacked .zip) fails with "can't open
    # patch: No such file or directory" and no explanation of why.
    fx = {"header": {"base": "abc123"}, "stat": bb.diffstat(SAMPLE_PATCH),
          "patch_text": SAMPLE_PATCH, "notes_text": None, "patch_kb": 7.1}
    out = bb.render_apply_page("cand-a-0001", "foo", fx, "artifact_dir__apply", "foo.html")
    # The cwd note must mention the artifact directory name
    assert "artifact_dir__apply" in out
    # The note must explain the cwd requirement
    assert "$PWD" in out and "directory" in out.lower()


def test_render_apply_section_links_to_the_page():
    fx = {"stat": bb.diffstat(SAMPLE_PATCH)}
    out = bb.render_apply_section(fx, "foo__cand-a-0001__apply.html")
    assert "One-shot apply" in out
    assert 'href="foo__cand-a-0001__apply.html"' in out
    assert "2 files changed, +4/−2" in out
    assert "verified" in out.lower()


def test_render_candidate_page_emits_apply_before_evolve():
    out = bb.render_candidate_page("# t\n", "t", "../index.html",
                                   "<div>EVOLVEBLOCK</div>", "<div>APPLYBLOCK</div>")
    assert out.index("APPLYBLOCK") < out.index("EVOLVEBLOCK")


def test_render_candidate_page_still_works_with_evolve_only():
    # Regression: the pre-existing 4-positional-arg call must keep working.
    out = bb.render_candidate_page("# t\n", "t", "../index.html", "<div>E</div>")
    assert "<div>E</div>" in out


def test_build_end_to_end_with_apply_bundle():
    root = Path(tempfile.mkdtemp())
    src = root / "sorted"
    src.mkdir(parents=True, exist_ok=True)
    (src / "sorted_candidates.md").write_text(SAMPLE_SORTED, encoding="utf-8")
    c1 = (src / ".." / "modules" / "qiskit_compiler" / "foo__cand-a-0001.md").resolve()
    c1.parent.mkdir(parents=True, exist_ok=True)
    c1.write_text("# foo\n", encoding="utf-8")
    # SAMPLE_SORTED's top row is cand-a-0001 -> module slug "a"
    _make_apply_tree(root, cand_id="cand-a-0001", slug="a")

    result = bb.build(str(src), top_n=1)
    assert result["apply_bundles"] == 1

    bundle = Path(result["bundle_dir"])
    d = bundle / "candidates" / "modules" / "qiskit_compiler"
    page = d / "foo__cand-a-0001__apply.html"
    assert page.exists()
    assert (d / "foo__cand-a-0001__apply" / "apply.patch").exists()
    assert (d / "foo__cand-a-0001__apply" / "APPLY-NOTES.md").exists()
    assert (d / "foo__cand-a-0001__apply" / "apply.prompt.txt").exists()
    assert (d / "foo__cand-a-0001__apply" / "apply.zip").exists()

    # Copies are byte-for-byte — compared as BYTES, not as decoded text. A
    # decode-and-rewrite (any encoding, any newline translation) produces text
    # that still compares equal while breaking the guarantee `git apply` needs,
    # so read_text() cannot pin this. filecmp with shallow=False can.
    src_apply = root / "apply" / "a" / "cand-a-0001"
    assert filecmp.cmp(src_apply / "apply.patch",
                       d / "foo__cand-a-0001__apply" / "apply.patch", shallow=False)
    assert filecmp.cmp(src_apply / "APPLY-NOTES.md",
                       d / "foo__cand-a-0001__apply" / "APPLY-NOTES.md", shallow=False)
    assert filecmp.cmp(src_apply / "apply.prompt.txt",
                       d / "foo__cand-a-0001__apply" / "apply.prompt.txt", shallow=False)

    # zip entries live under a top-level apply/ folder
    with _zip.ZipFile(d / "foo__cand-a-0001__apply" / "apply.zip") as zf:
        assert sorted(zf.namelist()) == [
            "apply/APPLY-NOTES.md", "apply/apply.patch", "apply/apply.prompt.txt"]

    # the candidate page links to the apply page; the apply page links back
    cand_html = (d / "foo__cand-a-0001.html").read_text(encoding="utf-8")
    assert 'href="foo__cand-a-0001__apply.html"' in cand_html
    assert 'href="foo__cand-a-0001.html"' in page.read_text(encoding="utf-8")

    # The INDEX-side wiring, end to end. render_index's unit tests are built from
    # hand-written row dicts, so nothing else connects build() to the index:
    # dropping r["apply_stat"] or r["apply_href"] in build() silently strips the badge
    # and the pill from every card while every other test stays green.
    idx = (bundle / "index.html").read_text(encoding="utf-8")
    st = bb.diffstat(SAMPLE_PATCH)
    # The badge carries the short stat, the pill the full prose one: build() has
    # to set both row keys, from the two different helpers.
    assert f'<span class="badge apply">apply · {bb.format_diffstat_short(st)}</span>' in idx
    assert f'🔧 One-shot apply: {bb.format_diffstat(st)} →' in idx
    apply_href = "candidates/modules/qiskit_compiler/foo__cand-a-0001__apply.html"
    assert f'class="apply-link" href="{apply_href}"' in idx
    # ...and that href actually resolves to the generated page
    assert (bundle / apply_href).exists()

    # and the whole thing is in the outer zip
    with _zip.ZipFile(Path(result["zip_path"])) as zf:
        names = zf.namelist()
    assert any(n.endswith("foo__cand-a-0001__apply.html") for n in names)
    assert any(n.endswith("foo__cand-a-0001__apply/apply.zip") for n in names)


def test_build_patch_without_notes_builds_a_bundle_of_one_file():
    # APPLY-NOTES.md and apply.prompt.txt are both optional. `_copy_apply_files`
    # guards on fx.get() for each, and without that guard `shutil.copy2(None, …)`
    # raises TypeError — *after* build() has already removed the previous
    # share-bundle/, so the user is left with no bundle at all. Same
    # abort-with-nothing class as a strict decode.
    root = Path(tempfile.mkdtemp())
    src = root / "sorted"
    src.mkdir(parents=True, exist_ok=True)
    (src / "sorted_candidates.md").write_text(SAMPLE_SORTED, encoding="utf-8")
    c1 = (src / ".." / "modules" / "qiskit_compiler" / "foo__cand-a-0001.md").resolve()
    c1.parent.mkdir(parents=True, exist_ok=True)
    c1.write_text("# foo\n", encoding="utf-8")
    _make_apply_tree(root, cand_id="cand-a-0001", slug="a", patch=True, notes=False,
                     prompt=False)

    result = bb.build(str(src), top_n=1)
    assert result["apply_bundles"] == 1        # a patch alone is enough

    bundle = Path(result["bundle_dir"])
    d = bundle / "candidates" / "modules" / "qiskit_compiler"
    applydir = d / "foo__cand-a-0001__apply"
    assert (d / "foo__cand-a-0001__apply.html").exists()
    assert (applydir / "apply.patch").exists()
    assert not (applydir / "APPLY-NOTES.md").exists()  # nothing invented
    with _zip.ZipFile(applydir / "apply.zip") as zf:
        assert zf.namelist() == ["apply/apply.patch"]
    # the page is still complete: the diff renders, the index still links to it
    page = (d / "foo__cand-a-0001__apply.html").read_text(encoding="utf-8")
    assert "The patch" in page and 'class="l d-add"' in page
    assert 'class="badge apply"' in (bundle / "index.html").read_text(encoding="utf-8")


def test_build_skips_notes_only_apply_directory():
    root = Path(tempfile.mkdtemp())
    src = root / "sorted"
    src.mkdir(parents=True, exist_ok=True)
    (src / "sorted_candidates.md").write_text(SAMPLE_SORTED, encoding="utf-8")
    c1 = (src / ".." / "modules" / "qiskit_compiler" / "foo__cand-a-0001.md").resolve()
    c1.parent.mkdir(parents=True, exist_ok=True)
    c1.write_text("# foo\n", encoding="utf-8")
    # prompt=False matters: a directory holding notes *and* a prompt is a prompt
    # bundle now, so "notes only" has to say so explicitly.
    _make_apply_tree(root, cand_id="cand-a-0001", slug="a",
                     patch=False, notes=True, prompt=False)

    result = bb.build(str(src), top_n=1)
    assert result["apply_bundles"] == 0
    assert result["apply_prompts"] == 0
    bundle = Path(result["bundle_dir"])
    d = bundle / "candidates" / "modules" / "qiskit_compiler"
    assert not (d / "foo__cand-a-0001__apply.html").exists()
    assert "One-shot apply" not in (d / "foo__cand-a-0001.html").read_text(encoding="utf-8")
    assert "badge apply" not in (bundle / "index.html").read_text(encoding="utf-8")


# ---------------------------------------------------------------------------
# Prompt-only apply bundles.
#
# `spotlights-engine apply` has a second, equally legitimate outcome: it writes
# apply.prompt.txt — the instruction it would have handed to its own agent — and
# no patch, because nothing was implemented. Before this existed, find_apply()
# returned None without a patch and such a run produced a silently empty share
# bundle: 5 candidates, 0 apply pages, no explanation anywhere. The prompt is now
# a first-class outcome with its own page behind the same `…__apply.html` name.
# ---------------------------------------------------------------------------

SAMPLE_PROMPT = """\
CANDIDATE: cand-a-0001
MODULE:    qiskit/compiler
BASE:      433942b8fb4e726f6fb512686a511f7ddc782081
REPO:      /Users/someone/checkouts/vllm
DIRTY:     true
WORKTREE:  (none — create one; it is yours to remove when you are done)
  git -C /Users/someone/checkouts/vllm worktree add --detach <dir> 433942b8
PROMPT:
You are implementing ONE proposed optimization in an isolated git worktree.

## Optimization goal
Reduce circuit depth after transpilation (direction: minimize)

MODULE: this line is inside the prompt body and is not a field
"""


def test_parse_prompt_header_extracts_the_fields_above_the_prompt_line():
    h = bb.parse_prompt_header(SAMPLE_PROMPT)
    assert h["candidate"] == "cand-a-0001"
    assert h["module"] == "qiskit/compiler"
    assert h["base"] == "433942b8fb4e726f6fb512686a511f7ddc782081"
    assert h["repo"] == "/Users/someone/checkouts/vllm"
    # Same lowercase keys as parse_patch_header, so both feed one renderer.
    assert h["dirty"] == "true"


def test_parse_prompt_header_stops_at_the_prompt_line():
    # `MODULE:` appears again inside the prompt body. Scanning the whole file
    # would let prose overwrite a real field — the last match would win and the
    # apply strip would print a sentence where a module path belongs.
    h = bb.parse_prompt_header(SAMPLE_PROMPT)
    assert h["module"] == "qiskit/compiler"
    # WORKTREE is deliberately not captured: its value is prose, and its
    # indented continuation line must not be read as a field either.
    assert "worktree" not in h
    assert not any(v.startswith("git -C") for v in h.values())


def test_parse_prompt_header_missing_fields_degrade():
    assert bb.parse_prompt_header("PROMPT:\ndo the thing\n") == {}
    assert bb.parse_prompt_header("BASE: abc123\nPROMPT:\nx\n") == {"base": "abc123"}
    # A `#`-prefixed patch-style comment is NOT a prompt field.
    assert bb.parse_prompt_header("# base: abc123\nPROMPT:\n") == {}


def test_find_apply_prompt_only_directory_is_prompt_mode():
    root = Path(tempfile.mkdtemp())
    d = _make_apply_tree(root, patch=False, notes=False, prompt=True)
    fx = bb.find_apply("cand-qiskit_compiler-0001", root / "apply")
    assert fx is not None
    assert fx["mode"] == "prompt"
    assert fx["patch"] is None and fx["stat"] is None and fx["patch_text"] is None
    assert fx["prompt"] == d / "apply.prompt.txt"
    assert fx["header"]["candidate"] == "cand-qiskit_compiler-0001"
    assert fx["prompt_kb"] > 0


def test_find_apply_patch_wins_when_both_ship():
    # The prompt ships beside the patch in every normal apply/ tree. Dispatching
    # on the prompt first would replace every patch page with a prompt page.
    root = Path(tempfile.mkdtemp())
    _make_apply_tree(root, patch=True, prompt=True)
    fx = bb.find_apply("cand-qiskit_compiler-0001", root / "apply")
    assert fx["mode"] == "patch"
    assert fx["stat"]["added"] == 4
    assert fx["prompt"] is not None          # still shipped, just not the mode


def test_find_apply_prompt_with_notes_is_prompt_mode_and_keeps_the_notes():
    root = Path(tempfile.mkdtemp())
    d = _make_apply_tree(root, patch=False, notes=True, prompt=True)
    fx = bb.find_apply("cand-qiskit_compiler-0001", root / "apply")
    assert fx["mode"] == "prompt"
    assert fx["notes"] == d / "APPLY-NOTES.md"
    assert "Apply notes" in fx["notes_text"]


def test_find_apply_reads_a_prompt_that_is_not_valid_utf8():
    # Same abort-with-nothing hazard as the patch: build() has already removed
    # the previous share-bundle/ by the time it gets here.
    root = Path(tempfile.mkdtemp())
    d = root / "apply" / "qiskit_compiler" / "cand-qiskit_compiler-0001"
    d.mkdir(parents=True)
    (d / "apply.prompt.txt").write_bytes(
        b"BASE:      abc123\nPROMPT:\nrename caf\xe9 to cafe\n")
    fx = bb.find_apply("cand-qiskit_compiler-0001", root / "apply")
    assert fx is not None and fx["mode"] == "prompt"
    assert fx["header"]["base"] == "abc123"


def test_copy_apply_files_without_a_patch_ships_the_prompt():
    # `members` used to start from fx["patch"] unconditionally; in prompt mode
    # that is None and shutil.copy2(None, …) raises TypeError.
    root = Path(tempfile.mkdtemp())
    d = _make_apply_tree(root, patch=False, notes=True, prompt=True)
    fx = bb.find_apply("cand-qiskit_compiler-0001", root / "apply")

    out = root / "out__apply"
    bb._copy_apply_files(fx, out)
    assert not (out / "apply.patch").exists()
    assert filecmp.cmp(d / "apply.prompt.txt", out / "apply.prompt.txt", shallow=False)
    with _zip.ZipFile(out / "apply.zip") as zf:
        assert sorted(zf.namelist()) == ["apply/APPLY-NOTES.md", "apply/apply.prompt.txt"]


def test_render_prompt_text_is_collapsed_with_a_raw_link():
    out = bb.render_prompt_text("prompt <body> & more", "foo__apply/apply.prompt.txt", 12.4)
    assert out.startswith('<details class="patch"><summary>apply.prompt.txt')
    assert "— 12 KB" in out
    assert 'href="foo__apply/apply.prompt.txt"' in out
    assert "prompt &lt;body&gt; &amp; more" in out      # escaped exactly once
    assert "<body>" not in out


def _prompt_fx(**over):
    fx = {"mode": "prompt", "patch": None, "stat": None, "patch_text": None,
          "notes": None, "notes_text": None,
          "prompt": Path("apply.prompt.txt"),
          "header": bb.parse_prompt_header(SAMPLE_PROMPT),
          "prompt_text": SAMPLE_PROMPT, "prompt_kb": 12.4}
    fx.update(over)
    return fx


def test_render_apply_page_prompt_mode_has_the_agent_recipe_and_downloads():
    out = bb.render_apply_page("cand-a-0001", "foo", _prompt_fx(),
                               "foo__apply", "foo.html")
    assert "<!DOCTYPE html>" in out
    assert 'href="foo.html"' in out                       # same back link
    assert "One-shot apply" in out
    # It says, up front, that there is no patch and nothing was run.
    assert "prompt only — no patch" in out
    assert "nothing was implemented" in out.lower()
    # The recipe: real base commit, placeholder repo, a worktree, an agent.
    assert "433942b8fb4e726f6fb512686a511f7ddc782081" in out
    assert "&lt;YOUR_VLLM_CHECKOUT&gt;" in out
    assert "worktree add --detach" in out
    assert "coding agent" in out
    assert "claude -p" in out and "codex exec" in out and "cursor-agent -p" in out
    # The producer's real path is never rendered as a command to run.
    assert "REPO=/Users/someone" not in out
    assert "producer" in out.lower() or "machine that produced" in out.lower()
    # Both downloads: the bare prompt file and the zip.
    assert 'href="foo__apply/apply.prompt.txt" download' in out
    assert 'href="foo__apply/apply.zip" download' in out
    # ...and the prompt itself is inlined.
    assert "Reduce circuit depth after transpilation" in out


def test_render_apply_page_prompt_mode_states_the_cwd_the_recipe_assumes():
    # `PROMPT="$PWD/<raw_reldir>/apply.prompt.txt"` is cwd-dependent exactly as
    # the patch recipe is, and fails the same opaque way from the wrong folder.
    out = bb.render_apply_page("cand-a-0001", "foo", _prompt_fx(),
                               "artifact_dir__apply", "foo.html")
    assert "artifact_dir__apply" in out
    assert "$PWD" in out and "folder holding this page" in out


def test_render_apply_page_prompt_mode_degrades_without_base_or_repo():
    out = bb.render_apply_page("cand-a-0001", "foo", _prompt_fx(header={}),
                               "foo__apply", "foo.html")
    assert "<!DOCTYPE html>" in out
    assert "&lt;BASE_COMMIT&gt;" in out
    assert "&lt;YOUR_REPO_CHECKOUT&gt;" in out
    # With no base there is no "line numbers only hold there" claim to make.
    assert "Base commit" not in out


def test_render_apply_page_prompt_mode_omits_notes_when_absent():
    with_notes = bb.render_apply_page(
        "cand-a-0001", "foo",
        _prompt_fx(notes_text="# Apply notes\n\n- **Objective:** speed\n"),
        "foo__apply", "foo.html")
    assert "Objective" in with_notes
    assert "Objective" not in bb.render_apply_page(
        "cand-a-0001", "foo", _prompt_fx(), "foo__apply", "foo.html")


def test_render_apply_page_without_a_mode_key_still_renders_the_patch_page():
    # Regression: the dispatch is `fx.get("mode") == "prompt"`, so every caller
    # and every fixture that predates `mode` keeps getting the patch page.
    fx = {"header": {"base": "abc123"}, "stat": bb.diffstat(SAMPLE_PATCH),
          "patch_text": SAMPLE_PATCH, "notes_text": None, "patch_kb": 7.1}
    out = bb.render_apply_page("cand-a-0001", "foo", fx, "foo__apply", "foo.html")
    assert "The patch" in out and 'class="l d-add"' in out
    assert "coding agent" not in out


def test_render_apply_section_prompt_mode_says_no_patch():
    out = bb.render_apply_section({"mode": "prompt"},
                                  "foo__cand-a-0001__apply.html")
    assert "One-shot apply" in out
    assert 'href="foo__cand-a-0001__apply.html"' in out
    assert "View the prompt →" in out
    assert "no patch was produced" in out
    # It must not reach for fx["stat"], which is None in prompt mode.
    assert "files changed" not in out


def test_build_end_to_end_with_prompt_only_apply():
    root = Path(tempfile.mkdtemp())
    src = root / "sorted"
    src.mkdir(parents=True, exist_ok=True)
    (src / "sorted_candidates.md").write_text(SAMPLE_SORTED, encoding="utf-8")
    c1 = (src / ".." / "modules" / "qiskit_compiler" / "foo__cand-a-0001.md").resolve()
    c1.parent.mkdir(parents=True, exist_ok=True)
    c1.write_text("# foo\n", encoding="utf-8")
    # SAMPLE_SORTED's top row is cand-a-0001 -> module slug "a"
    _make_apply_tree(root, cand_id="cand-a-0001", slug="a", patch=False)

    result = bb.build(str(src), top_n=1)
    # The two counters are disjoint: this run produced no patch at all.
    assert result["apply_prompts"] == 1
    assert result["apply_bundles"] == 0

    bundle = Path(result["bundle_dir"])
    d = bundle / "candidates" / "modules" / "qiskit_compiler"
    page = d / "foo__cand-a-0001__apply.html"
    assert page.exists()
    raw = d / "foo__cand-a-0001__apply"
    assert not (raw / "apply.patch").exists()
    assert (raw / "apply.prompt.txt").exists()
    assert (raw / "APPLY-NOTES.md").exists()
    assert (raw / "apply.zip").exists()

    # Byte-for-byte, compared as bytes: the recipient hands this exact file to
    # an agent, and a decode-and-rewrite would still pass a read_text() compare.
    src_apply = root / "apply" / "a" / "cand-a-0001"
    assert filecmp.cmp(src_apply / "apply.prompt.txt",
                       raw / "apply.prompt.txt", shallow=False)

    # The page's download button points at a file that is really there.
    html_txt = page.read_text(encoding="utf-8")
    assert 'href="foo__cand-a-0001__apply/apply.prompt.txt" download' in html_txt
    assert 'href="foo__cand-a-0001.html"' in html_txt          # back link

    # candidate page -> apply page, and it says there is no patch
    cand_html = (d / "foo__cand-a-0001.html").read_text(encoding="utf-8")
    assert 'href="foo__cand-a-0001__apply.html"' in cand_html
    assert "no patch was produced" in cand_html

    # INDEX-side wiring: build() has to set apply_stat/apply_badge_stat/apply_href
    # from the prompt arm too, or a prompt-only run gets a page nothing links to.
    idx = (bundle / "index.html").read_text(encoding="utf-8")
    assert '<span class="badge apply">apply · prompt</span>' in idx
    assert "🔧 One-shot apply: run with a coding agent →" in idx
    apply_href = "candidates/modules/qiskit_compiler/foo__cand-a-0001__apply.html"
    assert f'class="apply-link" href="{apply_href}"' in idx
    assert (bundle / apply_href).exists()

    # and the whole thing is in the outer zip
    with _zip.ZipFile(Path(result["zip_path"])) as zf:
        names = zf.namelist()
    assert any(n.endswith("foo__cand-a-0001__apply.html") for n in names)
    assert any(n.endswith("foo__cand-a-0001__apply/apply.prompt.txt") for n in names)


def test_render_index_prompt_badge_and_footer_link():
    # The badge and the pill are free text, so the prompt arm reuses the same
    # two row keys the patch arm sets. This pins the strings build() puts there.
    header = {"title": "T", "objective": "O"}
    rows = [
        {"rank": "1", "cand_id": "cand-a-0001", "module": "m", "symbol": "foo",
         "impact": "high", "score": "96", "rationale": "R",
         "html_href": "candidates/modules/m/foo__cand-a-0001.html",
         "apply_stat": "run with a coding agent",
         "apply_badge_stat": "prompt",
         "apply_href": "candidates/modules/m/foo__cand-a-0001__apply.html"},
    ]
    idx = bb.render_index(header, rows)
    assert '<span class="badge apply">apply · prompt</span>' in idx
    assert "🔧 One-shot apply: run with a coding agent →" in idx
    assert 'href="candidates/modules/m/foo__cand-a-0001__apply.html"' in idx


def test_build_reports_both_apply_counters_when_a_run_mixes_the_two():
    # A real run can have both: some candidates got a patch, others only a
    # prompt. The two counters must not double-count or cross-count.
    root = Path(tempfile.mkdtemp())
    src = root / "sorted"
    src.mkdir(parents=True, exist_ok=True)
    (src / "sorted_candidates.md").write_text(SAMPLE_SORTED, encoding="utf-8")
    # SAMPLE_SORTED's first two rows live under DIFFERENT module folders.
    mods = (src / ".." / "modules").resolve()
    (mods / "qiskit_compiler").mkdir(parents=True, exist_ok=True)
    (mods / "qiskit_transpiler_passes").mkdir(parents=True, exist_ok=True)
    (mods / "qiskit_compiler" / "foo__cand-a-0001.md").write_text(
        "# foo\n", encoding="utf-8")
    (mods / "qiskit_transpiler_passes" / "bar__cand-b-0002.md").write_text(
        "# bar\n", encoding="utf-8")
    _make_apply_tree(root, cand_id="cand-a-0001", slug="a", patch=True)
    _make_apply_tree(root, cand_id="cand-b-0002", slug="b", patch=False)

    result = bb.build(str(src), top_n=2)
    assert result["count"] == 2
    assert result["apply_bundles"] == 1
    assert result["apply_prompts"] == 1
    out = Path(result["bundle_dir"]) / "candidates" / "modules"
    assert (out / "qiskit_compiler" / "foo__cand-a-0001__apply"
            / "apply.patch").exists()
    assert not (out / "qiskit_transpiler_passes" / "bar__cand-b-0002__apply"
                / "apply.patch").exists()
    assert (out / "qiskit_transpiler_passes" / "bar__cand-b-0002__apply"
            / "apply.prompt.txt").exists()
    idx = (Path(result["bundle_dir"]) / "index.html").read_text(encoding="utf-8")
    assert '<span class="badge apply">apply · prompt</span>' in idx
    assert '<span class="badge apply">apply · +4/−2</span>' in idx


def test_build_end_to_end_with_both_arms():
    # The fourth tree combination: apply/ and evolve/ side by side.
    root = Path(tempfile.mkdtemp())
    src = root / "sorted"
    src.mkdir(parents=True, exist_ok=True)
    (src / "sorted_candidates.md").write_text(SAMPLE_SORTED, encoding="utf-8")
    c1 = (src / ".." / "modules" / "qiskit_compiler" / "foo__cand-a-0001.md").resolve()
    c1.parent.mkdir(parents=True, exist_ok=True)
    c1.write_text("# foo\n", encoding="utf-8")
    _make_apply_tree(root, cand_id="cand-a-0001", slug="a")
    ev = root / "evolve" / "a" / "cand-a-0001" / "coral"
    ev.mkdir(parents=True, exist_ok=True)
    (ev / "task.yaml").write_text("name: t\n", encoding="utf-8")

    result = bb.build(str(src), top_n=1)
    assert result["apply_bundles"] == 1
    assert result["evolve_bundles"] == 1

    bundle = Path(result["bundle_dir"])
    d = bundle / "candidates" / "modules" / "qiskit_compiler"
    assert (d / "foo__cand-a-0001__apply.html").exists()
    assert (d / "foo__cand-a-0001__evolve.html").exists()

    # both sections on the candidate page, apply first
    cand_html = (d / "foo__cand-a-0001.html").read_text(encoding="utf-8")
    assert cand_html.index("One-shot apply") < cand_html.index("Evolve bundles")

    # both arms' back links point at the same candidate page
    for page in ("foo__cand-a-0001__apply.html", "foo__cand-a-0001__evolve.html"):
        assert 'href="foo__cand-a-0001.html"' in (d / page).read_text(encoding="utf-8")


def test_build_without_apply_tree_reports_zero():
    # Regression: a run with no apply/ tree at all still builds.
    root = Path(tempfile.mkdtemp())
    src = root / "sorted"
    src.mkdir(parents=True, exist_ok=True)
    (src / "sorted_candidates.md").write_text(SAMPLE_SORTED, encoding="utf-8")
    c1 = (src / ".." / "modules" / "qiskit_compiler" / "foo__cand-a-0001.md").resolve()
    c1.parent.mkdir(parents=True, exist_ok=True)
    c1.write_text("# foo\n", encoding="utf-8")
    result = bb.build(str(src), top_n=1)
    assert result["apply_bundles"] == 0
    assert result["count"] == 1


def test_render_index_apply_badge_and_footer_link():
    header = {"title": "T", "objective": "O"}
    rows = [
        {"rank": "1", "cand_id": "cand-a-0001", "module": "m", "symbol": "foo",
         "impact": "high", "score": "96", "rationale": "R",
         "html_href": "candidates/modules/m/foo__cand-a-0001.html",
         "apply_stat": "1 file changed, +69/−26",
         "apply_badge_stat": "+69/−26",
         "apply_href": "candidates/modules/m/foo__cand-a-0001__apply.html"},
    ]
    idx = bb.render_index(header, rows)
    # The badge is the short form and the pill the long one, on the same card.
    # Collapsing them back to one string fails here.
    assert '<span class="badge apply">apply · +69/−26</span>' in idx
    assert '<span class="badge apply">apply · 1 file changed, +69/−26</span>' not in idx
    assert '🔧 One-shot apply: 1 file changed, +69/−26 →' in idx
    assert 'class="apply-link"' in idx
    assert 'href="candidates/modules/m/foo__cand-a-0001__apply.html"' in idx


def test_render_index_shows_both_pills_with_apply_first():
    header = {"title": "T", "objective": "O"}
    rows = [
        {"rank": "1", "cand_id": "cand-a-0001", "module": "m", "symbol": "foo",
         "impact": "high", "score": "96", "rationale": "R",
         "html_href": "candidates/modules/m/foo__cand-a-0001.html",
         "apply_stat": "1 file changed, +69/−26",
         "apply_badge_stat": "+69/−26",
         "apply_href": "candidates/modules/m/foo__cand-a-0001__apply.html",
         "evolve_count": 2, "evolve_engines": ["coral", "nous"],
         "evolve_href": "candidates/modules/m/foo__cand-a-0001__evolve.html"},
    ]
    idx = bb.render_index(header, rows)
    assert 'class="apply-link"' in idx and 'class="evolve-link"' in idx
    assert idx.index('class="apply-link"') < idx.index('class="evolve-link"')
    assert '<span class="badge apply">' in idx
    assert '<span class="badge evolve">evolve · 2</span>' in idx


def _css_rule(selector):
    """Return the declaration block for `selector` from the embedded stylesheet.

    Matches the selector as a whole line-leading token so `.badge` does not also
    pick up `.badge.apply`, and strips comments (which contain braces-free prose
    but would otherwise pad the match).
    """
    body = re.sub(r"/\*.*?\*/", "", bb._CSS, flags=re.S)
    m = re.search(r"(?m)^" + re.escape(selector) + r"\s*\{(.*?)\}", body, re.S)
    assert m, f"no rule found for {selector}"
    return " ".join(m.group(1).split())


def test_no_badge_modifier_shares_a_name_with_a_block_rule():
    # THE bug behind the misshapen apply badge. Every badge is `class="badge X"`,
    # so a bare `.X { }` rule anywhere in the stylesheet also matches it — at equal
    # specificity (0,1,0), and later in the sheet, so it WINS. `.apply` (the apply
    # page's "Apply this patch" strip: padding 10px 14px, border-radius 8px) was
    # therefore restyling `<span class="badge apply">` on the index: 10px of
    # vertical padding where the other badges had .12em, and an 8px radius instead
    # of the pill's 999px. Renaming the block to `.apply-strip` fixes it.
    #
    # Asserted over every modifier rather than just `apply`: the next badge colour
    # someone adds is one bare rule away from the same silent override.
    css = re.sub(r"/\*.*?\*/", "", bb._CSS, flags=re.S)
    modifiers = {"apply", "evolve", "score", "impact-high"}
    for mod in sorted(modifiers):
        bare = re.findall(r"(?m)^\." + re.escape(mod) + r"(?=[\s,{])[^{]*", css)
        assert not bare, f".{mod} is a badge modifier AND a block selector: {bare}"
    # the modifier set above is the real one — a new badge class must join it
    idx = bb.render_index(
        {"title": "T"},
        [{"rank": "1", "cand_id": "c", "module": "m", "symbol": "s", "impact": "high",
          "score": "9", "rationale": "R", "html_href": "c.html",
          "apply_stat": "1 file changed, +1/−1", "apply_badge_stat": "+1/−1",
          "apply_href": "a.html", "evolve_count": 1, "evolve_engines": ["coral"],
          "evolve_href": "e.html"}])
    used = {c for m in re.finditer(r'class="badge ([^"]*)"', idx) for c in m.group(1).split()}
    assert used <= modifiers, f"unchecked badge modifier(s): {used - modifiers}"


def test_badges_row_keeps_every_badge_a_pill():
    # Hardening around the fix above: the badges are flex items, so under the flex
    # default (align-items:stretch) a badge shorter than a taller sibling inherits
    # that height and its border-radius:999px turns it into a circle. Nothing
    # produces a taller sibling today, but a squeezed badge whose text wrapped
    # would — hence no-shrink and nowrap.
    badges = _css_rule(".badges")
    assert "align-items: center" in badges      # never stretch a rounded pill
    assert "flex-shrink: 0" in badges           # never squeeze one narrower than its text
    assert "white-space: nowrap" in _css_rule(".badge")   # so it cannot wrap to 2 lines
    # .row1 must be free to move the whole group to its own line instead
    assert "flex-wrap: wrap" in _css_rule(".card .row1")


def test_apply_page_strip_uses_the_renamed_block_class():
    fx = {"header": {"base": "abc123"}, "stat": bb.diffstat(SAMPLE_PATCH),
          "patch_text": SAMPLE_PATCH, "notes_text": None, "patch_kb": 7.1}
    out = bb.render_apply_page("cand-a-0001", "foo", fx, "foo__apply", "foo.html")
    assert '<div class="apply-strip">' in out
    assert '<div class="apply">' not in out
    # the strip's own styling must have survived the rename
    assert "border-radius:8px" in _css_rule(".apply-strip")
    assert "Apply this patch" in out and "apply -3" in out


def test_footer_pins_apply_left_and_evolve_right():
    # Both arms are independently optional, so alignment cannot come from
    # justify-content: space-between — that leaves a lone evolve pill on the left.
    # An auto left margin on evolve holds the right edge in all three shapes.
    assert "margin-left:auto" in _css_rule(".evolve-link").replace(" ", "")
    assert "margin-left:auto" not in _css_rule(".apply-link").replace(" ", "")
    assert "space-between" not in _css_rule(".card-foot")

    header = {"title": "T", "objective": "O"}

    def _row(**kw):
        r = {"rank": "1", "cand_id": "cand-a-0001", "module": "m", "symbol": "foo",
             "impact": "high", "score": "96", "rationale": "R",
             "html_href": "c.html"}
        r.update(kw)
        return r

    both = bb.render_index(header, [_row(
        apply_stat="1 file changed, +69/−26", apply_badge_stat="+69/−26",
        apply_href="a.html", evolve_count=2, evolve_engines=["coral", "nous"],
        evolve_href="e.html")])
    evolve_only = bb.render_index(header, [_row(
        evolve_count=2, evolve_engines=["coral", "nous"], evolve_href="e.html")])
    apply_only = bb.render_index(header, [_row(
        apply_stat="1 file changed, +69/−26", apply_badge_stat="+69/−26",
        apply_href="a.html")])

    # apply always precedes evolve in source order, which is what puts it left
    assert both.index('class="apply-link"') < both.index('class="evolve-link"')
    # and the right-pinning class is the one used in the evolve-only card too —
    # the case the old CSS got wrong, where the pill sat left with nothing beside it
    assert 'class="evolve-link"' in evolve_only
    assert 'class="apply-link"' in apply_only


def test_render_index_apply_only_row_has_no_evolve_markup():
    header = {"title": "T", "objective": "O"}
    rows = [
        {"rank": "1", "cand_id": "cand-a-0001", "module": "m", "symbol": "foo",
         "impact": "high", "score": "96", "rationale": "R",
         "html_href": "candidates/modules/m/foo__cand-a-0001.html",
         "apply_stat": "1 file changed, +69/−26",
         "apply_badge_stat": "+69/−26",
         "apply_href": "candidates/modules/m/foo__cand-a-0001__apply.html"},
    ]
    idx = bb.render_index(header, rows)
    # NB: the embedded _CSS mentions `.badge.evolve` and `.evolve-link`, so
    # assert on the *markup* the loop would emit, not on the bare word.
    assert 'class="evolve-link"' not in idx
    assert '<span class="badge evolve">' not in idx


if __name__ == "__main__":
    _run_all()
