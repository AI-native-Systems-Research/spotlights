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


# --- one-shot fix ----------------------------------------------------------


def test_cand_module_slug_strips_prefix_and_sequence():
    assert bb._cand_module_slug("cand-vllm_v1_worker-0001") == "vllm_v1_worker"
    assert bb._cand_module_slug("cand-qiskit_compiler-0042") == "qiskit_compiler"
    # a slug containing digits and underscores must survive intact
    assert bb._cand_module_slug("cand-vllm_v1_kv_offload-0002") == "vllm_v1_kv_offload"


SAMPLE_PATCH = """\
# spotlights one-shot fix
# candidate: cand-vllm_v1_worker-0001
# module:    vllm/v1/worker
# repo:      /Users/someone/checkouts/vllm
# base:      83ad767eed3be3ee7f2df63be693bfaca5c7c922
# apply with (from the directory containing this patch):
#   git -C /Users/someone/checkouts/vllm checkout 83ad767e
#   git -C /Users/someone/checkouts/vllm apply "$PWD/fix.patch"
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
    # The `#` header block `spotlights-engine fix` writes above the diff can
    # contain a line starting with +/-; nothing before the first `diff --git`
    # belongs to any file. Without the `cur is None` guard this raises
    # TypeError on a None subscript.
    st = bb.diffstat("# spotlights one-shot fix\n-stray\n+stray\n")
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
    # `core.quotePath` defaults to true and `fix.patch` is a plain `git diff`,
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
    assert bb._diff_git_path("# spotlights one-shot fix") is None


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
    st = bb.diffstat("# spotlights one-shot fix\n# base:  abc\n")
    assert st == {"files": [], "added": 0, "removed": 0}


def test_format_diffstat_singular_plural_and_minus_sign():
    assert bb.format_diffstat({"files": [{}], "added": 69, "removed": 26}) == \
        "1 file changed, +69/−26"
    assert bb.format_diffstat({"files": [{}, {}], "added": 70, "removed": 26}) == \
        "2 files changed, +70/−26"
    assert bb.format_diffstat({"files": [], "added": 0, "removed": 0}) == \
        "0 files changed, +0/−0"


def _make_fix_tree(root, cand_id="cand-qiskit_compiler-0001", slug="qiskit_compiler",
                   patch=True, notes=True):
    """Create <root>/fix/<slug>/<cand_id>/ with the requested artifacts."""
    d = root / "fix" / slug / cand_id
    d.mkdir(parents=True, exist_ok=True)
    if patch:
        (d / "fix.patch").write_text(SAMPLE_PATCH, encoding="utf-8")
    if notes:
        (d / "FIX-NOTES.md").write_text(
            "# Fix notes\n\n- **Candidate:** `%s`\n" % cand_id, encoding="utf-8")
    return d


def test_find_fix_joins_slug_path_and_returns_artifacts():
    root = Path(tempfile.mkdtemp())
    d = _make_fix_tree(root)
    fx = bb.find_fix("cand-qiskit_compiler-0001", root / "fix")
    assert fx is not None
    assert fx["patch"] == d / "fix.patch"
    assert fx["notes"] == d / "FIX-NOTES.md"
    assert fx["header"]["base"] == "83ad767eed3be3ee7f2df63be693bfaca5c7c922"
    assert fx["stat"]["added"] == 4


def test_find_fix_absent_candidate_is_none():
    root = Path(tempfile.mkdtemp())
    _make_fix_tree(root)
    assert bb.find_fix("cand-qiskit_compiler-9999", root / "fix") is None
    # an absent fix/ tree entirely
    assert bb.find_fix("cand-qiskit_compiler-0001", root / "nope") is None


def test_find_fix_notes_only_directory_is_skipped():
    # `fix` writes FIX-NOTES.md and no patch when the change was not made.
    # A share bundle skips those: no page, no badge, no copies.
    root = Path(tempfile.mkdtemp())
    _make_fix_tree(root, patch=False, notes=True)
    assert bb.find_fix("cand-qiskit_compiler-0001", root / "fix") is None


def test_find_fix_patch_without_notes_still_found():
    root = Path(tempfile.mkdtemp())
    _make_fix_tree(root, patch=True, notes=False)
    fx = bb.find_fix("cand-qiskit_compiler-0001", root / "fix")
    assert fx is not None and fx["notes"] is None


def test_find_fix_derives_the_slug_directory_from_the_candidate_id():
    # Two candidates under DIFFERENT slugs, each found only at its own path.
    # Every other test here uses one slug, which an implementation that ignored
    # `_cand_module_slug` and globbed `fix/*/<cand_id>` would also satisfy.
    root = Path(tempfile.mkdtemp())
    _make_fix_tree(root, cand_id="cand-qiskit_compiler-0001", slug="qiskit_compiler")
    _make_fix_tree(root, cand_id="cand-vllm_v1_kv_offload-0002", slug="vllm_v1_kv_offload")
    for cand_id, slug in (("cand-qiskit_compiler-0001", "qiskit_compiler"),
                          ("cand-vllm_v1_kv_offload-0002", "vllm_v1_kv_offload")):
        fx = bb.find_fix(cand_id, root / "fix")
        assert fx is not None, cand_id
        assert fx["patch"] == root / "fix" / slug / cand_id / "fix.patch"


def test_find_fix_reads_a_patch_that_is_not_valid_utf8():
    # The engine collects the diff as raw bytes and writes it with `write_bytes`,
    # so a non-UTF-8 context byte from the target repo reaches this file intact.
    # Strict decoding would raise UnicodeDecodeError, and `build()` gets here only
    # after removing the previous share-bundle/ — so one such candidate would
    # abort the whole run. It must degrade to a replacement character instead.
    root = Path(tempfile.mkdtemp())
    d = root / "fix" / "qiskit_compiler" / "cand-qiskit_compiler-0001"
    d.mkdir(parents=True)
    (d / "fix.patch").write_bytes(
        b"# spotlights one-shot fix\n"
        b"# base:      83ad767eed3be3ee7f2df63be693bfaca5c7c922\n"
        b"diff --git a/x.py b/x.py\n"
        b"--- a/x.py\n"
        b"+++ b/x.py\n"
        b"@@ -1,2 +1,2 @@\n"
        b"-caf\xe9\n"                      # latin-1 e-acute: invalid UTF-8
        b"+cafe\n")
    fx = bb.find_fix("cand-qiskit_compiler-0001", root / "fix")
    assert fx is not None
    assert fx["header"]["base"] == "83ad767eed3be3ee7f2df63be693bfaca5c7c922"
    assert fx["stat"]["added"] == 1 and fx["stat"]["removed"] == 1


def test_repo_placeholder_from_basename_and_fallback():
    assert bb._repo_placeholder("/Users/someone/checkouts/vllm") == "<YOUR_VLLM_CHECKOUT>"
    assert bb._repo_placeholder("/srv/qiskit-terra/") == "<YOUR_QISKIT_TERRA_CHECKOUT>"
    assert bb._repo_placeholder(None) == "<YOUR_REPO_CHECKOUT>"
    assert bb._repo_placeholder("") == "<YOUR_REPO_CHECKOUT>"


def test_render_patch_colorizes_and_escapes():
    out = bb.render_patch(SAMPLE_PATCH, "foo__fix/fix.patch", 7.1)
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
    assert 'href="foo__fix/fix.patch"' in out
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


def test_render_fix_page_has_warning_apply_strip_and_notes():
    fx = {"patch": None, "notes": None,
          "header": {"candidate": "cand-a-0001", "module": "m/n",
                     "repo": "/Users/someone/checkouts/vllm",
                     "base": "83ad767eed3be3ee7f2df63be693bfaca5c7c922"},
          "stat": bb.diffstat(SAMPLE_PATCH),
          "patch_text": SAMPLE_PATCH, "notes_text": "# Fix notes\n\n- **Objective:** speed\n",
          "patch_kb": 7.1}
    out = bb.render_fix_page("cand-a-0001", "foo", fx, "foo__fix", "foo.html")
    assert "<!DOCTYPE html>" in out
    assert 'href="foo.html"' in out                      # back link
    assert "One-shot fix" in out and "foo" in out
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
    assert 'href="foo__fix/fix.zip"' in out
    assert "Objective" in out


def test_render_fix_page_without_notes_omits_that_section():
    fx = {"header": {"base": "abc123"}, "stat": bb.diffstat(SAMPLE_PATCH),
          "patch_text": SAMPLE_PATCH, "notes_text": None, "patch_kb": 7.1}
    out = bb.render_fix_page("cand-a-0001", "foo", fx, "foo__fix", "foo.html")
    assert "<!DOCTYPE html>" in out          # still a valid page
    assert "The patch" in out                # the diff is still there
    assert "Objective" not in out            # no notes content leaked in
    # a missing base commit degrades to the placeholder, not a crash
    out2 = bb.render_fix_page("cand-a-0001", "foo",
                              {**fx, "header": {}}, "foo__fix", "foo.html")
    assert "&lt;BASE_COMMIT&gt;" in out2
    assert "&lt;YOUR_REPO_CHECKOUT&gt;" in out2


def test_render_fix_page_says_which_directory_the_apply_commands_assume():
    # The apply commands use "$PWD/{raw_reldir}/fix.patch", so they are
    # silently cwd-dependent. The page must state which directory to cd into.
    # Without this note, running the commands from a different directory
    # (e.g., share-bundle/ or from the unpacked .zip) fails with "can't open
    # patch: No such file or directory" and no explanation of why.
    fx = {"header": {"base": "abc123"}, "stat": bb.diffstat(SAMPLE_PATCH),
          "patch_text": SAMPLE_PATCH, "notes_text": None, "patch_kb": 7.1}
    out = bb.render_fix_page("cand-a-0001", "foo", fx, "artifact_dir__fix", "foo.html")
    # The cwd note must mention the artifact directory name
    assert "artifact_dir__fix" in out
    # The note must explain the cwd requirement
    assert "$PWD" in out and "directory" in out.lower()


def test_render_fix_section_links_to_the_page():
    fx = {"stat": bb.diffstat(SAMPLE_PATCH)}
    out = bb.render_fix_section(fx, "foo__cand-a-0001__fix.html")
    assert "One-shot fix" in out
    assert 'href="foo__cand-a-0001__fix.html"' in out
    assert "2 files changed, +4/−2" in out
    assert "verified" in out.lower()


def test_render_candidate_page_emits_fix_before_evolve():
    out = bb.render_candidate_page("# t\n", "t", "../index.html",
                                   "<div>EVOLVEBLOCK</div>", "<div>FIXBLOCK</div>")
    assert out.index("FIXBLOCK") < out.index("EVOLVEBLOCK")


def test_render_candidate_page_still_works_with_evolve_only():
    # Regression: the pre-existing 4-positional-arg call must keep working.
    out = bb.render_candidate_page("# t\n", "t", "../index.html", "<div>E</div>")
    assert "<div>E</div>" in out


if __name__ == "__main__":
    _run_all()
