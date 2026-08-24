#!/usr/bin/env python3
"""Build a browsable, offline ZIP of the top-N Spotlights candidates.

Stdlib only. See .claude/commands/spotlights-share-candidates/SKILL.md.
"""
import argparse
import html
import re
import shutil
import zipfile
from pathlib import Path

_CODE = re.compile(r"`([^`]+)`")
_BOLD = re.compile(r"\*\*([^*]+)\*\*")
_AUTOLINK = re.compile(r"<(https?://[^>\s]+)>")
_MDLINK = re.compile(r"\[([^\]]+)\]\(([^)]+)\)")
_TOKEN = re.compile(r"\x00(\d+)\x00")
_HEADING = re.compile(r"^(#{1,6})\s+(.*)$")
_OBJECTIVE = re.compile(r"\*\*Objective:\*\*\s*(.+?)\s*$", re.MULTILINE)
_TABLE_ROW = re.compile(r"^\|(.+)\|\s*$")
_CAND_CELL = re.compile(r"\[`?([^`\]]+)`?\]\(([^)]+)\)")
_PATCH_FIELD = re.compile(r"^#\s(candidate|module|repo|base):\s*(.+?)\s*$")
_DIFF_GIT = re.compile(r"^diff --git a/(?:.+?) b/(.+)$")


def _is_external(href: str) -> bool:
    return href.startswith("http://") or href.startswith("https://")


def render_inline(text: str) -> str:
    """Render inline markdown to HTML with the project's link rules.

    External URLs (autolinks and http(s) markdown links) become clickable
    anchors that open in a new tab. Every other link is stripped and its
    visible text is rendered as inline <code>. Bold and inline code are
    supported. All literal text is HTML-escaped.
    """
    tokens: list[str] = []

    def stash(html_fragment: str) -> str:
        tokens.append(html_fragment)
        return f"\x00{len(tokens) - 1}\x00"

    # 1. Autolinks: <https://...>
    def _auto(m: re.Match) -> str:
        url = m.group(1)
        esc = html.escape(url)
        return stash(f'<a href="{esc}" target="_blank" rel="noopener">{esc}</a>')

    text = _AUTOLINK.sub(_auto, text)

    # 2. Markdown links: [text](href)
    def _link(m: re.Match) -> str:
        label, href = m.group(1), m.group(2)
        label_html = _inline_no_links(label)
        if _is_external(href):
            esc = html.escape(href)
            return stash(f'<a href="{esc}" target="_blank" rel="noopener">{label_html}</a>')
        # internal target not in the bundle -> plain code, not clickable
        return stash(f"<code>{_strip_code_ticks(label_html)}</code>")

    text = _MDLINK.sub(_link, text)

    # 3. Remaining inline (bold, code) on the non-token text, then restore tokens.
    text = _inline_no_links(text)

    def _restore(m: re.Match) -> str:
        return tokens[int(m.group(1))]

    return _TOKEN.sub(_restore, text)


def _strip_code_ticks(s: str) -> str:
    """A label like `<code>x</code>` collapses to x so we don't double-wrap."""
    return s.replace("<code>", "").replace("</code>", "")


def _inline_no_links(text: str) -> str:
    """Escape HTML, then apply bold + inline code (no link processing)."""
    def _code(m: re.Match) -> str:
        return "\x01" + html.escape(m.group(1)) + "\x02"

    # Protect code spans from escaping their own content twice.
    text = _CODE.sub(_code, text)
    text = html.escape(text)
    text = text.replace("\x01", "<code>").replace("\x02", "</code>")
    text = _BOLD.sub(r"<strong>\1</strong>", text)
    return text


def md_to_html_body(md_text: str) -> str:
    """Render a block-level markdown subset to an HTML fragment.

    Supported blocks: ATX headings (# .. ######), bullet lists (- ...),
    horizontal rules (---), and paragraphs. Inline formatting within each
    block is delegated to render_inline.
    """
    lines = md_text.replace("\r\n", "\n").split("\n")
    out: list[str] = []
    i = 0
    n = len(lines)
    while i < n:
        line = lines[i]
        stripped = line.strip()

        if not stripped:
            i += 1
            continue

        if stripped == "---":
            out.append("<hr>")
            i += 1
            continue

        if stripped.startswith("```"):
            # Fenced code block: collect lines verbatim until the closing
            # fence (or EOF). The opening fence's info string is ignored.
            i += 1
            code_lines: list[str] = []
            while i < n and not lines[i].strip().startswith("```"):
                code_lines.append(lines[i])
                i += 1
            if i < n:  # skip the closing fence line (EOF also closes)
                i += 1
            code = html.escape("\n".join(code_lines))
            out.append(f"<pre><code>{code}</code></pre>")
            continue

        m = _HEADING.match(stripped)
        if m:
            level = len(m.group(1))
            out.append(f"<h{level}>{render_inline(m.group(2).strip())}</h{level}>")
            i += 1
            continue

        if stripped.startswith("- "):
            out.append("<ul>")
            while i < n and lines[i].strip().startswith("- "):
                item = lines[i].strip()[2:]
                out.append(f"<li>{render_inline(item)}</li>")
                i += 1
            out.append("</ul>")
            continue

        # paragraph: gather consecutive non-blank, non-special lines
        para: list[str] = []
        while i < n:
            s = lines[i].strip()
            if not s or s == "---" or _HEADING.match(s) or s.startswith("- "):
                break
            para.append(s)
            i += 1
        out.append(f"<p>{render_inline(' '.join(para))}</p>")

    return "\n".join(out)


def parse_header(md_text: str) -> dict:
    """Extract the bundle title (first H1) and objective line."""
    title = ""
    for line in md_text.splitlines():
        s = line.strip()
        if s.startswith("# "):
            title = s[2:].strip()
            break
    m = _OBJECTIVE.search(md_text)
    objective = m.group(1).strip() if m else ""
    return {"title": title, "objective": objective}


def _split_row(line: str) -> list[str]:
    inner = _TABLE_ROW.match(line).group(1)
    return [c.strip() for c in inner.split("|")]


def parse_ranking_table(md_text: str, top_n: int) -> list[dict]:
    """Parse the '## Ranking summary' table; return up to top_n data rows.

    Skips the header row and the |---| separator. Each returned dict has:
    rank, cand_id, rel_link, module, symbol, impact, score, rationale.
    """
    rows: list[dict] = []
    seen_header = False
    for line in md_text.splitlines():
        if not _TABLE_ROW.match(line):
            continue
        cells = _split_row(line)
        # separator row like |---|---|
        if all(set(c) <= set("-: ") for c in cells):
            continue
        if not seen_header:
            seen_header = True  # first table row is the column header
            continue
        if len(cells) < 7:
            continue
        cm = _CAND_CELL.search(cells[1])
        if not cm:
            continue
        rows.append(
            {
                "rank": cells[0],
                "cand_id": cm.group(1),
                "rel_link": cm.group(2),
                "module": cells[2],
                "symbol": cells[3].strip("`"),
                "impact": cells[4],
                "score": cells[5],
                "rationale": "|".join(cells[6:]),
            }
        )
        if len(rows) >= top_n:
            break
    return rows


_CSS = """
:root { color-scheme: light; }
* { box-sizing: border-box; }
body { margin: 0; background: #f6f7f9; color: #1a1d21;
  font: 16px/1.6 -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif; }
.wrap { max-width: 860px; margin: 0 auto; padding: 32px 24px 80px; }
a { color: #2b6cb0; }
h1 { font-size: 1.7rem; line-height: 1.25; margin: 0 0 .5rem; }
h2 { font-size: 1.25rem; margin: 1.8rem 0 .6rem; border-bottom: 1px solid #e2e5e9; padding-bottom: .25rem; }
h3 { font-size: 1.05rem; margin: 1.3rem 0 .4rem; }
p { margin: .6rem 0; }
ul { margin: .5rem 0 .8rem; padding-left: 1.4rem; }
li { margin: .2rem 0; }
code { background: #eef0f3; border-radius: 4px; padding: .1em .35em;
  font: .9em/1.4 ui-monospace, SFMono-Regular, Menlo, Consolas, monospace; }
hr { border: 0; border-top: 1px solid #e2e5e9; margin: 1.6rem 0; }
.back { display: inline-block; margin-bottom: 1.2rem; font-size: .95rem; }
.subtitle { color: #5a6169; margin: 0 0 1.6rem; }
.card { display: block; background: #fff; border: 1px solid #e2e5e9; border-radius: 10px;
  padding: 16px 18px; margin: 12px 0; text-decoration: none; color: inherit;
  box-shadow: 0 1px 2px rgba(0,0,0,.04); transition: box-shadow .12s, border-color .12s; }
.card:hover { box-shadow: 0 3px 10px rgba(0,0,0,.08); border-color: #c7cdd4; }
.card .row1 { display: flex; align-items: baseline; gap: 10px; flex-wrap: wrap; }
.rank { font-weight: 700; color: #2b6cb0; }
.sym { font-weight: 600; font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace; }
.mod { color: #5a6169; font-size: .9rem; }
.badges { margin-left: auto; display: flex; gap: 8px; }
.badge { font-size: .78rem; padding: .12em .6em; border-radius: 999px; background: #eef0f3; color: #3a4149; }
.badge.impact-high { background: #fde8e8; color: #9b1c1c; }
.badge.score { background: #e6f4ea; color: #1e6b33; }
.rationale { margin: .5rem 0 0; color: #3a4149; font-size: .95rem; }

/* --- evolve bundles --- */
.badge.evolve { background: #e8e3fd; color: #4c2a9b; }
.card-main { display:block; text-decoration:none; color:inherit; }
.card-foot { margin-top:.7rem; padding-top:.6rem; border-top:1px dashed #e2e5e9; }
.evolve-link { display:inline-flex; align-items:center; gap:.45em;
  font-size:.85rem; font-weight:600; color:#4c2a9b; background:#f1edfd;
  border:1px solid #d9cffb; border-radius:999px; padding:.32em .85em;
  text-decoration:none; }
.evolve-link:hover { background:#e6ddfc; border-color:#c3b1f7; }
.evolve-section { margin-top: 1.8rem; }
.engine { background:#fff; border:1px solid #e2e5e9; border-radius:10px;
  padding:14px 16px; margin:12px 0; }
.engine h3 { margin:.1rem 0 .5rem; }
.engine .cmd { background:#1a1d21; color:#e6e6e6; border-radius:6px;
  padding:.5em .7em; font:.85em/1.4 ui-monospace,Menlo,Consolas,monospace;
  overflow-x:auto; }
.engine pre { background:#f6f7f9; border:1px solid #e2e5e9; border-radius:6px;
  max-height:360px; overflow:auto; padding:.6em .8em; }
.engine details { margin:.5rem 0; }
.engine summary { cursor:pointer; font:.9rem/1.4 ui-monospace,Menlo,Consolas,monospace;
  color:#3a4149; padding:.25em 0; }
.engine summary .sz { color:#8a9099; }
.dl { display:inline-flex; align-items:center; gap:.5em; margin:.2rem 0 1.4rem;
  font-size:.95rem; font-weight:600; color:#fff; background:#4c2a9b;
  border-radius:8px; padding:.5em 1em; text-decoration:none; }
.dl:hover { background:#3d2080; }
.engine .dl { margin:1rem 0 .1rem; font-size:.85rem; font-weight:600;
  color:#4c2a9b; background:#f1edfd; border:1px solid #d9cffb; padding:.4em .85em; }
.engine .dl:hover { background:#e6ddfc; border-color:#c3b1f7; }
.orient { background:#f1edfd; border:1px solid #d9cffb; border-radius:10px;
  padding:14px 16px; margin:.2rem 0 1.2rem; }
.orient p { margin:.4rem 0; }
.orient .warn { color:#8a5a00; }
.about { background:#fbfbfc; border:1px solid #eef0f3; border-radius:8px;
  padding:10px 14px; margin:.2rem 0 .8rem; }
.about .tagline { margin:.1rem 0 .6rem; font-weight:600; }
.about dl { display:grid; grid-template-columns:max-content minmax(0,1fr);
  gap:.3em 1em; margin:0; }
.about dt { color:#5a6169; font-size:.82rem; padding-top:.05em; }
.about dd { margin:0; font-size:.9rem; min-width:0; }
.about dd .warn { color:#9b1c1c; display:block; margin-top:.3rem; }
.inst-label { font-size:.78rem; color:#5a6169; margin:.5rem 0 .15rem; }
.inst-label:first-child { margin-top:.1rem; }
.engine pre.inst { background:#1a1d21; color:#e6e6e6; border:0; border-radius:6px;
  padding:.5em .7em; margin:.1rem 0; max-height:none; overflow-x:auto;
  font:.85em/1.5 ui-monospace,Menlo,Consolas,monospace; }
.about dd p { margin:.1rem 0 .3rem; }
.about dd .ask { margin:.3rem 0 0; padding:.6em .8em; background:#eef6ff;
  border-left:3px solid #2b6cb0; border-radius:0 6px 6px 0; color:#1a1d21;
  font-style:italic; font-size:.88rem; }
/* --- one-shot fix --- */
.badge.fix { background:#fdf0d5; color:#8a5a00; }
.fix-link { display:inline-flex; align-items:center; gap:.45em;
  font-size:.85rem; font-weight:600; color:#8a5a00; background:#fdf6e7;
  border:1px solid #f0dfb8; border-radius:999px; padding:.32em .85em;
  text-decoration:none; }
.fix-link:hover { background:#fbeed2; border-color:#e6cf9c; }
.fix-section { margin-top: 1.8rem; }
.apply { background:#fbfbfc; border:1px solid #eef0f3; border-radius:8px;
  padding:10px 14px; margin:.2rem 0 1.2rem; }
.apply h2 { margin:.1rem 0 .5rem; border:0; padding:0; font-size:1.05rem; }
.apply pre { background:#1a1d21; color:#e6e6e6; border:0; border-radius:6px;
  padding:.5em .7em; margin:.3rem 0; overflow-x:auto;
  font:.85em/1.5 ui-monospace,Menlo,Consolas,monospace; }
.apply .note { font-size:.85rem; color:#5a6169; margin:.5rem 0 .1rem; }
.apply .sha { font:.85em/1.4 ui-monospace,Menlo,Consolas,monospace;
  background:#eef0f3; border-radius:4px; padding:.1em .35em; }
.apply .dl { margin:.7rem 0 .2rem; font-size:.85rem; color:#8a5a00;
  background:#fdf6e7; border:1px solid #f0dfb8; }
.apply .dl:hover { background:#fbeed2; border-color:#e6cf9c; }
.diff { border:1px solid #e2e5e9; border-radius:8px; overflow:hidden;
  margin:.6rem 0 1.2rem; }
.diff .file { background:#f6f7f9; border-bottom:1px solid #e2e5e9;
  padding:.4em .7em; display:flex; gap:.8em; align-items:baseline;
  font:.85rem/1.4 ui-monospace,Menlo,Consolas,monospace; }
.diff .file .p { font-weight:600; word-break:break-all; }
.diff .file .st { margin-left:auto; white-space:nowrap; }
.diff .file .st .a { color:#1e6b33; }
.diff .file .st .r { color:#9b1c1c; }
.diff pre { margin:0; padding:0; background:#fff; overflow-x:auto;
  font:.82em/1.5 ui-monospace,Menlo,Consolas,monospace; }
.diff .l { display:block; padding:0 .7em; white-space:pre; }
.diff .d-add { background:#e6f4ea; color:#1e6b33; }
.diff .d-del { background:#fde8e8; color:#9b1c1c; }
.diff .d-hunk { background:#eef0f3; color:#5a6169; }
/* the patch <details> is not inside .engine, so it needs its own summary rules
   — these mirror `.engine summary` / `.engine summary .sz` above */
.patch { margin:.5rem 0 1.2rem; }
.patch summary { cursor:pointer; padding:.25em 0; color:#3a4149;
  font:.9rem/1.4 ui-monospace,Menlo,Consolas,monospace; }
.patch summary .sz { color:#8a9099; }
"""


def _doc(title: str, body: str) -> str:
    return (
        "<!DOCTYPE html>\n<html lang=\"en\">\n<head>\n"
        "<meta charset=\"utf-8\">\n"
        "<meta name=\"viewport\" content=\"width=device-width, initial-scale=1\">\n"
        f"<title>{html.escape(title)}</title>\n"
        f"<style>{_CSS}</style>\n</head>\n<body>\n<div class=\"wrap\">\n"
        f"{body}\n</div>\n</body>\n</html>\n"
    )


def render_candidate_page(md_text: str, title: str, back_href: str,
                          evolve_section: str = "", fix_section: str = "") -> str:
    """Render one candidate page.

    `fix_section` is last so the pre-existing 4-positional-argument calls keep
    working, but it is *emitted* before `evolve_section`: fix is the cheap arm,
    evolve the expensive one.
    """
    back = f'<a class="back" href="{html.escape(back_href)}">← Back to index</a>'
    body = back + "\n" + md_to_html_body(md_text)
    if fix_section:
        body += "\n" + fix_section
    if evolve_section:
        body += "\n" + evolve_section
    return _doc(title, body)


def render_index(header: dict, rows: list[dict]) -> str:
    parts = [f"<h1>{html.escape(header.get('title', 'Candidates'))}</h1>"]
    if header.get("objective"):
        parts.append(f'<p class="subtitle">{render_inline(header["objective"])}</p>')
    for r in rows:
        impact_cls = "impact-high" if r["impact"].lower() == "high" else ""
        evolve_count = r.get("evolve_count", 0)
        evolve_badge = (f'<span class="badge evolve">evolve · {evolve_count}</span>'
                        if evolve_count else "")
        # The whole card is one anchor to the candidate page; the evolve link
        # lives in a separate footer anchor so we never nest <a> in <a>.
        card_main = (
            f'<a class="card-main" href="{html.escape(r["html_href"])}">'
            f'<div class="row1">'
            f'<span class="rank">#{html.escape(r["rank"])}</span>'
            f'<span class="sym">{html.escape(r["symbol"])}</span>'
            f'<span class="mod">{html.escape(r["module"])}</span>'
            f'<span class="badges">{evolve_badge}'
            f'<span class="badge {impact_cls}">{html.escape(r["impact"])}</span>'
            f'<span class="badge score">score {html.escape(r["score"])}</span>'
            f'</span></div>'
            f'<p class="rationale">{render_inline(r["rationale"])}</p>'
            f'</a>'
        )
        foot = ""
        if evolve_count:
            engines = " · ".join(r.get("evolve_engines", []))
            foot = (f'<div class="card-foot"><a class="evolve-link" '
                    f'href="{html.escape(r["evolve_href"])}">⚙ Evolve bundles: '
                    f'{html.escape(engines)} →</a></div>')
        parts.append(f'<div class="card">{card_main}{foot}</div>')
    return _doc(header.get("title", "Candidates"), "\n".join(parts))


def _module_rel_path(rel_link: str) -> str:
    """Turn a candidate rel link like '../modules/pkg/foo__cand.md' into the
    path under the bundle: 'modules/pkg/foo__cand.md'. If 'modules/' is not in
    the link, fall back to the basename under 'modules/'."""
    parts = Path(rel_link).parts
    if "modules" in parts:
        idx = parts.index("modules")
        return str(Path(*parts[idx:]))
    return str(Path("modules") / Path(rel_link).name)


# --- evolve bundles ---------------------------------------------------------
#
# When `prep-evolve` has run, a sibling `evolve/` tree lives beside `sorted/`:
#   <run>/evolve/<module_slug>/<cand_id>/<engine>/<native files>
# For each exported candidate we fold those files into the bundle, add an
# `…__evolve.html` page describing every engine, and surface a link on the
# candidate page and index card. If no `evolve/` tree exists the build behaves
# exactly as before.
#
# Per-engine metadata is descriptive framing for a reader unfamiliar with the
# frameworks; facts are sourced from docs/prep-evolve.md and each project's repo.

ENGINES: dict[str, dict] = {
    "coral": {
        "cmd": "coral start --config task.yaml",
        "repo": "https://github.com/Human-Agent-Society/CORAL",
        "tagline": "Agentic multi-file evolver — an LLM agent edits a git worktree of the repo.",
        "scope": "Multi-file: the agent may edit any in-scope file in a git worktree.",
        "config": "task.yaml",
        "writes": "the grader + seed directories",
        "install": [
            {"label": "Shell installer",
             "cmd": "curl -fsSL https://raw.githubusercontent.com/Human-Agent-Society/CORAL/main/install.sh | sh"},
            {"label": "Claude Code plugin",
             "cmd": "/plugin marketplace add Human-Agent-Society/CORAL\n/plugin install coral@coral-marketplace"},
        ],
        "pip_warn": None,
        "quickstart": (
            "Open the target repo (make sure task.yaml's repo_path points to the same "
            "location — update it if needed), then ask Claude Code — replacing "
            "<CORAL_BUNDLE_PATH> with the path to wherever you unpacked this bundle:",
            "use coral to optimize this — start from the bundle at {path}. Don't change "
            "what task.yaml defines — the goal, in-scope file, oracle, metric, and "
            "direction are fixed. Fill only the gaps: write the grader, set up the seed, "
            "and add whatever's needed so the grader cleanly scores the seed. The seed "
            "should be the full repo so the agent can read everything, but the grader "
            "must reject any attempt that modifies or adds a file outside the allowlist "
            "— configured as target_files (or allowed_paths) under grader.args in "
            "task.yaml."),
    },
    "skydiscover": {
        "cmd": "skydiscover-run seed.py evaluator.py -c config.yaml",
        "repo": "https://github.com/skydiscover-ai/skydiscover",
        "tagline": "LLM-driven evolutionary search over a single marked code region.",
        "scope": "Single file: only the region between # EVOLVE-BLOCK-START and # EVOLVE-BLOCK-END changes.",
        "config": "config.yaml + seed.py",
        "writes": "evaluator.py",
        "writes_ref": "https://github.com/skydiscover-ai/skydiscover/blob/main/README.md#%EF%B8%8F-what-you-write",
        "install": [{"label": None, "cmd": "pip install skydiscover"}],
        "pip_warn": None,
    },
    "nous": {
        "cmd": "NOUS_CAMPAIGN_PARENT=$PWD/nous_runs nous run campaign.yaml",
        "repo": "https://github.com/AI-native-Systems-Research/agentic-strategy-evolution",
        "tagline": "Runs experiment 'arms' that apply code_changes[] across the target "
                   "(alias: agentic-strategy-evolution).",
        "scope": "Multi-file: experiment arms with code_changes[].",
        "config": "campaign.yaml",
        # No evaluator code — the agents discover metrics and evaluate on their own.
        # The one gap is the rule deciding whether a measured number is a win.
        "writes_html": ("<code>ground_truth.pass_condition</code> in "
                        "<code>campaign.yaml</code> — a concrete pass/fail rule."),
        "install": [{"label": None,
                     "cmd": 'pip install "git+https://github.com/AI-native-Systems-Research/'
                            'agentic-strategy-evolution.git@reflective"'}],
        "pip_warn": None,
    },
}

# Prerequisites are the same for every engine; stated once.
_EVOLVE_PREREQ = (
    "Python 3, an LLM API key for the model named in the config "
    "(e.g. <code>ANTHROPIC_API_KEY</code>), the target repo checked out at the "
    "run's commit, and a server with all the hardware the benchmark needs "
    "(e.g. a GPU) to build and measure the target.")


def _cand_module_slug(cand_id: str) -> str:
    """cand-<module_slug>-NNNN -> <module_slug> (the trailing -NNNN is dropped).

    Shared by both follow-on arms: `evolve/` and `fix/` are both keyed by
    <module_slug>/<cand_id>/ on disk.
    """
    return cand_id.removeprefix("cand-").rsplit("-", 1)[0]


def find_evolve(cand_id: str, evolve_root: Path) -> dict | None:
    """Return {engine: [files]} for a candidate's evolve bundle, or None.

    Looks under <evolve_root>/<module_slug>/<cand_id>/<engine>/ and keeps only
    the engines known to ENGINES that actually have files on disk.
    """
    d = evolve_root / _cand_module_slug(cand_id) / cand_id
    if not d.is_dir():
        return None
    engines: dict[str, list[Path]] = {}
    for eng in ENGINES:
        ed = d / eng
        if ed.is_dir():
            files = sorted(f for f in ed.iterdir() if f.is_file())
            if files:
                engines[eng] = files
    return engines or None


def render_engine(eng: str, files: list[Path], raw_reldir: str) -> str:
    """Render one engine's block: an About strip, the run command, the inlined
    README (if any), collapsible raw views of the other files, and a per-engine
    download link. `raw_reldir` is the evolve folder name relative to the page."""
    m = ENGINES[eng]
    repo_link = (f'<a href="{html.escape(m["repo"])}" target="_blank" '
                 f'rel="noopener">{html.escape(m["repo"])} ↗</a>')
    inst_blocks = m.get("install") or []
    if inst_blocks:
        rendered = []
        for b in inst_blocks:
            if b.get("label"):
                rendered.append(f'<div class="inst-label">{html.escape(b["label"])}</div>')
            rendered.append(f'<pre class="inst">{html.escape(b["cmd"])}</pre>')
        install = "".join(rendered)
    else:
        install = "From source — see the repo. No official installer."
    if m.get("pip_warn"):
        install += f'<span class="warn">⚠️ {html.escape(m["pip_warn"])}.</span>'

    writes_row = ""
    if m.get("writes_html"):
        writes_row = f'<dt>You must write</dt><dd>{m["writes_html"]}</dd>'
    elif m.get("writes"):
        writes = f'<code>{html.escape(m["writes"])}</code>'
        if m.get("writes_ref"):
            writes += (f' — <a href="{html.escape(m["writes_ref"])}" target="_blank" '
                       f'rel="noopener">what to write ↗</a>')
        writes_row = f'<dt>You must write</dt><dd>{writes}</dd>'

    quickstart_row = ""
    if m.get("quickstart"):
        intro, prompt = m["quickstart"]
        prompt = prompt.format(path="<CORAL_BUNDLE_PATH>")
        intro_html = html.escape(intro).replace("repo_path", "<code>repo_path</code>")
        quickstart_row = (
            '<dt>Quickstart</dt><dd>'
            f'<p>{intro_html}</p>'
            f'<blockquote class="ask">{html.escape(prompt)}</blockquote></dd>')

    parts = [f'<div class="engine"><h3>{html.escape(eng)}</h3>']
    parts.append(
        '<div class="about">'
        f'<p class="tagline">{html.escape(m["tagline"])}</p>'
        '<dl>'
        f'<dt>Repository</dt><dd>{repo_link}</dd>'
        f'<dt>Install</dt><dd>{install}</dd>'
        f'<dt>Edit scope</dt><dd>{html.escape(m["scope"])}</dd>'
        f'<dt>Native config</dt><dd><code>{html.escape(m["config"])}</code></dd>'
        f'{writes_row}'
        f'{quickstart_row}'
        f'<dt>Prerequisites</dt><dd>{_EVOLVE_PREREQ}</dd>'
        '</dl></div>')
    parts.append(f'<div class="cmd">$ {html.escape(m["cmd"])}</div>')
    # READMEs render inline — they're the human-facing overview.
    for f in files:
        if f.name.endswith(".md"):
            parts.append(md_to_html_body(f.read_text(encoding="utf-8")))
    # Every other file is a collapsible raw view, also linked to its copied file.
    for f in files:
        if f.name.endswith(".md"):
            continue
        kb = f.stat().st_size / 1024
        href = f"{html.escape(raw_reldir)}/{eng}/{html.escape(f.name)}"
        code = html.escape(f.read_text(encoding="utf-8"))
        parts.append(
            f'<details><summary>{html.escape(f.name)} '
            f'<span class="sz">— {kb:.0f} KB · <a href="{href}">open raw ↗</a></span>'
            f'</summary><pre><code>{code}</code></pre></details>')
    parts.append(
        f'<a class="dl" href="{html.escape(raw_reldir)}/{html.escape(eng)}.zip" download>'
        f'⬇ Download {html.escape(eng)} files (.zip)</a>')
    parts.append("</div>")
    return "\n".join(parts)


def _copy_evolve_files(engines: dict, evolve_dir: Path) -> None:
    """Copy every original evolve file into the bundle and write a per-engine
    <engine>.zip whose entries live under a top-level <engine>/ folder."""
    for eng, files in engines.items():
        for f in files:
            dst = evolve_dir / eng / f.name
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(f, dst)
        with zipfile.ZipFile(evolve_dir / f"{eng}.zip", "w", zipfile.ZIP_DEFLATED) as zf:
            for f in files:
                zf.write(f, f"{eng}/{f.name}")


def render_evolve_page(cand_id: str, symbol: str, engines: dict,
                       raw_reldir: str, back_href: str) -> str:
    """The `…__evolve.html` page: orientation + one section per engine."""
    body = [
        f'<a class="back" href="{html.escape(back_href)}">← Back to candidate</a>',
        f"<h1>Evolve bundles — {html.escape(symbol)}</h1>",
        f'<p class="subtitle">{html.escape(cand_id)} · '
        f'{len(engines)} engine(s): {", ".join(engines)}</p>',
        '<div class="orient">'
        '<p><strong>What is this?</strong> Each section below is a ready-to-run '
        'bundle for a different <em>evolver</em> — an evolutionary code-search '
        'engine. Spotlights already chose <em>what</em> to optimize (the target '
        'symbol and objective); an evolver repeatedly mutates the in-scope code to '
        'improve it. <strong>You only need to run one.</strong></p>'
        '<p class="warn">⚠️ Every bundle is launchable, but produces no meaningful '
        'score until you complete its evaluator/grader — the performance measurement '
        'ships as a <code>TODO</code>. See each engine\'s README (or config) below for '
        'the exact oracle Spotlights inferred.</p></div>',
    ]
    for eng, files in engines.items():
        body.append(render_engine(eng, files, raw_reldir))
    return _doc(f"Evolve — {symbol}", "\n".join(body))


def render_evolve_section(engines: dict, evolve_page_name: str) -> str:
    """The "Evolve bundles" block appended to a candidate page."""
    return (
        '<div class="evolve-section"><h2>Evolve bundles</h2>'
        f'<p>{len(engines)} ready-to-launch evolve bundle(s) for this '
        f'candidate: <strong>{", ".join(engines)}</strong>.</p>'
        f'<p><a href="{html.escape(evolve_page_name)}">View evolve bundles →</a></p></div>')


# --- one-shot fix -----------------------------------------------------------
#
# When `spotlights-engine fix` has run, a sibling `fix/` tree lives beside
# `sorted/`:
#   <run>/fix/<module_slug>/<cand_id>/{fix.patch,FIX-NOTES.md}
# For each exported candidate with a patch we copy those files into the bundle,
# add a `…__fix.html` page, and surface a link on the candidate page and index
# card. If no `fix/` tree exists the build behaves exactly as before.


def _diff_git_path(line: str) -> str | None:
    """The post-image path of a `diff --git` header line, or None if not one.

    The single source of truth for recognising a file boundary — `diffstat` and
    `render_patch` both start a new file by it, and both reset `in_hunk` by it.

    It never returns None for a line starting with `diff --git`, and that is the
    point: recognising the boundary is what resets the caller's `in_hunk` state.
    Coupling the reset to a successful *path* capture is a live bug — git quotes
    any path holding a non-ASCII byte, a quote, a backslash, or a control
    character (`core.quotePath` defaults to true, and `fix.patch` comes from a
    plain `git diff`), so a real patch contains both forms:

        diff --git a/vllm/v1/worker/utils.py b/vllm/v1/worker/utils.py
        diff --git "a/caf\\303\\251.py" "b/caf\\303\\251.py"

    Miss the second and `in_hunk` stays True across the boundary, so that file's
    own `--- a/…` and `+++ b/…` markers are counted as a removal and an addition.
    Git's escapes are left in the returned path rather than decoded: an unusual
    filename displayed verbatim is better than a file missing from the table.
    """
    if not line.startswith("diff --git "):
        return None
    m = _DIFF_GIT.match(line)
    if m:
        return m.group(1)
    rest = line[len("diff --git "):]
    # Git quotes each side independently, so a rename can quote only its source:
    # `diff --git "a/caf\303\251.py" b/ascii.py`. Try the quoted post-image first,
    # then the bare one, so the label is the destination path and not the whole
    # header remainder (which would carry the pre-image side along with it).
    _, sep, post = rest.rpartition(' "b/')
    if sep and post.endswith('"'):
        return post[:-1]
    _, sep, post = rest.rpartition(' b/')
    return post if sep else rest


def parse_patch_header(patch_text: str) -> dict:
    """Read the `# candidate/module/repo/base` block above the first diff.

    Both `spotlights-engine fix` and the /spotlights-fix-candidate skill write
    this header field-for-field, which is what makes it parseable. Absent
    fields are absent keys — callers use .get(). Scanning stops at the first
    `diff --git` so a `#` line inside a diff body can never be read as a field.
    """
    out: dict = {}
    for line in patch_text.splitlines():
        if line.startswith("diff --git"):
            break
        m = _PATCH_FIELD.match(line)
        if m:
            out[m.group(1)] = m.group(2)
    return out


def _diff_line_kind(line: str, in_hunk: bool = False) -> str:
    """Classify one patch line: marker | hunk | add | del | context.

    The single source of truth for this rule — `diffstat` counts by it and
    `render_patch` colours by it, so the two can never disagree.

    The `+++ b/…` and `--- a/…` file markers are tested *first*: they start
    with `+`/`-` but are not changed lines, and checking them second inflates
    every count by one. A bare `+` or `-` is a real added/removed blank line.

    `in_hunk` is what keeps that first test from swallowing real content. File
    markers only ever appear in a file's header block, before its first `@@`.
    Inside a hunk, `---`/`+++` is a changed line whose *content* begins with
    `--`/`++` — deleting a markdown `---` rule emits `----`, and deleting a
    `-- flag` doc line emits `--- flag`. Treating those as markers drops them
    from the count, under-reporting removals on the one page that exists to
    inform an apply/don't-apply decision. Callers iterate in order, so they
    set `in_hunk=True` on a `@@` line and back to False on `diff --git`.
    """
    if not in_hunk and (line.startswith("+++") or line.startswith("---")):
        return "marker"
    if line.startswith("@@"):
        return "hunk"
    if line.startswith("+"):
        return "add"
    if line.startswith("-"):
        return "del"
    return "context"


def diffstat(patch_text: str) -> dict:
    """Per-file and total +added/-removed, computed from the patch itself.

    Deliberately not read from FIX-NOTES.md's "Files changed" table: those
    notes say the table is derived from the patch, and the patch is what
    ships. One source of truth, and it is the one the recipient applies.
    """
    files: list[dict] = []
    cur: dict | None = None
    in_hunk = False
    for line in patch_text.splitlines():
        path = _diff_git_path(line)           # never None for a `diff --git` line,
        if path is not None:                  # so the reset cannot be skipped
            cur = {"path": path, "added": 0, "removed": 0}
            files.append(cur)
            in_hunk = False                   # back in a file header block
            continue
        if cur is None:                       # still in the `#` header block
            continue
        kind = _diff_line_kind(line, in_hunk)
        if kind == "hunk":
            in_hunk = True
        elif kind == "add":
            cur["added"] += 1
        elif kind == "del":
            cur["removed"] += 1
    return {
        "files": files,
        "added": sum(f["added"] for f in files),
        "removed": sum(f["removed"] for f in files),
    }


def format_diffstat(stat: dict) -> str:
    """'1 file changed, +69/−26' — U+2212 MINUS SIGN, not a hyphen."""
    n = len(stat["files"])
    noun = "file" if n == 1 else "files"
    return f"{n} {noun} changed, +{stat['added']}/−{stat['removed']}"


def find_fix(cand_id: str, fix_root: Path) -> dict | None:
    """Return a candidate's fix artifacts, or None.

    Looks under <fix_root>/<module_slug>/<cand_id>/. `fix.patch` is required:
    a directory holding only FIX-NOTES.md is the legitimate "the change could
    not be made" outcome, and a share bundle skips it entirely — no page, no
    badge, no copies. FIX-NOTES.md itself is optional.
    """
    d = fix_root / _cand_module_slug(cand_id) / cand_id
    patch = d / "fix.patch"
    if not patch.is_file():
        return None
    notes = d / "FIX-NOTES.md"
    # `errors="replace"`, not strict: `fix.patch` is the one file here that is
    # deliberately NOT guaranteed to be UTF-8. The engine collects the diff as
    # raw bytes and writes it with `write_bytes`, because decoding and re-encoding
    # it would corrupt a patch that `git apply` has to accept byte-for-byte — so a
    # single non-UTF-8 context byte from the target repo lands in this file. Strict
    # decoding would raise UnicodeDecodeError here, and `build()` reaches this
    # point only after it has already removed the previous `share-bundle/`, so one
    # such candidate would abort the run and leave no bundle at all.
    #
    # Lossy decoding is safe *because it is only ever used for reading*: the
    # copied artifact is the raw file (`shutil.copy2`), never this text. What a
    # replacement character costs is one unreadable glyph in the rendered diff,
    # against a build that otherwise does not happen.
    patch_text = patch.read_text(encoding="utf-8", errors="replace")
    notes_text = notes.read_text(encoding="utf-8") if notes.is_file() else None
    return {
        "patch": patch,
        "notes": notes if notes.is_file() else None,
        "header": parse_patch_header(patch_text),
        "stat": diffstat(patch_text),
        "patch_text": patch_text,
        "notes_text": notes_text,
        "patch_kb": patch.stat().st_size / 1024,
    }


# The warning is the whole reason this page is careful: a colorized diff in a
# browser is the most authoritative-looking artifact Spotlights emits, and none
# of it was tested, benchmarked, or built.
_DIFF_CLASS = {"add": "d-add", "del": "d-del", "hunk": "d-hunk", "context": ""}

_FIX_UNVERIFIED = (
    "Nothing here was verified. No test was run, no benchmark was measured, no "
    "build was attempted. The patch was produced in a fresh detached worktree "
    "with no virtualenv and no compiled extensions, on a machine that may lack "
    "the hardware the performance oracle needs. Treat it as a proposal "
    "faithfully implemented — not as a measured win.")


def _repo_placeholder(repo: str | None) -> str:
    """'/Users/…/vllm' -> '<YOUR_VLLM_CHECKOUT>'; falsy -> '<YOUR_REPO_CHECKOUT>'."""
    name = Path(repo.rstrip("/")).name if repo else ""
    slug = re.sub(r"[^A-Za-z0-9]+", "_", name).strip("_").upper()
    return f"<YOUR_{slug or 'REPO'}_CHECKOUT>"


def render_patch(patch_text: str, patch_href: str, size_kb: float) -> str:
    """The patch as a collapsed colorized diff, one block per changed file.

    Closed by default: FIX-NOTES.md is the orientation, the diff is the detail —
    the same split that collapses evolve's non-README files.
    """
    stat = diffstat(patch_text)
    per_file = {f["path"]: f for f in stat["files"]}
    blocks: list[str] = []
    lines: list[str] = []

    def flush() -> None:
        if lines:
            blocks.append(f'<pre>{"".join(lines)}</pre>')
            lines.clear()

    in_hunk = False
    for line in patch_text.splitlines():
        # Same boundary helper diffstat uses (Task 3), so the file rows here and
        # the rows in `per_file` are keyed by identical paths and the `in_hunk`
        # reset happens at exactly the same lines in both.
        path = _diff_git_path(line)
        if path is not None:
            flush()
            in_hunk = False                   # back in a file header block
            f = per_file.get(path, {"added": 0, "removed": 0})
            blocks.append(
                f'<div class="file"><span class="p">{html.escape(path)}</span>'
                f'<span class="st"><span class="a">+{f["added"]}</span> '
                f'<span class="r">−{f["removed"]}</span></span></div>')
            continue
        if not blocks:
            continue                          # the `#` header block
        # One classifier, shared with diffstat (Task 3) — the counts and the
        # colours can never disagree about what a line is, including on the
        # `in_hunk` rule that keeps a deleted `---` from reading as a marker.
        kind = _diff_line_kind(line, in_hunk)
        if kind == "hunk":
            in_hunk = True
        if kind == "marker":
            continue                          # shown in the file header row instead
        cls = f"l {_DIFF_CLASS[kind]}".rstrip()
        lines.append(f'<span class="{cls}">{html.escape(line)}\n</span>')
    flush()

    return (
        f'<details class="patch"><summary>fix.patch '
        f'<span class="sz">— {size_kb:.0f} KB · '
        f'<a href="{html.escape(patch_href)}">open raw ↗</a></span></summary>'
        f'<div class="diff">{"".join(blocks)}</div></details>')


def render_fix_page(cand_id: str, symbol: str, fx: dict,
                    raw_reldir: str, back_href: str) -> str:
    """The `…__fix.html` page: orientation, apply recipe, notes, then the diff."""
    hdr = fx.get("header") or {}
    base = hdr.get("base", "")
    ph = _repo_placeholder(hdr.get("repo"))
    stat_line = format_diffstat(fx["stat"])
    body = [
        f'<a class="back" href="{html.escape(back_href)}">← Back to candidate</a>',
        f"<h1>One-shot fix — {html.escape(symbol)}</h1>",
        f'<p class="subtitle">{html.escape(cand_id)} · {html.escape(stat_line)}</p>',
        '<div class="orient">'
        '<p><strong>What is this?</strong> One attempt at implementing this '
        'candidate\'s proposal, as a reviewable patch — not an evolutionary '
        'search. Spotlights read the candidate and the research behind it, made '
        'the change in a throwaway worktree, and handed back the diff plus its '
        'notes. Nothing has been applied to any repository.</p>'
        f'<p class="warn">⚠️ {html.escape(_FIX_UNVERIFIED)}</p></div>',
    ]

    apply_cmds = (
        f"REPO={ph}\n"
        f"git -C \"$REPO\" checkout {base or '<BASE_COMMIT>'}\n"
        f"git -C \"$REPO\" apply --check \"$PWD/{raw_reldir}/fix.patch\" \\\n"
        f"  && git -C \"$REPO\" apply \"$PWD/{raw_reldir}/fix.patch\"")
    apply_parts = ['<div class="apply"><h2>Apply this patch</h2>']
    if base:
        apply_parts.append(
            f'<p class="note">Base commit <span class="sha">{html.escape(base)}</span>'
            ' — the patch assumes this exact commit.</p>')
    apply_parts.append(f'<pre>{html.escape(apply_cmds)}</pre>')
    apply_parts.append(
        '<p class="note">If it does not apply cleanly, '
        f'<code>git -C "$REPO" apply -3 "$PWD/{html.escape(raw_reldir)}/fix.patch"</code> '
        'falls back to a three-way merge. Without git, <code>patch -p1 &lt; '
        'fix.patch</code> works from the repo root.</p>')
    apply_parts.append(
        '<p class="note">The paths written inside <code>fix.patch</code> and '
        '<code>FIX-NOTES.md</code> name the machine that produced them — '
        'substitute your own checkout, as above. The files are copied here '
        'byte-for-byte and were not rewritten.</p>')
    apply_parts.append(
        f'<a class="dl" href="{html.escape(raw_reldir)}/fix.zip" download>'
        '⬇ Download fix (.zip)</a>')
    apply_parts.append("</div>")
    body.append("".join(apply_parts))

    if fx.get("notes_text"):
        body.append(md_to_html_body(fx["notes_text"]))
    body.append("<h2>The patch</h2>")
    body.append(render_patch(fx["patch_text"],
                             f"{raw_reldir}/fix.patch", fx.get("patch_kb", 0.0)))
    return _doc(f"One-shot fix — {symbol}", "\n".join(body))


def render_fix_section(fx: dict, fix_page_name: str) -> str:
    """The "One-shot fix" block appended to a candidate page."""
    return (
        '<div class="fix-section"><h2>One-shot fix</h2>'
        f'<p>A reviewable patch for this candidate: '
        f'<strong>{html.escape(format_diffstat(fx["stat"]))}</strong>. '
        'Nothing about it was verified — no test, no benchmark, no build.</p>'
        f'<p><a href="{html.escape(fix_page_name)}">View the fix →</a></p></div>')


def build(source_dir: str, top_n: int = 5) -> dict:
    """Parse sorted_candidates.md and build share-bundle/ + share-candidates.zip.

    Returns {title, count, bundle_dir, zip_path, skipped}. Candidates whose
    linked .md file cannot be found are skipped (recorded in 'skipped'), not
    fatal.
    """
    src = Path(source_dir).resolve()
    sorted_md = src / "sorted_candidates.md"
    if not sorted_md.is_file():
        raise FileNotFoundError(f"sorted_candidates.md not found in {src}")

    md_text = sorted_md.read_text(encoding="utf-8")
    header = parse_header(md_text)
    rows = parse_ranking_table(md_text, top_n)

    bundle = src / "share-bundle"
    if bundle.exists():
        shutil.rmtree(bundle)
    (bundle / "candidates").mkdir(parents=True)

    # `prep-evolve` writes its output to a sibling `evolve/` tree; fold it in
    # when present, otherwise the build proceeds exactly as before.
    evolve_root = src.parent / "evolve"

    kept: list[dict] = []
    skipped: list[str] = []
    evolve_bundles = 0
    for r in rows:
        cand_path = (src / r["rel_link"]).resolve()
        if not cand_path.is_file():
            skipped.append(r["cand_id"])
            continue
        cand_md = cand_path.read_text(encoding="utf-8")
        mod_rel = _module_rel_path(r["rel_link"])  # modules/pkg/file.md
        out_md = bundle / "candidates" / mod_rel
        out_html = out_md.with_suffix(".html")
        out_md.parent.mkdir(parents=True, exist_ok=True)
        out_md.write_text(cand_md, encoding="utf-8")
        depth = len(Path(mod_rel).parts)  # candidates/<...>; back to bundle root
        back_href = "../" * (depth) + "index.html"

        r = dict(r)
        r["html_href"] = str(Path("candidates") / Path(mod_rel).with_suffix(".html"))

        # Fold in evolve bundles for this candidate, if any exist on disk.
        engines = find_evolve(r["cand_id"], evolve_root)
        r["evolve_count"] = len(engines) if engines else 0
        r["evolve_engines"] = list(engines) if engines else []
        r["evolve_href"] = ""
        evolve_section = ""
        if engines:
            evolve_bundles += 1
            stem = Path(mod_rel).with_suffix("")               # modules/pkg/file
            evolve_stem = str(stem) + "__evolve"
            raw_reldir = Path(evolve_stem).name                # relative to the page
            evolve_page_name = raw_reldir + ".html"
            r["evolve_href"] = str(Path("candidates") / (evolve_stem + ".html"))
            evolve_dir = bundle / "candidates" / evolve_stem
            _copy_evolve_files(engines, evolve_dir)
            (bundle / "candidates" / (evolve_stem + ".html")).write_text(
                render_evolve_page(r["cand_id"], r["symbol"] or r["cand_id"],
                                   engines, raw_reldir, Path(stem).name + ".html"),
                encoding="utf-8")
            evolve_section = render_evolve_section(engines, evolve_page_name)

        out_html.write_text(
            render_candidate_page(cand_md, r["symbol"] or r["cand_id"], back_href,
                                  evolve_section),
            encoding="utf-8",
        )
        kept.append(r)

    (bundle / "index.html").write_text(render_index(header, kept), encoding="utf-8")

    zip_path = src / "share-candidates.zip"
    if zip_path.exists():
        zip_path.unlink()
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        for f in sorted(bundle.rglob("*")):
            if f.is_file():
                zf.write(f, str(f.relative_to(bundle.parent)))

    return {
        "title": header["title"],
        "count": len(kept),
        "bundle_dir": str(bundle),
        "zip_path": str(zip_path),
        "skipped": skipped,
        "evolve_bundles": evolve_bundles,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="Build a browsable ZIP of top-N candidates.")
    ap.add_argument("--source", required=True, help="Folder containing sorted_candidates.md")
    ap.add_argument("--top-n", type=int, default=5, help="How many top candidates (default 5)")
    args = ap.parse_args()
    result = build(args.source, args.top_n)
    print(f"Title:      {result['title']}")
    print(f"Candidates: {result['count']}")
    print(f"Bundle:     {result['bundle_dir']}")
    print(f"Zip:        {result['zip_path']}")
    if result.get("evolve_bundles"):
        print(f"Evolve:     {result['evolve_bundles']} candidate(s) with evolve bundles")
    if result["skipped"]:
        print(f"Skipped (file not found): {', '.join(result['skipped'])}")


if __name__ == "__main__":
    main()
