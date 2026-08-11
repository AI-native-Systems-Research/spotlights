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


def render_candidate_page(md_text: str, title: str, back_href: str) -> str:
    back = f'<a class="back" href="{html.escape(back_href)}">← Back to index</a>'
    return _doc(title, back + "\n" + md_to_html_body(md_text))


def render_index(header: dict, rows: list[dict]) -> str:
    parts = [f"<h1>{html.escape(header.get('title', 'Candidates'))}</h1>"]
    if header.get("objective"):
        parts.append(f'<p class="subtitle">{render_inline(header["objective"])}</p>')
    for r in rows:
        impact_cls = "impact-high" if r["impact"].lower() == "high" else ""
        card = (
            f'<a class="card" href="{html.escape(r["html_href"])}">'
            f'<div class="row1">'
            f'<span class="rank">#{html.escape(r["rank"])}</span>'
            f'<span class="sym">{html.escape(r["symbol"])}</span>'
            f'<span class="mod">{html.escape(r["module"])}</span>'
            f'<span class="badges">'
            f'<span class="badge {impact_cls}">{html.escape(r["impact"])}</span>'
            f'<span class="badge score">score {html.escape(r["score"])}</span>'
            f'</span></div>'
            f'<p class="rationale">{render_inline(r["rationale"])}</p>'
            f'</a>'
        )
        parts.append(card)
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

    kept: list[dict] = []
    skipped: list[str] = []
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
        out_html.write_text(
            render_candidate_page(cand_md, r["symbol"] or r["cand_id"], back_href),
            encoding="utf-8",
        )
        r = dict(r)
        r["html_href"] = str(Path("candidates") / Path(mod_rel).with_suffix(".html"))
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
    if result["skipped"]:
        print(f"Skipped (file not found): {', '.join(result['skipped'])}")


if __name__ == "__main__":
    main()
