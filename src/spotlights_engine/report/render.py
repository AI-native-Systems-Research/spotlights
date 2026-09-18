"""Render one finished Spotlights run directory into a single self-contained experiment page.

Reads result.json + run_manifest.json + sorted/sorted_candidates.json (plus the
evolve/ and apply/ artifact trees) and writes one experiment.html: inlined CSS/JS,
no external assets, theme-aware, every chart paired with a table view.

Usage:  spotlights-engine report <run-dir> [-o out.html]

Stdlib only.
"""
from __future__ import annotations

import html
import json
import re
from collections import Counter
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

# --------------------------------------------------------------------------- #
# formatting helpers
# --------------------------------------------------------------------------- #


def esc(x) -> str:
    return html.escape("" if x is None else str(x))


def mdi(x) -> str:
    """Escape, then honour the inline markdown the agents actually write.

    Escaping happens first, so the only tags in the result are the ones added
    here; `x` is turned into <code> and **x** into <b>. The bold pattern
    excludes < and > so it can never span a tag boundary.
    """
    t = esc(x)
    t = re.sub(r"`([^`\n]+)`", r"<code>\1</code>", t)
    t = re.sub(r"\*\*([^*<>\n]+)\*\*", r"<b>\1</b>", t)
    t = re.sub(r"(?<![*\w])\*([^*<>\n]+)\*(?![*\w])", r"<i>\1</i>", t)
    return t


def brk(path) -> str:
    """Escape a slash-separated path and allow it to wrap at the separators.

    Long mono paths carry no break opportunity of their own, so without this a
    narrow column either overflows or breaks mid-identifier.
    """
    return esc(path).replace("/", "/<wbr>").replace("_", "_<wbr>")


def n0(x) -> str:
    """Thousands-separated integer."""
    try:
        return f"{int(round(float(x))):,}"
    except (TypeError, ValueError):
        return "—"


def usd(x, digits: int = 2) -> str:
    try:
        return f"${float(x):,.{digits}f}"
    except (TypeError, ValueError):
        return "—"


def compact(x) -> str:
    """1284 -> 1,284 · 1_284_000 -> 1.28M — for stat-tile values."""
    try:
        v = float(x)
    except (TypeError, ValueError):
        return "—"
    for cut, suf in ((1e9, "B"), (1e6, "M"), (1e3, "K")):
        if abs(v) >= cut:
            return f"{v / cut:.2f}".rstrip("0").rstrip(".") + suf
    return f"{int(v):,}"


def dur(seconds) -> str:
    """4h 19m · 12m 04s · 44s"""
    try:
        s = int(round(float(seconds)))
    except (TypeError, ValueError):
        return "—"
    h, rem = divmod(s, 3600)
    m, sec = divmod(rem, 60)
    if h:
        return f"{h}h {m:02d}m"
    if m:
        return f"{m}m {sec:02d}s"
    return f"{sec}s"


def pct(part, whole, digits: int = 1) -> str:
    try:
        w = float(whole)
        if w == 0:
            return "—"
        return f"{100.0 * float(part) / w:.{digits}f}%"
    except (TypeError, ValueError):
        return "—"


def short_sha(sha, n: int = 8) -> str:
    s = str(sha or "")
    return s[:n] if s else "—"


def ink_on(hex_fill: str) -> str:
    """Pick white or near-black for a label set inside a colored fill."""
    h = hex_fill.lstrip("#")
    if len(h) != 6:
        return "#ffffff"
    r, g, b = (int(h[i:i + 2], 16) / 255 for i in (0, 2, 4))

    def lin(c):
        return c / 12.92 if c <= 0.03928 else ((c + 0.055) / 1.055) ** 2.4

    lum = 0.2126 * lin(r) + 0.7152 * lin(g) + 0.0722 * lin(b)
    return "#0b0b0b" if lum > 0.42 else "#ffffff"


# --------------------------------------------------------------------------- #
# design tokens + stylesheet
#
# Chrome palette is the Spotlights leaderboard's warm-amber system. The viz
# tokens below were validated with the dataviz validator against THIS page's
# real chart surface (light #fbfaf6 / dark #15171d):
#   * --viz-accent  #c47f1e light (3.14:1) / #f4b44a dark (9.79:1) — the single
#     hue for one-series magnitude bars and the emphasis mark.
#   * --viz-rest    de-emphasis gray for the "everything else" half of an
#     emphasis chart. Deliberately below the chroma floor: it must read gray.
#   * 4 categorical slots (blue/orange/aqua/violet) — ALL CHECKS PASS in both
#     modes on the adjacent pairlist (worst adjacent CVD dE 9.2 light / 9.4
#     dark; normal-vision 27.6 / 24.6). Light-mode aqua sits at 2.7:1, so the
#     relief rule applies and every stack ships direct labels + a table view.
#   * Status tokens (good/warning/critical) are the fixed reserved palette and
#     are never themed; they always ship an icon + a text label, never hue
#     alone. "Skipped" is not a status — it takes the de-emphasis gray.
# --------------------------------------------------------------------------- #

BULLET = "•"

VIZ: dict[str, dict[str, Any]] = {
    "light": {
        "accent": "#c47f1e",
        "rest": "#cfcabb",
        "slots": ["#2a78d6", "#eb6834", "#1baf7a", "#4a3aa7"],
        "good": "#0ca30c",
        "warning": "#fab219",
        "critical": "#d03b3b",
    },
    "dark": {
        "accent": "#f4b44a",
        "rest": "#454b5a",
        "slots": ["#3987e5", "#d95926", "#199e70", "#9085e9"],
        "good": "#0ca30c",
        "warning": "#fab219",
        "critical": "#d03b3b",
    },
}

CSS = """
:root {
  --bg:#f4f2ec; --panel:#fbfaf6; --panel-2:#f0eee7; --border:#e0ddd2;
  --border-strong:#cfcabb; --ink:#1c1b18; --muted:#6f6b60; --faint:#a29d90;
  --accent:#d8912a; --accent-glow:rgba(216,145,42,.28); --hi:#c47f1e;
  --shadow:0 1px 2px rgba(30,25,15,.06),0 8px 24px rgba(30,25,15,.05);
  --mono:ui-monospace,"SF Mono","JetBrains Mono",Menlo,Consolas,monospace;
  --sans:system-ui,-apple-system,"Segoe UI",Roboto,sans-serif;
  --grid:#e1e0d9;
  --viz-accent:#c47f1e; --viz-rest:#cfcabb;
  --viz-1:#2a78d6; --viz-2:#eb6834; --viz-3:#1baf7a; --viz-4:#4a3aa7;
  --st-good:#0ca30c; --st-warn:#fab219; --st-crit:#d03b3b;
}
@media (prefers-color-scheme: dark) {
  :root:not([data-theme="light"]) {
    --bg:#0d0e12; --panel:#15171d; --panel-2:#1b1e26; --border:#262a33;
    --border-strong:#333846; --ink:#eae8e2; --muted:#8b8f9a; --faint:#5b606c;
    --accent:#f4b44a; --accent-glow:rgba(244,180,74,.30); --hi:#f4b44a;
    --shadow:0 1px 2px rgba(0,0,0,.4),0 10px 30px rgba(0,0,0,.35);
    --grid:#2c2c2a;
    --viz-accent:#f4b44a; --viz-rest:#454b5a;
    --viz-1:#3987e5; --viz-2:#d95926; --viz-3:#199e70; --viz-4:#9085e9;
  }
}
:root[data-theme="dark"] {
  --bg:#0d0e12; --panel:#15171d; --panel-2:#1b1e26; --border:#262a33;
  --border-strong:#333846; --ink:#eae8e2; --muted:#8b8f9a; --faint:#5b606c;
  --accent:#f4b44a; --accent-glow:rgba(244,180,74,.30); --hi:#f4b44a;
  --shadow:0 1px 2px rgba(0,0,0,.4),0 10px 30px rgba(0,0,0,.35);
  --grid:#2c2c2a;
  --viz-accent:#f4b44a; --viz-rest:#454b5a;
  --viz-1:#3987e5; --viz-2:#d95926; --viz-3:#199e70; --viz-4:#9085e9;
}
* { box-sizing:border-box; }
body { margin:0; background:var(--bg); color:var(--ink); font-family:var(--sans);
  font-size:14px; line-height:1.5; -webkit-font-smoothing:antialiased; }
.mono { font-family:var(--mono); font-variant-numeric:tabular-nums; }
.wrap { max-width:1240px; margin:0 auto; padding:20px 20px 40px; }
a { color:var(--accent); }

/* ---------- topbar ---------- */
.topbar { display:flex; align-items:center; gap:22px; flex-wrap:wrap;
  padding:14px 18px; background:var(--panel); border:1px solid var(--border);
  border-radius:12px; box-shadow:var(--shadow); }
.brand { display:flex; align-items:center; gap:10px; font-weight:700;
  letter-spacing:.14em; font-size:13px; text-transform:uppercase; }
.beam { width:16px; height:16px; border-radius:50%;
  background:radial-gradient(circle at 50% 45%,var(--accent),transparent 68%);
  box-shadow:0 0 12px 2px var(--accent-glow); position:relative; flex:none; }
.beam::after { content:""; position:absolute; inset:5px; border-radius:50%;
  background:var(--accent); }
.obj { color:var(--muted); font-size:13px; border-left:1px solid var(--border-strong);
  padding-left:20px; max-width:760px; }
.obj b { color:var(--ink); font-weight:600; }
.spacer { flex:1; }
.tgl { font:inherit; font-size:12px; color:var(--muted); cursor:pointer;
  background:var(--panel-2); border:1px solid var(--border-strong);
  border-radius:999px; padding:5px 12px; white-space:nowrap; }
.tgl:hover { color:var(--ink); border-color:var(--accent); }

/* ---------- section furniture ---------- */
.sec { margin-top:30px; }
.sec > h2 { font-size:11px; letter-spacing:.16em; text-transform:uppercase;
  color:var(--faint); font-weight:700; margin:0 2px 12px;
  display:flex; align-items:center; gap:9px; }
.sec > h2::after { content:""; flex:1; height:1px; background:var(--border); }
.card { background:var(--panel); border:1px solid var(--border); border-radius:12px;
  box-shadow:var(--shadow); }
.pad { padding:16px 18px; }

/* ---------- hero + stat tiles ---------- */
.hero { display:flex; flex-wrap:wrap; gap:26px; align-items:flex-end;
  padding:20px 18px 18px; }
.hero .fig { font-size:52px; font-weight:600; line-height:1;
  letter-spacing:-.02em; }
.hero .figlab { font-size:12px; color:var(--muted); margin-top:6px; }
.hero .figlab b { color:var(--ink); font-weight:600; }
.meta { display:grid; grid-template-columns:repeat(auto-fit,minmax(190px,1fr));
  gap:12px 26px; flex:1; min-width:280px; }
.meta div { min-width:0; }
.meta .k { font-size:10px; letter-spacing:.12em; text-transform:uppercase;
  color:var(--faint); }
.meta .v { font-size:13px; margin-top:2px; overflow-wrap:anywhere; }

.tiles { display:grid; grid-template-columns:repeat(auto-fit,minmax(168px,1fr));
  gap:12px; margin-top:12px; }
.tile { background:var(--panel); border:1px solid var(--border); border-radius:12px;
  box-shadow:var(--shadow); padding:14px 16px; }
.tile .l { font-size:11px; color:var(--muted); }
.tile .n { font-size:26px; font-weight:600; line-height:1.15; margin-top:4px; }
.tile .d { font-size:11.5px; color:var(--faint); margin-top:3px; }
.tile .d b { color:var(--muted); font-weight:600; }

/* ---------- constraint list ---------- */
.hints { list-style:none; margin:0; padding:0; display:grid;
  grid-template-columns:repeat(auto-fit,minmax(260px,1fr)); gap:10px 22px; }
.hints li { display:flex; gap:9px; font-size:13px; color:var(--muted);
  line-height:1.45; }
.hints li::before { content:""; flex:none; width:3px; border-radius:2px;
  background:var(--border-strong); }

/* ---------- charts ---------- */
.charts { display:grid; grid-template-columns:repeat(auto-fit,minmax(430px,1fr));
  gap:14px; align-items:start; }
.chart.wide { grid-column:1/-1; }
.chart { background:var(--panel); border:1px solid var(--border);
  border-radius:12px; box-shadow:var(--shadow); padding:16px 18px 14px;
  display:flex; flex-direction:column; }
.chart h3 { font-size:14px; font-weight:600; margin:0; }
.chart .sub { font-size:12px; color:var(--muted); margin:3px 0 0; }
.chart .plot { margin-top:14px; }
.chart .note { font-size:11px; color:var(--faint); margin:12px 0 0;
  padding-top:10px; border-top:1px solid var(--border); line-height:1.5; }
.chart .head { display:flex; align-items:flex-start; gap:12px; }
.chart .head .spacer { flex:1; }

/* legend — always present for >=2 series */
.legend { display:flex; flex-wrap:wrap; gap:6px 16px; margin-top:12px; }
.legend span { display:inline-flex; align-items:center; gap:7px; font-size:12px;
  color:var(--muted); }
.legend i { width:10px; height:10px; border-radius:3px; flex:none; }
.legend b { color:var(--ink); font-weight:600; font-variant-numeric:tabular-nums; }

/* horizontal bar rows */
.bars { display:flex; flex-direction:column; gap:5px; }
.pmbars { margin-top:16px; }
.bar { display:grid; grid-template-columns:minmax(96px,auto) 1fr auto;
  align-items:center; gap:12px; min-height:26px; }
.bar > .lab { font-size:12px; color:var(--muted); text-align:right;
  white-space:nowrap; overflow:hidden; text-overflow:ellipsis; }
.bar > .val { font-family:var(--mono); font-size:12px; color:var(--ink);
  font-variant-numeric:tabular-nums; text-align:right; white-space:nowrap; }
.track { position:relative; height:14px; display:flex; gap:2px;
  background:var(--panel); }
.track::before { content:""; position:absolute; left:0; right:0; bottom:-1px;
  height:1px; background:var(--grid); }
.seg { height:100%; min-width:2px; }
.seg:last-child { border-radius:0 4px 4px 0; }
.bars .bar:hover > .lab, .bars .bar:hover > .val { color:var(--ink); }

/* one-row stacked bar */
.stack { display:flex; gap:2px; height:20px; background:var(--panel); }
.stack .seg { border-radius:0; display:flex; align-items:center;
  justify-content:center; overflow:visible; }
.stack .seg:first-child { border-radius:4px 0 0 4px; }
.stack .seg:last-child { border-radius:0 4px 4px 0; }
.stack .seg span { font-family:var(--mono); font-size:11px; font-weight:600;
  letter-spacing:.02em; white-space:nowrap; }
.axisnote { display:flex; justify-content:space-between; font-family:var(--mono);
  font-size:10.5px; color:var(--faint); margin-top:6px; }

/* status legend carries an icon + a label, never hue alone */
.stleg { display:flex; flex-wrap:wrap; gap:8px 18px; margin-top:12px; }
.stleg span { display:inline-flex; align-items:center; gap:7px; font-size:12px;
  color:var(--muted); }
.stleg .ic { width:15px; height:15px; border-radius:4px; flex:none; color:#fff;
  font-size:10px; line-height:15px; text-align:center; font-weight:700; }
.stleg b { color:var(--ink); font-weight:600; font-variant-numeric:tabular-nums; }

/* per-status module roster under the outcome chart. Collapsed by default:
   30 path-like names would otherwise be the tallest thing on the page. The
   status hue lives only in the group's icon chip -- every name and count wears
   a text token, so nothing here is identified by colour. */
details.mbrk { margin-top:14px; border-top:1px solid var(--border);
  padding-top:10px; }
details.mbrk summary { cursor:pointer; font-size:11px; letter-spacing:.1em;
  text-transform:uppercase; color:var(--faint); font-weight:600; }
details.mbrk summary:hover { color:var(--ink); }
.mgrid { display:grid; grid-template-columns:repeat(auto-fit,minmax(215px,1fr));
  gap:16px 18px; margin-top:12px; }
.mgrp { min-width:0; }
.mgh { display:flex; align-items:center; gap:7px; font-size:12px;
  color:var(--ink); font-weight:600; }
.mgh .ic { width:15px; height:15px; border-radius:4px; flex:none; color:#fff;
  font-size:10px; line-height:15px; text-align:center; font-weight:700; }
.mgh b { margin-left:auto; font-family:var(--mono); color:var(--muted);
  font-weight:600; font-variant-numeric:tabular-nums; }
.mgs { font-size:10.5px; color:var(--faint); margin:3px 0 8px 22px;
  font-family:var(--mono); }
.mgl { list-style:none; margin:0; padding:0; display:flex;
  flex-direction:column; gap:3px; }
.mgl li { display:flex; align-items:baseline; gap:8px; font-family:var(--mono);
  font-size:11px; color:var(--muted); line-height:1.45;
  border-top:1px solid var(--border); padding-top:3px; }
.mgl li:first-child { border-top:0; padding-top:0; }
.mgl .nm { min-width:0; overflow-wrap:anywhere; }
.mgl .ct { margin-left:auto; flex:none; color:var(--ink);
  font-variant-numeric:tabular-nums; }
.mgl .ct.z { color:var(--faint); font-weight:400; }

/* table view — the relief for every chart */
.tv { margin-top:12px; border-top:1px solid var(--border); padding-top:10px; }
.tv[hidden] { display:none; }
.tv table { width:100%; border-collapse:collapse; font-size:12px; }
.tv th, .tv td { text-align:left; padding:5px 8px; border-bottom:1px solid var(--border); }
.tv th { font-size:10px; letter-spacing:.1em; text-transform:uppercase;
  color:var(--faint); font-weight:600; }
.tv td.r, .tv th.r { text-align:right; font-family:var(--mono);
  font-variant-numeric:tabular-nums; }
.tv tr:last-child td { border-bottom:none; }
/* A seven-column per-module table will not fit a phone. It scrolls inside its own
   box rather than making the page scroll sideways. */
.tvscroll { overflow-x:auto; }
.tvscroll table { min-width:620px; }
.tv tr.tot td { border-top:1px solid var(--border-strong); border-bottom:none;
  font-weight:700; color:var(--ink); }
.tvbtn { font:inherit; font-size:10.5px; letter-spacing:.1em; text-transform:uppercase;
  color:var(--faint); background:none; border:1px solid var(--border);
  border-radius:6px; padding:4px 9px; cursor:pointer; white-space:nowrap; flex:none; }
.tvbtn:hover { color:var(--ink); border-color:var(--border-strong); }

/* shared tooltip */
#tip { position:fixed; z-index:60; pointer-events:none; opacity:0;
  transition:opacity .1s; background:var(--panel); color:var(--ink);
  border:1px solid var(--border-strong); border-radius:8px; padding:7px 10px;
  font-size:12px; box-shadow:var(--shadow); max-width:280px; }
#tip.on { opacity:1; }
#tip .tt { font-weight:600; }
#tip .tv2 { font-family:var(--mono); color:var(--muted); margin-top:2px; }
"""

CSS += """
/* ---------- agent contribution panel ---------- */
.agents { display:grid; grid-template-columns:repeat(auto-fit,minmax(300px,1fr));
  gap:14px; }
.atab { width:100%; border-collapse:collapse; font-size:12.5px; }
.atab th, .atab td { padding:7px 6px; border-bottom:1px solid var(--border);
  text-align:right; }
.atab th:first-child, .atab td:first-child { text-align:left; }
.atab th { font-size:10px; letter-spacing:.1em; text-transform:uppercase;
  color:var(--faint); font-weight:600; }
.atab td { font-family:var(--mono); font-variant-numeric:tabular-nums; }
.atab td:first-child { font-family:var(--sans); }
.atab tr:last-child td { border-bottom:none; }
.atab .who { display:inline-flex; align-items:center; gap:8px; }
.atab .who i { width:9px; height:9px; border-radius:3px; flex:none; }

/* ---------- controls ---------- */
.controls { display:flex; gap:10px; align-items:center; flex-wrap:wrap;
  margin:0 2px 12px; }
.controls input, .controls select { font:inherit; color:var(--ink);
  background:var(--panel); border:1px solid var(--border-strong);
  border-radius:8px; padding:7px 10px; }
.controls input { flex:1; min-width:200px; }
.controls .count { color:var(--faint); font-family:var(--mono); font-size:12px;
  margin-left:auto; white-space:nowrap; }
/* Says so when opening a candidate from the tree had to clear a filter. The
   reset is otherwise invisible, and a filter row that silently empties itself
   reads as a glitch. margin-left:auto lives on .count, so the note sits after
   it and the pair stays right-aligned. */
.controls .lbnote { color:var(--accent); font-family:var(--mono); font-size:11px;
  white-space:nowrap; }
.controls .lbnote:empty { display:none; }

/* ---------- leaderboard ---------- */
.tbl-scroll { overflow-x:auto; border-radius:12px; }
/* A list long enough to outgrow the screen scrolls inside its own card rather
   than pushing everything after it off the page. Only the long ones: a scroll
   frame around eight rows is furniture. 70vh leaves the filter row above and the
   note below it in view, which is the whole point -- controls you have to scroll
   away from the list to reach are controls you stop using. The `th`s were
   already `position:sticky`; with no scrolling ancestor that did nothing, and
   with one the header now holds while the rows move under it. */
.tbl-scroll.tall { max-height:70vh; }
/* The leaderboard gets two more rows than that. Its rows are tall -- symbol,
   location, and a rationale that wraps to two lines in its 560px column, ~113px
   measured over vllm_subset's 173 rows -- so two rows is 226px. Absolute rather
   than a bigger `vh`: "two more candidates" is a row count, and a percentage
   would deliver a different number of rows on every screen. On a short window
   this does put the bottom of the card past the fold; that is ordinary page
   scrolling, and it is the cost of the two rows. */
.tbl-scroll.tall.lbtall { max-height:calc(70vh + 226px); }
table.lb { width:100%; border-collapse:collapse; min-width:860px; }
table.lb th { text-align:left; font-size:10px; letter-spacing:.1em;
  text-transform:uppercase; color:var(--faint); font-weight:600; padding:9px 12px;
  border-bottom:1px solid var(--border); background:var(--panel);
  user-select:none; position:sticky; top:0; z-index:2; }
table.lb th[data-key] { cursor:pointer; }
table.lb th[data-key]:hover { color:var(--ink); }
table.lb th .ar { opacity:0; font-size:9px; }
table.lb th.on .ar { opacity:1; color:var(--accent); }
table.lb th.r { text-align:right; }
table.lb td { padding:11px 12px; border-bottom:1px solid var(--border);
  vertical-align:middle; }
tbody.cand:last-child tr:last-child td { border-bottom:none; }
tr.row { cursor:pointer; }
tr.row:hover td { background:var(--panel-2); }
tbody.cand.open tr.row td { background:var(--panel-2); }
/* Arriving at one row out of 173 needs a moment of "here": the pane may have
   swapped, the row may be mid-page, and an opened detail alone is easy to miss.
   It animates back to transparent so the .open background underneath survives. */
@keyframes lbflash { from { background:var(--accent-glow); }
  to { background:transparent; } }
tbody.cand.flash tr.row td { animation:lbflash 1.5s ease-out 1; }
@media (prefers-reduced-motion:reduce) {
  tbody.cand.flash tr.row td { animation:none; }
}
.rank { font-family:var(--mono); color:var(--faint); font-size:12px; width:34px; }
.pred { display:flex; align-items:center; gap:9px; min-width:104px; }
.pred .v { font-family:var(--mono); font-weight:700; font-size:15px; width:26px;
  font-variant-numeric:tabular-nums; }
.glowbar { flex:1; height:5px; border-radius:3px; background:var(--panel-2);
  overflow:hidden; min-width:44px; }
.glowbar i { display:block; height:100%;
  background:linear-gradient(90deg,var(--accent),var(--hi));
  box-shadow:0 0 8px var(--accent-glow); }
.pill { display:inline-flex; align-items:center; gap:5px; font-size:10.5px;
  font-weight:600; letter-spacing:.06em; text-transform:uppercase; padding:3px 9px;
  border-radius:999px; border:1px solid transparent; white-space:nowrap; }
.imp-high { color:var(--accent); border-color:var(--accent); background:var(--accent-glow); }
.imp-medium { color:var(--muted); border-color:var(--border-strong); }
.imp-low { color:var(--faint); border-color:var(--border); }
.sym { font-family:var(--mono); font-weight:600; font-size:13px;
  overflow-wrap:anywhere; }
.loc { font-family:var(--mono); font-size:11px; color:var(--faint); margin-top:2px;
  overflow-wrap:anywhere; }
.rat { color:var(--muted); font-size:12px; margin-top:4px; max-width:560px; }
.modtag { font-family:var(--mono); font-size:11px; color:var(--muted);
  overflow-wrap:break-word; }
.depth { font-family:var(--mono); font-size:11.5px; color:var(--muted);
  white-space:nowrap; }
.stage { display:flex; flex-direction:column; gap:4px; align-items:flex-start; }
.tag { display:inline-flex; align-items:center; gap:5px; font-size:10px;
  letter-spacing:.06em; text-transform:uppercase; font-weight:600;
  padding:2px 7px; border-radius:5px; white-space:nowrap;
  border:1px solid var(--border-strong); color:var(--muted); }
.tag.on { color:var(--accent); border-color:var(--accent); background:var(--accent-glow); }
.tag .ic { font-size:9px; }
.stage .none { font-family:var(--mono); font-size:11px; color:var(--faint); }

.d-sec code, .prop code { font-family:var(--mono); font-size:.92em;
  background:var(--panel-2); border-radius:4px; padding:0 3px; }

/* ---------- detail panel ---------- */
tbody.cand .detail { display:none; }
tbody.cand.open .detail { display:table-row; }
.d-wrap { padding:4px 4px 12px; }
.d-grid { display:grid; grid-template-columns:repeat(auto-fit,minmax(300px,1fr));
  gap:14px 22px; }
.d-sec h4 { font-size:10px; letter-spacing:.14em; text-transform:uppercase;
  color:var(--faint); margin:0 0 5px; font-weight:700; }
.d-sec p { margin:0; color:var(--muted); font-size:12.5px; line-height:1.55; }
.d-sec p.ink { color:var(--ink); }
.d-props { margin-top:16px; }
.prop { border:1px solid var(--border); border-radius:9px; padding:11px 13px;
  background:var(--panel-2); margin-top:8px; }
.prop .ph { display:flex; align-items:baseline; gap:9px; flex-wrap:wrap; }
.prop .ph b { font-size:13px; }
.prop .by { font-family:var(--mono); font-size:10.5px; color:var(--faint);
  border:1px solid var(--border-strong); border-radius:4px; padding:1px 6px;
  white-space:nowrap; }
.prop p { margin:6px 0 0; color:var(--muted); font-size:12.5px; line-height:1.55;
  white-space:pre-wrap; }
.prop .eff { color:var(--accent); font-family:var(--mono); font-size:11px; }
.d-arts { margin-top:14px; font-family:var(--mono); font-size:11.5px;
  color:var(--muted); }
.d-arts code { background:var(--panel-2); border:1px solid var(--border);
  border-radius:4px; padding:1px 5px; }

/* ---------- issues ---------- */
.iss { display:grid; grid-template-columns:repeat(auto-fit,minmax(230px,1fr));
  gap:12px; }
.issrow { display:flex; align-items:center; gap:10px; font-size:13px; }
.issrow .ic { width:17px; height:17px; border-radius:5px; flex:none; color:#fff;
  font-size:11px; line-height:17px; text-align:center; font-weight:700; }
.issrow .n { font-family:var(--mono); font-weight:600; }
.issrow .t { color:var(--muted); font-size:12px; }
details.more { margin-top:14px; }
details.more summary { cursor:pointer; font-size:11px; letter-spacing:.1em;
  text-transform:uppercase; color:var(--faint); font-weight:600; }
details.more summary:hover { color:var(--ink); }
details.more ul { margin:10px 0 0; padding-left:18px; font-family:var(--mono);
  font-size:11.5px; color:var(--muted); line-height:1.7; max-height:280px;
  overflow-y:auto; }

.foot { color:var(--faint); font-size:11px; text-align:center; margin-top:26px;
  line-height:1.8; }
.foot code { font-family:var(--mono); }

@media (max-width:640px) {
  .obj { border-left:none; padding-left:0; }
  .hero .fig { font-size:42px; }
  .bar { grid-template-columns:1fr; gap:2px; }
  .bar > .lab { text-align:left; }
  .bar > .val { text-align:left; }
}
.tvbtn[aria-pressed="true"] { color:var(--accent); border-color:var(--accent);
  background:var(--accent-glow); }
/* the badge only earns its width in hot mode, where it explains the order */
.hp { display:none; }
.lb.hotmode .hp { display:block; margin-top:3px; font-size:10px; color:var(--accent);
  font-family:var(--mono); letter-spacing:0; }
/* In a bar row the badge is inline: the row is one grid line tall and the label
   column grows to fit rather than the row growing taller. */
.bars.hotmode .hp { display:inline; margin-left:6px; font-size:10px;
  color:var(--accent); font-family:var(--mono); }
.tv.hotmode tbody tr.dim { display:none; }

/* ---------- section head that carries a view switch ---------------------- */
.sechead { display:flex; align-items:baseline; gap:12px; }
.sechead .sp { flex:1 1 auto; }

/* ---------- radial collapsible candidate tree ---------------------------- */
.rtwrap { background:var(--panel); border:1px solid var(--border);
  border-radius:12px; padding:6px 6px 12px; }
.rtbar { display:flex; align-items:center; gap:8px; flex-wrap:wrap;
  padding:8px 10px 4px; font-size:11px; color:var(--muted); }
.rtbar .sp { flex:1 1 auto; }
.rtplot { width:100%; }
.rtsvg { display:block; width:100%; height:auto; margin:0 auto; }
.rtsvg .lnk { fill:none; stroke:var(--border); stroke-width:1; }
.rtsvg text { font-family:var(--mono); font-size:9px; fill:var(--muted);
  dominant-baseline:middle; pointer-events:none; }
/* A node's label is part of the node. The dot is about 9px across and the name
   beside it is the part that is actually legible -- and in hot mode it carries
   the score too -- so it is what a reader aims at. The labels stay outside the
   .nd groups, after them in document order, so they still paint above every
   dot and link; carrying the node's data-k is what makes them clickable
   without moving them. The hub count in the middle names no node and keeps the
   inherited pointer-events:none. */
.rtsvg text.lab { pointer-events:auto; cursor:pointer; }
.rtsvg text.br { fill:var(--faint); font-size:9.5px; letter-spacing:.02em; }
.rtsvg text.hub { fill:var(--faint); font-size:10px; text-anchor:middle;
  font-family:var(--mono); }
/* Breadcrumb for the drill-down. The tree can be rooted at any node, so the
   path from the run down to what is on screen has to be on screen too, and
   every step of it clickable: a zoom with no way back reads as a broken view. */
.rtcrumb { display:flex; align-items:center; gap:3px; flex-wrap:wrap;
  padding:0 10px 6px; font-family:var(--mono); font-size:10.5px;
  color:var(--muted); }
.rtcrumb button { font:inherit; background:none; border:0; padding:1px 2px;
  color:var(--accent); cursor:pointer; }
.rtcrumb button:hover { text-decoration:underline; }
.rtcrumb .here { color:var(--ink); padding:1px 2px; }
.rtcrumb .sep { color:var(--faint); }
.rtsvg .hubhit { cursor:pointer; }
.rtsvg .nd { cursor:pointer; }
.rtsvg .nd:focus { outline:none; }
.rtsvg .nd:focus circle.dot { stroke:var(--accent); stroke-width:2.5; }
.rtleg { display:flex; gap:16px; flex-wrap:wrap; padding:2px 10px 0;
  font-size:11px; color:var(--muted); }
.rtleg i { display:inline-block; width:9px; height:9px; border-radius:50%;
  margin-right:5px; vertical-align:-1px; border:1px solid transparent; }

@media print { .tgl, .tvbtn { display:none; } .tv { display:block !important; } }
"""


CSS += """
/* ---------- deep research layer ------------------------------------------ */
.works { display:grid; gap:8px; }
details.work { border:1px solid var(--border); border-radius:10px;
  background:var(--panel-2); }
details.work > summary { cursor:pointer; padding:10px 13px; display:flex;
  gap:10px; align-items:baseline; flex-wrap:wrap; list-style:none; }
details.work > summary::-webkit-details-marker { display:none; }
details.work > summary::before { content:"\\25B8"; color:var(--faint);
  font-size:10px; flex:none; }
details.work[open] > summary::before { content:"\\25BE"; }
details.work > summary:hover { color:var(--ink); }
.work .wt { font-size:12.5px; font-weight:600; overflow-wrap:anywhere; }
.work .wn { font-family:var(--mono); font-size:11px; color:var(--accent);
  margin-left:auto; white-space:nowrap; }
.work .wbody { padding:0 13px 12px; }
.work .wu { font-family:var(--mono); font-size:11px; color:var(--faint);
  overflow-wrap:anywhere; margin:0 0 9px; }
/* One line per distinct URL, because a work reached under two URLs is the case
   the reader most needs to see: a scraper mirror and a pinned archive version
   look identical from a count. Stacked, not comma-joined -- these are long. */
.work .wul { list-style:none; margin:-4px 0 9px; padding:0;
  font-family:var(--mono); font-size:11px; }
.work .wul li { margin:0 0 3px; overflow-wrap:anywhere; }
.work .wul .wm { color:var(--faint); }
.work .cols { display:grid;
  grid-template-columns:repeat(auto-fit,minmax(230px,1fr)); gap:12px; }
.work .col h5 { margin:0 0 4px; font-family:var(--mono); font-size:11px;
  color:var(--accent); font-weight:600; overflow-wrap:anywhere; }
.work .col p { margin:0; color:var(--muted); font-size:12px; line-height:1.5; }

/* The findings catalogue borrows the leaderboard's table grammar but keeps its
   own class: the leaderboard script claims `table.lb` by querySelector, and two
   tables answering to one selector would leave the second one dead. */
table.fnd { width:100%; border-collapse:collapse; min-width:960px; }
table.fnd th { text-align:left; font-size:10px; letter-spacing:.1em;
  text-transform:uppercase; color:var(--faint); font-weight:600; padding:9px 12px;
  border-bottom:1px solid var(--border); background:var(--panel);
  user-select:none; position:sticky; top:0; z-index:2; }
table.fnd th[data-key] { cursor:pointer; }
table.fnd th[data-key]:hover { color:var(--ink); }
table.fnd th .ar { opacity:0; font-size:9px; }
table.fnd th.on .ar { opacity:1; color:var(--accent); }
table.fnd th.r, table.fnd td.r { text-align:right; }
table.fnd td { padding:10px 12px; border-bottom:1px solid var(--border);
  vertical-align:top; }
tbody.fnd:last-child tr:last-child td { border-bottom:none; }
tbody.fnd tr.frow { cursor:pointer; }
tbody.fnd tr.frow:hover td { background:var(--panel-2); }
tbody.fnd.open tr.frow td { background:var(--panel-2); }
tbody.fnd .fdetail { display:none; }
tbody.fnd.open .fdetail { display:table-row; }
.ftitle { font-size:12.5px; font-weight:600; overflow-wrap:anywhere;
  max-width:520px; }
.furl { font-family:var(--mono); font-size:11px; color:var(--faint);
  margin-top:3px; overflow-wrap:anywhere; }
.furl a { color:var(--faint); }
.furl a:hover { color:var(--accent); }
/* Published: the site over the date. Both are sort keys and neither was
   visible in a column before, so a reader could not check the sort against the
   rows. `white-space:nowrap` on the date only -- a host may wrap, a date never
   should. */
table.fnd td.pub { white-space:nowrap; }
.fhost { font-family:var(--mono); font-size:11.5px; overflow-wrap:anywhere;
  white-space:normal; max-width:180px; }
.fhost.none { color:var(--faint); }
.fdate { font-family:var(--mono); font-size:11px; color:var(--faint);
  font-variant-numeric:tabular-nums; margin-top:3px; }
.fnum { font-family:var(--mono); font-variant-numeric:tabular-nums;
  font-size:12.5px; }
.fnum.none { color:var(--faint); }
/* The base of the fraction, held back so the hit count reads first. */
.fden { color:var(--faint); }
.fbody { padding:2px 4px 12px; display:grid;
  grid-template-columns:repeat(auto-fit,minmax(300px,1fr)); gap:14px 22px; }
.fbody h4 { font-size:10px; letter-spacing:.14em; text-transform:uppercase;
  color:var(--faint); margin:0 0 5px; font-weight:700; }
.fbody p { margin:0; color:var(--muted); font-size:12.5px; line-height:1.55; }
.fbody ul { margin:0; padding-left:17px; color:var(--muted); font-size:12.5px;
  line-height:1.6; }
.fbody code { font-family:var(--mono); font-size:.92em;
  background:var(--panel-2); border-radius:4px; padding:0 3px; }
/* A proposal is only meaningful as the code it landed on, so the candidate id is
   demoted to a tooltip and the line carries what a reader can actually judge:
   impact, symbol, file range, and rank where the candidate has one. */
.fbody .pgo { display:flex; align-items:baseline; gap:7px; flex-wrap:wrap;
  margin-top:3px; }
.fbody .pgo a { font-family:var(--mono); font-size:12px; font-weight:600;
  overflow-wrap:anywhere; }
.fbody .prank { font-family:var(--mono); font-size:11px; color:var(--faint);
  white-space:nowrap; }
.fbody .ploc { display:block; font-family:var(--mono); font-size:11px;
  color:var(--faint); overflow-wrap:anywhere; margin-top:1px; }
.fbody li + li { margin-top:9px; }
.fbody .fhint { margin-top:7px; font-size:11px; color:var(--faint); }
/* The proposals block spans the grid. It is a list of rows now, not a paragraph,
   and 300px of column would wrap every symbol onto three lines. */
.fbody .fprops { grid-column:1/-1; }
.fscope { margin:0 0 8px; font-size:11.5px; color:var(--faint); line-height:1.5; }
details.pitem { border:1px solid var(--border); border-radius:8px;
  background:var(--panel-2); margin-top:7px; }
details.pitem > summary { cursor:pointer; list-style:none; padding:9px 11px;
  display:flex; align-items:baseline; gap:7px; flex-wrap:wrap; }
details.pitem > summary::-webkit-details-marker { display:none; }
details.pitem > summary::before { content:"\\25B8"; color:var(--faint);
  font-size:10px; flex:none; }
details.pitem[open] > summary::before { content:"\\25BE"; }
.pitem .ptitle { flex:1 1 100%; font-size:12.5px; font-weight:600;
  color:var(--ink); line-height:1.45; }
.pitem .pwhy { margin:0; padding:0 12px 11px 26px; color:var(--muted);
  font-size:12.5px; line-height:1.55; }
.pitem .pwl { display:block; font-size:10px; letter-spacing:.14em;
  text-transform:uppercase; color:var(--faint); font-weight:700; margin-bottom:4px; }

/* ---------- provenance on a rendered proposal ---------------------------- */
.prop .src { font-family:var(--mono); font-size:10.5px; border-radius:4px;
  padding:1px 6px; border:1px solid var(--border-strong); color:var(--faint);
  white-space:nowrap; }
.prop .src.res { color:var(--accent); border-color:var(--accent);
  background:var(--accent-glow); }
.prop .fref { margin:7px 0 0; font-size:11.5px; color:var(--faint);
  line-height:1.5; }
.prop .fref b { color:var(--muted); font-weight:600; }
.atab td.k { text-align:left; font-family:var(--sans); color:var(--muted); }
"""

JS = """
/* ---------- theme toggle (wins over the OS setting in both directions) ---- */
(function () {
  var btn = document.getElementById('theme');
  if (!btn) return;
  function label() {
    var t = document.documentElement.getAttribute('data-theme');
    btn.textContent = t === 'dark' ? 'light theme' : t === 'light' ? 'dark theme' : 'flip theme';
  }
  btn.addEventListener('click', function () {
    var root = document.documentElement;
    var cur = root.getAttribute('data-theme');
    if (!cur) {
      var dark = window.matchMedia('(prefers-color-scheme: dark)').matches;
      cur = dark ? 'dark' : 'light';
    }
    root.setAttribute('data-theme', cur === 'dark' ? 'light' : 'dark');
    label();
  });
  label();
})();

/* ---------- chart table views (the relief for every chart) --------------- */
document.querySelectorAll('.tvbtn').forEach(function (b) {
  b.addEventListener('click', function () {
    var t = document.getElementById(b.dataset.target);
    if (!t) return;
    t.hidden = !t.hidden;
    b.textContent = t.hidden ? 'table' : 'chart';
    b.setAttribute('aria-expanded', String(!t.hidden));
  });
});

/* ---------- shared hover tooltip; keyboard focus shows the same ---------- */
(function () {
  var tip = document.getElementById('tip');
  if (!tip) return;
  function show(el) {
    var t = el.dataset.tip, v = el.dataset.tipval || '';
    if (!t) return;
    tip.innerHTML = '<div class="tt"></div>' + (v ? '<div class="tv2"></div>' : '');
    tip.firstChild.textContent = t;
    if (v) tip.lastChild.textContent = v;
    tip.classList.add('on');
  }
  function place(x, y) {
    var r = tip.getBoundingClientRect();
    var left = Math.min(x + 14, window.innerWidth - r.width - 8);
    var top = y - r.height - 12;
    if (top < 8) top = y + 18;
    tip.style.left = Math.max(8, left) + 'px';
    tip.style.top = top + 'px';
  }
  function hide() { tip.classList.remove('on'); }
  document.addEventListener('mousemove', function (e) {
    var el = e.target.closest('[data-tip]');
    if (!el) { hide(); return; }
    show(el);
    place(e.clientX, e.clientY);
  });
  document.addEventListener('focusin', function (e) {
    var el = e.target.closest('[data-tip]');
    if (!el) { hide(); return; }
    show(el);
    var r = el.getBoundingClientRect();
    place(r.left + r.width / 2, r.top);
  });
  document.addEventListener('focusout', hide);
  window.addEventListener('scroll', hide, { passive: true });
})();

/* ---------- leaderboard: filter, sort, expand ---------------------------- */
(function () {
  var table = document.querySelector('table.lb');
  if (!table) return;
  var bodies = Array.from(table.querySelectorAll('tbody.cand'));
  var q = document.getElementById('q');
  var modSel = document.getElementById('mod');
  var impSel = document.getElementById('imp');
  var stgSel = document.getElementById('stg');
  var counter = document.getElementById('count');
  var hotBtn = document.getElementById('hotbtn');
  var hot = false;

  var noteEl = document.getElementById('lbnote'), noteT = null;
  function note(t) {
    if (!noteEl) return;
    noteEl.textContent = t;
    if (noteT) { clearTimeout(noteT); noteT = null; }
    if (t) noteT = setTimeout(function () { noteEl.textContent = ''; }, 6000);
  }

  function apply() {
    var text = q.value.trim().toLowerCase();
    var mod = modSel.value, imp = impSel.value, stg = stgSel.value;
    var shown = 0;
    for (var i = 0; i < bodies.length; i++) {
      var b = bodies[i];
      var ok = (!hot || b.dataset.hoton === '1')
        && (!mod || b.dataset.module === mod)
        && (!imp || b.dataset.impact === imp)
        && (!stg || (b.dataset.stage || '').indexOf(stg) >= 0)
        && (!text || (b.dataset.search || '').indexOf(text) >= 0);
      b.style.display = ok ? '' : 'none';
      if (ok) shown++;
    }
    counter.textContent = shown + ' / ' + bodies.length + ' shown';
  }

  var impactOrder = { high: 0, medium: 1, low: 2 };
  var sortKey = null, sortDir = 1;
  function sortBy(key, th) {
    if (sortKey === key) { sortDir = -sortDir; } else { sortKey = key; sortDir = 1; }
    var numeric = (key === 'score' || key === 'rank' || key === 'props');
    var sorted = bodies.slice().sort(function (a, b) {
      var av = a.dataset[key] || '', bv = b.dataset[key] || '';
      if (numeric) {
        av = parseFloat(av); bv = parseFloat(bv);
        if (isNaN(av)) av = -Infinity;
        if (isNaN(bv)) bv = -Infinity;
        return (av - bv) * sortDir;
      }
      if (key === 'impact') {
        av = impactOrder[av] === undefined ? 3 : impactOrder[av];
        bv = impactOrder[bv] === undefined ? 3 : impactOrder[bv];
        return (av - bv) * sortDir;
      }
      return String(av).localeCompare(String(bv)) * sortDir;
    });
    sorted.forEach(function (b) { table.appendChild(b); });
    table.querySelectorAll('th[data-key]').forEach(function (h) {
      h.classList.toggle('on', h === th);
      var ar = h.querySelector('.ar');
      if (ar) ar.textContent = (h === th && sortDir < 0) ? '\\u25B2' : '\\u25BC';
    });
  }
  /* Hot order: the module's hot score descending. High-impact count breaks a
     tie -- two modules the score cannot separate are separated by how much
     high-impact work each actually holds -- then module name, so a module's rows
     stay contiguous instead of interleaving, and rank orders rows inside one. */
  function sortHot() {
    var s2 = bodies.slice().sort(function (a, b) {
      var d = (parseFloat(b.dataset.hot) || 0) - (parseFloat(a.dataset.hot) || 0);
      if (d) return d;
      d = (+b.dataset.hotn || 0) - (+a.dataset.hotn || 0);
      if (d) return d;
      d = String(a.dataset.module).localeCompare(String(b.dataset.module));
      if (d) return d;
      var ar = parseFloat(a.dataset.rank), br = parseFloat(b.dataset.rank);
      if (isNaN(ar)) ar = Infinity;
      if (isNaN(br)) br = Infinity;
      return ar - br;
    });
    s2.forEach(function (b) { table.appendChild(b); });
    sortKey = null;   /* a later header click starts a fresh ascending sort */
    table.querySelectorAll('th[data-key]').forEach(function (h) {
      h.classList.remove('on');
      var ar2 = h.querySelector('.ar');
      if (ar2) ar2.textContent = '';
    });
  }

  /* The module filter has to agree with the rows it filters: in hot mode a cold
     module in the list is a control that can only ever return nothing. The
     options are rebuilt rather than hidden -- option[hidden] is honoured
     unevenly across browsers, and the list is being reordered anyway. */
  var allOpt = modSel.querySelector('option[value=""]');
  var modOpts = [].slice.call(modSel.options).filter(function (o) { return o.value; });
  for (var mi = 0; mi < modOpts.length; mi++) modOpts[mi]._t = modOpts[mi].textContent;

  function setModOpts() {
    /* Read the selection first: detaching the options resets it. */
    var want = modSel.value;
    var list = modOpts;
    if (hot) {
      list = modOpts.filter(function (o) { return o.dataset.hoton === '1'; })
        .sort(function (a, b) {
          var d = (parseFloat(b.dataset.hot) || 0) - (parseFloat(a.dataset.hot) || 0);
          if (d) return d;
          d = (+b.dataset.hotn || 0) - (+a.dataset.hotn || 0);
          return d ? d : String(a.value).localeCompare(String(b.value));
        });
    }
    var i, still = false;
    for (i = 0; i < modOpts.length; i++) {
      modOpts[i].textContent = modOpts[i]._t;
      if (modOpts[i].parentNode) modSel.removeChild(modOpts[i]);
    }
    for (i = 0; i < list.length; i++) {
      if (hot) {
        list[i].textContent = list[i]._t + '  ' +
          Math.round(parseFloat(list[i].dataset.hot)) + '%';
      }
      modSel.appendChild(list[i]);
      if (list[i].value === want) still = true;
    }
    /* A module that just left the list cannot stay selected, or the table would
       show nothing with no visible cause. */
    modSel.value = still ? want : '';
    if (allOpt) allOpt.textContent = hot ? 'all hot modules' : 'all modules';
  }

  if (hotBtn) hotBtn.addEventListener('click', function () {
    hot = !hot;
    hotBtn.setAttribute('aria-pressed', String(hot));
    table.classList.toggle('hotmode', hot);
    setModOpts();
    if (hot) sortHot();
    else sortBy('rank', table.querySelector('th[data-key="rank"]'));
    apply();
    if (window.__rtsethot) window.__rtsethot(hot);
  });

  table.querySelectorAll('th[data-key]').forEach(function (th) {
    th.addEventListener('click', function () { sortBy(th.dataset.key, th); });
  });

  bodies.forEach(function (b) {
    var row = b.querySelector('tr.row');
    row.setAttribute('tabindex', '0');
    function toggle() { b.classList.toggle('open'); }
    row.addEventListener('click', toggle);
    row.addEventListener('keydown', function (e) {
      if (e.key === 'Enter' || e.key === ' ') { e.preventDefault(); toggle(); }
    });
  });

  /* Opening a candidate that the table is currently filtering out.
     The tree deliberately ignores these filters, so a leaf clicked there can be
     a row the table is hiding; switching panes would then land on a
     display:none row and the click would read as dead. The tree asks the
     leaderboard to reveal a candidate rather than reaching into the filter
     controls itself, and a filter is cleared only when it is the thing doing
     the hiding -- a filter the reader set is worth keeping when it costs
     nothing. Returns whether anything had to be cleared, so the caller can say
     so. */
  window.__lbreveal = function (id) {
    var b = table.querySelector('tbody.cand[data-id="' + id + '"]');
    if (!b) return false;
    if (b.style.display !== 'none') return false;
    q.value = ''; modSel.value = ''; impSel.value = ''; stgSel.value = '';
    apply();
    if (b.style.display === 'none' && hot) {
      /* Not reachable while one switch drives both panes' hot mode -- the tree
         only offers hot modules' candidates then, and their rows are the ones
         hot mode keeps. Kept so a row can never lose the argument with a
         filter if those two ever drift apart. */
      hot = false;
      hotBtn.setAttribute('aria-pressed', 'false');
      table.classList.remove('hotmode');
      setModOpts();
      apply();
      if (window.__rtsethot) window.__rtsethot(false);
    }
    note('filters cleared to show this candidate');
    return true;
  };

  /* The whole journey to one row -- clear what hides it, swap back from the tree
     pane, open it, land on it -- in one place, because two panes now ask for it:
     the radial tree's leaves and the findings catalogue's proposal links. Both
     were about to reach for the same four element ids and the same open/scroll
     pair, and a second copy of that is a second thing to forget when the table
     changes. Returns false for an id the table does not hold, which is what lets
     a caller fall back to the plain #fragment. */
  window.__lbjump = function (id) {
    var b = table.querySelector('tbody.cand[data-id="' + id + '"]');
    if (!b) return false;
    window.__lbreveal(id);
    var tbl = document.getElementById('lbtable');
    var tree = document.getElementById('lbtree');
    var btn = document.getElementById('lbview');
    if (tbl && tree && btn) {
      tbl.hidden = false; tree.hidden = true;
      btn.textContent = 'radial tree';
      btn.setAttribute('aria-pressed', 'false');
    }
    b.classList.add('open');
    b.scrollIntoView({ block: 'center' });
    /* Restart the flash on a repeat click. Without the forced reflow the class
       is already present, the animation never replays, and clicking the same
       link twice looks like the second click did nothing. */
    b.classList.remove('flash');
    void b.offsetWidth;
    b.classList.add('flash');
    return true;
  };

  [q, modSel, impSel, stgSel].forEach(function (el) {
    el.addEventListener(el === q ? 'input' : 'change', function () {
      note('');       /* the reader is filtering again; the notice is spent */
      apply();
    });
  });
  apply();
})();

/* ---------- leaderboard view switch: table <-> radial tree --------------- */
(function () {
  var btn = document.getElementById('lbview');
  var tbl = document.getElementById('lbtable');
  var tree = document.getElementById('lbtree');
  if (!btn || !tbl || !tree) return;
  btn.addEventListener('click', function () {
    var toTree = tbl.hidden === false;
    tbl.hidden = toTree;
    tree.hidden = !toTree;
    btn.textContent = toTree ? 'table view' : 'radial tree';
    btn.setAttribute('aria-pressed', String(toTree));
    if (toTree && window.__rtdraw) window.__rtdraw();
  });
})();

/* ---------- radial collapsible candidate tree ---------------------------- */
(function () {
  var plot = document.getElementById('rtplot');
  var src = document.getElementById('rtdata');
  if (!plot || !src) return;
  var root = JSON.parse(src.textContent);

  var IMP = { high: 'var(--accent)', medium: 'var(--muted)', low: 'var(--faint)' };
  var RLEAF = 250, RIN0 = 70, RIN1 = 215, PADMIN = 196, LABMAX = 30;
  /* Advance of the label monospace face, in px per character, measured off a
     rendered page. The outward labels are the only thing that can leave the
     viewBox, so the pad is derived from the widest one drawn rather than fixed:
     a folded group in hot mode carries its name plus " (19 * 32% hot)" and
     overruns any constant that fits the plain view. PADMIN keeps the unfolded
     view exactly the size it was.
     Two faces, because a short pad clips and a long one only adds white space:
     a folded module label is class .br, which is 9.5px with .02em of letter
     spacing, so it advances wider than the 9px leaf face -- and the folded
     labels are precisely the ones the hot suffix lengthens. Measuring every
     label in characters against the narrower face is what cut the ")" off the
     longest one. */
  var CHW = 5.45, CHWB = 5.95;
  var flat = [], hotOnly = false;
  /* The node the view is rooted at. Clicking a branch means "show me this and
     everything under it", not "hide what is under it": the ring is rebuilt from
     the clicked node, so a module's candidates get the whole circle instead of
     the slice they had at run level. The way back out is the breadcrumb above
     the plot and the hub in the middle. */
  var focus = root;
  var crumb = document.getElementById('rtcrumb');

  /* depth, parent, subtree candidate count, subtree high-impact count, and the
     two hot fields, all in one post-order pass.
       _ho  does this subtree contain a hot module?
       _sx  the best hot score in it, which is the node's own score when the node
            IS a hot module.
     Only a node whose path is a module carries s/hn/ho from the generator. An
     intermediate path segment is deliberately unscored: aggregating a hot and a
     cold child can land below the baseline, and scoring the segment on that
     aggregate would prune a hot module out of the view. */
  function tag(n, d, par) {
    n._d = d; n._p = par;
    var own = (n.ho === 1);
    n._sx = own ? (+n.s || 0) : -1;
    n._ho = own;
    if (n.c && n.c.length) {
      var s = 0, h = 0;
      for (var i = 0; i < n.c.length; i++) {
        s += tag(n.c[i], d + 1, n);
        h += n.c[i]._hi;
        if (n.c[i]._ho) { n._ho = true; }
        if (n.c[i]._sx > n._sx) { n._sx = n.c[i]._sx; }
      }
      n._lv = s; n._hi = h;
      return s;
    }
    n._lv = 1; n._hi = (n.i === 'high') ? 1 : 0;
    return 1;
  }
  tag(root, 0, null);

  function isGroup(n) { return !!(n.c && n.c.length); }

  /* Hot mode keeps a branch if it contains a hot module and orders branches by
     the best hot score inside them, so the hottest module leads its ring and
     leads the branch that holds it. High-impact count breaks a tie, matching the
     table. Candidate leaves are never dropped: "hot modules" means every
     candidate of a hot module, not only its high-impact ones. */
  function vis(n) {
    var c = n.c || [];
    if (!hotOnly) return c;
    var g = [], l = [];
    for (var i = 0; i < c.length; i++) {
      if (!isGroup(c[i])) { l.push(c[i]); }
      else if (c[i]._ho) { g.push(c[i]); }
    }
    g.sort(function (a, b) {
      return (b._sx - a._sx) || (b._hi - a._hi) ||
             String(a.p).localeCompare(String(b.p));
    });
    return g.concat(l);
  }
  function isOpen(n) {
    /* Whatever fold state the focus carried when it was a mere branch, it is
       open now -- it is the thing being looked at. */
    if (n === focus) return isGroup(n) && !!(n._k && n._k.length);
    return isGroup(n) && !n._col && !!(n._k && n._k.length);
  }

  function chain(n) {
    var c = [];
    while (n) { c.unshift(n); n = n._p; }
    return c;
  }
  /* Is this node still on a visible path from the run? Hot mode drops cold
     groups, and a focus sitting inside one has to be let go of. */
  function reachable(n) {
    while (n && n !== root) {
      var k = vis(n._p), ok = false;
      for (var i = 0; i < k.length; i++) if (k[i] === n) ok = true;
      if (!ok) return false;
      n = n._p;
    }
    return n === root;
  }

  function esc(t) {
    return String(t == null ? '' : t).replace(/&/g, '&amp;')
      .replace(/</g, '&lt;').replace(/>/g, '&gt;').replace(/"/g, '&quot;');
  }
  function trunc(t, k) {
    t = String(t == null ? '' : t);
    return t.length > k ? t.slice(0, k - 1) + '…' : t;
  }
  function pos(r, a) {
    var t = (a - 90) * Math.PI / 180;
    return [r * Math.cos(t), r * Math.sin(t)];
  }
  function f1(v) { return v.toFixed(1); }

  /* the outer ring is candidates plus any collapsed group standing in for its
     subtree; a collapsed group therefore reads exactly like a leaf. */
  function layout() {
    var ring = [], maxd = 0;
    (function walk(n) {
      var rd = n._d - focus._d;
      if (rd > maxd) maxd = rd;
      if (isOpen(n)) { for (var i = 0; i < n._k.length; i++) walk(n._k[i]); }
      else if (n !== focus) { ring.push(n); }
    })(focus);
    var N = ring.length || 1;
    for (var i = 0; i < N; i++) ring[i]._a = (i + 0.5) * 360 / N;
    (function up(n) {
      if (!isOpen(n)) return;
      for (var i = 0; i < n._k.length; i++) up(n._k[i]);
      n._a = (n._k[0]._a + n._k[n._k.length - 1]._a) / 2;
    })(focus);
    return { ring: ring, maxd: Math.max(1, maxd) };
  }

  function draw() {
    (function prep(n) {
      if (!(n.c && n.c.length)) return;
      n._k = vis(n);
      for (var i = 0; i < n.c.length; i++) prep(n.c[i]);
    })(root);
    /* A focus whose children have all been filtered away has nothing to show,
       so climb until there is something. */
    while (focus !== root && !(focus._k && focus._k.length)) focus = focus._p;
    var L = layout(), maxd = L.maxd, parts = [], labs = [], maxw = 0;
    /* How much of a candidate name fits depends on how many of them share the
       circle. LABMAX is the crowded run-level budget; zoom into one module and
       seven names have the whole circumference, so truncating them to 30
       characters throws away the readability the zoom was for. The pad is
       derived from the widest label drawn, so a longer budget widens the box
       instead of overrunning it. */
    var lim = L.ring.length > 60 ? LABMAX : (L.ring.length > 28 ? 46 : 72);
    var span = Math.max(1, maxd - 2);
    flat = [];

    /* Dendrogram convention: every ring member sits on the SAME outer circle.
       Placing nodes at depth * step instead crushes the candidates of a shallow
       module into a small radius that has no circumference to hold their
       labels. Only the open path nodes are spread by depth, and across an inner
       annulus that starts at RIN0 so the hub text stays clear. */
    function rad_(n) {
      if (!isOpen(n)) return RLEAF;
      if (n === focus) return 0;
      return RIN0 + (n._d - focus._d - 1) / span * (RIN1 - RIN0);
    }

    (function link(n) {
      if (!isOpen(n)) return;
      var pr = rad_(n), p = pos(pr, n._a);
      for (var i = 0; i < n._k.length; i++) {
        var k = n._k[i], kr = rad_(k), m = (pr + kr) / 2;
        var c1 = pos(m, n._a), c2 = pos(m, k._a), e = pos(kr, k._a);
        parts.push('<path class="lnk" d="M' + f1(p[0]) + ',' + f1(p[1]) +
          'C' + f1(c1[0]) + ',' + f1(c1[1]) + ' ' + f1(c2[0]) + ',' + f1(c2[1]) +
          ' ' + f1(e[0]) + ',' + f1(e[1]) + '"/>');
        link(k);
      }
    })(focus);

    var all = [];
    (function walk(n) {
      all.push(n);
      if (isOpen(n)) { for (var i = 0; i < n._k.length; i++) walk(n._k[i]); }
    })(focus);

    for (var i = 0; i < all.length; i++) {
      var n = all[i];
      if (n === focus) continue;
      var grp = isGroup(n), shut = grp && !!n._col, leafish = !grp || shut;
      var r = rad_(n), a = n._a, xy = pos(r, a);
      var rad = grp ? (shut ? 5 : 3.5) : 3;
      var fill = grp ? (shut ? 'var(--accent)' : 'var(--panel)')
                     : (IMP[n.i] || 'var(--faint)');
      var stroke = grp ? 'var(--border-strong)' : 'none';
      var tip, tval;
      if (grp) {
        tip = n.q || n.p;
        tval = n._lv + (n._lv === 1 ? ' candidate · ' : ' candidates · ') +
          'click to zoom in';
      } else {
        tip = n.n;
        tval = (n.k == null ? '' : '#' + n.k + ' · ') + n.i + ' impact · ' + n.m;
      }
      var idx = flat.length;
      flat.push(n);
      parts.push('<g class="nd" tabindex="0" role="button" data-k="' + idx +
        '" data-tip="' + esc(tip) + '" data-tipval="' + esc(tval) +
        '" transform="translate(' + f1(xy[0]) + ',' + f1(xy[1]) + ')">' +
        '<circle class="dot" r="' + rad + '" fill="' + fill + '" stroke="' +
        stroke + '" stroke-width="1"/>' +
        '<circle r="9" fill="transparent"/></g>');

      /* labels: leafish outward, an open group inward and only when its span
         is wide enough to carry text without colliding with its siblings. */
      var flip = a > 180, txt, cls, dx, anc, rr, outward = false;
      if (leafish) {
        outward = true;
        txt = shut
          ? n.p + ' (' + n._lv +
            ((hotOnly && n.s != null) ? ' · ' + Math.round(n.s) + '% hot' : '') + ')'
          : trunc(n.n, lim);
        cls = shut ? 'br' : '';
        rr = r; dx = flip ? -(rad + 5) : (rad + 5);
        anc = flip ? 'end' : 'start';
      } else if (n._d - focus._d <= 2 && n._lv >= 4) {
        /* Only the shallow groups are named in place. A deep group's label sits
           at a radius close to the leaf ring and collides with it; its name is
           still one hover away, and "collapse to modules" gives every module a
           leaf-style label of its own. */
        txt = n.p; cls = 'br';
        rr = r; dx = flip ? -(rad + 6) : (rad + 6);
        anc = flip ? 'end' : 'start';
      } else { continue; }
      if (outward) {
        var w = txt.length * (cls === 'br' ? CHWB : CHW);
        if (w > maxw) maxw = w;
      }
      labs.push('<text class="' + (cls ? cls + ' lab' : 'lab') +
        '" data-k="' + idx + '" data-tip="' + esc(tip) +
        '" data-tipval="' + esc(tval) +
        '" text-anchor="' + anc + '" x="' + f1(dx) + '" transform="rotate(' +
        f1(a - 90) + ') translate(' + f1(rr) + ',0)' +
        (flip ? ' rotate(180)' : '') + '">' + esc(txt) + '</text>');
    }

    var shown = (function cnt(n) {
      if (!(n.c && n.c.length)) return 1;
      var s2 = 0, k = n._k || [];
      for (var i = 0; i < k.length; i++) s2 += cnt(k[i]);
      return s2;
    })(focus);
    var PAD = Math.max(PADMIN, Math.ceil(maxw) + 12);
    var vb = -(RLEAF + PAD), side = 2 * (RLEAF + PAD);
    /* The hub is built from a list rather than fixed offsets because it now has
       between two and four lines: the count, what is being counted, the hot
       qualifier, and -- once the view is zoomed -- the way back up. Lines are
       centred on the same y the two-line version used, so the plain view is
       unchanged. */
    var hl = [String(shown), 'candidates'];
    if (hotOnly) hl.push('hot modules');
    if (focus !== root) hl.push('↑ up one level');
    var hub = '';
    for (var hi = 0; hi < hl.length; hi++) {
      hub += '<text class="hub" y="' +
        f1(6.5 + (hi - (hl.length - 1) / 2) * 13) + '">' + esc(hl[hi]) +
        '</text>';
    }
    /* Under the text, because the text is pointer-events:none and the click has
       to land on something. RIN0 keeps every drawn node well outside r=44. */
    if (focus !== root) {
      hub = '<circle class="hubhit" r="44" fill="transparent" data-up="1"/>' + hub;
    }
    plot.innerHTML = '<svg class="rtsvg" viewBox="' + vb + ' ' + vb + ' ' +
      side + ' ' + side + '" role="img" aria-label="Radial tree of ' +
      shown + ' candidates grouped by module path">' +
      parts.join('') + labs.join('') + hub + '</svg>';
    drawCrumb();
  }

  /* isOpen() already ignores the fold state of the focus -- the thing being
     looked at is open by definition -- so this must NOT clear _col. Clearing it
     would silently unfold the node, and coming back up would land on a view
     reshaped by the trip: zoom into a module from "collapse to modules", come
     back, and it would no longer be a folded dot like its siblings. */
  function setFocus(n) {
    if (!n || !isGroup(n)) return;
    focus = n; draw();
  }

  function drawCrumb() {
    if (!crumb) return;
    var c = chain(focus), out = [], i;
    for (i = 0; i < c.length; i++) {
      var nm = (c[i] === root) ? 'whole run' : c[i].p;
      out.push(i === c.length - 1
        ? '<span class="here">' + esc(nm) + '</span>'
        : '<button type="button" data-c="' + i + '">' + esc(nm) + '</button>');
      if (i < c.length - 1) out.push('<span class="sep">/</span>');
    }
    out.push('<span class="sep">&middot; ' + (c.length === 1
      ? 'click a branch to zoom into it'
      : 'showing ' + c[c.length - 1]._lv + ' of ' + root._lv +
        ' candidates') + '</span>');
    crumb.innerHTML = out.join('');
    var bs = crumb.querySelectorAll('button');
    for (i = 0; i < bs.length; i++) {
      (function (node, b) {
        b.addEventListener('click', function () { setFocus(node); });
      })(c[+bs[i].dataset.c], bs[i]);
    }
  }

  function setAll(v) {
    (function walk(n) {
      if (!isGroup(n)) return;
      if (n._d > 0 && n !== focus) n._col = v;
      for (var i = 0; i < n.c.length; i++) walk(n.c[i]);
    })(root);
  }
  /* "collapse to modules" shuts exactly the groups that own candidates, so the
     outer ring becomes the module list rather than an arbitrary depth cut. */
  function toModules() {
    (function walk(n) {
      if (!isGroup(n)) return;
      var owns = false;
      for (var i = 0; i < n.c.length; i++) if (!isGroup(n.c[i])) owns = true;
      n._col = (n._d > 0 && owns && n !== focus);
      for (var j = 0; j < n.c.length; j++) walk(n.c[j]);
    })(root);
  }

  /* The tree shows every candidate and the table is filtered, so a leaf clicked
     here can be a row the table is hiding -- which is exactly what __lbjump
     handles, filters and pane swap included. */
  function openRow(id) {
    if (window.__lbjump) window.__lbjump(id);
  }

  function hit(e) {
    if (e.target.closest && e.target.closest('[data-up]')) {
      setFocus(focus._p || root);
      return;
    }
    /* Either the node group or its label -- both carry data-k, so the reader
       can aim at the name instead of hunting the dot. */
    var g = e.target.closest ? e.target.closest('[data-k]') : null;
    if (!g) return;
    var n = flat[+g.dataset.k];
    if (!n) return;
    if (isGroup(n)) setFocus(n);
    else if (n.d) openRow(n.d);
  }
  plot.addEventListener('click', hit);
  plot.addEventListener('keydown', function (e) {
    if (e.key !== 'Enter' && e.key !== ' ') return;
    e.preventDefault(); hit(e);
  });

  var bx = document.getElementById('rtexp');
  var bc = document.getElementById('rtmod');
  if (bx) bx.addEventListener('click', function () { setAll(false); draw(); });
  if (bc) bc.addEventListener('click', function () { toModules(); draw(); });

  window.__rtdraw = draw;
  window.__rtsethot = function (v) {
    hotOnly = !!v;
    /* A cold module cannot stay on screen once hot mode is on, so a focus
       inside one goes back to the run rather than showing a subtree the rest
       of the page is filtering out. */
    if (!reachable(focus)) focus = root;
    draw();
  };
  window.__rtfocus = function () { return focus; };
  draw();
})();

/* ---- hot modules in the per-module chart -------------------------------- *
 * Independent of the leaderboard's toggle: they are separate sections and a
 * reader who reorders one has not asked to reorder the other. Both read the
 * same data-hot key, so the two orders agree when both are on.              */
(function () {
  var btn = document.getElementById('modhot');
  if (!btn || !btn.closest) return;
  var chart = btn.closest('section.chart');
  var bars = chart ? chart.querySelector('.bars') : null;
  var tv = document.getElementById('tv-mods');
  var tb = tv ? tv.querySelector('tbody') : null;
  if (!bars) return;

  /* Captured in document order so toggling off restores the by-total sort
     exactly, without re-deriving it from the totals. */
  var rows = [].slice.call(bars.children);
  var trs = tb ? [].slice.call(tb.children) : [];
  var hot = false;

  function reorder(list, parent) {
    var s2 = list.slice().sort(function (a, b) {
      var d = (parseFloat(b.dataset.hot) || 0) - (parseFloat(a.dataset.hot) || 0);
      if (d) return d;
      d = (+b.dataset.hotn || 0) - (+a.dataset.hotn || 0);
      return d ? d : String(a.dataset.module).localeCompare(String(b.dataset.module));
    });
    for (var i = 0; i < s2.length; i++) parent.appendChild(s2[i]);
  }
  function restore(list, parent) {
    for (var i = 0; i < list.length; i++) parent.appendChild(list[i]);
  }

  btn.addEventListener('click', function () {
    hot = !hot;
    btn.setAttribute('aria-pressed', String(hot));
    bars.classList.toggle('hotmode', hot);
    if (tv) tv.classList.toggle('hotmode', hot);
    if (hot) { reorder(rows, bars); if (tb) reorder(trs, tb); }
    else { restore(rows, bars); if (tb) restore(trs, tb); }
    var all = rows.concat(trs);
    for (var i = 0; i < all.length; i++) {
      var cold = hot && all[i].dataset.hoton !== '1';
      /* Bars hide by display, table rows by a class the .tv rule owns: a <tr>
         with display:none set inline would fight the table's own layout. */
      if (all[i].tagName === 'TR') all[i].classList.toggle('dim', cold);
      else all[i].style.display = cold ? 'none' : '';
    }
  });
})();
"""


JS += """
/* ---------- findings catalogue: filter, sort, expand --------------------- */
(function () {
  var table = document.querySelector('table.fnd');
  if (!table) return;
  var bodies = Array.from(table.querySelectorAll('tbody.fnd'));
  var q = document.getElementById('fq');
  var srcSel = document.getElementById('fsrc');
  var modSel = document.getElementById('fmod');
  var useSel = document.getElementById('fuse');
  var sortSel = document.getElementById('fsort');
  var counter = document.getElementById('fcount');
  /* Document order is a sort option, not just the starting state, so it has to
     be kept: once the rows have been reordered there is nothing left to derive
     it from. */
  var order0 = bodies.slice();

  function apply() {
    var text = q.value.trim().toLowerCase();
    var src = srcSel.value, mod = modSel.value, use = useSel.value;
    var shown = 0;
    for (var i = 0; i < bodies.length; i++) {
      var b = bodies[i];
      var used = b.dataset.used === '1';
      var ok = (!src || b.dataset.src === src)
        && (!mod || b.dataset.module === mod)
        && (!use || (use === 'used') === used)
        && (!text || (b.dataset.search || '').indexOf(text) >= 0);
      b.style.display = ok ? '' : 'none';
      if (ok) shown++;
    }
    counter.textContent = shown + ' / ' + bodies.length + ' shown';
  }

  /* The direction a key is worth reading first: dates and counts descend, text
     ascends. A header click flips whatever is current; the menu states its
     direction outright, since "newest first" is the thing being asked for and
     an invisible toggle would hide it. */
  var DIR0 = {pub: -1, props: -1, rate: -1, src: 1, host: 1, module: 1,
              title: 1};
  var sortKey = null, sortDir = 1;

  function sortBy(key, dir) {
    sortKey = key || null;
    sortDir = dir || 1;
    if (!sortKey) {
      order0.forEach(function (b) { table.appendChild(b); });
    } else {
      var numeric = (sortKey === 'props' || sortKey === 'rate');
      bodies.slice().sort(function (a, b) {
        var av = a.dataset[sortKey] || '', bv = b.dataset[sortKey] || '';
        /* Missing values sink in BOTH directions. Otherwise "newest first" on a
           run where 3 of 578 findings are dated opens with 575 blanks, which
           reads as a broken control rather than a sparse field. */
        if (!av !== !bv) return av ? -1 : 1;
        if (numeric) return ((parseFloat(av) || 0) - (parseFloat(bv) || 0)) * sortDir;
        return String(av).localeCompare(String(bv)) * sortDir;
      }).forEach(function (b) { table.appendChild(b); });
    }
    /* One state, two controls: whichever was used, the other has to show it, or
       the menu will claim a sort the headers have since replaced. */
    table.querySelectorAll('th[data-key]').forEach(function (h) {
      var on = h.dataset.key === sortKey;
      h.classList.toggle('on', on);
      var ar = h.querySelector('.ar');
      if (ar) ar.textContent = (on && sortDir < 0) ? '\\u25B2' : '\\u25BC';
    });
    if (sortSel) {
      var want = sortKey ? sortKey + ':' + sortDir : 'doc';
      var has = Array.prototype.some.call(sortSel.options, function (o) {
        return o.value === want;
      });
      sortSel.value = has ? want : '';
    }
  }

  table.querySelectorAll('th[data-key]').forEach(function (th) {
    th.addEventListener('click', function () {
      var k = th.dataset.key;
      sortBy(k, sortKey === k ? -sortDir : (DIR0[k] || 1));
    });
  });
  if (sortSel) {
    sortSel.addEventListener('change', function () {
      if (!sortSel.value) return;            /* the placeholder is not an action */
      var v = sortSel.value.split(':');
      sortBy(v[0] === 'doc' ? null : v[0], parseInt(v[1], 10) || 1);
    });
  }

  /* A finding's proposals link to the leaderboard rows they landed on. The hrefs
     are real fragments, so they still work with scripting off; the handler is
     what makes them land somewhere visible rather than on a filtered-out row.
     Delegated from the table, because those links live in detail rows that are
     display:none until the reader opens one. */
  table.addEventListener('click', function (e) {
    var a = e.target.closest && e.target.closest('a[data-cand]');
    if (!a) return;
    /* Unconditional preventDefault, because the link sits inside a <summary>:
       toggling that open is the same click's default action, so following the
       link would otherwise also collapse the proposal the reader is reading.
       The fragment is then applied by hand for the case the leaderboard cannot
       take the jump, so the link still goes somewhere either way. */
    e.preventDefault();
    if (!(window.__lbjump && window.__lbjump(a.dataset.cand)))
      location.hash = a.dataset.cand;
  });

  bodies.forEach(function (b) {
    var row = b.querySelector('tr.frow');
    if (!row) return;
    row.setAttribute('tabindex', '0');
    function toggle() { b.classList.toggle('open'); }
    /* The outbound source link is the one thing in the row that is not the
       expander: following it must not also open the row behind the new tab. */
    row.addEventListener('click', function (e) {
      if (e.target.closest('a')) return;
      toggle();
    });
    row.addEventListener('keydown', function (e) {
      if (e.key === 'Enter' || e.key === ' ') { e.preventDefault(); toggle(); }
    });
  });

  [q, srcSel, modSel, useSel].forEach(function (el) {
    el.addEventListener(el === q ? 'input' : 'change', apply);
  });
  apply();
})();
"""


# --------------------------------------------------------------------------- #
# load
# --------------------------------------------------------------------------- #

IMPACT_ORDER = {"high": 0, "medium": 1, "low": 2}
STATUS_ORDER = ["SUCCEEDED", "DEGRADED", "FAILED", "SKIPPED"]


def _read_json(p: Path):
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def _cand_ids_under(root: Path) -> set[str]:
    """Candidate ids that have an artifact tree under evolve/ or apply/."""
    out: set[str] = set()
    if not root.is_dir():
        return out
    for module_dir in root.iterdir():
        if not module_dir.is_dir():
            continue
        for cand_dir in module_dir.iterdir():
            if cand_dir.is_dir() and cand_dir.name.startswith("cand-"):
                out.add(cand_dir.name)
    return out


def _span0(locations):
    """(symbol, file, line_start, line_end) from the first location/span."""
    if not locations:
        return None, None, None, None
    loc = locations[0] or {}
    spans = loc.get("spans") or []
    if not spans:
        return None, loc.get("file"), None, None
    s = spans[0] or {}
    return s.get("symbol"), loc.get("file"), s.get("line_start"), s.get("line_end")


# --------------------------------------------------------------------------- #
# deep-research layer
#
# The `deep-research` pipeline carries a stage the older pipelines do not: each
# module is researched against the objective, producing FINDINGS (a work that was
# read, plus the technique extracted from it), and every (candidate, finding)
# pair the run chooses to evaluate is then handed to a proposal writer. Until
# this section existed the findings were absent from the page entirely, and every
# rendered proposal dropped the two fields that say where it came from --
# `source` and `finding_ref_id`.
#
# Everything here is derived from the artifact. Three facts the code has to
# respect rather than assume:
#   * Findings carry no module field. The module is recoverable only from the id
#     (`find-<module_qualified_name with "/" -> "_">-NNNN`), and module names
#     themselves contain "-", so the id is matched longest-first against the
#     run's own module list instead of being split on the last "-".
#   * The pair grid is NOT candidates x findings. On
#     iocr/page-latency-2026-08-18-idanfr it is 1,215 pairs where the cross
#     product would be 578 x 159, so the count is read from the per-pair
#     telemetry and never multiplied out.
#   * `publication_date` is present on some runs only, so it is optional.
# --------------------------------------------------------------------------- #

# Label -> per_module_telemetry key. Order is the pipeline's own order, which is
# also the order the stacked bar reads left to right.
STAGES = [
    ("Module discovery", "discovery_total_duration_s"),
    ("Deep research", "deep_research_duration_s"),
    ("Finding to proposal", "proposal_from_finding_duration_s"),
    ("Agent proposals", "agent_proposals_duration_s"),
]

# Source types the researcher emits, ordered strongest-provenance first rather
# than by count, so the chart's row order is stable across runs.
SOURCE_ORDER = ["paper", "docs", "blog", "issue", "pr", "codebase", "other"]


def _work_key(title: str) -> str:
    """Group the findings that read the same work.

    Keyed on the title, not the URL: the same work reaches two modules under
    different URLs -- a scraper mirror, a pinned docs version, arXiv against the
    publisher -- so URL equality finds only a third of the works two modules
    actually share.
    """
    return " ".join(re.sub(r"[^a-z0-9 ]+", " ", (title or "").lower()).split())


def _module_matcher(names):
    """Build finding_id -> module_qualified_name, by longest match.

    A module name can contain "-" (`vllm-core`) and can be a prefix of another
    module's name, so neither rsplit("-") nor a first match is safe; the longest
    module slug that the id continues with "-" is the right one.
    """
    slug = {qn.replace("/", "_"): qn for qn in names if qn}

    def of(finding_id: str) -> str:
        body = (finding_id or "").removeprefix("find-")
        best = ""
        for cand in slug:
            if (len(cand) > len(best) and body.startswith(cand)
                    and body[len(cand):].startswith("-")):
                best = cand
        return slug.get(best, "")

    return of


def _host(url: str) -> str:
    """The site a finding was published on, as a sort key and a group label.

    This is the closest thing the artifact has to a publication venue. The
    finding schema records `title`, `url`, `source_type`, `technique_summary`,
    `supporting_evidence` and an optional `publication_date` -- and nothing else:
    no author, no venue, no journal, no conference, anywhere in `result.json`.
    (`proposals[].author` exists but names the *agent* that wrote the proposal,
    not a paper's author.) So arxiv.org groups the preprints and usenix.org the
    USENIX papers, but NSDI cannot be told from OSDI: the host is where it was
    published, not what published it. 100% of the corpus's 4,540 findings carry
    a URL, so this key is never empty in practice.
    """
    try:
        h = (urlparse(url).netloc or "").lower()
    except ValueError:
        return ""
    return h[4:] if h.startswith("www.") else h


# A pair whose agent call errored. The step records one issue per failed pair
# naming both ids, so the failures are recoverable exactly rather than inferred
# from a missing proposal: `agent failure (candidate_id=..., finding_id=...): ...`.
# The `agent_proposals` stage logs a similar line with `agent=` in place of
# `finding_id=`, which this deliberately does not match.
_PAIR_FAIL = re.compile(
    r"^agent failure \(candidate_id=([^,]+), finding_id=([^)]+)\): (.*)$", re.S)


def research_layer(result: dict, rows) -> dict | None:
    """The deep-research aggregates, or None when the run has no findings.

    Returning None is what keeps every non-research run rendering exactly as it
    did before: each consumer of this dict is gated on it.

    Side effect, deliberate: a proposal that resolves to a finding gets an
    `_finding` key attached, so the per-candidate detail panel can name its
    source without `detail()` having to grow a second argument.
    """
    report = result.get("report") or {}
    findings = report.get("findings") or []
    if not findings:
        return None

    names = set((result.get("module_runs") or {}).keys())
    names |= {r["module"] for r in rows if r["module"]}
    mod_of = _module_matcher(names)

    F = {}
    for f in findings:
        fid = f.get("finding_id") or ""
        F[fid] = {
            "id": fid,
            "module": mod_of(fid),
            "title": f.get("title") or "(untitled)",
            "url": f.get("url") or "",
            "source_type": (f.get("source_type") or "other").lower(),
            "summary": f.get("technique_summary") or "",
            "evidence": f.get("supporting_evidence") or "",
            "published": f.get("publication_date") or "",
            "work": _work_key(f.get("title") or ""),
            # Candidates this finding was paired with, filled in from the pair
            # telemetry below. A set, because the key is what makes it countable
            # and a candidate appears once per finding.
            "cands": set(),
            # Pairings that never returned a verdict: the agent call failed, so
            # "no proposal" here is an error, not a judgement of fit.
            "failed": set(),
            "props": [],
        }

    # ---- proposals, back onto the finding each one cites ------------------
    grounded = ungrounded = dangling = 0
    for r in rows:
        for p in r["proposals"]:
            if (p.get("source") or "") == "research_finding":
                grounded += 1
            else:
                ungrounded += 1
            ref = p.get("finding_ref_id")
            if not ref:
                continue
            hit = F.get(ref)
            if hit is None:
                dangling += 1
                continue
            # The candidate's own leaderboard identity travels with the
            # proposal so the catalogue can name the code it landed on. Module is
            # deliberately absent: the pair grid is built per module, so a
            # candidate's module always equals its finding's -- verified 0
            # mismatches over 3,865 linked proposals across the corpus -- and
            # repeating it in the row that already shows it says nothing.
            hit["props"].append({"cand": r["id"], "symbol": r["symbol"],
                                 "title": p.get("title") or "Untitled",
                                 "rank": r["rank"], "impact": r["impact"],
                                 "loc": loc_str(r), "showrank": False,
                                 # The per-pair bridge: why this technique
                                 # applied to this candidate. It is the only
                                 # proposal field that differs between the
                                 # proposals of one finding -- `description`
                                 # differs too but belongs to the candidate
                                 # panel, and `mechanism`,
                                 # `required_changes`, `expected_effect` and
                                 # `evaluation_metric` are null on all 8,664
                                 # proposals in the corpus.
                                 "why": p.get("rationale") or ""})
            p["_finding"] = hit

    # Ranking is all-or-nothing per run: 8 of the 19 runs in the corpus rank
    # every candidate and the other 8 rank none, because the rank comes from a
    # sorted shortlist that a run either produced or did not. So an "unranked"
    # label on every link is the same empty repetition as a constant provenance
    # chip -- the chip appears only where a rank is a distinguishing fact.
    if any(r["rank"] is not None for r in rows):
        for f in F.values():
            for pp in f["props"]:
                pp["showrank"] = True

    # The provenance chip earns its place only where provenance actually varies.
    # A run with no findings has one kind of proposal, and a chip repeated on all
    # of them says nothing; this flag is what keeps those pages unchanged.
    if grounded and ungrounded:
        for r in rows:
            for p in r["proposals"]:
                p["_prov"] = True

    # ---- the pair grid and the stage budget ------------------------------
    # Per module as well as in total, because the total alone misrepresents the
    # grid. Run-wide, 2,064 pairs against 173 candidates x 97 findings looks like
    # heavy pruning; per module it is 21 x 15 exactly, eight times over. The
    # shortfall is partitioning -- a candidate is never paired with another
    # module's findings -- and that is only visible one module at a time.
    ncand = Counter(r["module"] for r in rows)
    nfind = Counter(f["module"] for f in F.values())
    nprop = Counter(r["module"] for r in rows for p in r["proposals"]
                    if (p.get("source") or "") == "research_finding")

    telemetry = result.get("per_module_telemetry") or {}
    totals = {k: 0.0 for _label, k in STAGES}
    pairs = 0
    pair_fails = 0
    pair_accum_s = 0.0
    per_module = []
    for mod in sorted(telemetry):
        t = telemetry[mod] or {}
        per_pair = t.get("proposal_from_finding_per_pair_durations_s") or {}
        # A pair key is `<candidate id>__<finding id>`, so the pairs a finding
        # was actually in are countable per finding rather than inferred from its
        # module's candidate count. The two differ: under a per-module pair cap
        # the truncation lands mid-candidate, and 20 of idanfr's 25 modules have
        # findings paired with different numbers of candidates as a result.
        fails = 0
        for it in (t.get("issues") or []):
            m = _PAIR_FAIL.match(str((it or {}).get("message") or ""))
            if not m:
                continue
            hit = F.get(m.group(2))
            if hit is not None:
                hit["failed"].add(m.group(1))
                fails += 1
        pair_fails += fails
        for key in per_pair:
            _c, _sep, _f = key.partition("__find-")
            hit = F.get(f"find-{_f}") if _sep else None
            if hit is not None:
                hit["cands"].add(_c)
        acc = 0.0
        for v in per_pair.values():
            try:
                acc += float(v or 0.0)
            except (TypeError, ValueError):
                pass
        pairs += len(per_pair)
        pair_accum_s += acc
        # Stage totals cover every module, pairs or not: a module can be
        # discovered and researched and still contribute no pair.
        for _label, k in STAGES:
            totals[k] += float(t.get(k) or 0.0)
        if not per_pair:
            continue
        per_module.append({
            "module": mod,
            "cands": ncand.get(mod, 0),
            "findings": nfind.get(mod, 0),
            "pairs": len(per_pair),
            "cross": ncand.get(mod, 0) * nfind.get(mod, 0),
            "props": nprop.get(mod, 0),
            "fails": fails,
            "accum_s": acc,
            "stage_s": float(t.get("proposal_from_finding_duration_s") or 0.0),
        })
    per_module.sort(key=lambda x: (-x["pairs"], x["module"]))
    stages = [(label, totals[k]) for label, k in STAGES if totals[k]]

    # A module whose pair count falls short of its own candidates x findings was
    # cut off by something. When every short module stopped at the same number,
    # that number is a per-module pair budget and worth naming -- on the idanfr
    # run 23 of 25 modules stop at exactly 50. One short module is not evidence
    # of a rule, so the cap is only claimed from two.
    short = [x for x in per_module if x["pairs"] < x["cross"]]
    cap = None
    if len(short) > 1 and len({x["pairs"] for x in short}) == 1:
        cap = short[0]["pairs"]

    # ---- yield by source type --------------------------------------------
    by_source: dict[str, dict[str, Any]] = {}
    for f in F.values():
        b = by_source.setdefault(
            f["source_type"], {"type": f["source_type"], "n": 0, "used": 0, "props": 0})
        b["n"] += 1
        b["props"] += len(f["props"])
        if f["props"]:
            b["used"] += 1
    rank = {s: i for i, s in enumerate(SOURCE_ORDER)}
    by_source_rows = sorted(by_source.values(),
                            key=lambda b: (rank.get(b["type"], 99), -b["n"]))

    # ---- works more than one module read --------------------------------
    works: dict[str, list[dict[str, Any]]] = {}
    for f in F.values():
        works.setdefault(f["work"], []).append(f)
    shared = [sorted(g, key=lambda f: (f["module"], f["id"]))
              for g in works.values() if len({f["module"] for f in g}) > 1]
    shared.sort(key=lambda g: (-len({f["module"] for f in g}), g[0]["title"]))

    return {
        "findings": sorted(F.values(), key=lambda f: (f["module"], f["id"])),
        "by_id": F,
        "by_source": by_source_rows,
        "works": len(works),
        "shared": shared,
        "orphans": [f for f in F.values() if not f["props"]],
        "researched": sorted({f["module"] for f in F.values() if f["module"]}),
        "pairs": pairs,
        "pair_fails": pair_fails,
        "per_module": per_module,
        "short": len(short),
        "cap": cap,
        "pair_accum_s": pair_accum_s,
        "pair_stage_s": totals["proposal_from_finding_duration_s"],
        "stages": stages,
        "grounded": grounded,
        "ungrounded": ungrounded,
        "dangling": dangling,
    }


def load(run: Path) -> dict:
    result = _read_json(run / "result.json")
    if result is None:
        raise FileNotFoundError(f"no readable result.json under {run}")
    manifest = _read_json(run / "run_manifest.json") or {}
    ranking = _read_json(run / "sorted" / "sorted_candidates.json") or {}

    report = result.get("report") or {}
    ctx = report.get("context") or {}
    run_meta = report.get("run") or {}

    overlay = {}
    for c in ranking.get("candidates") or []:
        cid = c.get("id")
        if cid:
            overlay[cid] = c
    ranked = bool(overlay)

    evolved = _cand_ids_under(run / "evolve")
    applied = _cand_ids_under(run / "apply")

    # ---- per-module status -------------------------------------------------
    module_runs = result.get("module_runs") or {}
    status_of = {}
    for qn, mrun in module_runs.items():
        status_of[qn] = ((mrun or {}).get("status") or "UNKNOWN").upper()

    # ---- rows -------------------------------------------------------------
    rows = []
    for c in report.get("candidates") or []:
        cid = c.get("id") or ""
        symbol, file, ls, le = _span0(c.get("locations") or [])
        ov = overlay.get(cid) or {}
        module = c.get("module_qualified_name") or ""
        rows.append({
            "id": cid,
            "rank": ov.get("rank"),
            "score": ov.get("score"),
            "module": module,
            "module_status": status_of.get(module, "UNKNOWN"),
            "symbol": ov.get("symbol") or symbol or cid,
            "file": file,
            "line_start": ls,
            "line_end": le,
            "impact": str(ov.get("impact") or c.get("estimated_impact") or "low").lower(),
            "rationale": ov.get("rationale") or c.get("description") or "",
            "description": c.get("description") or "",
            "current_approach": c.get("current_approach") or "",
            "evolve_rationale": c.get("evolve_rationale") or "",
            "impact_why": c.get("estimated_impact_explanation") or "",
            "origin": c.get("origin") or "",
            "proposals": c.get("proposals") or [],
            "evolved": cid in evolved,
            "applied": cid in applied,
        })

    if ranked:
        rows.sort(key=lambda r: (r["rank"] is None, r["rank"] or 0))
    else:
        rows.sort(key=lambda r: (IMPACT_ORDER.get(r["impact"], 3), r["id"]))

    # ---- per-module aggregates -------------------------------------------
    modules = {}
    for qn, status in status_of.items():
        modules[qn] = {"name": qn, "status": status, "total": 0, "high": 0}
    for r in rows:
        m = modules.setdefault(
            r["module"], {"name": r["module"], "status": r["module_status"], "total": 0, "high": 0}
        )
        m["total"] += 1
        if r["impact"] == "high":
            m["high"] += 1

    status_counts = {s: 0 for s in STATUS_ORDER}
    for m in modules.values():
        status_counts[m["status"]] = status_counts.get(m["status"], 0) + 1

    # ---- agent effort ----------------------------------------------------
    telemetry = result.get("per_module_telemetry") or {}
    discovery: dict[str, dict[str, Any]] = {}
    for _qn, t in telemetry.items():
        for it in (t or {}).get("discovery_iterations") or []:
            a = it.get("agent") or "unknown"
            d = discovery.setdefault(a, {"agent": a, "rounds": 0, "seconds": 0.0, "added": 0})
            d["rounds"] += 1
            d["seconds"] += float(it.get("duration_s") or 0.0)
            d["added"] += len(it.get("added") or [])
    # Authorship is counted with the provenance of what was written, because the
    # two are not the same question: one of the "authors" in a deep-research run
    # is the finding->proposal stage, not a coding agent, and the panel has to be
    # able to tell them apart without hardcoding a name.
    authored: dict[str, dict[str, Any]] = {}
    for r in rows:
        for p in r["proposals"]:
            a = p.get("author") or "unknown"
            x = authored.setdefault(a, {"author": a, "n": 0, "research": 0})
            x["n"] += 1
            if (p.get("source") or "") == "research_finding":
                x["research"] += 1

    # ---- token mix -------------------------------------------------------
    buckets = {"cache_read": 0, "cache_create": 0, "input": 0, "output": 0}
    for m in manifest.get("models_used") or []:
        u = m.get("usage") or {}
        for k in buckets:
            buckets[k] += int(u.get(k) or 0)

    # Gated on the run actually having findings: None for every other pipeline,
    # and every consumer renders nothing when it is None.
    research_d = research_layer(result, rows)

    issues = report.get("issues") or []
    issue_groups: dict[tuple[str, str], int] = {}
    for i in issues:
        key = (i.get("step") or "unknown", (i.get("severity") or "unknown").lower())
        issue_groups[key] = issue_groups.get(key, 0) + 1

    return {
        "run_dir": run,
        "run_id": manifest.get("run_id") or run_meta.get("run_id") or ranking.get("run_id") or "",
        "objective": ctx.get("objective") or ranking.get("objective") or "",
        "hints": ctx.get("workload_hints") or [],
        "method": ranking.get("method") or "",
        "pipeline": (manifest.get("spotlights") or {}).get("pipeline")
                    or run_meta.get("pipeline") or "",
        "target": manifest.get("target") or {},
        "engine_sha": (manifest.get("spotlights") or {}).get("commit_sha") or "",
        "started": run_meta.get("started_at") or manifest.get("date") or "",
        "finished": run_meta.get("finished_at") or "",
        "cost": manifest.get("cost") or {},
        "external_cost": manifest.get("external_cost") or {},
        "timing": manifest.get("timing") or {},
        "total_tokens": manifest.get("total_tokens"),
        "buckets": buckets,
        "models_used": manifest.get("models_used") or [],
        "rows": rows,
        "ranked": ranked,
        "modules": sorted(modules.values(), key=lambda m: (-m["total"], m["name"])),
        "status_counts": status_counts,
        "discovery": sorted(discovery.values(), key=lambda d: -d["added"]),
        "authored": authored,
        "research": research_d,
        "issues": issues,
        "issue_groups": issue_groups,
        "evolved": evolved,
        "applied": applied,
    }


# --------------------------------------------------------------------------- #
# components
#
# Charts are plain HTML/CSS marks, not SVG: no text measurement, no collision
# risk, responsive for free. Mark specs honoured — bars <=24px thick, 4px
# rounded data-end square at the baseline, a 2px surface gap between touching
# fills, hairline solid baseline, hover tooltip + keyboard focus, and a table
# view twin so no value is ever gated behind hue or hover.
# --------------------------------------------------------------------------- #

# An inline label only goes inside a mark when it comfortably fits.
INLINE_LABEL_MIN_SHARE = 0.09


def tile(label: str, value: str, detail: str = "") -> str:
    d = f'<div class="d">{detail}</div>' if detail else ""
    return (f'<div class="tile"><div class="l">{esc(label)}</div>'
            f'<div class="n">{esc(value)}</div>{d}</div>')


def legend(items) -> str:
    """items: (color, label, value|None) — always present for >=2 series."""
    out = []
    for color, label, value in items:
        v = f' <b>{esc(value)}</b>' if value is not None else ""
        out.append(f'<span><i style="background:{esc(color)}"></i>{esc(label)}{v}</span>')
    return f'<div class="legend">{"".join(out)}</div>'


def status_roster(d: dict, fills: dict, glyphs: dict) -> str:
    """Which modules ended in each outcome, grouped, with what each contributed.

    Every count comes from d["modules"] -- the same rows-derived aggregate the
    candidates-per-module chart and the leaderboard use -- so a module cannot
    report one candidate count here and a different one two cards down. Note in
    particular that this is the FINAL candidate set: a module_runs entry may list
    candidates that did not survive into it, and quoting that field here would
    make the roster disagree with every other number on the page.
    """
    groups: dict[str, list[dict[str, Any]]] = {}
    for m in d["modules"]:
        groups.setdefault(m["status"], []).append(m)
    present = [st for st in STATUS_ORDER if groups.get(st)]
    if not present:
        return ""
    cards = []
    for st in present:
        ms = sorted(groups[st], key=lambda m: (-m["total"], m["name"]))
        n_c = sum(m["total"] for m in ms)
        n_h = sum(m["high"] for m in ms)
        fill = fills.get(st, "var(--viz-rest)")
        chip = fill if fill.startswith("#") else "#8a8578"
        sub = (f'{n0(n_c)} candidates \u00b7 {n0(n_h)} high impact' if n_c
               else 'nothing reached the final set')
        lis = "".join(
            f'<li><span class="nm">{brk(m["name"])}</span>'
            f'<span class="ct{"" if m["total"] else " z"}">'
            f'{esc(n0(m["total"])) if m["total"] else "&mdash;"}</span></li>'
            for m in ms
        )
        cards.append(
            f'<div class="mgrp"><div class="mgh">'
            f'<i class="ic" style="background:{esc(chip)};color:{ink_on(chip)}">'
            f'{esc(glyphs.get(st, BULLET))}</i>'
            f'<span class="lb">{esc(st.title())}</span><b>{esc(n0(len(ms)))}</b></div>'
            f'<div class="mgs">{sub}</div>'
            f'<ul class="mgl">{lis}</ul></div>'
        )
    return (f'<details class="mbrk"><summary>Which modules ended where</summary>'
            f'<div class="mgrid">{"".join(cards)}</div></details>')


def status_legend(items) -> str:
    """items: (color, glyph, label, value) — status always ships icon + label."""
    out = []
    for color, glyph, label, value in items:
        out.append(
            f'<span><i class="ic" style="background:{esc(color)};'
            f'color:{ink_on(color)}">{esc(glyph)}</i>{esc(label)} <b>{esc(value)}</b></span>'
        )
    return f'<div class="stleg">{"".join(out)}</div>'


def table_view(cid: str, headers, body_rows, right_from: int = 1,
               rows_meta=None) -> str:
    """The WCAG-clean twin of a chart. headers/body_rows are plain sequences.

    rows_meta, when given, is a list of attribute strings parallel to body_rows.
    It exists so a chart whose rows can be filtered client-side can carry the
    same keys in its table twin; omitted, the markup is what it always was.

    """
    th = "".join(
        f'<th class="{"r" if i >= right_from else ""}">{esc(h)}</th>'
        for i, h in enumerate(headers)
    )
    trs = []
    for j, r in enumerate(body_rows):
        tds = "".join(
            f'<td class="{"r" if i >= right_from else ""}">{esc(v)}</td>'
            for i, v in enumerate(r)
        )
        at = f" {rows_meta[j]}" if rows_meta else ""
        trs.append(f"<tr{at}>{tds}</tr>")
    return (f'<div class="tv" id="{esc(cid)}" hidden>'
            f'<table><thead><tr>{th}</tr></thead>'
            f'<tbody>{"".join(trs)}</tbody></table></div>')


def chart(cid: str, title: str, sub: str, plot: str, tail: str, tv: str,
          note: str = "", wide: bool = False, head_extra: str = "") -> str:
    """head_extra rides in the chart head, immediately left of the table button."""
    n = f'<p class="note">{note}</p>' if note else ""
    return (
        f'<section class="chart{" wide" if wide else ""}">'
        f'<div class="head"><div><h3>{esc(title)}</h3><p class="sub">{esc(sub)}</p></div>'
        f'<div class="spacer"></div>{head_extra}'
        f'<button class="tvbtn" data-target="{esc(cid)}" aria-expanded="false">table</button>'
        f'</div>'
        f'<div class="plot">{plot}</div>{tail}{tv}{n}</section>'
    )


def stack(segments, inline_labels: bool = True, fmt=n0) -> str:
    """One-row stacked bar. segments: (label, value, color). Part-to-whole.

    inline_labels=False when a segment fill themes light<->dark, since no single
    ink clears contrast against both; the legend + tooltip + table carry the
    values instead. Never clip a label to make it fit.

    fmt renders the value in its own unit -- counts read as counts, seconds read
    as a duration. It reaches the inline label AND the tooltip, so a chart cannot
    end up labelled in one unit and hovered in another.
    """
    total = sum(v for _l, v, _c in segments) or 1
    out = []
    for label, value, color in segments:
        share = value / total
        inner = ""
        if inline_labels and share >= INLINE_LABEL_MIN_SHARE:
            inner = f'<span style="color:{ink_on(color)}">{esc(fmt(value))}</span>'
        out.append(
            f'<div class="seg" tabindex="0" style="flex:{share:.6f} 1 0;'
            f'background:{esc(color)}" data-tip="{esc(label)}" '
            f'data-tipval="{esc(fmt(value))} · {esc(pct(value, total))}">{inner}</div>'
        )
    return f'<div class="stack">{"".join(out)}</div>'


def bars_single(items, fmt=n0, label_width: str = "minmax(96px,auto)") -> str:
    """Horizontal bars, ONE series in one hue — magnitude comparison.

    One color for every bar: a value-ramp across nominal categories would just
    re-encode bar length as hue. items: (label, value, tip).
    """
    top = max((v for _l, v, _t in items), default=0) or 1
    rows = []
    for label, value, tip in items:
        w = 100.0 * value / top
        rows.append(
            f'<div class="bar" style="grid-template-columns:{label_width} 1fr auto" '
            f'tabindex="0" data-tip="{esc(label)}" data-tipval="{esc(tip)}">'
            f'<div class="lab">{esc(label)}</div>'
            f'<div class="track"><div class="seg" style="width:{w:.4f}%;'
            f'background:var(--viz-accent)"></div></div>'
            f'<div class="val">{esc(fmt(value))}</div></div>'
        )
    return f'<div class="bars">{"".join(rows)}</div>'


def bars_emphasis(items, label_width: str = "minmax(150px,auto)",
                  rows_meta=None) -> str:
    """Emphasis form: the part that matters in the accent hue, the rest in the
    de-emphasis gray. items: (label, part, total, tip).

    rows_meta, when given, is a list of (attributes, badge_html) parallel to
    items. The bar scale stays keyed to the widest total in the full set even
    when rows are hidden, so a bar means the same length before and after a
    filter.
    """
    top = max((t for _l, _p, t, _tip in items), default=0) or 1
    rows = []
    for j, (label, part, total, tip) in enumerate(items):
        at, badge = (rows_meta[j] if rows_meta else ("", ""))
        rest = max(0, total - part)
        wp = 100.0 * part / top
        wr = 100.0 * rest / top
        segs = ""
        if part:
            segs += (f'<div class="seg" style="width:{wp:.4f}%;'
                     f'background:var(--viz-accent)"></div>')
        if rest:
            segs += (f'<div class="seg" style="width:{wr:.4f}%;'
                     f'background:var(--viz-rest)"></div>')
        rows.append(
            f'<div class="bar" style="grid-template-columns:{label_width} 1fr auto" '
            f'tabindex="0"{" " + at if at else ""} data-tip="{esc(label)}" '
            f'data-tipval="{esc(tip)}">'
            f'<div class="lab">{esc(label)}{badge}</div>'
            f'<div class="track">{segs}</div>'
            f'<div class="val">{esc(n0(total))}</div></div>'
        )
    return f'<div class="bars">{"".join(rows)}</div>'


# --------------------------------------------------------------------------- #
# sections
# --------------------------------------------------------------------------- #


def sec(title: str, body: str, head_extra: str = "") -> str:
    """A titled section. head_extra rides on the title line, right-aligned.

    Without it the markup is unchanged from before the view switch existed, so
    every other section keeps its original single-element header.
    """
    if not head_extra:
        return f'<section class="sec"><h2>{esc(title)}</h2>{body}</section>'
    return (f'<section class="sec"><div class="sechead"><h2>{esc(title)}</h2>'
            f'<div class="sp"></div>{head_extra}</div>{body}</section>')


def overview(d: dict) -> str:
    rows = d["rows"]
    total = len(rows)
    high = sum(1 for r in rows if r["impact"] == "high")
    with_cands = [m for m in d["modules"] if m["total"]]
    props = sum(len(r["proposals"]) for r in rows)
    covered = sum(1 for r in rows if r["proposals"])
    tgt = d["target"]
    tok = d["total_tokens"] or sum(d["buckets"].values())
    cached = d["buckets"]["cache_read"]
    contracted = (d["cost"] or {}).get("amount_usd")
    listed = (d["external_cost"] or {}).get("amount_usd")
    t = d["timing"] or {}

    repo = str(tgt.get("repo_url") or "")
    repo_name = repo.rstrip("/").split("/")[-1].removesuffix(".git") or "—"

    meta_items = [
        ("Run id", d["run_id"] or "—"),
        ("Target repo", repo_name),
        ("Target commit", short_sha(tgt.get("commit_sha"), 10)),
        ("Engine commit", short_sha(d["engine_sha"], 10)),
        ("Pipeline", (d["pipeline"] or "—").replace("_", "-")),
        ("Ranking", d["method"] or ("ranked" if d["ranked"] else "unranked")),
        ("Started", (d["started"] or "—")[:16].replace("T", " ")),
        ("Finished", (d["finished"] or "—")[:16].replace("T", " ")),
    ]
    meta = "".join(
        f'<div><div class="k">{esc(k)}</div><div class="v mono">{esc(v)}</div></div>'
        for k, v in meta_items
    )

    hero = (
        f'<div class="card hero">'
        f'<div><div class="fig">{esc(n0(total))}</div>'
        f'<div class="figlab">optimisation candidates from '
        f'<b>{esc(n0(len(with_cands)))}</b> of '
        f'<b>{esc(n0(len(d["modules"])))}</b> modules</div></div>'
        f'<div class="meta">{meta}</div></div>'
    )

    # Who wrote the proposals, not only what they landed on. A deep-research run
    # has two or three populations writing at once -- one coding agent per stream
    # plus the finding->proposal stage -- and a single total hides which of them
    # the number belongs to. The stage is recognised the way the authorship panel
    # recognises it, by every one of its proposals citing a finding, so a renamed
    # stage does not turn into a third agent here either; and it is labelled
    # "from findings" rather than "findings" because a bare "221 findings" in a
    # tile sitting beside the Findings tile reads as a count of findings.
    who = sorted(d["authored"].values(), key=lambda x: -x["n"])
    mix = " \u00b7 ".join(
        f'<b>{esc(n0(x["n"]))}</b> '
        + esc("from findings" if x["n"] and x["research"] == x["n"] else x["author"])
        for x in who)
    # With one author the split is the total restated, so it is left off.
    pdetail = (f'across <b>{esc(n0(covered))}</b> candidates'
               + (f'<br>{mix}' if len(who) > 1 else ''))

    tiles = "".join([
        tile("High impact", n0(high),
             f"{esc(pct(high, total, 0))} of all candidates"),
        tile("Proposals", n0(props), pdetail),
        tile("Ranked top score", n0(max((r["score"] or 0) for r in rows)) if rows else "—",
             "listwise sub-agent judge, 0–100"),
        tile("Cost", usd(contracted),
             f"<b>{esc(usd(listed))}</b> at public list price"),
        tile("Tokens", compact(tok),
             f"<b>{esc(pct(cached, tok, 0))}</b> served from cache"),
        tile("Wall clock", dur(t.get("wall_clock_s")),
             f"<b>{esc(dur(t.get('api_time_s')))}</b> of API time"),
    ])

    # Two more tiles, and only when the run has a research layer to describe.
    R = d.get("research")
    if R:
        tiles += "".join([
            tile("Findings", n0(len(R["findings"])),
                 f'<b>{esc(n0(R["works"]))}</b> distinct works over '
                 f'<b>{esc(n0(len(R["researched"])))}</b> researched modules'),
            tile("Grounded proposals", n0(R["grounded"]),
                 f'<b>{esc(pct(R["grounded"], props, 0))}</b> of proposals cite '
                 f'a finding'),
        ])

    hints = ""
    if d["hints"]:
        lis = "".join(f"<li>{esc(h)}</li>" for h in d["hints"])
        hints = sec("Workload constraints the objective is scored under",
                    f'<div class="card pad"><ul class="hints">{lis}</ul></div>')

    return hero + f'<div class="tiles">{tiles}</div>' + hints


def charts(d: dict) -> str:
    L = VIZ["light"]
    out = []

    # -- 1. module outcome: part-to-whole with status meaning ---------------
    counts = d["status_counts"]
    glyphs = {"SUCCEEDED": "✓", "DEGRADED": "!", "FAILED": "✕", "SKIPPED": "–"}
    fills = {
        "SUCCEEDED": L["good"], "DEGRADED": L["warning"],
        "FAILED": L["critical"], "SKIPPED": "var(--viz-rest)",
    }
    present = [s for s in STATUS_ORDER if counts.get(s)]
    n_mod = sum(counts.values()) or 1
    # The clean-completion count is read off the run, not written into the prose:
    # the same note has to hold for a run where every module succeeded.
    n_clean = counts.get("SUCCEEDED", 0)
    clean_line = (
        f'All {n0(n_mod)} modules completed cleanly.' if n_clean == n_mod
        else ('No module completed cleanly.' if n_clean == 0
              else f'Only {n0(n_clean)} of {n0(n_mod)} modules completed cleanly.')
    )
    # What each outcome actually contributed, so the note reports the barren
    # statuses instead of asserting that they are barren. Same rows-derived
    # totals as the roster and the per-module chart.
    contrib: dict[str, int] = {}
    for m in d["modules"]:
        contrib[m["status"]] = contrib.get(m["status"], 0) + m["total"]
    barren = [st for st in present if not contrib.get(st)]
    barren_line = ""
    if barren:
        names = " and ".join(f'<b>{st.lower()}</b>' for st in barren)
        n_b = sum(counts[st] for st in barren)
        barren_line = (
            f' The {esc(n0(n_b))} {names} '
            f'{"module" if n_b == 1 else "modules"} contributed nothing to the '
            f'final candidate set.')
    out.append(chart(
        "tv-status",
        "Module analysis outcome",
        f"{n0(n_mod)} modules walked by the extractor",
        stack([(s.title(), counts[s], fills[s]) for s in present], inline_labels=False),
        status_legend([(fills[s] if fills[s].startswith("#") else "#8a8578",
                        glyphs[s], s.title(), n0(counts[s])) for s in present])
        + status_roster(d, fills, glyphs),
        table_view("tv-status",
                   ["Outcome", "Modules", "Share", "Candidates", "High impact"],
                   [(s.title(), n0(counts[s]), pct(counts[s], n_mod),
                     n0(contrib.get(s, 0)),
                     n0(sum(m["high"] for m in d["modules"] if m["status"] == s)))
                    for s in present]),
        note=(f"{esc(clean_line)}{barren_line}"
              # Gated: the sentence describes a status, so it may not exist.
              + (" <b>Degraded</b> modules still returned candidates but recorded "
                 "at least one failure along the way." if counts.get("DEGRADED")
                 else "")
              + " Open <i>which modules ended where</i> for the roster and what "
              "each module contributed. Status colour is the reserved "
              "good/warning/critical palette and always ships with its icon and "
              "label &mdash; never hue alone."),
    ))

    # -- 2. candidates per module: emphasis on the high-impact share -------
    # The bars are sorted by total, which answers "who produced the most". The
    # hot toggle re-sorts by high-impact share, which answers "whose output is
    # densest" -- the same key and the same ordering the leaderboard uses, so a
    # module keeps its place across the two sections.
    mods = [m for m in d["modules"] if m["total"]]
    # One definition of hot for the whole page: the same function the leaderboard
    # uses, over the same rows, so a module cannot be hot in one section and cold
    # in another.
    H = hotness(d["rows"])
    hm = H["mods"]
    meta = [(f'data-hot="{hm[m["name"]]["score"]:.4f}" '
             f'data-hotn="{m["high"]}" '
             f'data-hoton="{1 if hm[m["name"]]["hot"] else 0}" '
             f'data-module="{esc(m["name"])}"', m) for m in mods]
    out.append(chart(
        "tv-mods",
        "Candidates per module",
        "high-impact share highlighted; every other candidate greyed",
        bars_emphasis(
            [(m["name"], m["high"], m["total"],
              f'{n0(m["total"])} candidates · {n0(m["high"])} high impact · '
              f'{hm[m["name"]]["share"]:.0f}% high · '
              f'{hm[m["name"]]["score"]:.0f}% hot score · {m["status"].lower()}')
             for m in mods],
            rows_meta=[(at, f'<span class="hp">{hm[m["name"]]["score"]:.0f}%</span>')
                       for at, m in meta],
        ),
        legend([("var(--viz-accent)", "High impact", None),
                ("var(--viz-rest)", "Medium or low", None)]),
        table_view("tv-mods",
                   ["Module", "Candidates", "High impact", "High share",
                    "Hot score", "Outcome"],
                   [(m["name"], n0(m["total"]), n0(m["high"]),
                     f'{hm[m["name"]]["share"]:.0f}%',
                     f'{hm[m["name"]]["score"]:.1f}%', m["status"].title())
                    for m in mods],
                   rows_meta=[at for at, _m in meta]),
        note=(f"Emphasis rather than eight hues: the question is where the "
              f"high-impact mass sits, so that part carries the accent and "
              f"everything else recedes. <b>Hot modules</b> is {hot_blurb(H)}, "
              f"which is why a rate over 8 candidates outranks the same rate "
              f"over 4. Bar lengths keep their original scale, so a bar means "
              f"the same thing in both orders &mdash; in hot order, length is "
              f"still volume while position is density."),
        wide=True,
        head_extra=(f'<button class="tvbtn" id="modhot" aria-pressed="false" '
                    f'title="Hot modules: {esc(hot_blurb(H, plain=True))}">'
                    f'hot modules</button>'),
    ))

    # -- 3. spend by model and role: one series, one hue -------------------
    by_model = (d["cost"] or {}).get("by_model") or []
    ext_by = {(m.get("model"), m.get("role")): m.get("amount_usd")
              for m in ((d["external_cost"] or {}).get("by_model") or [])}
    spend = sorted(by_model, key=lambda m: -(m.get("amount_usd") or 0))
    total_spend = sum(m.get("amount_usd") or 0 for m in spend) or 1
    out.append(chart(
        "tv-spend",
        "Where the spend went",
        "contracted rate, by model and pipeline role",
        bars_single(
            [(f'{m.get("model")} · {(m.get("role") or "").replace("_", " ")}',
              m.get("amount_usd") or 0,
              f'{usd(m.get("amount_usd"))} · {pct(m.get("amount_usd"), total_spend)} of run')
             for m in spend],
            fmt=lambda v: usd(v, 2),
            label_width="minmax(190px,auto)",
        ),
        "",
        table_view("tv-spend",
                   ["Model · role", "Contracted", "Public list", "Share"],
                   [(f'{m.get("model")} · {(m.get("role") or "").replace("_", " ")}',
                     usd(m.get("amount_usd")),
                     usd(ext_by.get((m.get("model"), m.get("role")))),
                     pct(m.get("amount_usd"), total_spend))
                    for m in spend]),
        note=("One series, so one hue for every bar &mdash; shading bars by size "
              "would re-encode length as colour. The public-list column in the "
              "table view is the same run priced at published API rates."),
        wide=True,
    ))

    # -- 4. token mix: the cache story -------------------------------------
    b = d["buckets"]
    tok_total = sum(b.values()) or 1
    seg_defs = [
        ("Cache read", b["cache_read"], "var(--viz-1)"),
        ("Input", b["input"], "var(--viz-3)"),
        ("Cache write", b["cache_create"], "var(--viz-2)"),
        ("Output", b["output"], "var(--viz-4)"),
    ]
    out.append(chart(
        "tv-tokens",
        "Token mix",
        f"{compact(tok_total)} tokens across every agent invocation",
        stack(seg_defs),
        legend([(c, lab, f"{compact(v)} · {pct(v, tok_total, 0)}")
                for lab, v, c in seg_defs]),
        table_view("tv-tokens", ["Bucket", "Tokens", "Share"],
                   [(lab, n0(v), pct(v, tok_total)) for lab, v, _c in seg_defs]),
        note=("Only the widest segment is labelled in place; the rest would not "
              "fit without clipping, so the legend and table view carry them. "
              "The four hues pass every colour-vision gate in both themes."),
    ))

    # Display order pairs the two short cards on one row so neither sits alone
    # beside empty space; the two long ones follow full width.
    status_c, mods_c, spend_c, tokens_c = out
    ordered = [status_c, tokens_c, mods_c, spend_c]
    return sec("Run shape", f'<div class="charts">{"".join(ordered)}</div>')


def agents(d: dict) -> str:
    hue = {"claude_code": "var(--viz-1)", "claude": "var(--viz-1)",
           "codex": "var(--viz-2)", "gpt-5.5": "var(--viz-3)"}

    disc = d["discovery"]
    disc_added = sum(x["added"] for x in disc) or 1
    disc_rows = "".join(
        f'<tr><td><span class="who"><i style="background:'
        f'{esc(hue.get(x["agent"], "var(--viz-4)"))}"></i>{esc(x["agent"])}</span></td>'
        f'<td>{esc(n0(x["rounds"]))}</td><td>{esc(dur(x["seconds"]))}</td>'
        f'<td>{esc(n0(x["added"]))}</td><td>{esc(pct(x["added"], disc_added, 0))}</td></tr>'
        for x in disc
    )
    # Derived, not asserted. The sentence that used to sit here claimed both
    # agents ran the same number of rounds and that one dominated -- true of the
    # run it was written against and not of every run this page renders.
    if not disc:
        disc_note = "No discovery iterations are recorded for this run."
    else:
        top = max(disc, key=lambda x: x["added"])
        same = len({x["rounds"] for x in disc}) == 1
        if len(disc) == 1:
            lead = (f'One agent, <b>{esc(top["agent"])}</b>, ran discovery for '
                    f'this run.')
        else:
            rounds = ("ran the same number of rounds" if same
                      else "ran different numbers of rounds")
            share = top["added"] / disc_added
            if share >= 0.6:
                lead = (f'The {esc(n0(len(disc)))} agents {rounds}, but the '
                        f'candidate set is mostly <b>{esc(top["agent"])}</b>'
                        f'&rsquo;s &mdash; '
                        f'<b>{esc(pct(top["added"], disc_added, 0))}</b> of all '
                        f'introductions.')
            else:
                lead = (f'The {esc(n0(len(disc)))} agents {rounds} and '
                        f'contributed comparably; the largest share is '
                        f'<b>{esc(top["agent"])}</b>&rsquo;s at '
                        f'<b>{esc(pct(top["added"], disc_added, 0))}</b>.')
        n_rows = len(d["rows"])
        rel = ("exceeds" if disc_added > n_rows
               else "falls short of" if disc_added < n_rows else "matches")
        disc_note = (f'{lead} &ldquo;Added&rdquo; counts candidate-introduction '
                     f'events across rounds, so it {rel} the '
                     f'{esc(n0(n_rows))} that survived to the final set.')

    disc_tab = (
        f'<div class="card pad"><h3 style="margin:0 0 2px;font-size:14px">'
        f'Discovery rounds</h3>'
        f'<p class="sub" style="color:var(--muted);font-size:12px;margin:0 0 10px">'
        f'who proposed the candidate set</p>'
        f'<table class="atab"><thead><tr><th>Agent</th><th>Rounds</th>'
        f'<th>Agent time</th><th>Added</th><th>Share</th></tr></thead>'
        f'<tbody>{disc_rows}</tbody></table>'
        f'<p class="note" style="margin-top:12px">{disc_note}</p></div>'
    )

    # A writer every one of whose proposals cites a finding is the pipeline's
    # finding->proposal stage, not a coding agent. This table used to list it as
    # an agent -- with the largest share of the three -- which read as though a
    # third agent had out-proposed both real ones. The test is the data, not the
    # writer's name, so a renamed stage cannot quietly become an agent again.
    auth = sorted(d["authored"].values(), key=lambda x: -x["n"])
    for x in auth:
        x["pipeline"] = bool(x["n"]) and x["research"] == x["n"]
    show_kind = any(x["pipeline"] for x in auth)
    auth_total = sum(x["n"] for x in auth) or 1
    auth_rows = "".join(
        f'<tr><td><span class="who"><i style="background:'
        f'{esc("var(--viz-rest)" if x["pipeline"] else hue.get(x["author"], "var(--viz-4)"))}"'
        f'></i>{esc(x["author"])}</span></td>'
        + (f'<td class="k">{"pipeline stage" if x["pipeline"] else "coding agent"}</td>'
           if show_kind else "")
        + f'<td>{esc(n0(x["n"]))}</td><td>{esc(pct(x["n"], auth_total, 0))}</td></tr>'
        for x in auth
    )
    kind_th = "<th>Kind</th>" if show_kind else ""

    ags = [x for x in auth if not x["pipeline"]]
    ag_total = sum(x["n"] for x in ags) or 1
    parts = []
    if len(ags) >= 2:
        hi, lo = ags[0], ags[-1]
        if (hi["n"] - lo["n"]) / ag_total <= 0.10:
            parts.append(
                f'Authorship across the {esc(n0(len(ags)))} coding agents is near '
                f'even ({esc(" / ".join(pct(x["n"], ag_total, 0) for x in ags))}).')
        else:
            parts.append(
                f'<b>{esc(hi["author"])}</b> wrote '
                f'{esc(pct(hi["n"], ag_total, 0))} of the {esc(n0(ag_total))} '
                f'agent-authored proposals and <b>{esc(lo["author"])}</b> '
                f'{esc(pct(lo["n"], ag_total, 0))}.')
    elif len(ags) == 1:
        parts.append(f'A single coding agent, <b>{esc(ags[0]["author"])}</b>, wrote '
                     f'all {esc(n0(ags[0]["n"]))} agent-authored proposals.')
    pipe_n = sum(x["n"] for x in auth if x["pipeline"])
    if pipe_n:
        parts.append(
            f'The other <b>{esc(n0(pipe_n))}</b> come from the '
            f'finding&rarr;proposal stage &mdash; pipeline machinery reading the '
            f'deep-research findings, not a competing agent. The '
            f'&ldquo;What the deep research read&rdquo; section breaks it down.')
    parts.append('Per-agent cost is not split out here: the manifest prices spend '
                 'by model and role, and the Codex stream does not report a cost '
                 'per invocation.')

    auth_tab = (
        f'<div class="card pad"><h3 style="margin:0 0 2px;font-size:14px">'
        f'Proposals authored</h3>'
        f'<p class="sub" style="color:var(--muted);font-size:12px;margin:0 0 10px">'
        f'who wrote the optimisation proposals</p>'
        f'<table class="atab"><thead><tr><th>Writer</th>{kind_th}'
        f'<th>Proposals</th><th>Share</th></tr></thead>'
        f'<tbody>{auth_rows}</tbody></table>'
        f'<p class="note" style="margin-top:12px">{" ".join(parts)}</p></div>'
    )
    return sec("Which agent did what", f'<div class="agents">{disc_tab}{auth_tab}</div>')


def work_card(g) -> str:
    """One work, and the technique each module took away from it, side by side.

    Collapsed by default and built from <details>, because the panel is 6 works
    on one run and 71 on another: an always-open list would be the longest thing
    on the page for no gain. The comparison is the point of the card -- two
    modules reading the same paper and extracting the same technique is
    redundant work, extracting different techniques is not, and only the
    summaries side by side can tell you which happened.
    """
    mods = sorted({f["module"] for f in g})
    head = g[0]
    cols = "".join(
        f'<div class="col"><h5>{brk(f["module"])}</h5>'
        f'<p>{mdi(f["summary"]) if f["summary"] else "(no technique summary)"}</p></div>'
        for f in g
    )

    # Distinct URLs in first-seen order, each carrying the modules that reached
    # the work through it. dict.fromkeys, not a set: card order has to be stable
    # across regenerations of the same run.
    urls: dict[str, list[str]] = {}
    for f in g:
        if f["url"]:
            urls.setdefault(f["url"], [])
            if f["module"] and f["module"] not in urls[f["url"]]:
                urls[f["url"]].append(f["module"])

    def _a(u: str) -> str:
        return (f'<a href="{esc(u)}" target="_blank" rel="noopener noreferrer">'
                f'{brk(u)}</a>')

    if len(urls) > 1:
        # More than one URL is the interesting case, so every URL is shown and
        # attributed. Without the module beside it the list cannot answer the
        # question it exists to answer -- which module read which copy.
        tag, inline = f"{len(urls)} URLs", ""
        items = "".join(
            f'<li>{_a(u)}'
            + (f' <span class="wm">&larr; {brk(", ".join(ms))}</span>' if ms else "")
            + "</li>"
            for u, ms in urls.items()
        )
        listing = f'<ul class="wul">{items}</ul>'
    else:
        tag = "same URL" if urls else "no URL"
        inline = f" &middot; {_a(next(iter(urls)))}" if urls else ""
        listing = ""

    return (
        f'<details class="work"><summary>'
        f'<span class="wt">{esc(head["title"])}</span>'
        f'<span class="wn">{esc(n0(len(mods)))} modules &middot; '
        f'{esc(n0(len(g)))} findings</span></summary>'
        f'<div class="wbody"><p class="wu">{esc(tag)} &middot; '
        f'{esc(head["source_type"])}{inline}</p>{listing}'
        f'<div class="cols">{cols}</div></div></details>'
    )


def _prop_target(pp: dict) -> str:
    """One proposal in a finding's detail: the candidate it landed on, and why.

    The raw candidate id was what this line used to show, and it identifies the
    row without describing it -- `cand-vllm_distributed_kv_transfer-0001` tells a
    reader nothing they can weigh. Impact, symbol and file range are what they
    can weigh, so those are the summary.

    Collapsed, because the reasoning is the part that differs between the
    proposals of one finding and it runs to a paragraph each: `technique_summary`
    and `supporting_evidence` live on the *finding*, so every proposal here
    shares them by construction and no per-proposal copy of them could ever
    differ. `rationale` is the per-pair bridge and is distinct in every one of
    the 1,062 multi-proposal findings in the corpus.

    The href is a real fragment to the row's tbody, so the jump works with
    scripting off. `data-cand` is what upgrades it: with JS the click also clears
    a filter that would be hiding the row, swaps back from the tree pane, opens
    the detail, and flashes it.
    """
    pill = f'imp-{pp["impact"]}' if pp["impact"] in IMPACT_ORDER else "imp-low"
    rank = ""
    if pp["showrank"]:
        rank = (f'<span class="prank">#{esc(n0(pp["rank"]))}</span>'
                if pp["rank"] is not None else
                '<span class="prank" title="not in the ranked shortlist">'
                'unranked</span>')
    head = (
        f'<summary><span class="ptitle">{esc(pp["title"])}</span>{rank}'
        f'<span class="pill {pill}">{esc(pp["impact"])}</span>'
        f'<a href="#{esc(pp["cand"])}" data-cand="{esc(pp["cand"])}">'
        f'{esc(pp["symbol"])}</a>'
        + (f'<span class="ploc">{esc(pp["loc"])}</span>' if pp["loc"] else "")
        + "</summary>"
    )
    why = (f'<p class="pwhy"><span class="pwl">Why the technique applied here'
           f'</span>{mdi(pp["why"])}</p>' if pp["why"] else "")
    return f'<details class="pitem">{head}{why}</details>'


def _paired_line(nc: int, n: int, failed: int) -> str:
    """What this finding's pairings did, in one sentence.

    A pair is a question asked, not a match found: the pipeline pairs every
    candidate in a module with every finding in it (`_build_pair_keys` is a plain
    double loop) and asks one agent per pair whether the finding supports a change
    to that candidate. The per-pair schema allows an array of 0 or 1 proposals and
    the prompt says to prefer the empty one -- "only emit a proposal when the
    finding contributes a concrete, transferable idea", "prefer emptiness when the
    finding is only topically adjacent" -- so most pairs are answered "no" by
    design. Corpus-wide 24,814 of 28,900 pairs (86%) came back empty.

    Failures are separated out because they are not that answer. 430 pairs across
    the corpus never returned a verdict at all -- the agent call errored or timed
    out -- and calling those "did not fit" would put words in the agent's mouth.
    """
    part = (f"Paired with {n0(nc)} candidate{'' if nc == 1 else 's'} in this "
            f"module; ")
    if n:
        part += (f"{'it' if n == 1 else n0(n)} "
                 f"produced {'a proposal' if n == 1 else 'proposals'}")
    else:
        part += "none produced a proposal"
    answered = nc - failed
    if failed:
        part += (f", and {n0(failed)} of the pairings never returned a verdict "
                 f"— the agent call failed or timed out, so "
                 f"{'that pairing' if failed == 1 else 'those pairings'} says "
                 f"nothing about fit")
        if not n:
            those = "that candidate" if answered == 1 else "those candidates"
            part += (f". The other {n0(answered)} were answered: the technique "
                     f"did not apply to {those}")
    elif not n:
        part += " — the agent judged the technique inapplicable to every one"
    return part + "."


def finding_row(f: dict) -> str:
    """One catalogue row plus its collapsed detail, mirroring lb_row/detail."""
    used = 1 if f["props"] else 0
    search = " ".join([f["title"], f["module"], f["url"], f["summary"],
                       f["source_type"]]).lower()
    rate = ((len({p["cand"] for p in f["props"]}) / len(f["cands"])
             + len(f["cands"]) / 1e6) if f["cands"] else "")
    at = (f'class="fnd" data-src="{esc(f["source_type"])}" '
          f'data-module="{esc(f["module"])}" data-title="{esc(f["title"].lower())}" '
          f'data-props="{len(f["props"])}" data-cands="{len(f["cands"])}" '
          # Sorting a fraction column by its denominator would order the rows by
          # a number the column no longer leads with. The key is the rate, with
          # the base as tiebreak so 1/1 does not outrank 12/21 on a run where
          # both exist; a finding paired with nothing sorts as missing, not as 0.
          f'data-rate="{rate}" '
          f'data-used="{used}" '
          f'data-pub="{esc(f["published"].strip())}" '
          f'data-host="{esc(_host(f["url"]))}" '
          f'data-search="{esc(search)}"')

    link = (f'<a href="{esc(f["url"])}" target="_blank" rel="noopener noreferrer">'
            f'{brk(f["url"])}</a>' if f["url"] else "&mdash;")
    # The two sort keys the row could not previously show. The date used to sit
    # on the url line; it belongs next to the site, because "where and when this
    # was published" is one fact, and because a sort you cannot see the key of is
    # a sort you have to take on trust.
    host = _host(f["url"])
    pub = (f'<div class="fhost">{brk(host)}</div>' if host
           else '<div class="fhost none">&mdash;</div>')
    if f["published"]:
        pub += f'<div class="fdate">{esc(f["published"])}</div>'
    n = len(f["props"])
    # Pairs this finding was in, and the candidates among them it landed on.
    # `hit` is counted over the proposals' own candidate ids rather than taken as
    # `n`: the two are equal everywhere in the corpus (no finding has two
    # proposals on one candidate, checked over all 4,540) but that is a fact
    # about these runs, not a guarantee, and a fraction that can read 4/3 if it
    # ever breaks is worse than one that cannot.
    nc = len(f["cands"])
    hit = len({p["cand"] for p in f["props"]})
    # The fraction, not the count: "3 of the 21 candidates this finding was
    # paired with" is the row's own hit rate, where a bare 21 was only the size
    # of the question asked. Denominator dimmed because the eye wants the
    # numerator; both are here because a rate without its base is unreadable
    # (1/1 and 12/21 are not the same claim).
    frac = (f'{esc(n0(hit))}<span class="fden">/{esc(n0(nc))}</span>' if nc
            else '&mdash;')
    row = (
        f'<tr class="frow">'
        f'<td><span class="tag">{esc(f["source_type"])}</span></td>'
        f'<td><div class="ftitle">{esc(f["title"])}</div>'
        f'<div class="furl">{link}</div></td>'
        f'<td><span class="modtag">{brk(f["module"])}</span></td>'
        f'<td class="pub">{pub}</td>'
        f'<td class="r"><span class="fnum{"" if hit else " none"}">{frac}</span></td>'
        f'<td class="r"><span class="fnum{"" if n else " none"}">'
        f'{esc(n0(n)) if n else "0"}</span></td>'
        f'</tr>'
    )

    blocks = []
    if f["summary"]:
        blocks.append(f'<div><h4>Technique extracted</h4><p>{mdi(f["summary"])}</p></div>')
    if f["evidence"]:
        blocks.append(f'<div><h4>Supporting evidence</h4>'
                      f'<p>{mdi(f["evidence"])}</p></div>')
    if f["props"]:
        items = "".join(_prop_target(pp) for pp in f["props"])
        many = len(f["props"]) > 1
        # Naming the scope is the point of this line. The two blocks above are
        # fields of the finding, not of any proposal, so they are identical for
        # every proposal here -- a reader looking at three proposals under one
        # technique summary would otherwise reasonably wonder which of them it
        # describes.
        # The pair arithmetic for this one row: paired with nc, landed on n.
        hit_of = _paired_line(nc, n, len(f["failed"]))
        scope = (hit_of + " The technique and evidence above are the finding's own, so all "
                 f"{n0(len(f['props']))} proposals share them. What differs is "
                 "the code each one targets and its reasoning."
                 if many else
                 hit_of + " The technique and evidence above are the finding's own.")
        blocks.append(f'<div class="fprops"><h4>{esc(n0(len(f["props"])))} '
                      f'proposal{"" if len(f["props"]) == 1 else "s"} it '
                      f'produced</h4>'
                      f'<p class="fscope">{esc(scope)}</p>{items}'
                      f'<p class="fhint">Each row expands to the reasoning and '
                      f'links to its candidate in the leaderboard below.</p>'
                      f'</div>')
    else:
        blocks.append(f'<div><h4>Proposals</h4><p>None. '
                      f'{esc(_paired_line(nc, 0, len(f["failed"])))}</p></div>')
    detail_row = (f'<tr class="fdetail"><td colspan="6">'
                  f'<div class="fbody">{"".join(blocks)}</div></td></tr>')
    return f"<tbody {at}>{row}{detail_row}</tbody>"


def research(d: dict) -> str:
    """The deep-research layer: what was read, what it cost, what it produced.

    Returns "" for a run without findings, so every non-research run renders
    byte-identically to before this section existed.
    """
    R = d.get("research")
    if not R:
        return ""
    F = R["findings"]
    n = len(F)
    used = n - len(R["orphans"])
    out = []

    # -- 1. stage budget ---------------------------------------------------
    if R["stages"]:
        st = dict(R["stages"])
        st_total = sum(st.values()) or 1
        pal = ["var(--viz-1)", "var(--viz-3)", "var(--viz-2)", "var(--viz-4)"]
        segs = [(label, v, pal[i % len(pal)])
                for i, (label, v) in enumerate(R["stages"])]
        dr = st.get("Deep research", 0.0)
        pf = st.get("Finding to proposal", 0.0)
        note = (f"Reading is the cheap half. Deep research itself takes "
                f"<b>{esc(pct(dr, st_total, 0))}</b> of stage time; turning those "
                f"findings into proposals takes "
                f"<b>{esc(pct(pf, st_total, 0))}</b>. The pair grid, not the "
                f"reading, is where this layer spends. Stage time is summed over "
                f"modules, which run concurrently, so it is accumulated agent "
                f"work rather than wall clock.")
        out.append(chart(
            "tv-rstage", "Where the pipeline's agent time went",
            f"{dur(st_total)} of stage time, summed over every module",
            stack(segs, fmt=dur),
            legend([(c, label, f"{dur(v)} · {pct(v, st_total, 0)}")
                    for label, v, c in segs]),
            table_view("tv-rstage", ["Stage", "Agent time", "Share"],
                       [(label, dur(v), pct(v, st_total)) for label, v, _c in segs]),
            note=note, wide=True))

    # -- 2. what was read, and which source types paid off ------------------
    bs = R["by_source"]
    out.append(chart(
        "tv-rsrc", "What deep research read, and which sources paid off",
        "findings per source type; the accented part produced at least one proposal",
        bars_emphasis([(b["type"], b["used"], b["n"],
                        f'{n0(b["n"])} findings · {n0(b["used"])} produced a '
                        f'proposal · {n0(b["props"])} proposals')
                       for b in bs],
                      label_width="minmax(96px,auto)"),
        legend([("var(--viz-accent)", "Produced a proposal", None),
                ("var(--viz-rest)", "Never used", None)]),
        table_view("tv-rsrc",
                   ["Source type", "Findings", "Produced a proposal", "Hit rate",
                    "Proposals", "Proposals per finding"],
                   [(b["type"], n0(b["n"]), n0(b["used"]), pct(b["used"], b["n"], 0),
                     n0(b["props"]),
                     f'{b["props"] / b["n"]:.2f}' if b["n"] else "—")
                    for b in bs]),
        note=(f"<b>{esc(n0(used))}</b> of {esc(n0(n))} findings "
              f"(<b>{esc(pct(used, n, 0))}</b>) reached at least one proposal; the "
              f"grey remainder was read and never used. A source type is a "
              f"nominal category, so the accent marks the part that mattered and "
              f"nothing is encoded by hue alone."),
        wide=True))

    # -- 3. the pair grid, per module and in total --------------------------
    pairs = R["pairs"]
    conc = (R["pair_accum_s"] / R["pair_stage_s"]) if R["pair_stage_s"] else 0.0
    grid_tiles = "".join([
        tile("Pairs evaluated", n0(pairs),
             "each timed individually in the run telemetry"),
        tile("Proposals produced", n0(R["grounded"]),
             f'<b>{esc(pct(R["grounded"], pairs, 1))}</b> of pairs yielded one'
             if pairs else "no pair telemetry recorded"),
        tile("Accumulated agent work", dur(R["pair_accum_s"]),
             (f'<b>{conc:.1f}&times;</b> concurrent, so '
              f'<b>{esc(dur(R["pair_stage_s"]))}</b> of stage time'
              if conc else "no stage duration recorded")),
    ])

    # Emphasis bars, part = proposals against total = pairs, and the two are
    # commensurable: exactly one proposal comes out of a pair that works --
    # 4,086 grounded proposals across the corpus and not a single pair produced
    # two. So bar length is the work the module did and the accent is literally
    # the share of that work which paid off. It also shows the thing a density
    # column could not: a module that evaluated every one of its pairs and got
    # nothing is a full grey bar, and 17 of the corpus's 291 modules are.
    pm = R["per_module"]
    plot = f'<div class="tiles">{grid_tiles}</div>'
    tail = tv = ""
    if pm:
        plot += ('<div class="pmbars">' + bars_emphasis(
            [(x["module"], x["props"], x["pairs"],
              # Candidates and findings first, because they are what the pair
              # count is made of: 6 x 14 = 84 reads as arithmetic, where 84 on
              # its own is a number the reader has to go find the parts of.
              f'{n0(x["cands"])} candidates \u00b7 {n0(x["findings"])} findings '
              f'\u00b7 {n0(x["pairs"])} pairs \u00b7 {n0(x["props"])} proposals \u00b7 '
              f'{pct(x["props"], x["pairs"], 1)} of pairs yielded \u00b7 '
              f'{dur(x["accum_s"])} agent work'
              + (f' \u00b7 {n0(x["fails"])} failed' if x["fails"] else ''))
             for x in pm]) + '</div>')
        tail = legend([
            ("var(--viz-accent)", "Pair yielded a proposal", None),
            # Deliberately not "the finding did not fit": a pair is a question
            # asked, and most answers are a considered no, but 430 pairs across
            # the corpus never answered at all.
            ("var(--viz-rest)", "Pair yielded nothing", None)])
        # Proposals carries its own yield in parentheses rather than in a column
        # of its own: the count is meaningless without the base it is a share of,
        # and reading a rate off two adjacent columns is work the cell can do.
        rows_pm = [
            (x["module"], n0(x["cands"]), n0(x["findings"]), n0(x["pairs"]),
             f'{n0(x["props"])} ({pct(x["props"], x["pairs"], 1)})',
             dur(x["accum_s"]),
             f'{x["accum_s"] / x["stage_s"]:.1f}&times;' if x["stage_s"] else "\u2014")
            for x in pm
        ]
        tp = sum(x["props"] for x in pm)
        rows_pm.append((
            f'all {n0(len(pm))} modules',
            n0(sum(x["cands"] for x in pm)), n0(sum(x["findings"] for x in pm)),
            n0(pairs), f'{n0(tp)} ({pct(tp, pairs, 1)})',
            dur(R["pair_accum_s"]), f"{conc:.1f}&times;" if conc else "\u2014"))
        # &times; has to reach the browser as markup and table_view escapes every
        # cell, so that one column is unescaped here after the fact rather than
        # by teaching the helper an exception.
        tv = ('<div class="tvscroll">' + table_view(
            "tv-rgrid",
            ["Module", "Candidates", "Findings", "Pairs", "Proposals",
             "Agent work", "Concurrency"],
            rows_pm, rows_meta=[""] * len(pm) + ['class="tot"'],
        ).replace("&amp;times;", "&times;") + '</div>')

    # The note is derived, because the cases read very differently and only one
    # of them was true of the run this section was first written against.
    cross_all = len(d["rows"]) * len(R["findings"]) if R["findings"] else 0
    why = ("Pairing happens inside a module: a candidate is only ever paired "
           "with findings from its own module. That is why the total sits far "
           "below every candidate against every finding"
           + (f" ({esc(n0(len(d['rows'])))} &times; "
              f"{esc(n0(len(R['findings'])))} = {esc(n0(cross_all))})"
              if cross_all else "")
           + ". ")
    if pm and R["cap"] is not None:
        why += (f"<b>{esc(n0(R['short']))}</b> of {esc(n0(len(pm)))} modules "
                f"stop at exactly <b>{esc(n0(R['cap']))}</b> pairs, so this run "
                f"also applied a per-module pair budget; the rest evaluate "
                f"their own cross product in full.")
    elif pm and R["short"]:
        why += (f"<b>{esc(n0(R['short']))}</b> of {esc(n0(len(pm)))} modules "
                f"evaluated fewer pairs than their own candidates &times; "
                f"findings, with no single cut-off in common.")
    elif pm:
        why += ("Every module evaluated its own cross product in full &mdash; "
                "each module's pair count is exactly its own candidates "
                "&times; findings.")
    # The yield percentages divide by pairs asked. Where some of those pairs
    # never came back, that is a caveat on every one of them, so it is stated
    # once here rather than repeated per number -- and it is the run-level answer
    # to why a pair can produce nothing without the technique being a bad fit.
    pf = R["pair_fails"]
    if pf:
        why += (f" <b>{esc(n0(pf))}</b> of the {esc(n0(pairs))} pairs "
                f"(<b>{esc(pct(pf, pairs, 1))}</b>) never returned a verdict "
                f"&mdash; the agent call failed or timed out &mdash; so the yield "
                f"percentages are over pairs <i>asked</i>, not pairs answered.")
    dead = sum(1 for x in pm if not x["props"])
    if dead:
        why += (f" <b>{esc(n0(dead))}</b> of {esc(n0(len(pm)))} modules "
                f"evaluated every pair they had and produced nothing at all.")
    why += (" Bar length is the pairs a module evaluated &mdash; its share of "
            "the work &mdash; and the accent inside it is the pairs that "
            "produced something, on the same scale, so a long mostly-grey bar "
            "is a module that worked hard for little."
            " Pair counts are read from the per-pair telemetry, never "
            "multiplied out. Per-pair durations overlap, so their sum is "
            "accumulated agent work &mdash; the concurrency column in the table "
            "is what converts it to each stage&rsquo;s own elapsed time.")

    out.append(chart(
        "tv-rgrid", "The (candidate \u00d7 finding) pair grid",
        "pairs evaluated per module; the accented part yielded a proposal",
        plot, tail, tv, note=why, wide=True))

    # -- 4. works more than one module read ---------------------------------
    shared_card = ""
    sh = R["shared"]
    if sh:
        shared_n = sum(len(g) for g in sh)
        shared_card = (
            f'<div class="card pad">'
            f'<h3 style="margin:0 0 2px;font-size:14px">Works more than one '
            f'module read</h3>'
            f'<p class="sub" style="color:var(--muted);font-size:12px;margin:0 0 12px">'
            f'{esc(n0(len(sh)))} of {esc(n0(R["works"]))} works, '
            f'{esc(n0(shared_n))} findings &mdash; open one to compare what each '
            f'module took from it</p>'
            f'<div class="works">{"".join(work_card(g) for g in sh)}</div>'
            f'<p class="note" style="margin-top:12px">Research runs per module, so '
            f'two modules can independently reach the same source. That is reuse, '
            f'not necessarily waste: the cost is only duplicated where the two '
            f'modules also extracted the <i>same</i> technique, which is what the '
            f'side-by-side summaries are for. Works are matched on the title '
            f'rather than the URL, because the same work arrives at different '
            f'modules under different URLs.</p></div>'
        )

    # -- 5. the catalogue ---------------------------------------------------
    srcs = sorted({f["source_type"] for f in F})
    mods = sorted({f["module"] for f in F if f["module"]})

    # Sorting by date is offered only when the run has dates, and
    # `publication_date` is all-or-nothing per run the way `rank` is: 5 of the 17
    # runs with findings carry it on 100% of them, 11 carry it on none, and
    # idanfr has 3 of 578. A control that reorders nothing is the same empty
    # repetition as a label that reads the same on every row -- and when the
    # coverage is partial the option says so, because sorting 578 rows by a
    # field 3 of them have is a trap unless the reader is told first.
    dated = sum(1 for f in F if f["published"].strip())
    # A placeholder first, then the reset, and they are different things. The
    # headers sort too, and they toggle direction, so a reader can reach states
    # the menu has no preset for -- "proposals, fewest first". A menu that fell
    # back to its first option would then claim a sort that is not in effect, so
    # the fallback is a placeholder that claims nothing and `as listed` is the
    # explicit way back to document order.
    opts = ['<option value="">sort by\u2026</option>',
            '<option value="doc">as listed</option>']
    if dated:
        cov = ("" if dated == n else
               f" \u00b7 {n0(dated)} of {n0(n)} dated")
        opts += [f'<option value="pub:-1">newest first{cov}</option>',
                 f'<option value="pub:1">oldest first{cov}</option>']
    hosts = {_host(f["url"]) for f in F if f["url"]}
    if len(hosts) > 1:
        opts.append(f'<option value="host:1">where published '
                    f'\u00b7 {n0(len(hosts))} sites</option>')
    if len(srcs) > 1:
        opts.append('<option value="src:1">source type</option>')
    opts.append('<option value="props:-1">most proposals</option>')
    sort_sel = ('<select id="fsort" aria-label="Sort findings">'
                + "".join(opts) + '</select>')
    controls = (
        '<div class="controls">'
        '<input id="fq" type="search" placeholder="search title, technique, '
        'module, url…" aria-label="Search findings">'
        '<select id="fsrc" aria-label="Filter by source type">'
        '<option value="">all source types</option>'
        + "".join(f'<option value="{esc(x)}">{esc(x)}</option>' for x in srcs)
        + '</select>'
        '<select id="fmod" aria-label="Filter by module">'
        '<option value="">all modules</option>'
        + "".join(f'<option value="{esc(m)}">{esc(m)}</option>' for m in mods)
        + '</select>'
        '<select id="fuse" aria-label="Filter by whether the finding was used">'
        '<option value="">used or not</option>'
        '<option value="used">produced a proposal</option>'
        '<option value="unused">never used</option></select>'
        + sort_sel
        + '<span class="count" id="fcount"></span>'
        '</div>'
    )
    head = (
        '<thead><tr>'
        '<th data-key="src">Source<span class="ar"></span></th>'
        '<th data-key="title">Finding<span class="ar"></span></th>'
        '<th data-key="module">Module<span class="ar"></span></th>'
        '<th data-key="host">Published<span class="ar"></span></th>'
        '<th class="r" data-key="rate">Pair candidates<span class="ar"></span></th>'
        '<th class="r" data-key="props">Proposals<span class="ar"></span></th>'
        '</tr></thead>'
    )
    # Past roughly twenty rows the card is taller than the screen; below that the
    # cap would only add a scrollbar to a table that already fits.
    tall = " tall" if len(F) > 20 else ""
    table = (f'<div class="card"><div class="tbl-scroll{tall}">'
             f'<table class="fnd">{head}'
             f'{"".join(finding_row(f) for f in F)}</table></div></div>')
    hint = (f'<p class="note" style="color:var(--faint);font-size:11px;'
            f'margin:10px 2px 0">Click a row to open the technique the researcher '
            f'extracted, the evidence it cited, and the proposals it went on to '
            f'produce. Every column header sorts; the <b>sort</b> menu adds '
            f'<b>publication date</b>, which is the second line of the '
            f'<b>Published</b> column. <b>Pair candidates</b> reads '
            f'<i>landed / paired</i>: the denominator is how many of the '
            f'module&rsquo;s candidates this finding was paired with &mdash; read '
            f'from the per-pair telemetry, not multiplied out &mdash; and the '
            f'numerator is how many of those it produced a proposal for. '
            f'Clicking that header sorts by the fraction, not the base. '
            f'<b>Published</b> is the site the '
            f'URL points to &mdash; the artifact records no venue and no author, '
            f'so arXiv preprints group together but NSDI cannot be told from '
            f'OSDI. '
            f'A finding with <b>0 proposals</b> was '
            f'read and paired but never used &mdash; '
            f'<b>{esc(n0(len(R["orphans"])))}</b> of {esc(n0(n))} findings here '
            f'({esc(pct(len(R["orphans"]), n, 0))}).</p>')

    dangling = ""
    if R["dangling"]:
        dangling = (f'<p class="note" style="color:var(--faint);font-size:11px;'
                    f'margin:6px 2px 0"><b>{esc(n0(R["dangling"]))}</b> proposals '
                    f'cite a finding id that is not in the report, so they are '
                    f'counted as grounded but cannot be linked to a source.</p>')

    return (
        sec("What the deep research read",
            f'<div class="charts">{"".join(out)}</div>'
            f'<div class="agents" style="margin-top:14px">{shared_card}</div>')
        + sec(f"Findings catalogue — {n0(n)} findings",
              controls + table + hint + dangling)
    )


def issues(d: dict) -> str:
    """The run's own failure record: what broke, how badly, and what it cost.

    Nothing here is asserted about the *kind* of failure -- v1 hard-coded
    "every issue in this run is a 600 s agent timeout", which is untrue of every
    run that is not the one it was written for (missing runner binaries,
    unparseable CLI envelopes and non-zero exits all land in this list). The note
    reports only what the data says, and the raw messages carry the rest.
    """
    L = VIZ["light"]
    groups = sorted(d["issue_groups"].items(), key=lambda kv: -kv[1])
    if not groups:
        return ""
    sev_fill = {"error": L["critical"], "warning": L["warning"], "info": L["good"]}
    sev_glyph = {"error": "\u2715", "warning": "!", "info": "i"}
    cells = "".join(
        f'<div class="issrow">'
        f'<i class="ic" style="background:{esc(sev_fill.get(sev, L["warning"]))};'
        f'color:{ink_on(sev_fill.get(sev, L["warning"]))}">'
        f'{esc(sev_glyph.get(sev, "!"))}</i>'
        f'<span class="n">{esc(n0(n))}</span>'
        f'<span class="t">{esc(sev)} in {esc(step.replace("_", " "))}</span></div>'
        for (step, sev), n in groups
    )
    msgs = "".join(
        f'<li>{esc(i.get("message") or "")}</li>' for i in d["issues"][:80]
    )
    more = (f'<details class="more"><summary>All '
            f'{esc(n0(len(d["issues"])))} issue messages</summary>'
            f'<ul>{msgs}</ul></details>') if msgs else ""
    n_err = sum(n for (_st, sev), n in groups if sev == "error")
    n_steps = len({st for (st, _sev), _n in groups})
    no_props = sum(1 for r in d["rows"] if not r["proposals"])
    counts = d["status_counts"]
    n_lost = counts.get("DEGRADED", 0) + counts.get("FAILED", 0)
    body = (
        f'<div class="card pad"><div class="iss">{cells}</div>'
        f'<p class="note">'
        f'<b>{esc(n0(len(d["issues"])))}</b> issues across '
        f'<b>{esc(n0(n_steps))}</b> pipeline '
        f'{"step" if n_steps == 1 else "steps"}, '
        f'<b>{esc(n0(n_err))}</b> of them errors. The visible cost is '
        f'<b>{esc(n0(no_props))}</b> candidates that reached the final set with no '
        f'proposal attached, plus the <b>{esc(n0(n_lost))}</b> degraded or failed '
        f'modules above. Individual failure modes are in the messages below &mdash; '
        f'they are not all the same fault, and the page does not claim they are.</p>'
        f'{more}</div>'
    )
    return sec("Run issues", body)


def artifacts(d: dict) -> str:
    ev, ap = d["evolved"], d["applied"]
    if not ev and not ap:
        return ""
    ev_mods = len({c.rsplit("-", 1)[0].removeprefix("cand-") for c in ev})
    tiles = "".join([
        tile("Evolve scaffolds", n0(len(ev)),
             f"candidates packaged across <b>{esc(n0(ev_mods))}</b> modules"),
        tile("Patches applied", n0(len(ap)),
             "candidate carried through to a diff" if len(ap) == 1
             else "candidates carried through to a diff"),
        tile("Awaiting measurement", n0(len(d["rows"])),
             "no candidate has a measured speedup yet"),
    ])
    return sec("Downstream artifacts", f'<div class="tiles">{tiles}</div>')


def score_cell(r: dict) -> str:
    s = r["score"]
    if s is None:
        return '<div class="pred"><span class="v">—</span></div>'
    w = max(0, min(100, int(s)))
    return (f'<div class="pred"><span class="v">{w}</span>'
            f'<span class="glowbar"><i style="width:{w}%"></i></span></div>')


def loc_str(r: dict) -> str:
    if not r["file"]:
        return ""
    if r["line_start"] and r["line_end"]:
        return f'{r["file"]}:{r["line_start"]}–{r["line_end"]}'
    return str(r["file"])


def detail(r: dict) -> str:
    blocks = []
    for head, text, ink in (
        ("What it is", r["description"], True),
        ("How it works today", r["current_approach"], False),
        ("Why it is a candidate", r["evolve_rationale"], False),
        ("Why this impact rating", r["impact_why"], False),
    ):
        if text:
            cls = ' class="ink"' if ink else ""
            blocks.append(f'<div class="d-sec"><h4>{esc(head)}</h4>'
                          f'<p{cls}>{mdi(text)}</p></div>')
    grid = f'<div class="d-grid">{"".join(blocks)}</div>' if blocks else ""

    props = ""
    if r["proposals"]:
        items = []
        for p in r["proposals"]:
            eff = (f'<span class="eff">{esc(p.get("expected_effect"))}</span>'
                   if p.get("expected_effect") else "")
            by = (f'<span class="by">{esc(p.get("author") or "?")}</span>')
            # Provenance. The artifact has carried `source` and `finding_ref_id`
            # all along and this panel used to drop both, so a proposal derived
            # from a researched source was indistinguishable from one an agent
            # wrote out of its own knowledge.
            grounded = (p.get("source") or "") == "research_finding"
            src = ""
            if p.get("_prov"):
                src = (f'<span class="src{" res" if grounded else ""}">'
                       f'{"research" if grounded else "agent knowledge"}</span>')
            body = p.get("description") or p.get("mechanism") or ""
            rat = p.get("rationale") or ""
            extra = (f'<p style="color:var(--faint);font-size:12px">{mdi(rat)}</p>'
                     if rat else "")
            fnd = p.get("_finding")
            fref = ""
            if fnd:
                link = (f'<a href="{esc(fnd["url"])}" target="_blank" '
                        f'rel="noopener noreferrer">{esc(fnd["title"])}</a>'
                        if fnd["url"] else f'<b>{esc(fnd["title"])}</b>')
                fref = (f'<p class="fref">from {esc(fnd["source_type"])}: {link}'
                        f'</p>')
            elif grounded and p.get("finding_ref_id"):
                # Cited a finding the report does not contain. Say so rather than
                # showing a bare chip that implies a source the reader can reach.
                fref = (f'<p class="fref">cites <b>{esc(p["finding_ref_id"])}</b>, '
                        f'which is not in this report</p>')
            items.append(
                f'<div class="prop"><div class="ph"><b>{esc(p.get("title") or "Untitled")}</b>'
                f'{by}{src}{eff}</div><p>{mdi(body)}</p>{extra}{fref}</div>'
            )
        props = (f'<div class="d-props"><h4 style="font-size:10px;letter-spacing:.14em;'
                 f'text-transform:uppercase;color:var(--faint);margin:0;font-weight:700">'
                 f'{esc(len(r["proposals"]))} proposal'
                 f'{"" if len(r["proposals"]) == 1 else "s"}</h4>{"".join(items)}</div>')
    else:
        props = ('<div class="d-props"><p style="color:var(--faint);font-size:12.5px;'
                 'margin:0">No proposal attached — every agent round for this '
                 'candidate timed out.</p></div>')

    arts = []
    if r["evolved"]:
        arts.append(f'evolve scaffold: <code>evolve/{esc(r["module"].replace("/", "_"))}/'
                    f'{esc(r["id"])}/</code>')
    if r["applied"]:
        arts.append(f'applied patch: <code>apply/{esc(r["module"].replace("/", "_"))}/'
                    f'{esc(r["id"])}/apply.patch</code>')
    arts_html = f'<div class="d-arts">{" · ".join(arts)}</div>' if arts else ""

    return f'<div class="d-wrap">{grid}{props}{arts_html}</div>'


HOT_PRIOR = 4.0   # prior weight, in pseudo-candidates


def hotness(rows) -> dict:
    """Per-module hotness, plus the run-level numbers the explanations quote.

    A raw high-impact share says nothing about how much evidence is behind it:
    2-of-4 and 4-of-8 are both 50%, but one candidate changing impact moves the
    first to 25% or 75% and the second only to 37.5% or 62.5%. So the raw share
    is pulled toward the run baseline by HOT_PRIOR pseudo-candidates held at the
    baseline rate:

        score = (high + K * base) / (total + K)

    A well-evidenced rate barely moves; a thin one is dragged most of the way
    home. Volume therefore enters as how far a module is trusted, not as a
    second term to be weighted against the share.

    The cutoff is the baseline itself -- hot means "still above the run average
    once sample size is accounted for" -- so it is the run's own number rather
    than a constant picked to yield a pleasing count.

    Scores are kept at full precision for the sort and rounded only for display:
    two modules that both read 39% must not swap order between views.
    """
    agg: dict[str, tuple[int, int]] = {}
    for r in rows:
        h, t = agg.get(r["module"], (0, 0))
        agg[r["module"]] = (h + (1 if r["impact"] == "high" else 0), t + 1)
    high = sum(h for h, _t in agg.values())
    total = sum(t for _h, t in agg.values())
    base = (100.0 * high / total) if total else 0.0
    mods = {}
    for m, (h, t) in agg.items():
        score = ((100.0 * h + HOT_PRIOR * base) / (t + HOT_PRIOR)) if t else 0.0
        # The high >= 1 gate is belt and braces. With a small total the prior
        # alone can lift a module toward the baseline, and a module that produced
        # no high-impact candidate must never be hot whatever K does.
        mods[m] = {
            "high": h, "total": t,
            "share": (100.0 * h / t) if t else 0.0,
            "score": score,
            "hot": h >= 1 and score >= base,
        }
    hot = [v for v in mods.values() if v["hot"]]
    return {
        "mods": mods, "base": base, "k": HOT_PRIOR,
        "n_hot": len(hot), "n_mods": len(mods),
        "high_hot": sum(v["high"] for v in hot), "high_all": high,
        "cand_hot": sum(v["total"] for v in hot), "cand_all": total,
    }


def hot_blurb(H: dict, plain: bool = False) -> str:
    """The one sentence that defines hot, shared by every control that offers it.

    plain=True for an attribute value: esc() escapes '&', so entity markup cannot
    travel into a title=""; a literal dash can.
    """
    dash = "\u2014" if plain else "&mdash;"
    return (f'ranked by high-impact share after shrinking it toward the run '
            f'baseline of {H["base"]:.1f}% by {n0(int(H["k"]))} pseudo-candidates, '
            f'then cut at that baseline {dash} {H["n_hot"]} of {H["n_mods"]} '
            f'modules, holding {H["high_hot"]} of the {H["high_all"]} '
            f'high-impact candidates')


def hot_example(H: dict) -> str:
    """A worked example drawn from this run: two modules on the same raw share
    that the shrinkage separates.

    The abstract statement ("more evidence wins") is easy to nod at and hard to
    act on. Naming the two modules that actually tie, and the scores they end up
    with, lets a reader check the rule against the table in front of them. If no
    two modules tie, there is nothing to illustrate and the sentence is dropped
    rather than invented.
    """
    ms = [(m, v) for m, v in H["mods"].items() if v["total"] and v["high"]]
    best = None
    for i, (m1, a) in enumerate(ms):
        for m2, b in ms[i + 1:]:
            if abs(a["share"] - b["share"]) > 0.05 or a["total"] == b["total"]:
                continue
            big, small = (a, b) if a["total"] > b["total"] else (b, a)
            bn, sn = (m1, m2) if a["total"] > b["total"] else (m2, m1)
            # Rank candidate pairs by the score gap, not by the count ratio: the
            # sentence exists to show the shrinkage separating a tie, and a pair
            # that ends up 33% against 32% demonstrates nothing a reader can see.
            gap = big["score"] - small["score"]
            if best is None or gap > best[0]:
                best = (gap, bn, big, sn, small)
    if best is None:
        return ("The same rate measured over more candidates is the better bet, "
                "so volume enters as how far a rate is trusted rather than as a "
                "second thing to weigh against it.")
    _g, bn, big, sn, small = best
    return (f'That is what separates two modules on the same raw share: '
            f'<b>{esc(bn.rsplit("/", 1)[-1])}</b> ({big["high"]} of '
            f'{big["total"]} high) and <b>{esc(sn.rsplit("/", 1)[-1])}</b> '
            f'({small["high"]} of {small["total"]}) are both '
            f'{big["share"]:.0f}%, and the one measured over '
            f'{big["total"]} candidates rather than {small["total"]} scores '
            f'{big["score"]:.0f}% against {small["score"]:.0f}%.')


def cand_tree(rows, hm: dict) -> dict:
    """Nest the candidates under their module path: one node per path segment.

    The module qualified name is already a slash-separated path, so the tree is
    read straight out of it rather than invented here. Modules that produced no
    candidates have no rows and so never appear -- the tree is a view of the
    126 candidates, not of the 30 module runs.
    """
    root: dict[str, Any] = {"p": "run", "q": "", "c": []}
    index = {"": root}
    for r in rows:
        cur, path = root, ""
        for seg in [x for x in (r["module"] or "?").split("/") if x]:
            path = f"{path}/{seg}" if path else seg
            node = index.get(path)
            if node is None:
                node = {"p": seg, "q": path, "c": []}
                # Hotness is a property of a module run, so it is stamped on the
                # node whose path IS a module and nowhere else. The browser then
                # only has to read the flag: an intermediate segment must not be
                # scored as if it were a module, because the aggregate of a hot
                # and a cold child can fall below the baseline and would prune a
                # hot module out of the view.
                v = hm.get(path)
                if v:
                    node["s"] = round(v["score"], 4)
                    node["hn"] = v["high"]
                    node["ho"] = 1 if v["hot"] else 0
                index[path] = node
                cur["c"].append(node)
            cur = node
        cur["c"].append({
            "n": r["symbol"], "d": r["id"], "m": r["module"],
            "i": r["impact"], "k": r["rank"],
        })

    def order(n):
        groups = sorted((k for k in n["c"] if "c" in k), key=lambda k: k["p"])
        leaves = sorted((k for k in n["c"] if "c" not in k),
                        key=lambda k: (k["k"] is None, k["k"] or 0, k["n"]))
        n["c"] = groups + leaves
        for k in groups:
            order(k)

    order(root)
    return root


def radial_tree(rows, hm: dict) -> str:
    """The tree view's markup: controls, an empty plot, and the data it needs.

    Layout and collapsing happen in the browser -- a collapse changes every
    angle, so the geometry cannot be baked in here.
    """
    data = json.dumps(cand_tree(rows, hm), separators=(",", ":")).replace("</", "<\\/")
    bar = ('<div class="rtbar">'
           '<button class="tvbtn" id="rtexp">expand all</button>'
           '<button class="tvbtn" id="rtmod">collapse to modules</button>'
           '<span class="sp"></span>'
           '<span>click a branch to zoom into it &middot; click the middle to go '
           'back up &middot; click a leaf to open the row</span>'
           '</div>'
           '<div class="rtcrumb" id="rtcrumb"></div>')
    leg = ('<div class="rtleg">'
           '<span><i style="background:var(--accent)"></i>high impact</span>'
           '<span><i style="background:var(--muted)"></i>medium</span>'
           '<span><i style="background:var(--faint)"></i>low</span>'
           '<span><i style="background:var(--panel);border-color:var(--border-strong)">'
           '</i>module path</span>'
           '<span><i style="background:var(--accent)"></i>folded branch (larger dot)</span>'
           '<span><i style="background:var(--panel);border-color:var(--accent)">'
           '</i>zoom: the ring is rebuilt from the branch you click</span>'
           '</div>')
    return (f'<div class="rtwrap">{bar}<div class="rtplot" id="rtplot"></div>{leg}</div>'
            f'<script type="application/json" id="rtdata">{data}</script>')


def lb_row(r: dict, hot: dict) -> str:
    pill = f'imp-{r["impact"]}' if r["impact"] in IMPACT_ORDER else "imp-low"
    rank = "" if r["rank"] is None else str(r["rank"])
    stage_bits = []
    if r["applied"]:
        stage_bits.append('<span class="tag on"><span class="ic">◆</span>applied</span>')
    if r["evolved"]:
        stage_bits.append('<span class="tag"><span class="ic">▲</span>evolve</span>')
    stage = ("".join(stage_bits) if stage_bits
             else '<span class="none">—</span>')
    stage_key = ("applied " if r["applied"] else "") + ("evolve" if r["evolved"] else "")
    search = " ".join(filter(None, [
        r["symbol"], r["file"] or "", r["rationale"], r["module"], r["id"],
    ])).lower()
    nprops = len(r["proposals"])
    v = hot.get(r["module"]) or {}
    # The badge is always in the markup but only shown in hot mode, so the sort
    # the reader is looking at is legible without widening the table by default.
    # It carries the raw share as well as the score: the score is why the row
    # sits where it does, the share is the number a reader can check by eye.
    hp = (f'<span class="hp">{v["score"]:.0f}% hot &middot; '
          f'{v["high"]}/{v["total"]} high</span>' if v.get("total") else "")
    return (
        f'<tbody class="cand" id="{esc(r["id"])}" data-id="{esc(r["id"])}" '
        f'data-rank="{esc(rank)}" '
        f'data-hot="{v.get("score", 0.0):.4f}" data-hotn="{v.get("high", 0)}" '
        f'data-hoton="{1 if v.get("hot") else 0}" '
        f'data-score="{esc("" if r["score"] is None else r["score"])}" '
        f'data-module="{esc(r["module"])}" data-impact="{esc(r["impact"])}" '
        f'data-props="{nprops}" data-symbol="{esc(r["symbol"])}" '
        f'data-stage="{esc(stage_key.strip())}" data-search="{esc(search)}">'
        f'<tr class="row">'
        f'<td class="r rank">{esc(rank)}</td>'
        f'<td>{score_cell(r)}</td>'
        f'<td><span class="pill {pill}">{esc(r["impact"])}</span></td>'
        f'<td><div class="sym">{esc(r["symbol"])}</div>'
        f'<div class="loc">{esc(loc_str(r))}</div>'
        f'<div class="rat">{esc(r["rationale"])}</div></td>'
        f'<td class="modtag">{brk(r["module"])}{hp}</td>'
        f'<td class="depth">{nprops} prop{"" if nprops == 1 else "s"}</td>'
        f'<td><div class="stage">{stage}</div></td>'
        f'</tr>'
        f'<tr class="detail"><td colspan="7">{detail(r)}</td></tr>'
        f'</tbody>'
    )


def leaderboard(d: dict) -> str:
    rows = d["rows"]
    H = hotness(rows)
    hot = H["mods"]
    mods = sorted({r["module"] for r in rows})
    # Each option carries its module's score and membership so the hot toggle can
    # narrow and reorder the list without re-deriving the rule in JS.
    mod_opts = '<option value="">all modules</option>' + "".join(
        f'<option value="{esc(m)}" data-hot="{hot[m]["score"]:.4f}" '
        f'data-hotn="{hot[m]["high"]}" '
        f'data-hoton="{1 if hot[m]["hot"] else 0}">{esc(m)}</option>' for m in mods)
    controls = (
        '<div class="controls">'
        '<input id="q" type="search" placeholder="search symbol, file, module, rationale…" '
        'aria-label="Search candidates">'
        f'<select id="mod" aria-label="Filter by module">{mod_opts}</select>'
        '<select id="imp" aria-label="Filter by impact">'
        '<option value="">all impact</option><option value="high">high</option>'
        '<option value="medium">medium</option><option value="low">low</option></select>'
        '<select id="stg" aria-label="Filter by downstream stage">'
        '<option value="">any stage</option>'
        '<option value="evolve">has evolve scaffold</option>'
        '<option value="applied">patch applied</option></select>'
        '<span class="count" id="count"></span>'
        '<span class="lbnote" id="lbnote" role="status" aria-live="polite"></span>'
        '</div>'
    )
    head = (
        '<thead><tr>'
        '<th class="r" data-key="rank">#<span class="ar"></span></th>'
        '<th data-key="score">Predicted<span class="ar"></span></th>'
        '<th data-key="impact">Impact<span class="ar"></span></th>'
        '<th data-key="symbol">Candidate<span class="ar"></span></th>'
        '<th data-key="module">Module<span class="ar"></span></th>'
        '<th data-key="props">Depth<span class="ar"></span></th>'
        '<th>Next stage</th>'
        '</tr></thead>'
    )
    body = "".join(lb_row(r, hot) for r in rows)
    # Same rule as the findings catalogue: the long lists get their own viewport,
    # this one two rows deeper (see `.lbtall`).
    tall = " tall lbtall" if len(rows) > 20 else ""
    table = (f'<div class="card"><div class="tbl-scroll{tall}">'
             f'<table class="lb">{head}{body}</table></div></div>')
    hint = (f'<p class="note" style="color:var(--faint);font-size:11px;margin:10px 2px 0">'
            f'Click any row to open the candidate. Column headers sort; the score is '
            f'the judge&rsquo;s predicted 0&ndash;100 ranking, not a measured speedup. '
            f'<b>Hot modules</b> is {hot_blurb(H)}; the badge on each module reads '
            f'that score and the raw count it came from. {hot_example(H)}</p>')
    tree_hint = ('<p class="note" style="color:var(--faint);font-size:11px;margin:10px 2px 0">'
                 'Branches are module-path segments, leaves are candidates coloured by '
                 'impact. The tree shows all candidates and ignores the filters above. '
                 'With <b>hot modules</b> on, a branch survives if it contains a hot '
                 'module &mdash; a path segment is not itself scored, so a hot module '
                 'is never hidden by a lukewarm parent.</p>')
    # The two views are exclusive: this is a change of view, not a second panel
    # revealed beside the first, so the search and filter row travels with the
    # table it drives.
    # The hot toggle sits in the header, not in the filter row: the filter row
    # travels with the table, and this control has to stay reachable while the
    # tree is the visible view.
    switch = (f'<button class="tvbtn" id="hotbtn" aria-pressed="false" '
              f'title="Hot modules: {esc(hot_blurb(H, plain=True))}">hot modules</button>'
              f'<button class="tvbtn" id="lbview" aria-pressed="false">'
              f'radial tree</button>')
    panes = (f'<div id="lbtable">{controls}{table}{hint}</div>'
             f'<div id="lbtree" hidden>{radial_tree(rows, hot)}{tree_hint}</div>')
    return sec(f"Candidate leaderboard — {n0(len(rows))} ranked", panes,
               head_extra=switch)


# --------------------------------------------------------------------------- #
# document
# --------------------------------------------------------------------------- #


def render(d: dict) -> str:
    title = f'Spotlights run — {d["run_id"] or d["run_dir"].name}'
    topbar = (
        '<div class="topbar">'
        '<div class="brand"><span class="beam"></span> Spotlights</div>'
        f'<div class="obj">objective: <b>{esc(d["objective"] or "(none)")}</b></div>'
        '<div class="spacer"></div>'
        '<button class="tgl" id="theme" type="button">flip theme</button>'
        '</div>'
    )
    t = d["timing"] or {}
    wall = t.get("wall_clock_s")
    acc = t.get("accumulated_duration_s")
    speedup = ""
    if wall and acc:
        try:
            speedup = f" · {float(acc) / float(wall):.1f}× parallel across modules"
        except (TypeError, ValueError, ZeroDivisionError):
            speedup = ""
    foot = (
        '<div class="foot">'
        f'run <code>{esc(d["run_id"] or "unknown")}</code> · '
        f'{esc(dur(wall))} wall clock{esc(speedup)}<br>'
        f'rendered from <code>result.json</code>, <code>run_manifest.json</code> and '
        f'<code>sorted/sorted_candidates.json</code> · '
        'no candidate carries a measured speedup yet'
        '</div>'
    )
    body = (
        topbar
        + overview(d)
        + charts(d)
        + agents(d)
        + research(d)
        + artifacts(d)
        + leaderboard(d)
        + issues(d)
        + foot
    )
    return (
        '<!DOCTYPE html>\n<html lang="en">\n<head>\n<meta charset="utf-8">\n'
        '<meta name="viewport" content="width=device-width, initial-scale=1">\n'
        f'<title>{esc(title)}</title>\n<style>{CSS}</style>\n</head>\n<body>\n'
        f'<div class="wrap">\n{body}\n</div>\n<div id="tip" role="status"></div>\n'
        f'<script>{JS}</script>\n</body>\n</html>\n'
    )


