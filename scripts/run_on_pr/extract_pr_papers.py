#!/usr/bin/env python3
"""extract_pr_papers.py — extract the paper URLs a PR's prose cites.

Reads the PR's title + body (+ optionally linked-issue bodies), all fetched by
the harness and **never** shown to the engine, and pulls out *paper* references:
arxiv links, DOIs, and a small allow-list of known paper hosts. Each URL is
canonicalized through the **engine's own** `_normalize_url`
(`module_deep_research/orchestration.py`) so an arxiv ref becomes `arxiv:<id>` —
the same key space `match_papers.py` compares against `Finding.url`.

Extraction is deliberately conservative (design "keep extraction conservative"):
a PR body links many non-paper URLs (the repo, CI, docs), and a false-positive
"cited paper" would corrupt the paper-recall signal. Only arxiv / DOI / known
paper hosts count. When a URL appears inside a markdown link `[title](url)`, the
link text is captured as an optional human `title` (the matcher uses it as a
fallback key).

For **arxiv** refs the link text is an unreliable title source — it is often a
citation label (`[Zandieh et al., arXiv:2504.19874](...)`), a bare word like
`[paper](...)`, or entirely absent (a bare `https://arxiv.org/pdf/<id>` link has
no link text at all). Since the engine surfaces the finding under the paper's
*real* name, such link text never matches as a `--paper-title` fallback — worse,
a generic word could spuriously match an unrelated finding. So for arxiv refs we
resolve the **canonical title from the arxiv API** (keyed on the normalized
`arxiv:<id>`) and use *only* that, never the scraped link text. The API fetch is
best-effort with a short retry; on persistent network/parse failure we leave the
title **absent** rather than degrade to junk link text — the normalized `arxiv:`
URL key still matches, and an absent title simply disables the (unhelpful)
fallback. Pass `--offline` to skip the fetch entirely (title then absent).

Emits `cited_papers.json`:
    {
      "papers": [
        {"raw_url": "https://arxiv.org/abs/2504.19874v2",
         "normalized": "arxiv:2504.19874",
         "kind": "arxiv",                 # arxiv | doi | host
         "title": "Preble: ..."}          # optional
      ],
      "source_fields": ["title", "body"]   # which inputs were scanned
    }

An empty `papers` list is a normal, non-error outcome (the paper signal is then
`n/a` for this PR).

Inputs (any combination; all scanned):
    --pr-json FILE    JSON from `gh pr view <url> --json title,body[,...]`
                      ('-' reads stdin). title/body and any nested issue bodies
                      under closingIssuesReferences[].body are scanned.
    --text STR        Extra raw prose to scan (repeatable).

Usage:
    gh pr view <url> --json title,body | extract_pr_papers.py --pr-json -
    extract_pr_papers.py --pr-json pr_prose.json -o cited_papers.json
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import urllib.error
import urllib.request
from pathlib import Path

# Reuse the engine's URL normalizer verbatim (design: do NOT hand-roll).
_REPO_ROOT = Path(__file__).resolve().parents[2]
_SRC = _REPO_ROOT / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from spotlights_engine.module_deep_research.orchestration import (  # noqa: E402
    _normalize_url,
)

# A bare URL token (stops at whitespace and common markdown/paren delimiters).
_URL_RE = re.compile(r"https?://[^\s<>\)\]\"'}]+", re.IGNORECASE)
# Markdown link: capture link text + url so we can attach an optional title.
_MD_LINK_RE = re.compile(r"\[([^\]]+)\]\((https?://[^\s)]+)\)", re.IGNORECASE)
# A bare DOI reference, e.g. "doi:10.1145/3600006" or "10.1145/3600006.3613145".
_DOI_RE = re.compile(r"\b(?:doi:\s*)?(10\.\d{4,9}/[-._;()/:A-Za-z0-9]+)", re.IGNORECASE)

# Known academic paper hosts (besides arxiv, which is handled explicitly).
_PAPER_HOSTS = (
    "doi.org",
    "dl.acm.org",
    "openreview.net",
    "proceedings.mlr.press",
    "proceedings.neurips.cc",
    "papers.nips.cc",
    "proceedings.iclr.cc",
    "aclanthology.org",
    "ieeexplore.ieee.org",
    "link.springer.com",
    "usenix.org",
    "semanticscholar.org",
)

_TRAILING_PUNCT = ".,;:!?)]}>\"'"


def _classify(url: str) -> str | None:
    """Return the paper kind for `url`, or None if it is not a paper ref.

    Note: a `www.arxiv.org/abs/X` or arxiv-DOI (`doi.org/10.48550/arXiv.X`)
    citation is recognized as a paper, but the engine's `_normalize_url` only
    canonicalizes the bare `arxiv.org/abs|pdf/` prefix to `arxiv:<id>`. So those
    forms normalize to a plain URL key, not `arxiv:<id>`, and URL-matching to an
    engine finding at the canonical arxiv URL relies on the title fallback in
    `match_papers.py`. This is deliberate fidelity to the engine's own dedup
    keys (the design forbids hand-rolling a divergent normalization); we mirror
    `_normalize_url` exactly rather than pre-stripping `www.`/DOI forms.
    """
    low = url.lower()
    if "arxiv.org/abs/" in low or "arxiv.org/pdf/" in low:
        return "arxiv"
    # host match (strip scheme)
    host = re.sub(r"^https?://", "", low).split("/", 1)[0]
    host = host.removeprefix("www.")
    for ph in _PAPER_HOSTS:
        if host == ph or host.endswith("." + ph):
            return "doi" if ph == "doi.org" else "host"
    return None


def _strip_trailing(url: str) -> str:
    return url.rstrip(_TRAILING_PUNCT)


# arxiv Atom API: one entry per id, <title> holds the canonical paper name.
_ARXIV_API = "http://export.arxiv.org/api/query?id_list={id}"
_ARXIV_TITLE_RE = re.compile(r"<entry>.*?<title>(.*?)</title>", re.IGNORECASE | re.DOTALL)


def _fetch_arxiv_title(
    arxiv_id: str, *, timeout: float = 10.0, attempts: int = 3
) -> str | None:
    """Best-effort canonical title for an arxiv id via the public API.

    Returns the whitespace-collapsed <title> of the matching entry, or None on
    persistent network / HTTP / parse failure. The transient failure modes
    (timeout, reset, 5xx) are retried up to `attempts` times before giving up;
    on a final None the caller leaves the title absent rather than falling back
    to unreliable link text.
    """
    req = urllib.request.Request(
        _ARXIV_API.format(id=arxiv_id),
        headers={"User-Agent": "spotlights-run-on-pr/1.0"},
    )
    for _ in range(max(1, attempts)):
        try:
            with urllib.request.urlopen(req, timeout=timeout) as resp:  # noqa: S310 (fixed host)
                body = resp.read().decode("utf-8", "replace")
        except (urllib.error.URLError, OSError, ValueError):
            continue
        m = _ARXIV_TITLE_RE.search(body)
        if not m:
            return None
        title = re.sub(r"\s+", " ", m.group(1)).strip()
        return title or None
    return None


def extract(texts: list[str], *, resolve_arxiv_titles: bool = True) -> list[dict]:
    """Extract conservative paper references from the given prose blocks.

    Returns a deduped list (keyed by normalized URL) of
    {raw_url, normalized, kind, title?}, in first-seen order.

    When `resolve_arxiv_titles` is set (the default), an arxiv ref's title is
    resolved from the arxiv API and overrides any scraped link text (the link
    text is often a citation label or absent). The fetch is best-effort: on
    failure the link-text title, if any, is kept.
    """
    titles: dict[str, str] = {}  # raw_url -> link text (first seen)
    raw_urls: list[str] = []

    for text in texts:
        if not text:
            continue
        # Markdown links first, so we can attach titles.
        for m in _MD_LINK_RE.finditer(text):
            link_text, url = m.group(1).strip(), _strip_trailing(m.group(2))
            if url not in titles and link_text:
                titles[url] = link_text
            raw_urls.append(url)
        # All bare URLs (markdown ones re-appear here harmlessly; deduped below).
        for m in _URL_RE.finditer(text):
            raw_urls.append(_strip_trailing(m.group(0)))
        # Bare DOI references → doi.org URL form.
        for m in _DOI_RE.finditer(text):
            raw_urls.append("https://doi.org/" + m.group(1).rstrip(_TRAILING_PUNCT))

    papers: list[dict] = []
    seen_norm: set[str] = set()
    for url in raw_urls:
        kind = _classify(url)
        if kind is None:
            continue
        normalized = _normalize_url(url)
        if not normalized or normalized in seen_norm:
            continue
        seen_norm.add(normalized)
        entry: dict = {"raw_url": url, "normalized": normalized, "kind": kind}
        is_arxiv = kind == "arxiv" and normalized.startswith("arxiv:")
        if is_arxiv:
            # arxiv link text is unreliable (a citation label, a bare "paper", or
            # absent). Use ONLY the canonical API title; on failure leave the
            # title absent rather than fall back to junk link text — the arxiv:
            # URL key still matches, and a bad title could spuriously match an
            # unrelated finding.
            if resolve_arxiv_titles:
                arxiv_title = _fetch_arxiv_title(normalized.removeprefix("arxiv:"))
                if arxiv_title:
                    entry["title"] = arxiv_title
        elif url in titles:
            entry["title"] = titles[url]
        papers.append(entry)
    return papers


def _collect_texts(args: argparse.Namespace) -> tuple[list[str], list[str]]:
    """Return (texts, source_fields) gathered from the inputs."""
    texts: list[str] = []
    fields: list[str] = []

    if args.pr_json:
        if args.pr_json == "-":
            raw = sys.stdin.read()
        else:
            raw = Path(args.pr_json).read_text(encoding="utf-8")
        obj = json.loads(raw) if raw.strip() else {}
        if isinstance(obj, dict):
            if obj.get("title"):
                texts.append(str(obj["title"]))
                fields.append("title")
            if obj.get("body"):
                texts.append(str(obj["body"]))
                fields.append("body")
            issues = obj.get("closingIssuesReferences")
            if isinstance(issues, list):
                issue_bodies = [str(it.get("body", "")) for it in issues if isinstance(it, dict)]
                issue_bodies = [b for b in issue_bodies if b]
                if issue_bodies:
                    texts.extend(issue_bodies)
                    fields.append("closing_issues")

    for t in args.text or []:
        texts.append(t)
        fields.append("text")

    return texts, fields


def main() -> None:
    ap = argparse.ArgumentParser(description="Extract PR-cited paper URLs.")
    ap.add_argument("--pr-json", help="gh pr view JSON file ('-' for stdin)")
    ap.add_argument("--text", action="append", help="Extra prose to scan (repeatable)")
    ap.add_argument("-o", "--output", default=None, help="Write JSON here (default: stdout)")
    ap.add_argument(
        "--offline",
        action="store_true",
        help="Skip arxiv API title resolution; use scraped link text only.",
    )
    args = ap.parse_args()

    texts, fields = _collect_texts(args)
    papers = extract(texts, resolve_arxiv_titles=not args.offline)
    out = {"papers": papers, "source_fields": fields}

    text = json.dumps(out, indent=2) + "\n"
    if args.output:
        Path(args.output).write_text(text, encoding="utf-8")
    else:
        sys.stdout.write(text)


if __name__ == "__main__":
    main()
