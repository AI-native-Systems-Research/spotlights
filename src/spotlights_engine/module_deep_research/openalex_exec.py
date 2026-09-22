"""OpenAlex deep-research runner.

Queries the free OpenAlex REST API (`https://api.openalex.org/works`) with the
full-text `search` parameter — results come back relevance-ranked
(`relevance_score` desc) by default — then emits the same
`ModuleDeepResearchOutput` wire JSON the Codex/Claude runners produce, so it
fans out through the same `ThreadPoolExecutor` in `orchestration.run_runners`
and its findings merge with the others' verbatim.

No API key required: the public API is free on the "polite pool". Set a
contact `mailto` (option or env) to land in the polite pool and get better
rate limits. An `OPENALEX_API_KEY`, if present, is sent as the `api_key`
param to draw on premium credits — but it is entirely optional.

Query derivation: the runner only receives the rendered research *prompt*
(the `ModuleResearchRunner` protocol), so the search terms are parsed back out
of that prompt — the target module's qualified name plus the caller objective.
Result count is bounded by `max_results` (OpenAlex `per_page`); downstream
`normalize_module_deep_research_output` applies the real per-module cap.
"""

from __future__ import annotations

import json
import os
import re
import time
import urllib.error
import urllib.parse
import urllib.request
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Literal, Protocol

from pydantic import BaseModel, ConfigDict, Field

from spotlights_engine.local_agent.base import AgentExecResult

OPENALEX_API_KEY_ENV = "OPENALEX_API_KEY"
OPENALEX_MAILTO_ENV = "OPENALEX_MAILTO"
OPENALEX_WORKS_URL = "https://api.openalex.org/works"

# Transient-failure retry budget for a single works fetch (see `_fetch_works`).
_FETCH_RETRIES = 4
_FETCH_BACKOFF_SECONDS = 1.5

# Minimum wall-clock gap between consecutive works fetches in this process, to
# stay under the OpenAlex free-pool burst limit (bursts trigger HTTP 429).
# Override via OPENALEX_MIN_INTERVAL_SECONDS. Cross-process bursts are handled
# by retrying 429 with backoff in `_fetch_works`.
_MIN_INTERVAL_SECONDS = float(os.environ.get("OPENALEX_MIN_INTERVAL_SECONDS", "1.2"))
_last_fetch_ts = 0.0

# Fields we ask OpenAlex to return — keeps the response small and fast.
_SELECT_FIELDS = (
    "id",
    "doi",
    "title",
    "display_name",
    "abstract_inverted_index",
    "publication_year",
    "cited_by_count",
    "authorships",
    "primary_location",
)


class OpenAlexRunnerOptions(BaseModel):
    """Options for the OpenAlex REST runner."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    cwd: Path | str = Field(default_factory=Path.cwd)
    # Accepted for runner-interface parity (and the --openalex-model CLI flag);
    # OpenAlex has no model concept, so it is unused.
    model: str | None = None
    # Optional premium credit key. None/absent => free polite-pool access.
    api_key: str | None = None
    api_key_env: str = OPENALEX_API_KEY_ENV
    timeout_seconds: int | None = 30
    env: Mapping[str, str] | None = None

    base_url: str = OPENALEX_WORKS_URL
    # Contact email for the polite pool (better free rate limits). Optional.
    mailto: str | None = None
    mailto_env: str = OPENALEX_MAILTO_ENV
    # Number of works fetched per query (OpenAlex per_page; downstream caps).
    # 20 per keyword pool: each of the (up to `max_queries`) facet queries is its
    # own relevance-ranked pool, so a deep-enough page lets the exact-match paper
    # surface even when it is not the #1 hit of its facet. Trimmed 30->20 to
    # offset the added semantic slice below (fewer pre-dedup findings => fewer
    # downstream step-4 pairs) with negligible recall loss: when a query carries
    # the paper's title/method tokens the exact match collapses into the top ~20.
    max_results: int = 20
    # Per query, ALSO fetch this many meaning-matched works via OpenAlex
    # `search.semantic` (GTE-Large-EN embeddings; cosine similarity), merged
    # after the keyword hits and deduped.
    #   Why: keyword `search` ANDs stemmed tokens and is citation-weighted, so a
    #   fresh, low-cited GT preprint can be MISSED entirely even with the right
    #   words (proven: the Muon paper is absent from the keyword page). The
    #   semantic slice matches by meaning and is NOT citation-weighted, so it
    #   surfaced that same Muon GT at rank #1 on the identical query. It is the
    #   only lever that rescued a hard fresh GT that keyword search misses.
    #   Endpoint per_page ceiling is 50, but 30 is the default: a live probe on
    #   the hard tiny GTs (Muon, ZeroBubble, SiLU/Swish) put every GT the
    #   semantic slice recovers at rank <=12 given a title/method-token query, so
    #   slots 31-50 added zero recall while doubling pre-dedup findings (=> step-4
    #   pairs). 30 keeps a safety margin over the observed rank-12 worst case.
    #   Recall is decided by query phrasing (title/method tokens), not depth: a
    #   generic query MISSES at any size. Set 0 to disable. NOT recency-sorted:
    #   semantic ignores citations, so fresh works already rank fairly.
    semantic_results: int = 30
    # Per query, ALSO fetch this many most-recent works (sort=publication_date
    # :desc) as a second slice, merged after the relevance hits and deduped.
    #   Why: OpenAlex `relevance_score` is citation-weighted, so a brand-new
    #   arXiv preprint with cited_by=1 (exactly the profile of most GT papers,
    #   e.g. the Muon paper 2502.16982, cited=1) ranks BELOW its established,
    #   higher-cited family papers and never makes the relevance page — even
    #   though it matches the query and passes the field/preprint gate. A small
    #   recency slice per query rescues these low-cited-but-exact preprints
    #   without disturbing the relevance ordering (it only appends to the tail).
    #   Set 0 to disable. Bounded by the same `search` + `base_filter`.
    recency_results: int = 3
    # Max distinct OpenAlex searches issued per run = number of facet POOLS. In
    # "codex" mode the query-writer emits up to this many one-per-line queries
    # (each facet gets its own relevance-ranked search); results are merged +
    # deduped. "regex" mode always issues a single query regardless.
    max_queries: int = 5
    # OpenAlex `filter` value ANDed with the relevance search.
    #   type:article|preprint — GT papers are overwhelmingly arXiv PREPRINTS
    #     (e.g. the Muon paper 2502.16982); an `article`-only gate silently
    #     excluded every one of them, so preprints must be allowed through.
    #   primary_topic.field.id — a QUANTITATIVE-TECHNICAL allow-list, not just CS:
    #     17 Computer Science, 26 Mathematics, 22 Engineering,
    #     31 Physics & Astronomy, 18 Decision Sciences.
    #     Rationale: OpenAlex routinely MIS-TOPICS ML/optimization methods into
    #     adjacent technical fields — the Muon optimizer is tagged Physics (the
    #     word "muon" is a particle), "How Much Orthogonalization Does Muon Need?"
    #     is tagged Engineering, operations-research optimizers land in Decision
    #     Sciences. A CS-only (or CS+Math) gate silently drops these true hits.
    #     The five technical fields keep them while still excluding the real noise
    #     domains (biology, medicine, chemistry, social sciences, humanities) that
    #     a bare full-text `search` otherwise drags in. NOTE: 15 is Chemical
    #     Engineering (NOT Mathematics — that is 26); the original 17|15 value was
    #     a bug that leaked chemistry/spectroscopy papers. Precision is recovered
    #     upstream by the query-writer (specific, application-anchored queries)
    #     rather than by a narrow field gate, which trades away recall.
    base_filter: str = "type:article|preprint,primary_topic.field.id:17|26|22|31|18"
    # How the OpenAlex search query is built from the research prompt:
    #   "regex" — deterministic: parse qualified name + objective (no LLM).
    #   "codex" — a Codex query-writer distills the prompt into keywords, with
    #             a regex fallback if Codex yields nothing. Default.
    query_mode: Literal["regex", "codex"] = "codex"


class _QueryWriter(Protocol):
    """Minimal runner interface used to distill a search query from the prompt."""

    def run(self, prompt: str, *, check: bool = ...) -> AgentExecResult: ...


class OpenAlexRunner:
    """Run OpenAlex paper research via the free REST API."""

    name = "openalex"

    def __init__(
        self,
        options: OpenAlexRunnerOptions | None = None,
        *,
        query_writer: _QueryWriter | None = None,
    ) -> None:
        self.options = options or OpenAlexRunnerOptions()
        # Injectable for tests; built lazily from Codex in "codex" query mode.
        self._query_writer = query_writer

    def _optional_api_key(self) -> str | None:
        """Resolve the premium key if configured; None means free access."""
        opt = self.options
        if opt.api_key:
            return opt.api_key
        if opt.env and opt.api_key_env in opt.env:
            return opt.env[opt.api_key_env]
        return os.environ.get(opt.api_key_env) or None

    def _resolve_mailto(self) -> str | None:
        opt = self.options
        if opt.mailto:
            return opt.mailto
        if opt.env and opt.mailto_env in opt.env:
            return opt.env[opt.mailto_env]
        return os.environ.get(opt.mailto_env) or None

    def _build_url(
        self,
        terms: str,
        *,
        sort: str | None = None,
        per_page: int | None = None,
        semantic: bool = False,
    ) -> str:
        # The semantic endpoint caps per_page at 50; keyword search allows 200.
        page_cap = 50 if semantic else 200
        params: dict[str, str] = {
            "per_page": str(max(1, min(per_page or self.options.max_results, page_cap))),
            "select": ",".join(_SELECT_FIELDS),
        }
        if terms:
            # search.semantic ranks by embedding similarity (meaning), not the
            # citation-weighted token match of the default `search`.
            params["search.semantic" if semantic else "search"] = terms
        if sort:
            params["sort"] = sort  # overrides the default relevance ordering
        if self.options.base_filter:
            # search.semantic accepts only a limited filter set — notably NOT
            # primary_topic.field.id — so drop the unsupported clauses (keeping
            # e.g. type:article|preprint) rather than 400 the whole slice.
            flt = (
                _semantic_safe_filter(self.options.base_filter)
                if semantic
                else self.options.base_filter
            )
            if flt:
                params["filter"] = flt
        mailto = self._resolve_mailto()
        if mailto:
            params["mailto"] = mailto
        api_key = self._optional_api_key()
        if api_key:
            params["api_key"] = api_key
        return f"{self.options.base_url}?{urllib.parse.urlencode(params)}"

    def _resolve_queries(self, prompt: str) -> list[str]:
        """Build the list of OpenAlex searches to issue, per `query_mode`.

        "codex" asks a Codex query-writer for up to `max_queries` one-per-line
        searches (one per facet); empty/failed output falls back to the single
        deterministic regex query so a search always runs. "regex" always
        returns exactly one query.
        """
        if self.options.query_mode == "codex":
            queries = self._codex_queries(prompt)
            if queries:
                return queries
        return [_derive_query(prompt)]

    def _codex_queries(self, prompt: str) -> list[str]:
        writer = self._query_writer or self._default_query_writer()
        if writer is None:
            return []
        try:
            result = writer.run(_QUERY_WRITER_INSTRUCTION + _query_brief(prompt), check=False)
        except Exception:
            return []
        if result.returncode != 0 or not result.final_message:
            return []
        return _parse_query_lines(result.final_message, limit=self.options.max_queries)

    def _default_query_writer(self) -> _QueryWriter | None:
        from spotlights_engine.module_deep_research.codex_exec import (
            CodexExecClient,
            CodexExecOptions,
        )

        return CodexExecClient(
            CodexExecOptions(
                cwd=self.options.cwd,
                model=self.options.model,
                # Query writing is a pure text transform: no web/tools needed,
                # and bound the turn so a stuck session can't hang the runner.
                search=False,
                timeout_seconds=self.options.timeout_seconds,
            )
        )

    def run(self, prompt: str, *, check: bool = True) -> AgentExecResult:
        queries = self._resolve_queries(prompt)
        plans = [(q, self._query_urls(q)) for q in queries]
        safe_cmd = ["GET", *(_redact_api_key(u) for _q, us in plans for u in us)]

        per_query: list[tuple[str, str, list]] = []
        first_error: tuple[int, str] | None = None
        for query, urls in plans:
            primary_url = urls[0]
            try:
                works = self._fetch_works(primary_url)
            except urllib.error.HTTPError as exc:
                if check:
                    raise RuntimeError(
                        f"OpenAlex request failed: HTTP {exc.code} {exc.reason}"
                    ) from exc
                if first_error is None:
                    first_error = (exc.code, f"HTTP {exc.code} {exc.reason}")
                continue
            except (
                urllib.error.URLError,
                TimeoutError,
                json.JSONDecodeError,
            ) as exc:
                if check:
                    raise RuntimeError(f"OpenAlex request failed: {exc}") from exc
                if first_error is None:
                    first_error = (1, str(exc))
                continue
            # Best-effort recency slice(s): appended after relevance hits and
            # deduped. A recency-slice failure must never sink the query.
            for extra_url in urls[1:]:
                try:
                    works = _concat_dedup(works, self._fetch_works(extra_url))
                except Exception:
                    pass
            per_query.append((query, primary_url, works))

        # Every query failed: surface the first error like the single-query path.
        if not per_query and first_error is not None:
            return AgentExecResult(
                command=safe_cmd,
                returncode=first_error[0],
                stdout="",
                stderr=first_error[1],
                final_message=None,
                usage=None,
            )

        merged = _merge_works(per_query)
        payload = _works_to_output_json(per_query, merged)
        return AgentExecResult(
            command=safe_cmd,
            returncode=0,
            stdout=payload,
            stderr="",
            final_message=payload,
            usage=None,
        )

    def _query_urls(self, query: str) -> list[str]:
        """URLs to fetch for one query: keyword, then recency, then semantic.

        All extra slices require a query to bound them (`search` present). The
        recency slice (`recency_results > 0`) re-uses the keyword `search` with
        a most-recent-first sort; the semantic slice (`semantic_results > 0`)
        swaps to `search.semantic` for embedding-similarity recall of fresh,
        low-cited GTs the citation-weighted keyword ranking misses.
        """
        urls = [self._build_url(query)]
        if query:
            n = self.options.recency_results
            if n > 0:
                urls.append(
                    self._build_url(query, sort="publication_date:desc", per_page=n)
                )
            s = self.options.semantic_results
            if s > 0:
                urls.append(self._build_url(query, per_page=s, semantic=True))
        return urls

    def _fetch_works(self, url: str) -> list:
        # The semantic endpoint (and occasionally the main one) returns transient
        # 5xx / times out under load; retry a couple of times with backoff before
        # giving up so a slow semantic slice still lands its rescue hits.
        global _last_fetch_ts
        last_exc: Exception | None = None
        for attempt in range(_FETCH_RETRIES + 1):
            # Throttle: never fire two fetches closer than _MIN_INTERVAL_SECONDS.
            gap = time.monotonic() - _last_fetch_ts
            if gap < _MIN_INTERVAL_SECONDS:
                time.sleep(_MIN_INTERVAL_SECONDS - gap)
            _last_fetch_ts = time.monotonic()
            try:
                request = urllib.request.Request(url, headers=self._headers())
                with urllib.request.urlopen(
                    request, timeout=self.options.timeout_seconds
                ) as resp:
                    body = resp.read().decode("utf-8")
                data = json.loads(body)
                works = data.get("results") if isinstance(data, dict) else None
                return works if isinstance(works, list) else []
            except (urllib.error.HTTPError, urllib.error.URLError, TimeoutError) as exc:
                # 429 (Too Many Requests) is retryable — the burst limit clears
                # after a short wait; honor Retry-After when the server sends it.
                is_http = isinstance(exc, urllib.error.HTTPError)
                retryable = (not is_http) or exc.code >= 500 or exc.code == 429
                if not retryable or attempt == _FETCH_RETRIES:
                    raise
                last_exc = exc
                delay = _FETCH_BACKOFF_SECONDS * (attempt + 1)
                if is_http and exc.code == 429:
                    try:
                        delay = max(delay, float(exc.headers.get("Retry-After", 0)))
                    except (TypeError, ValueError):
                        pass
                time.sleep(delay)
        assert last_exc is not None  # unreachable; loop either returns or raises
        raise last_exc

    def _headers(self) -> dict[str, str]:
        mailto = self._resolve_mailto()
        ua = "spotlights-engine/openalex"
        if mailto:
            ua += f" (mailto:{mailto})"
        return {"User-Agent": ua, "Accept": "application/json"}


# --------------------------------------------------------------------------
# Query derivation
# --------------------------------------------------------------------------

_QUALIFIED_NAME_RE = re.compile(r"^Qualified name:\s*(.+)$", re.MULTILINE)
_OBJECTIVE_RE = re.compile(r"^Objective:\s*(.+)$", re.MULTILINE)

# Prepended to the research prompt when query_mode="codex". Teaches the model
# how OpenAlex `search` ranks, then asks for SEVERAL one-per-line queries — one
# per facet — which the runner searches independently and merges. Multiple
# focused queries beat one bloated query because OpenAlex ANDs words together,
# so a long single query collapses to zero hits.
_QUERY_WRITER_INSTRUCTION = (
    "You are constructing queries for the OpenAlex academic-paper search API. "
    "Read the research brief below — a target code module plus the caller's "
    "objective — and output a SHORT LIST of OpenAlex `search` strings that "
    "together surface papers describing the underlying technique or algorithm. "
    "Each query is searched separately and the results are merged, so make the "
    "queries cover DIFFERENT facets (e.g. the core method, the problem it "
    "solves, a well-known synonym) rather than restating one idea.\n"
    "\n"
    "How OpenAlex search works, so write to exploit it:\n"
    "- It searches titles, abstracts, and full text with stemming and "
    "stop-word removal, and ranks by relevance (text match + citation weight; "
    "terms appearing close together rank higher).\n"
    "- Space-separated words are ANDed by default, so every extra word narrows "
    "the results. Keep each query tight: 2-6 high-signal concepts, no filler.\n"
    '- Wrap a canonical multi-word term in double quotes for an exact phrase, '
    'e.g. \"key-value cache\".\n'
    "- Use UPPERCASE OR with parentheses to allow a synonym for one concept, "
    'e.g. (\"KV cache\" OR \"key-value cache\") compression.\n'
    "- Use research-paper vocabulary describing the method/problem "
    "(\"attention\", \"quantization\", \"speculative decoding\"); NEVER file "
    "paths, class or variable names, or the host framework/library name unless "
    "it is itself the research subject.\n"
    "- Do NOT emit a bare generic term on its own (\"optimization\", \"deep "
    "learning\", \"neural network\", \"machine learning\", \"gradient descent\", "
    "\"matrix\"): alone they match tens of thousands of off-topic papers. Always "
    "pair a generic with the SPECIFIC method, structure, or problem "
    '(e.g. \"orthogonalized momentum matrix optimizer\", not \"optimization\").\n'
    "- Prefer the method's proper name and its algorithmic mechanism if known "
    '(e.g. \"Newton-Schulz iteration orthogonalization\", \"Muon optimizer\").\n'
    "- ANCHOR a method/algorithm name with its application domain in the SAME "
    "query — a bare method name matches the wrong field (e.g. \"Newton-Schulz "
    "orthogonalization\" alone returns pure-mathematics matrix-iteration papers; "
    "\"Newton-Schulz orthogonalization momentum optimizer neural network\" "
    "returns the machine-learning application). Add 2-3 application words "
    "(the task, the model class, or \"optimizer\"/\"training\") to every "
    "method-named query.\n"
    "- The target module and its hot spots (above) name the concrete method — "
    "USE those names verbatim; they are the research subject, not forbidden "
    "framework identifiers.\n"
    "- Do not use the NOT operator.\n"
    "\n"
    "Output up to 5 queries, ONE PER LINE, ordered most to least important. "
    "Favor specificity over breadth — a precise 3-5 concept query beats a vague "
    "1-2 word one. No numbering, no bullets, no markdown, no blank lines, no "
    "explanation — just the query strings, each on its own line.\n"
    "\n"
    "--- RESEARCH BRIEF ---\n"
)

# The rendered research prompt continues past the semantic brief into agent
# execution instructions (Workflow, Output rules, the ModuleDeepResearchOutput
# JSON schema dump). Those trailing sections name wire fields like `findings`,
# `issues`, `severity`, and the step id `module_deep_research`; a query-writer
# fed the whole prompt latches onto those tokens (recency) and emits them as
# OpenAlex queries. Cut at the first execution-section header so the writer
# only sees module + objective + hot spots.
_QUERY_BRIEF_CUT_RE = re.compile(r"\n(?:Workflow:|Output rules:)\n")


def _query_brief(prompt: str) -> str:
    """Trim the rendered research prompt to its semantic brief for query writing."""
    return _QUERY_BRIEF_CUT_RE.split(prompt, maxsplit=1)[0].rstrip()

# Characters allowed through in a sanitized query: word chars, spaces, and the
# OpenAlex operators we let the model use (double quotes, parentheses, hyphen).
_QUERY_ALLOWED_RE = re.compile(r'[^0-9A-Za-z "()\-]+')
# Leading label ("Query:"), enumerator ("1.", "-", "*") the model may prepend.
_QUERY_PREFIX_RE = re.compile(r"(?i)^(?:\d+[.)]|[-*•]|query|keywords|search)\s*[:.\-)]*\s*")


def _sanitize_line(line: str) -> str:
    """Clean one candidate query line into a valid OpenAlex `search` string.

    Preserves the operators the instruction permits (quoted phrases, `OR`,
    parentheses) while stripping labels/enumerators, markdown, and stray
    punctuation. Drops all quotes if unbalanced, since a dangling quote breaks
    the phrase match.
    """
    line = line.strip().strip("`").strip()
    line = _QUERY_PREFIX_RE.sub("", line).strip()
    line = _QUERY_ALLOWED_RE.sub(" ", line)
    if line.count('"') % 2:
        line = line.replace('"', " ")
    return " ".join(line.split())[:256].strip()


def _sanitize_query(raw: str) -> str:
    """Reduce a reply to a single clean query (the last non-empty line)."""
    lines = [ln for ln in raw.splitlines() if ln.strip()]
    return _sanitize_line(lines[-1]) if lines else ""


def _parse_query_lines(raw: str, *, limit: int) -> list[str]:
    """Parse a multi-line reply into up to `limit` distinct OpenAlex queries."""
    out: list[str] = []
    seen: set[str] = set()
    for line in raw.splitlines():
        query = _sanitize_line(line)
        if not query:
            continue
        key = query.lower()
        if key in seen:
            continue
        seen.add(key)
        out.append(query)
        if len(out) >= max(1, limit):
            break
    return out


def _derive_query(prompt: str) -> str:
    """Extract OpenAlex search terms from the rendered research prompt.

    Combines the target module's qualified name (path separators and dashes
    become spaces) with the caller objective. Returns "" when neither is
    present, in which case the caller searches on the base filter alone.
    """
    parts: list[str] = []

    qn_match = _QUALIFIED_NAME_RE.search(prompt)
    if qn_match:
        module_terms = re.sub(r"[/._-]+", " ", qn_match.group(1)).strip()
        if module_terms:
            parts.append(module_terms)

    obj_match = _OBJECTIVE_RE.search(prompt)
    if obj_match:
        objective = obj_match.group(1).strip()
        if objective and objective.lower() != "(none)":
            parts.append(objective)

    return " ".join(" ".join(parts).split()).strip()


# --------------------------------------------------------------------------
# OpenAlex work -> wire findings
# --------------------------------------------------------------------------


def _reconstruct_abstract(inverted_index: object) -> str:
    """Rebuild abstract text from OpenAlex's `abstract_inverted_index`."""
    if not isinstance(inverted_index, Mapping) or not inverted_index:
        return ""
    positioned: list[tuple[int, str]] = []
    for word, positions in inverted_index.items():
        if not isinstance(positions, Sequence):
            continue
        for pos in positions:
            if isinstance(pos, int):
                positioned.append((pos, str(word)))
    positioned.sort(key=lambda item: item[0])
    return " ".join(word for _, word in positioned)


def _first_sentences(text: str, *, limit: int = 2) -> str:
    if not text:
        return ""
    sentences = re.split(r"(?<=[.!?])\s+", text.strip())
    return " ".join(sentences[:limit]).strip()


def _venue(work: dict) -> str:
    location = work.get("primary_location")
    if isinstance(location, dict):
        source = location.get("source")
        if isinstance(source, dict):
            name = source.get("display_name")
            if isinstance(name, str) and name.strip():
                return name.strip()
    return ""


def _authors(work: dict, *, limit: int = 3) -> str:
    authorships = work.get("authorships")
    if not isinstance(authorships, list):
        return ""
    names: list[str] = []
    for entry in authorships:
        if not isinstance(entry, dict):
            continue
        author = entry.get("author")
        if isinstance(author, dict):
            name = author.get("display_name")
            if isinstance(name, str) and name.strip():
                names.append(name.strip())
        if len(names) >= limit:
            break
    if not names:
        return ""
    joined = ", ".join(names)
    total = len(authorships)
    if total > len(names):
        joined += " et al."
    return joined


def _work_to_finding(work: dict, index: int, *, query: str) -> dict | None:
    if not isinstance(work, dict):
        return None
    title = work.get("title") or work.get("display_name")
    if not isinstance(title, str) or not title.strip():
        return None

    url = work.get("doi") or work.get("id")
    if not isinstance(url, str) or not url.strip():
        return None

    abstract = _reconstruct_abstract(work.get("abstract_inverted_index"))
    year = work.get("publication_year")
    cited = work.get("cited_by_count")
    oa_id = work.get("id") or "unknown"
    venue = _venue(work)
    authors = _authors(work)

    summary_body = _first_sentences(abstract) or (
        "OpenAlex work matched the module search; no abstract was available "
        "to summarize the contributed technique."
    )
    context_bits = [b for b in (venue, str(year) if isinstance(year, int) else "") if b]
    context = f" ({'; '.join(context_bits)})" if context_bits else ""
    technique_summary = (
        f"{summary_body}{context} Surfaced by OpenAlex for query '{query}'."
        if query
        else f"{summary_body}{context}"
    )

    evidence_bits: list[str] = []
    if abstract:
        quote = _first_sentences(abstract, limit=1) or abstract[:280]
        evidence_bits.append(f'"{quote}"')
    if authors:
        evidence_bits.append(authors)
    pointer = f"OpenAlex {oa_id}"
    if isinstance(cited, int):
        pointer += f", cited_by={cited}"
    evidence_bits.append(pointer)
    supporting_evidence = " — ".join(evidence_bits)

    return {
        "finding_id": f"find-{index:04d}",
        "title": title.strip(),
        "url": url.strip(),
        "source_type": "paper",
        "technique_summary": technique_summary,
        "supporting_evidence": supporting_evidence,
    }


def _work_key(work: object) -> str | None:
    """Dedup key for a work: normalized DOI, else OpenAlex id, else None."""
    if not isinstance(work, dict):
        return None
    doi = work.get("doi")
    if isinstance(doi, str) and doi.strip():
        return doi.strip().lower()
    oa_id = work.get("id")
    if isinstance(oa_id, str) and oa_id.strip():
        return oa_id.strip().lower()
    return None


def _concat_dedup(primary: list, extra: list) -> list:
    """Append `extra` works onto `primary`, dropping DOI/OpenAlex-id repeats.

    Order-preserving: relevance hits stay first, recency-slice hits follow.
    Works without a usable key are always kept (never deduped).
    """
    seen = {k for w in primary if (k := _work_key(w)) is not None}
    out = list(primary)
    for work in extra:
        key = _work_key(work)
        if key is not None:
            if key in seen:
                continue
            seen.add(key)
        out.append(work)
    return out


def _merge_works(
    per_query: Sequence[tuple[str, str, list]],
) -> list[tuple[dict, str]]:
    """Round-robin interleave works across queries, deduped by DOI/OpenAlex id.

    Takes the top hit of each query first, then each second hit, and so on, so
    no single facet dominates the merged list. Each kept work is paired with the
    query that surfaced it (first occurrence wins). Works without a usable key
    are kept but never deduped.
    """
    from itertools import zip_longest

    columns = [
        [(w, query) for w in works if isinstance(w, dict)]
        for query, _url, works in per_query
    ]
    merged: list[tuple[dict, str]] = []
    seen: set[str] = set()
    for row in zip_longest(*columns):
        for item in row:
            if item is None:
                continue
            work, query = item
            key = _work_key(work)
            if key is not None:
                if key in seen:
                    continue
                seen.add(key)
            merged.append((work, query))
    return merged


def _works_to_output_json(
    per_query: Sequence[tuple[str, str, list]],
    merged: Sequence[tuple[dict, str]],
) -> str:
    """Serialize merged OpenAlex works into ModuleDeepResearchOutput wire JSON.

    `merged` is the deduped round-robin list of (work, source_query) pairs that
    becomes the findings; `per_query` supplies one `search_queries` entry per
    issued query, each previewing that query's own (pre-merge) hits.
    """
    findings: list[dict] = []
    for work, query in merged:
        finding = _work_to_finding(work, len(findings) + 1, query=query)
        if finding is not None:
            findings.append(finding)

    issues: list[dict] = []
    if not findings:
        joined = ", ".join(f"'{q}'" for q, _u, _w in per_query) or "(none)"
        issues.append(
            {
                "step": "module_deep_research",
                "severity": "error",
                "message": f"OpenAlex returned no usable works for queries {joined}.",
                "recoverable": True,
            }
        )

    search_queries: list[dict] = []
    for query, url, works in per_query:
        results: list[dict] = []
        for work in works:
            finding = _work_to_finding(work, len(results) + 1, query=query)
            if finding is not None:
                results.append(
                    {
                        "title": finding["title"],
                        "url": finding["url"],
                        "snippet": finding["technique_summary"],
                    }
                )
        search_queries.append(
            {
                "query": query or _redact_api_key(url),
                "tool": "openalex",
                "results": results,
            }
        )

    output = {
        "findings": findings,
        "issues": issues,
        "search_queries": search_queries,
    }
    return json.dumps(output)


# Filter keys OpenAlex `search.semantic` accepts (others 400 the request).
# From the endpoint's own error message; notably excludes topic/field filters.
_SEMANTIC_FILTER_KEYS = frozenset(
    {
        "author.id",
        "authorships.author.id",
        "authorships.institutions.id",
        "authorships.institutions.lineage",
        "funders.id",
        "has_abstract",
        "has_fulltext",
        "institution.id",
        "institutions.id",
        "is_oa",
        "is_retracted",
        "language",
        "open_access.is_oa",
        "primary_location.license",
        "primary_location.source.id",
        "publication_year",
        "type",
    }
)


def _semantic_safe_filter(base_filter: str) -> str:
    """Keep only the comma-clauses `search.semantic` supports (e.g. `type`).

    OpenAlex `filter` is comma-separated `key:value` AND-clauses. The semantic
    endpoint rejects most keys (topic/field especially), so drop any clause
    whose key is not in `_SEMANTIC_FILTER_KEYS`; returns "" if none survive.
    """
    kept = [
        clause
        for clause in base_filter.split(",")
        if clause.split(":", 1)[0].strip() in _SEMANTIC_FILTER_KEYS
    ]
    return ",".join(kept)


def _redact_api_key(url: str) -> str:
    return re.sub(r"(api_key=)[^&]+", r"\1***", url)


__all__ = [
    "OPENALEX_API_KEY_ENV",
    "OPENALEX_MAILTO_ENV",
    "OPENALEX_WORKS_URL",
    "OpenAlexRunner",
    "OpenAlexRunnerOptions",
]
