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
import logging
import os
import re
import threading
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
_log = logging.getLogger(__name__)

_FETCH_RETRIES = 4
_FETCH_BACKOFF_SECONDS = 1.5

# Hard cap on any single retry backoff, including a server-sent Retry-After.
# OpenAlex answers a quota-exhausted 429 with a multi-hour Retry-After (observed
# ~25000s); honoring it verbatim makes the fetch sleep for hours at 0% CPU,
# which looks exactly like a hung run. Cap the wait so the retry budget is spent
# quickly and the fetch fails fast (letting deep_research finish empty) instead
# of blocking the whole run. Override via OPENALEX_MAX_BACKOFF_SECONDS.
_MAX_BACKOFF_SECONDS = float(os.environ.get("OPENALEX_MAX_BACKOFF_SECONDS", "60"))

# Extra backoff FLOOR for gateway 5xx (502/503/504). A 504 Gateway Timeout means
# the OpenAlex edge is overloaded/slow; the default linear backoff (1.5s..6s) is
# often too short to let it recover, so every retry re-hits the timeout and the
# fetch gives up with 0 works. When >0, a gateway 5xx waits at least this long
# (still clamped by _MAX_BACKOFF_SECONDS). 0 disables (default) → byte-identical
# to prior behavior. Override via OPENALEX_GATEWAY_BACKOFF_SECONDS.
_GATEWAY_BACKOFF_SECONDS = float(
    os.environ.get("OPENALEX_GATEWAY_BACKOFF_SECONDS", "0")
)

# Minimum wall-clock gap between consecutive works fetches in this process, to
# stay under the OpenAlex free-pool burst limit (bursts trigger HTTP 429).
# Override via OPENALEX_MIN_INTERVAL_SECONDS. Cross-process bursts are handled
# by retrying 429 with backoff in `_fetch_works`.
_MIN_INTERVAL_SECONDS = float(os.environ.get("OPENALEX_MIN_INTERVAL_SECONDS", "0.5"))
_last_fetch_ts = 0.0

# Extra wall-clock slack over the socket timeout for the hard fetch bound below.
_FETCH_HARD_MARGIN_SECONDS = 10.0


def _read_url_hard_bounded(request: urllib.request.Request, timeout_s: float | None) -> str:
    """Fetch a URL body under a hard wall-clock bound the socket timeout can't reach.

    `urllib`'s `timeout=` only covers individual socket recv/connect ops once the
    kernel is in a normal blocking read. It does NOT cover a native `getaddrinfo`
    or a TLS/connect that black-holes — the exact failure that hung a real run
    for 6h with the nominal 30s timeout never firing (14 CLOSE_WAIT + 99 stuck
    ESTABLISHED sockets to a Cloudflare IPv6 route). So we run the blocking fetch
    on a daemon thread and `join` it under a hard deadline: if the socket op is
    wedged in a native call, the join returns anyway and we raise `TimeoutError`
    (retryable in `_fetch_works`). The orphaned daemon thread dies at process
    exit — acceptable in a short-lived per-run engine.
    """
    socket_timeout = timeout_s if timeout_s is not None else 30.0
    box: dict[str, object] = {}

    def _work() -> None:
        try:
            with urllib.request.urlopen(request, timeout=socket_timeout) as resp:
                box["body"] = resp.read().decode("utf-8")
        except BaseException as exc:  # noqa: BLE001 - surfaced to the caller below
            box["exc"] = exc

    worker = threading.Thread(target=_work, daemon=True)
    worker.start()
    hard = socket_timeout + _FETCH_HARD_MARGIN_SECONDS
    worker.join(hard)
    if worker.is_alive():
        raise TimeoutError(
            f"OpenAlex fetch exceeded hard {hard:.0f}s bound "
            "— socket op stuck in a native call (DNS/connect/TLS black-hole)"
        )
    if "exc" in box:
        raise box["exc"]  # type: ignore[misc]
    return box["body"]  # type: ignore[return-value]

def _int_env(name: str, default: int) -> int:
    """Read an int from env, falling back to `default` on unset/blank/garbage."""
    raw = os.environ.get(name)
    if raw is None or not raw.strip():
        return default
    try:
        return int(raw)
    except ValueError:
        return default


def _bool_env(name: str) -> bool:
    return os.environ.get(name, "").strip().lower() in {"1", "true", "yes", "on"}


# Post-fetch relevance gate (opt-in). A fetched work is kept only when its
# title+abstract share at least this many significant tokens with the query that
# surfaced it. 0 disables the gate (default) → byte-identical to prior behavior.
#   Why: run-to-run finding overlap was low (Jaccard ~0.14) because the recall
#   slices — especially the semantic slice, which drops the field filter (only
#   `type` survives `_semantic_safe_filter`) and ignores citations — drag in
#   ~200 off-topic papers whose set churns between runs. Requiring a minimum
#   token overlap with the query strips that volatile tail while keeping the
#   stable relevant core, which both raises precision AND stabilizes the set.
#   Override via OPENALEX_RELEVANCE_MIN_OVERLAP (2 is a good starting point).
_RELEVANCE_MIN_OVERLAP = _int_env("OPENALEX_RELEVANCE_MIN_OVERLAP", 0)

# Canonical, citation-neutral ordering of the merged findings (opt-in). When on,
# the deduped findings are sorted by their OpenAlex work key so two runs over the
# same work set produce the same finding ORDER — which stabilizes downstream
# step-4 finding pairing (order-sensitive) and thus the proposals. Off (default)
# keeps the round-robin merge order. Override via OPENALEX_STABLE_SORT.
_STABLE_SORT = _bool_env("OPENALEX_STABLE_SORT")

# Canonicalize the research brief before query-gen (opt-in). The candidate
# description is written by the discovery agent at temperature>0, so two runs
# over the SAME code emit differently-worded briefs (different tensor notation,
# synonyms, even impact rating). The query-writer faithfully extracts from that
# brief, so different briefs → different queries → different papers → low
# run-to-run finding overlap. When on, an LLM first rewrites the brief into a
# FIXED canonical form (7 labeled lines, canonical notation, controlled-vocab
# TERMS) so two equivalent briefs collapse to byte-identical text; the writer
# then extracts from the stable canonical brief. Off (default) → byte-identical
# to prior behavior. Override via OPENALEX_CANONICALIZE_BRIEF.
_CANONICALIZE_BRIEF = _bool_env("OPENALEX_CANONICALIZE_BRIEF")

# Query tokens that carry no topical signal — OpenAlex boolean operators and
# common stop words the query-writer may emit; excluded from the overlap count.
_QUERY_NOISE_TOKENS = frozenset(
    {"or", "and", "the", "for", "with", "of", "in", "on", "to", "an"}
)
_WORD_RE = re.compile(r"[A-Za-z][A-Za-z0-9+\-]*")

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
    max_results: int = Field(default_factory=lambda: _int_env("OPENALEX_MAX_RESULTS", 6))
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
    semantic_results: int = Field(
        default_factory=lambda: _int_env("OPENALEX_SEMANTIC_RESULTS", 32)
    )
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
    recency_results: int = Field(
        default_factory=lambda: _int_env("OPENALEX_RECENCY_RESULTS", 2)
    )
    # Max distinct OpenAlex searches issued per run = number of facet POOLS. In
    # "codex" mode the query-writer emits up to this many one-per-line queries
    # (each facet gets its own relevance-ranked search); results are merged +
    # deduped. "regex" mode always issues a single query regardless.
    max_queries: int = Field(default_factory=lambda: _int_env("OPENALEX_MAX_QUERIES", 5))
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
    # Determinism knobs for the "codex" query-writer. The writer is a free-text
    # LLM: at default sampling it re-paraphrases the same intent into different
    # keyword strings each run (measured wording Jaccard ~0.05 across whole-run
    # reruns), and since OpenAlex keyword `search` is wording-sensitive, that
    # alone tanks cross-run paper overlap. Pinning temperature=0 + a fixed seed
    # makes the writer emit a near-stable query set (0.2 keeps a little slack so
    # facet coverage isn't collapsed to one phrasing). Complemented by post-hoc
    # query canonicalization (see `_canonicalize_queries`) and the semantic-heavy
    # pool (semantic recall is wording-insensitive). None = leave the CLI default.
    query_writer_temperature: float | None = 0.1
    query_writer_seed: int | None = 7
    # Brief canonicalization is a stricter transform than query-gen: its whole
    # job is to collapse two differently-worded briefs to ONE byte-identical
    # text, so it wants the lowest usable temperature (no facet-coverage slack).
    # A sweep (no seed, codex CLI) held m3/m4 5/5 at 0.001 and 0.1 but broke at
    # 0.5 (a GOAL-CLASS flip pass-2 can't re-derive), so canonicalization runs at
    # 0.05 — deep inside the stable band, a touch of slack over 0.001.
    canonicalize_temperature: float | None = 0.05
    canonicalize_seed: int | None = 7


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
        # Rotating pointer into the resolved key pool; advanced on a blocked
        # (429/402/403) fetch so an exhausted key hands off to the next one.
        self._key_idx = 0

    def _api_keys(self) -> list[str]:
        """Ordered, de-duped premium-key pool. Empty => free polite-pool access.

        Sources, in order: explicit ``options.api_key``; a comma/space list in
        ``OPENALEX_API_KEYS``; then the numbered singletons ``OPENALEX_API_KEY``,
        ``OPENALEX_API_KEY_2``, ``OPENALEX_API_KEY_3`` ... (env or ``options.env``).
        Rotation across this pool lets a 429/exhausted key fail over to the next.
        """
        opt = self.options
        env = dict(os.environ)
        if opt.env:
            env.update(opt.env)
        keys: list[str] = []
        if opt.api_key:
            keys.append(opt.api_key)
        for raw in re.split(r"[,\s]+", env.get("OPENALEX_API_KEYS", "")):
            if raw.strip():
                keys.append(raw.strip())
        base = opt.api_key_env
        for name in (base, *(f"{base}_{i}" for i in range(2, 10))):
            v = env.get(name)
            if v and v.strip():
                keys.append(v.strip())
        seen: set[str] = set()
        return [k for k in keys if not (k in seen or seen.add(k))]

    def _optional_api_key(self) -> str | None:
        """Current premium key from the rotating pool; None means free access."""
        keys = self._api_keys()
        if not keys:
            return None
        return keys[self._key_idx % len(keys)]

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
        # Opt-in: query the full expanded corpus (datasets + repository records,
        # formerly "XPAC") instead of the curated core, for ~10-60% more works.
        if _bool_env("OPENALEX_CORPUS_ALL"):
            params["corpus"] = "all"
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
        brief = _query_brief(prompt)
        if _CANONICALIZE_BRIEF:
            canon_writer = self._query_writer or self._canonicalize_writer()
            brief = self._canonicalize_brief(canon_writer or writer, brief)
        _log.info(
            "openalex: codex query-gen start (timeout=%ss)",
            self.options.timeout_seconds,
        )
        started = time.monotonic()
        try:
            result = writer.run(_QUERY_WRITER_INSTRUCTION + brief, check=False)
        except Exception as exc:
            _log.warning(
                "openalex: codex query-gen failed in %.1fs (%s) — "
                "falling back to regex query",
                time.monotonic() - started,
                exc,
            )
            return []
        if result.returncode != 0 or not result.final_message:
            _log.warning(
                "openalex: codex query-gen empty in %.1fs (rc=%s) — "
                "falling back to regex query",
                time.monotonic() - started,
                result.returncode,
            )
            return []
        queries = _parse_query_lines(result.final_message, limit=self.options.max_queries)
        _log.info(
            "openalex: codex query-gen done in %.1fs — %d quer%s",
            time.monotonic() - started,
            len(queries),
            "y" if len(queries) == 1 else "ies",
        )
        return queries

    def _canonicalize_brief(self, writer: "_QueryWriter", brief: str) -> str:
        """Rewrite `brief` into the fixed canonical form (opt-in, best-effort).

        The discovery agent writes the candidate description at temperature>0, so
        two runs over the same code produce differently-worded briefs and thus
        different queries. Normalizing the brief first collapses those variants to
        one canonical text, stabilizing the queries. Any failure (non-zero exit,
        empty reply, exception) falls back to the original brief so canonicalize
        can never make a run worse than the un-canonicalized path.
        """
        started = time.monotonic()
        _log.info("openalex: canonicalize brief start (timeout=%ss)", self.options.timeout_seconds)
        try:
            result = writer.run(_CANONICALIZE_INSTRUCTION + brief, check=False)
        except Exception as exc:
            _log.warning(
                "openalex: canonicalize brief failed in %.1fs (%s) — using raw brief",
                time.monotonic() - started,
                exc,
            )
            return brief
        canonical = (result.final_message or "").strip()
        if result.returncode != 0 or not canonical:
            _log.warning(
                "openalex: canonicalize brief empty in %.1fs (rc=%s) — using raw brief",
                time.monotonic() - started,
                result.returncode,
            )
            return brief
        # Pass 2: lock the derived slots deterministically (model-independent).
        canonical = _normalize_canonical(canonical)
        _log.info(
            "openalex: canonicalize brief done in %.1fs (%d chars)",
            time.monotonic() - started,
            len(canonical),
        )
        return canonical

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
                temperature=self.options.query_writer_temperature,
                seed=self.options.query_writer_seed,
                timeout_seconds=self.options.timeout_seconds,
            )
        )

    def _canonicalize_writer(self) -> _QueryWriter | None:
        """Dedicated codex writer for brief canonicalization at its own (lower)
        temperature, so query-gen keeps its facet-coverage slack (0.2) while the
        canonicalizer runs as deterministically as possible (0.001)."""
        from spotlights_engine.module_deep_research.codex_exec import (
            CodexExecClient,
            CodexExecOptions,
        )

        return CodexExecClient(
            CodexExecOptions(
                cwd=self.options.cwd,
                model=self.options.model,
                search=False,
                temperature=self.options.canonicalize_temperature,
                seed=self.options.canonicalize_seed,
                timeout_seconds=self.options.timeout_seconds,
            )
        )

    def run(self, prompt: str, *, check: bool = True) -> AgentExecResult:
        queries = self._resolve_queries(prompt)
        plans = [(q, self._query_urls(q)) for q in queries]
        safe_cmd = ["GET", *(_redact_api_key(u) for _q, us in plans for u in us)]
        _log.info(
            "openalex: %d quer%s → %d URL(s) to fetch (mode=%s)",
            len(plans),
            "y" if len(plans) == 1 else "ies",
            sum(len(us) for _q, us in plans),
            self.options.query_mode,
        )

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
            if _RELEVANCE_MIN_OVERLAP > 0:
                before = len(works)
                works = _filter_relevant(works, query, _RELEVANCE_MIN_OVERLAP)
                if len(works) != before:
                    _log.info(
                        "openalex: relevance gate (min_overlap=%d) kept %d/%d "
                        "work(s) for query %r",
                        _RELEVANCE_MIN_OVERLAP,
                        len(works),
                        before,
                        query,
                    )
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
        if _STABLE_SORT:
            merged = _stable_sort(merged)
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

    @staticmethod
    def _swap_api_key(url: str, new_key: str) -> str:
        """Return `url` with its api_key query param replaced by `new_key`."""
        parts = urllib.parse.urlsplit(url)
        q = dict(urllib.parse.parse_qsl(parts.query, keep_blank_values=True))
        q["api_key"] = new_key
        return urllib.parse.urlunsplit(
            parts._replace(query=urllib.parse.urlencode(q))
        )

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
            _log.info(
                "openalex: fetch attempt %d/%d (timeout=%ss) %s",
                attempt + 1,
                _FETCH_RETRIES + 1,
                self.options.timeout_seconds,
                _redact_api_key(url),
            )
            fetch_started = time.monotonic()
            try:
                request = urllib.request.Request(url, headers=self._headers())
                body = _read_url_hard_bounded(request, self.options.timeout_seconds)
                data = json.loads(body)
                works = data.get("results") if isinstance(data, dict) else None
                works = works if isinstance(works, list) else []
                _log.info(
                    "openalex: fetch done in %.1fs — %d work(s)",
                    time.monotonic() - fetch_started,
                    len(works),
                )
                return works
            except (urllib.error.HTTPError, urllib.error.URLError, TimeoutError) as exc:
                _log.warning(
                    "openalex: fetch attempt %d failed in %.1fs: %s",
                    attempt + 1,
                    time.monotonic() - fetch_started,
                    exc,
                )
                # 429 (Too Many Requests) is retryable — the burst limit clears
                # after a short wait; honor Retry-After when the server sends it.
                is_http = isinstance(exc, urllib.error.HTTPError)
                # Key-block codes: a spent/blocked key (402 no credit, 401/403
                # auth, 429 rate) fails over to the next key in the pool before we
                # fall back to plain backoff. Rewrite the api_key in-place and
                # retry immediately (no wait) as long as an unused key remains.
                if is_http and exc.code in (401, 402, 403, 429):
                    keys = self._api_keys()
                    if len(keys) > 1 and self._key_idx + 1 < len(keys) and attempt < _FETCH_RETRIES:
                        self._key_idx += 1
                        url = self._swap_api_key(url, keys[self._key_idx])
                        _log.warning(
                            "openalex: key blocked (HTTP %d); rotating to key #%d/%d",
                            exc.code, self._key_idx + 1, len(keys),
                        )
                        last_exc = exc
                        continue
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
                if (
                    is_http
                    and exc.code in (502, 503, 504)
                    and _GATEWAY_BACKOFF_SECONDS > 0
                ):
                    delay = max(delay, _GATEWAY_BACKOFF_SECONDS)
                if delay > _MAX_BACKOFF_SECONDS:
                    _log.warning(
                        "openalex: capping %.1fs backoff (server Retry-After?) "
                        "to %.1fs — quota likely exhausted",
                        delay,
                        _MAX_BACKOFF_SECONDS,
                    )
                    delay = _MAX_BACKOFF_SECONDS
                _log.info("openalex: retrying after %.1fs backoff", delay)
                time.sleep(delay)
        assert last_exc is not None  # unreachable; loop either returns or raises
        raise last_exc

    def _headers(self) -> dict[str, str]:
        mailto = self._resolve_mailto()
        ua = "spotlights-engine/openalex"
        if mailto:
            ua += f" (mailto:{mailto})"
        # Connection: close — no keep-alive pooling. A pooled socket left
        # half-open by a wedged read stacks up as leaked ESTABLISHED/CLOSE_WAIT
        # fds; closing per response bounds the leak to at most one live socket.
        return {
            "User-Agent": ua,
            "Accept": "application/json",
            "Connection": "close",
        }


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
    "Output EXACTLY 5 queries, ONE PER LINE, filling these FIXED facet slots in "
    "this order. Treat this as an EXTRACTION task, not creative writing: for a "
    "given brief the same 5 lines must be produced every time.\n"
    "  slot 1 — the method's proper name copied VERBATIM from the brief + the "
    'single most specific application-domain noun in the brief (e.g. \"Muon '
    'optimizer neural network training\")\n'
    "  slot 2 — the core algorithmic mechanism, named with the brief's own terms "
    '(e.g. \"Newton-Schulz iteration orthogonalization momentum\")\n'
    "  slot 3 — the problem the method solves, in research-paper vocabulary drawn "
    "from the brief\n"
    "  slot 4 — the ONE most established published synonym for the method (the "
    "only slot that may use a term not in the brief) + domain\n"
    "  slot 5 — the broader technique class, anchored to the specific method name\n"
    "Determinism rules — follow exactly; they are what make runs reproducible:\n"
    "  1. Prefer terms that appear VERBATIM in the brief; copy them, do not "
    "paraphrase.\n"
    "  2. For each concept pick the SINGLE most established name — never invent "
    "novel synonyms or alternate phrasings.\n"
    "  3. Do NOT reorder concepts within a line, change casing, add adjectives, "
    "or expand/contract acronyms unless the brief itself does.\n"
    "  4. If a slot has no distinct content, restate the nearest higher-priority "
    "slot more tightly rather than inventing a vague generic.\n"
    "Favor specificity over breadth (a precise 3-5 concept query beats a vague "
    "1-2 word one). No numbering, no bullets, no markdown, no blank lines, no "
    "explanation — just the 5 query strings, each on its own line.\n"
    "\n"
    "--- RESEARCH BRIEF ---\n"
)

# Prepended to the semantic brief when OPENALEX_CANONICALIZE_BRIEF is on. Rewrites
# a free-form candidate brief into a FIXED CANONICAL QUERY PATTERN so two
# differently-worded briefs describing the SAME code + SAME optimization collapse to
# byte-identical text — which is what makes the downstream query-writer reproducible
# across runs. The pattern is 4 sorted slots general to ANY candidate:
# SUBJECT (the operation), OBJECTS (the data it touches), ADJECTIVES (how+why it is
# optimized, from closed vocabularies), EXAMPLES (established named methods). Every
# free-vocabulary slot is normalized to controlled terms and alphabetically sorted so
# wording and ordering variance cannot survive.
_CANONICALIZE_INSTRUCTION = (
    "You normalize a code-optimization research brief into a FIXED CANONICAL QUERY "
    "PATTERN. Two briefs describing the SAME code and the SAME optimization — however "
    "differently worded — MUST produce BYTE-IDENTICAL output. UNDERSTAND the material "
    "and map every phrase to the controlled vocabulary below; do NOT echo the brief's "
    "wording. This is normalization, not writing.\n"
    "\n"
    "Output EXACTLY these 4 lines, each starting with its label, in this order, and "
    "NOTHING else:\n"
    "SUBJECT: one canonical noun-phrase for the operation being optimized "
    "(e.g. \"maxsim late-interaction scoring\", \"padding removal\", "
    "\"per-cluster mean pooling\", \"image-token masking\", \"ragged image-patch "
    "batching\"). Lowercase. No symbol/class/method name, no file path, no framework "
    "name.\n"
    "OBJECTS: comma-separated data structures the change reads or writes, mapped to "
    "controlled nouns and sorted ALPHABETICALLY. Controlled nouns: cluster labels, "
    "embedding tensor, index tensor, offset vector, padding mask, patch tensor, "
    "similarity tensor, token mask. Map synonyms (multi-vector embeddings->embedding "
    "tensor; attention/score matrix->similarity tensor; zero-padding->padding mask; "
    "pixel values->patch tensor; cluster ids->cluster labels). List ONLY the primary "
    "input/output structures; EXCLUDE internal scratch tensors the algorithm builds on "
    "the way (e.g. a cumsum/argsort index used only to slice or gather) UNLESS that "
    "structure is itself the operation's output.\n"
    "ADJECTIVES: EXACTLY two terms, sorted ALPHABETICALLY — one TECHNIQUE and one "
    "GOAL-CLASS, each from its closed list. TECHNIQUE in {batched gather, deferred "
    "materialization, in-place update, kernel fusion, precompute-and-cache, segmented "
    "reduction, tiled reduction, vectorized cumulative-count, vectorized reduction}; "
    "if none fits use \"other reduction\". GOAL-CLASS in {allocation reduction, "
    "host-sync elimination, kernel vectorization, memory reduction}. When more than "
    "one GOAL-CLASS could apply, emit the FIRST in this priority order: memory "
    "reduction, host-sync elimination, allocation reduction, kernel vectorization.\n"
    "EXAMPLES: derived DETERMINISTICALLY from the chosen TECHNIQUE via this fixed "
    "table (NOT from the brief's wording), so it never drifts: batched gather->gather, "
    "index_select; deferred materialization->pad_sequence, torch.split; in-place "
    "update->masked_fill_, scatter_; kernel fusion->flash attention; "
    "precompute-and-cache->memoization; segmented reduction->scatter-add, segment_csr; "
    "tiled reduction->flash attention; vectorized cumulative-count->cumsum; vectorized "
    "reduction->amax, logsumexp; other reduction->(empty). Emit that slot's value "
    "verbatim.\n"
    "\n"
    "RULES (generalized from real drifts):\n"
    "- UNDERSTAND then NORMALIZE. Collapse synonyms to ONE canonical term: "
    "\"unpad\"/\"strip zero-padding\"/\"unbind padded embeddings\"/\"remove padding\" "
    "all -> SUBJECT \"padding removal\", OBJECTS \"embedding tensor, padding mask\".\n"
    "- CLOSED vocabulary is mandatory for OBJECTS, ADJECTIVES. Never invent a "
    "synonym; pick the listed term whose meaning matches.\n"
    "- SORT every multi-item slot alphabetically. Ordering must not depend on the "
    "brief's phrasing.\n"
    "- Map \"deferred padding\" AND \"packed representation\" AND \"ragged/packed "
    "layout\" -> TECHNIQUE \"deferred materialization\".\n"
    "- Map \"avoids host round-trip\"/\"removes .item() sync\"/\"no CPU-GPU stall\" -> "
    "GOAL-CLASS \"host-sync elimination\"; \"lower peak memory\"/\"no big intermediate "
    "tensor\" -> \"memory reduction\"; \"fewer allocations\"/\"no pad buffer\" -> "
    "\"allocation reduction\"; \"replace python loop with tensor ops\" -> \"kernel "
    "vectorization\".\n"
    "- Include ONLY what the change itself requires; drop upstream/context names one "
    "wording might mention but the other omits (e.g. the clustering method feeding a "
    "pooling loop).\n"
    "\n"
    "WORKED EXAMPLES:\n"
    "  brief: 'unbinds a padded [B,L,d] multivector batch into per-sequence tensors "
    "by scanning a zero mask per row (O(B) .item() syncs)'\n"
    "  ->\n"
    "  SUBJECT: padding removal\n"
    "  OBJECTS: embedding tensor, padding mask\n"
    "  ADJECTIVES: host-sync elimination, vectorized reduction\n"
    "  EXAMPLES: cumsum\n"
    "  brief: 'process_images pads per-image pixel patches to the batch max and "
    "returns a dense [B,Lmax,d]; padding is discarded downstream'\n"
    "  ->\n"
    "  SUBJECT: ragged image-patch batching\n"
    "  OBJECTS: offset vector, patch tensor\n"
    "  ADJECTIVES: deferred materialization, memory reduction\n"
    "  EXAMPLES: pad_sequence, torch.split\n"
    "\n"
    "DO NOT: emit impact/severity/priority/effort; line numbers, paths, or framework "
    "names; adjectives outside the closed lists; blank lines, markdown, bullets, or "
    "any text outside the 4 labeled lines.\n"
    "\n"
    "--- BRIEF ---\n"
)

# Deterministic pass-2 over the LLM's canonical draft. The LLM (pass 1) picks the
# semantic slots (SUBJECT, OBJECTS, ADJECTIVES); this code locks the DERIVED
# slots so model-to-model variance can't reintroduce drift: EXAMPLES is a pure
# function of the chosen TECHNIQUE (never the wording), scratch OBJECTS derivable
# from a listed structure are dropped, and every multi-item slot is alpha-sorted.
# Proven to converge m3/m4 5/5 byte-identical over 10 rounds through the codex
# CLI at temperature 0.001 (weaker instruction-follower than sonnet), so it holds
# regardless of which model backs the writer.
_CANON_TECHNIQUES = frozenset({
    "batched gather", "deferred materialization", "in-place update", "kernel fusion",
    "precompute-and-cache", "segmented reduction", "tiled reduction",
    "vectorized cumulative-count", "vectorized reduction", "other reduction",
})
_CANON_EXAMPLES_TABLE = {
    "batched gather": "gather, index_select",
    "deferred materialization": "pad_sequence, torch.split",
    "in-place update": "masked_fill_, scatter_",
    "kernel fusion": "flash attention",
    "precompute-and-cache": "memoization",
    "segmented reduction": "scatter-add, segment_csr",
    "tiled reduction": "flash attention",
    "vectorized cumulative-count": "cumsum",
    "vectorized reduction": "amax, logsumexp",
    "other reduction": "",
}
# An "index tensor" is scratch (drop it) when it is derivable from another listed
# structure — a cumsum/argsort index built to slice cluster labels, a padding
# mask, or an offset vector, none of which the operation itself outputs.
_CANON_INDEX_SOURCES = frozenset({"cluster labels", "padding mask", "offset vector"})
_CANON_SLOT_RE = re.compile(r"\s*(SUBJECT|OBJECTS|ADJECTIVES|EXAMPLES)\s*:\s*(.*)")


def _normalize_canonical(raw: str) -> str:
    """Lock the derived slots of an LLM canonical draft; return raw if unparseable."""
    slots: dict[str, str] = {}
    for line in raw.splitlines():
        m = _CANON_SLOT_RE.match(line)
        if m:
            slots[m.group(1)] = m.group(2).strip()
    if "SUBJECT" not in slots or "ADJECTIVES" not in slots:
        return raw.strip()
    _items = lambda v: [x.strip() for x in v.split(",") if x.strip()]
    objs = _items(slots.get("OBJECTS", ""))
    if "index tensor" in objs and any(s in objs for s in _CANON_INDEX_SOURCES):
        objs = [o for o in objs if o != "index tensor"]
    objs = sorted(set(objs))
    adjs = sorted(set(_items(slots.get("ADJECTIVES", ""))))
    technique = next((a for a in adjs if a in _CANON_TECHNIQUES), "other reduction")
    examples = _CANON_EXAMPLES_TABLE.get(technique, "")
    out = [
        f"SUBJECT: {slots['SUBJECT'].strip().lower()}",
        f"OBJECTS: {', '.join(objs)}",
        f"ADJECTIVES: {', '.join(adjs)}",
        f"EXAMPLES: {examples}" if examples else "EXAMPLES:",
    ]
    return "\n".join(out)

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


# Function words dropped when building a query's token-set key, so paraphrases
# that differ only in glue words ("ColBERT retrieval" vs "retrieval via ColBERT")
# collapse to the same key. Content tokens (maxsim, colbert, retrieval, memory,
# …) are kept — they carry the facet.
_QUERY_STOPWORDS = frozenset(
    {"a", "an", "the", "of", "for", "and", "or", "to", "in", "on", "with",
     "by", "via", "using", "based", "vs", "at"}
)
# Two queries whose content-token sets overlap at least this much (Jaccard) are
# treated as the same facet; the later one is dropped. Subset/superset pairs
# ("ColBERT retrieval" ⊂ "ColBERT late interaction retrieval") always collapse.
_QUERY_NEAR_DUPE_JACCARD = 0.8


def _query_token_key(query: str) -> frozenset[str]:
    """Order-independent content-token signature of a query for dedup."""
    return frozenset(
        t for t in re.findall(r"[a-z0-9]+", query.lower())
        if t not in _QUERY_STOPWORDS
    )


def _canonicalize_queries(queries: list[str], *, limit: int) -> list[str]:
    """Collapse paraphrase-equivalent queries to a stable, deduped subset.

    The codex query-writer re-words the same facet differently each run; since
    OpenAlex keyword `search` is wording-sensitive, those near-duplicates
    fragment recall and destroy cross-run overlap. We keep the first query of
    each facet (identity fixed by its content-token set) and drop later exact,
    subset/superset, or high-Jaccard variants, then cap at `limit`.
    """
    kept: list[str] = []
    kept_keys: list[frozenset[str]] = []
    for query in queries:
        key = _query_token_key(query)
        if not key:
            continue
        dup = False
        for prior in kept_keys:
            if key == prior or key <= prior or prior <= key:
                dup = True
                break
            union = key | prior
            if union and len(key & prior) / len(union) >= _QUERY_NEAR_DUPE_JACCARD:
                dup = True
                break
        if dup:
            continue
        kept.append(query)
        kept_keys.append(key)
        if len(kept) >= max(1, limit):
            break
    return kept


def _parse_query_lines(raw: str, *, limit: int) -> list[str]:
    """Parse a multi-line reply into up to `limit` distinct OpenAlex queries.

    Sanitizes every line, then canonicalizes: paraphrase-equivalent queries
    (same content-token set, subset/superset, or Jaccard >= threshold) collapse
    so the wording drift between reruns does not fragment the search pool.
    """
    sanitized = [q for q in (_sanitize_line(ln) for ln in raw.splitlines()) if q]
    return _canonicalize_queries(sanitized, limit=limit)


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


def _significant_tokens(text: str) -> set[str]:
    """Lowercased word tokens of length >= 3, minus boolean/stop-word noise."""
    return {
        tok
        for tok in (m.group(0).lower() for m in _WORD_RE.finditer(text))
        if len(tok) >= 3 and tok not in _QUERY_NOISE_TOKENS
    }


def _work_text(work: dict) -> str:
    title = work.get("title") or work.get("display_name") or ""
    abstract = _reconstruct_abstract(work.get("abstract_inverted_index"))
    return f"{title} {abstract}"


def _filter_relevant(works: list, query: str, min_overlap: int) -> list:
    """Keep works whose title+abstract share >= `min_overlap` query tokens.

    No-op (returns `works` unchanged) when `min_overlap` <= 0, the query is
    empty, or the query has no significant tokens — so the gate is strictly
    opt-in and never removes hits when it cannot score them. Non-dict entries
    are always kept (never scored).
    """
    if min_overlap <= 0 or not query:
        return works
    q_tokens = _significant_tokens(query)
    if not q_tokens:
        return works
    kept: list = []
    for work in works:
        if not isinstance(work, dict):
            kept.append(work)
            continue
        if len(q_tokens & _significant_tokens(_work_text(work))) >= min_overlap:
            kept.append(work)
    return kept


def _stable_sort(merged: list[tuple[dict, str]]) -> list[tuple[dict, str]]:
    """Canonical, citation-neutral order: sort merged findings by work key.

    Stabilizes the finding ORDER across runs over the same work set (the merge
    order otherwise depends on query arrival), which in turn stabilizes the
    order-sensitive step-4 finding pairing. Citation-neutral so it does not
    re-bias against the fresh, low-cited GTs the recall slices rescue.
    """
    return sorted(merged, key=lambda item: _work_key(item[0]) or "")


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
