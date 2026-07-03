"""arXiv-search research runner for the module deep-research step.

This runner satisfies the `ModuleResearchRunner` protocol but does *not* use the
rendered NL research prompt: it derives its own structured queries from the
module model and candidate hot spots captured at construction, executes them
against the open arXiv API, and normalizes the Atom results into the same
`AgentModuleDeepResearchOutput` wire shape the agent runners emit. It owns its
HTTP end-to-end (stdlib `urllib`), so — unlike the hosted-search agent runners —
it does not depend on any vendor tool surviving the LiteLLM proxy.

Query planning has two seams behind one interface:

- `plan_queries_claude` runs one read-only, no-web-tool Claude session whose
  sole job is vocabulary translation (repo idioms -> paper-abstract terms) and
  emits a structured CNF plan; Python renders the arXiv syntax deterministically.
- `plan_queries_template` is a pure deterministic RAKE-style planner, used as the
  fallback on any Claude-planner failure and as the only planner when
  `query_planner="template"`.

Any Claude-planner failure degrades to the template planner plus one recoverable
`StepIssue`; a dead CLI/proxy therefore never fails the runner.
"""

from __future__ import annotations

import json
import logging
import re
import time
import urllib.error
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from collections.abc import Iterable
from datetime import UTC
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from spotlights_engine.module_deep_research.agent_exec import AgentExecResult
from spotlights_engine.module_deep_research.claude_exec import (
    ClaudeExecClient,
    ClaudeExecOptions,
)
from spotlights_engine.module_deep_research.debug import (
    write_arxiv_plan,
    write_arxiv_planner_output,
    write_arxiv_planner_prompt,
)
from spotlights_engine.module_deep_research.validation import (
    AgentFinding,
    AgentModuleDeepResearchOutput,
    _extract_json_object,
)
from spotlights_engine.schemas.candidate import Candidate
from spotlights_engine.schemas.common import StepIssue
from spotlights_engine.schemas.pipeline import ModuleDeepResearchInput
from spotlights_engine.schemas.project import Module

_log = logging.getLogger(__name__)

_ARXIV_API = "https://export.arxiv.org/api/query"

# arXiv asks for courteous spacing (~1 request / 3s, no parallel requests).
_REQUEST_INTERVAL_S = 3.0
# Rendered `search_query` cap: the API returns HTTP 414 well above this; the cap
# is a guard, hit only on pathological input (verified 2.5KB OK, 8.5KB fails).
_MAX_QUERY_CHARS = 1024
_MAX_AND_CLAUSES = 3
_MAX_PHRASE_WORDS = 4
_FETCH_ATTEMPTS = 3

# Small built-in stopword list for RAKE-style phrase splitting. Kept minimal on
# purpose: it only needs to break prose into domain noun-phrases.
_STOPWORDS = frozenset(
    {
        "a", "an", "the", "and", "or", "but", "if", "then", "else", "of", "to",
        "in", "on", "for", "with", "without", "into", "onto", "from", "by", "at",
        "as", "is", "are", "was", "were", "be", "been", "being", "that", "this",
        "these", "those", "it", "its", "their", "them", "they", "which", "who",
        "whom", "whose", "what", "when", "where", "while", "during", "under",
        "over", "up", "down", "out", "so", "than", "too", "very", "can", "could",
        "should", "would", "may", "might", "will", "shall", "do", "does", "did",
        "not", "no", "yes", "per", "via", "using", "use", "used", "uses", "each",
        "any", "all", "some", "such", "only", "also", "both", "more", "most",
        "less", "least", "get", "gets", "picks", "pick", "make", "makes",
    }
)

_IMPACT_RANK = {"high": 0, "medium": 1, "low": 2}


class ArxivSearchOptions(BaseModel):
    """Options for the arXiv-search runner.

    The runner is enabled by the *presence* of this object on the manager config
    (`None` = disabled), not by a field inside it.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    # Max distinct query strings derived per module by the query planner.
    # [no CLI flag — internal default]
    max_queries: int = Field(default=4, ge=0)

    # Query-planning mode. [CLI: --arxiv-query-planner {claude,template}]
    query_planner: Literal["claude", "template"] = "claude"

    # Claude planner session knobs. Model None = the claude CLI default.
    # [no CLI flag — internal defaults]
    planner_model: str | None = None
    planner_max_turns: int = 8
    planner_timeout_s: int = 240

    # Results requested per query (arXiv `max_results`). [no CLI flag]
    results_per_query: int = Field(default=10, ge=0)

    # Per-request HTTP timeout (seconds). [no CLI flag]
    http_timeout_s: float = 20.0

    # Local cap on findings before merge — deliberately lower than the agent
    # runners' cap: every surviving finding costs `len(candidates)` step-4
    # sessions downstream. Bounded by 9999 so the advisory `find-NNNN` wire id
    # (4-digit pattern) can never overflow. [no CLI flag]
    max_findings: int = Field(default=10, ge=0, le=9999)

    # Contact email embedded in the User-Agent, per arXiv API etiquette.
    # [CLI: --arxiv-mailto EMAIL]
    mailto: str | None = None


class PlannedArxivQuery(BaseModel):
    """One planned query in conjunctive normal form: the outer list is AND-ed,
    the inner lists are OR-ed. Each item is one 1-4-word lowercase phrase as it
    would appear in a paper abstract."""

    model_config = ConfigDict(extra="forbid")

    clauses: list[list[str]] = Field(default_factory=list)


class ArxivQueryPlan(BaseModel):
    """The structured plan the Claude planner session emits (never rendered
    query strings — Python renders the arXiv syntax deterministically)."""

    model_config = ConfigDict(extra="forbid")

    queries: list[PlannedArxivQuery] = Field(default_factory=list)


# ---------------------------------------------------------------------------
# Deterministic phrase machinery (shared by both planner paths)
# ---------------------------------------------------------------------------


def _looks_like_code_identifier(atom: str) -> bool:
    """True for tokens that are code identifiers, not natural-language words.

    Matches `_`, `()`, an internal `.` between word characters (`KVCache.evict`),
    or mid-word caps (camelCase). Case-sensitive on purpose: the caller passes
    the original-case atom so camelCase survives detection.
    """
    if "_" in atom or "(" in atom or ")" in atom:
        return True
    # A dotted path like `KVCache.evict` is code; a decimal like `gpt-4.5` or
    # `llama 3.1` is not — require a letter on both sides of the dot.
    if re.search(r"[A-Za-z]\.[A-Za-z]", atom):
        return True
    if re.search(r"[a-z][A-Z]", atom):
        return True
    # Acronym glued to a word (`KVCache`, `LRUPolicy`) is code-ish. The
    # natural-language forms the query should use have a space (`kv cache`).
    if re.search(r"[A-Z]{2,}(?=[A-Z][a-z])", atom):
        return True
    return False


def _humanize_identifier(identifier: str) -> str:
    """Split a code identifier into space-separated lowercase words.

    Splits on `_`, `.`, `()`, and camelCase boundaries while preserving all-caps
    acronym runs (`kv_offload` -> `kv offload`, `LRUPolicy` -> `lru policy`).
    """
    base = re.sub(r"\(\)", " ", identifier)
    base = re.sub(r"[._]+", " ", base)
    # camelCase / acronym boundaries: `aB` -> `a B`, `ABc` -> `AB c`.
    base = re.sub(r"(?<=[a-z0-9])(?=[A-Z])", " ", base)
    base = re.sub(r"(?<=[A-Z])(?=[A-Z][a-z])", " ", base)
    words = [w.lower() for w in base.split() if w]
    return " ".join(words)


def _strip_outer_punct(atom: str) -> str:
    """Strip surrounding punctuation while keeping internal hyphens/apostrophes."""
    return atom.strip("\"'`.,;:!?()[]{}<>")


def _extract_phrases(text: str) -> list[str]:
    """RAKE-style phrase extraction from prose.

    Splits `text` into candidate phrases at punctuation and stopwords, keeps
    phrases of 1-4 words, and treats code-identifier tokens as delimiters
    (they never appear in paper abstracts). Returns lowercase phrases in first-
    occurrence order (duplicates preserved so callers can score by frequency).
    """
    phrases: list[str] = []
    current: list[str] = []

    def flush() -> None:
        if 1 <= len(current) <= _MAX_PHRASE_WORDS:
            phrases.append(" ".join(current))
        current.clear()

    for raw_atom in text.split():
        atom = _strip_outer_punct(raw_atom)
        if not atom:
            flush()
            continue
        if _looks_like_code_identifier(atom):
            flush()
            continue
        lowered = atom.lower()
        if lowered in _STOPWORDS or not any(ch.isalnum() for ch in lowered):
            flush()
            continue
        current.append(lowered)
    flush()
    return phrases


def _rank_phrases(occurrences: Iterable[str]) -> list[str]:
    """Rank phrases by `frequency × word_count`, tie-broken by first occurrence.

    `occurrences` is the concatenation of `_extract_phrases` output across all
    input fields (duplicates carry the frequency signal).
    """
    freq: dict[str, int] = {}
    first_seen: dict[str, int] = {}
    for idx, phrase in enumerate(occurrences):
        if phrase not in first_seen:
            first_seen[phrase] = idx
        freq[phrase] = freq.get(phrase, 0) + 1

    def score(phrase: str) -> tuple[int, int]:
        words = len(phrase.split())
        return (-(freq[phrase] * words), first_seen[phrase])

    return sorted(freq, key=score)


def _dedup_keep_order(values: Iterable[str]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for v in values:
        if v and v not in seen:
            seen.add(v)
            out.append(v)
    return out


# ---------------------------------------------------------------------------
# Validation + render (deterministic, shared by both planner paths)
# ---------------------------------------------------------------------------


class _Rejection(BaseModel):
    """One phrase or query dropped by validation, with the reason."""

    model_config = ConfigDict(extra="forbid")

    value: str
    reason: Literal[
        "code_identifier",
        "too_many_words",
        "too_many_clauses",
        "over_length_cap",
        "duplicate",
    ]


def _clean_phrase(phrase: str) -> str:
    return " ".join(phrase.lower().split())


def _validate_phrase(phrase: str, rejections: list[_Rejection]) -> str | None:
    """Lowercase/strip and validate one phrase; record + drop invalid ones."""
    raw_cleaned = " ".join(phrase.strip().split())
    cleaned = raw_cleaned.lower()
    if not cleaned:
        return None
    if _looks_like_code_identifier(raw_cleaned) or _looks_like_code_identifier(cleaned):
        rejections.append(_Rejection(value=cleaned, reason="code_identifier"))
        return None
    if len(cleaned.split()) > _MAX_PHRASE_WORDS:
        rejections.append(_Rejection(value=cleaned, reason="too_many_words"))
        return None
    return cleaned


def _render_query(clauses: list[list[str]]) -> str:
    """Render CNF clauses into an arXiv `search_query`:
    `(all:"p1" OR all:"p2") AND (all:"p3")`."""
    rendered_clauses = []
    for clause in clauses:
        rendered_clauses.append(
            "(" + " OR ".join(f'all:"{phrase}"' for phrase in clause) + ")"
        )
    return " AND ".join(rendered_clauses)


def _validate_and_render_plan(
    plan: ArxivQueryPlan, *, max_queries: int
) -> tuple[list[PlannedArxivQuery], list[str], list[_Rejection]]:
    """Validate a CNF plan and render it to `search_query` strings.

    Returns `(surviving_structured_queries, rendered_query_strings, rejections)`.
    Shared by the Claude path and the template path (the template planner builds
    an `ArxivQueryPlan` and runs it through here too, so both share the shape
    guards, dedup, cap, and truncation).
    """
    rejections: list[_Rejection] = []
    survived: list[PlannedArxivQuery] = []
    rendered: list[str] = []
    seen_rendered: set[str] = set()

    if max_queries <= 0:
        return survived, rendered, rejections

    for query in plan.queries:
        clauses: list[list[str]] = []
        for clause in query.clauses:
            phrases = _dedup_keep_order(
                p for raw in clause if (p := _validate_phrase(raw, rejections))
            )
            if phrases:
                clauses.append(phrases)
        if not clauses:
            continue
        if len(clauses) > _MAX_AND_CLAUSES:
            rejections.append(
                _Rejection(value=_render_query(clauses), reason="too_many_clauses")
            )
            continue
        rendered_query = _render_query(clauses)
        if len(rendered_query) > _MAX_QUERY_CHARS:
            rejections.append(
                _Rejection(value=rendered_query[:120] + "…", reason="over_length_cap")
            )
            continue
        if rendered_query in seen_rendered:
            rejections.append(_Rejection(value=rendered_query, reason="duplicate"))
            continue
        seen_rendered.add(rendered_query)
        survived.append(PlannedArxivQuery(clauses=clauses))
        rendered.append(rendered_query)
        if len(rendered) >= max_queries:
            break

    return survived, rendered, rejections


# ---------------------------------------------------------------------------
# Template planner (pure deterministic fallback)
# ---------------------------------------------------------------------------


def _sorted_candidates(request: ModuleDeepResearchInput) -> list[Candidate]:
    """Candidates ranked high>medium>low, tie-broken by id, when hot spots are
    enabled; otherwise empty."""
    if not (request.include_candidate_hotspots and request.candidates):
        return []
    return sorted(
        request.candidates,
        key=lambda c: (_IMPACT_RANK.get(c.estimated_impact, 3), c.id),
    )


def _candidate_phrase(candidate: Candidate) -> str | None:
    """Best phrase from a candidate's prose (`description` + `evolve_rationale`);
    falls back to the humanized first-span symbol only when prose yields none."""
    ranked = _rank_phrases(
        _extract_phrases(candidate.description)
        + _extract_phrases(candidate.evolve_rationale)
    )
    if ranked:
        return ranked[0]
    symbol = candidate.locations[0].spans[0].symbol
    humanized = _humanize_identifier(symbol)
    return humanized or None


def _plan_template_with_rejections(
    request: ModuleDeepResearchInput,
    module: Module,
    options: ArxivSearchOptions,
) -> tuple[list[PlannedArxivQuery], list[str], list[_Rejection]]:
    """Deterministic RAKE-style planner, returning debug-trace details."""
    hint_phrases = _dedup_keep_order(
        _clean_phrase(h) for h in request.context.workload_hints if h.strip()
    )
    description_phrases = _rank_phrases(_extract_phrases(module.description or ""))
    module_name_phrase = _humanize_identifier(module.name)

    module_pool = _rank_phrases(
        _extract_phrases(module.description or "")
        + _extract_phrases(request.context.objective)
        + hint_phrases
    )
    objective_hint_pool = _rank_phrases(
        _extract_phrases(request.context.objective) + hint_phrases
    )

    primary_anchor = module_pool[0] if module_pool else module_name_phrase
    queries: list[PlannedArxivQuery] = []

    # Q1 — broad recall: top module phrases + the (weak) humanized module name.
    broad = _dedup_keep_order(module_pool[:3] + [module_name_phrase])[:4]
    if broad:
        queries.append(PlannedArxivQuery(clauses=[broad]))

    # Q2 — module precision: top module phrase AND top objective/hint phrase.
    top_module = description_phrases[0] if description_phrases else (
        module_pool[0] if module_pool else module_name_phrase
    )
    top_obj = objective_hint_pool[0] if objective_hint_pool else primary_anchor
    if top_module and top_obj and top_module != top_obj:
        queries.append(PlannedArxivQuery(clauses=[[top_module], [top_obj]]))

    # Q3..Qn — candidate precision, one query per candidate in impact order.
    candidates = _sorted_candidates(request)
    slots_for_candidates = max(options.max_queries - 2, 0)
    for candidate in candidates[:slots_for_candidates]:
        phrase = _candidate_phrase(candidate)
        if phrase and primary_anchor and phrase != primary_anchor:
            queries.append(PlannedArxivQuery(clauses=[[phrase], [primary_anchor]]))

    # Freed slots (no/gated candidates, or slots remain) -> module-level variants
    # from next-best phrases, pairing each with the anchor.
    if len(queries) < options.max_queries:
        used = {primary_anchor, top_module, top_obj}
        for phrase in module_pool:
            if len(queries) >= options.max_queries:
                break
            if phrase in used or phrase == primary_anchor:
                continue
            used.add(phrase)
            queries.append(PlannedArxivQuery(clauses=[[phrase], [primary_anchor]]))

    survived, rendered, rejections = _validate_and_render_plan(
        ArxivQueryPlan(queries=queries), max_queries=options.max_queries
    )
    return survived, rendered, rejections


def plan_queries_template(
    request: ModuleDeepResearchInput,
    module: Module,
    options: ArxivSearchOptions,
) -> list[str]:
    """Deterministic RAKE-style planner. Same input twice -> identical output."""
    _, rendered, _ = _plan_template_with_rejections(request, module, options)
    return rendered


# ---------------------------------------------------------------------------
# Claude planner
# ---------------------------------------------------------------------------


def render_arxiv_query_planner_prompt(
    request: ModuleDeepResearchInput,
    module: Module,
    options: ArxivSearchOptions,
) -> str:
    """Render the Claude query-planner prompt.

    Reuses the research-prompt formatters so the planner sees the module the same
    way the research runners do, with three deliberate differences: candidates
    are pre-sorted by `estimated_impact`, `validation_plan` is omitted, and the
    embedded schema is `ArxivQueryPlan`'s.
    """
    # Imported lazily to avoid a heavy import at module load and any cycle.
    from spotlights_engine.module_deep_research.prompts import (
        _format_candidates,
        _format_list,
        _format_module,
        _format_repository,
    )

    repo_path = str(request.repo_path)
    schema_json = json.dumps(ArxivQueryPlan.model_json_schema(), indent=2)

    hotspots_section = ""
    sorted_candidates = _sorted_candidates(request)
    if sorted_candidates:
        hotspots_section = (
            "\nIdentified hot spots (from candidate_discovery), highest "
            "estimated impact first:\n"
            f"{_format_candidates(sorted_candidates)}\n"
        )

    return f"""You are the arXiv query planner for the Spotlights
module_deep_research pipeline step. Do not modify files. Do not ask
questions.

Goal:
Design up to {options.max_queries} arXiv API search queries that will
surface papers carrying a concrete method, algorithm, technique, or design
idea this module could adopt to meet the caller objective. You only design
the queries: a separate non-LLM step executes them against the arXiv API,
and a later step filters the results. Query quality is the only lever you
control.

Repository:
{_format_repository(request.project_tree.repository)}

Repository working directory (read-only; open files via relative paths from
this root): {repo_path}

Target module:
{_format_module(module, request.module_qualified_name)}
{hotspots_section}
Caller context:
Objective: {request.context.objective}
Workload hints:
{_format_list(request.context.workload_hints)}

Workflow:
1. Skim the module's main files (read-only) just enough to name the
   techniques, algorithms, and data structures it actually uses, and where
   it falls short relative to the objective. Budget this tightly: a handful
   of file reads, not a code review.
2. Translate what you learned into the vocabulary of paper titles and
   abstracts. This is the core of the task. The module's own names never
   appear in papers (`kv_offload`, `_compute_scores` match nothing); the
   field's standard terms do ("KV cache offloading", "attention
   approximation"). When the literature uses two common names for one
   concept, use both - as OR alternatives inside one clause, or spread
   across query variants.
3. Emit the query plan following the slot structure below.

Query semantics (how your plan is executed):
- A query is a list of clauses. Clauses are AND-ed together; the phrases
  inside one clause are OR-ed. Each phrase is matched as an exact quoted
  phrase over the paper record (title, abstract, and other fields).
- One clause = broad recall. Two clauses = precision. Three or more AND
  clauses almost always over-constrain arXiv search: never emit more than
  three, and prefer two.
- A phrase is 1-4 lowercase natural-language words, written as it would
  appear in an abstract. No code identifiers, no quotes, no boolean words,
  no arXiv field prefixes - rendering and escaping happen outside your
  output.

Query plan (up to {options.max_queries} slots):
- Slot 1, broad recall: one clause OR-ing 3-5 phrases that together cover
  the module's core topic and the objective.
- Slot 2, module precision: the best module-topic phrase AND the best
  objective/workload phrase.
- Slots 3+, hot-spot precision: one query per identified hot spot, in the
  listed order (highest estimated impact first): a phrase naming the
  technique that hot spot needs AND one module/objective anchor phrase.
- If there are no hot spots, or slots remain after covering them, spend the
  remaining slots on module-precision variants using different synonym
  phrasings - not near-duplicates of earlier slots.
- Precision beats recall here: every returned paper costs downstream LLM
  sessions, so make each precision query specific enough that a matching
  paper is probably adoptable. Fewer, sharper queries beat padded slots;
  it is fine to return fewer than {options.max_queries} queries.

Output rules:
- Return a single bare JSON object matching the ArxivQueryPlan schema
  below. No Markdown fences, no explanatory prose.
- If the module context is too thin to produce useful queries, return
  {{"queries": []}} - the caller falls back to a deterministic planner.

ArxivQueryPlan JSON schema:
{schema_json}
""".strip()


def _parse_query_plan(text: str) -> ArxivQueryPlan:
    """Parse a planner session response into `ArxivQueryPlan` (lenient JSON)."""
    payload = json.loads(_extract_json_object(text))
    return ArxivQueryPlan.model_validate(payload)


# ---------------------------------------------------------------------------
# HTTP + Atom parsing
# ---------------------------------------------------------------------------

_ATOM_NS = "{http://www.w3.org/2005/Atom}"
_ARXIV_NS = "{http://arxiv.org/schemas/atom}"


class _HttpResponse(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    body: str
    status: int
    retry_after: float | None = None


def _parse_retry_after(raw: str | None) -> float | None:
    """Parse a `Retry-After` header value: integer seconds or an HTTP-date.

    Returns the number of seconds to wait, or None if absent/unparseable. The
    HTTP-date form is measured against the current time; a past date yields 0.
    """
    if not raw:
        return None
    raw = raw.strip()
    if raw.isdigit():
        return float(raw)
    try:
        from datetime import datetime
        from email.utils import parsedate_to_datetime

        when = parsedate_to_datetime(raw)
        if when is None:
            return None
        if when.tzinfo is None:
            when = when.replace(tzinfo=UTC)
        return max(0.0, (when - datetime.now(UTC)).total_seconds())
    except (TypeError, ValueError):
        return None


def _get_text(
    url: str, *, timeout: float, headers: dict[str, str]
) -> _HttpResponse:
    """Fetch `url` and return body + status. Raises `urllib.error.URLError` on
    transport failure; HTTP error statuses are returned (not raised) so the
    caller can distinguish 429 from a dead socket."""
    req = urllib.request.Request(url, headers=headers, method="GET")
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:  # noqa: S310
            body = resp.read().decode("utf-8", errors="replace")
            return _HttpResponse(body=body, status=resp.status)
    except urllib.error.HTTPError as exc:
        raw = exc.headers.get("Retry-After") if exc.headers else None
        retry_after = _parse_retry_after(raw)
        body = ""
        try:
            body = exc.read().decode("utf-8", errors="replace")
        except Exception:  # noqa: BLE001
            pass
        return _HttpResponse(body=body, status=exc.code, retry_after=retry_after)


def _parse_arxiv_id(raw_id: str) -> str:
    """Extract the version-stripped arXiv id from an entry `id`/abs URL.

    Preserves the category prefix of old-style ids (`hep-th/9901001`,
    `math.GT/0309136`): only the `.../abs/` prefix is stripped, NOT every path
    segment — otherwise the id, the canonical abs URL, and the dedup/paper-filter
    key all disagree with orchestration's `_normalize_arxiv_id`, which keeps the
    category.
    """
    tail = raw_id.strip().rstrip("/")
    for marker in ("/abs/", "/pdf/"):
        idx = tail.find(marker)
        if idx != -1:
            tail = tail[idx + len(marker) :]
            break
    tail = tail.split("?", 1)[0].split("#", 1)[0].removesuffix(".pdf")
    return re.sub(r"v\d+$", "", tail)


def _parse_atom(body: str) -> list[dict[str, str]]:
    """Parse an arXiv Atom response into a list of entry dicts.

    Each dict has `arxiv_id`, `url` (the canonical abs page), `title`,
    `abstract`, and optional `doi`. Malformed XML yields an empty list.
    """
    try:
        root = ET.fromstring(body)
    except ET.ParseError:
        return []

    entries: list[dict[str, str]] = []
    for entry in root.findall(f"{_ATOM_NS}entry"):
        raw_id = (entry.findtext(f"{_ATOM_NS}id") or "").strip()
        if not raw_id:
            continue
        arxiv_id = _parse_arxiv_id(raw_id)
        title = " ".join((entry.findtext(f"{_ATOM_NS}title") or "").split())
        abstract = " ".join((entry.findtext(f"{_ATOM_NS}summary") or "").split())
        doi = (entry.findtext(f"{_ARXIV_NS}doi") or "").strip()
        if not arxiv_id or not title:
            continue
        entries.append(
            {
                "arxiv_id": arxiv_id,
                "url": f"https://arxiv.org/abs/{arxiv_id}",
                "title": title,
                "abstract": abstract,
                "doi": doi,
            }
        )
    return entries


def _sleep(seconds: float) -> None:
    """Indirection so tests can monkeypatch out courtesy/backoff sleeps."""
    if seconds > 0:
        time.sleep(seconds)


# ---------------------------------------------------------------------------
# Finding normalization
# ---------------------------------------------------------------------------


def _abstract_lead(abstract: str) -> str:
    """First sentence(s) of an abstract, capped for a technique summary."""
    if not abstract:
        return "(no abstract available)"
    sentences = re.split(r"(?<=[.!?])\s+", abstract)
    lead = sentences[0]
    if len(lead) < 160 and len(sentences) > 1:
        lead = lead + " " + sentences[1]
    return lead[:400].strip()


def _abstract_quote(abstract: str) -> str:
    """Short verbatim quote (first sentence) for supporting evidence."""
    if not abstract:
        return ""
    first = re.split(r"(?<=[.!?])\s+", abstract)[0]
    return first[:280].strip()


def _entry_to_finding(entry: dict[str, str], index: int) -> AgentFinding:
    """Map one Atom entry to an `AgentFinding` with a bare `find-NNNN` id.

    `technique_summary` is extractive (templated, `[arxiv]`-prefixed): the arXiv
    API exposes no citation counts, so there is no popularity signal to fake.
    """
    quote = _abstract_quote(entry["abstract"])
    evidence_parts = []
    if quote:
        evidence_parts.append(f'"{quote}"')
    evidence_parts.append(f"arXiv:{entry['arxiv_id']}")
    if entry.get("doi"):
        evidence_parts.append(f"DOI:{entry['doi']}")

    return AgentFinding(
        finding_id=f"find-{index:04d}",
        title=entry["title"],
        url=entry["url"],
        source_type="paper",
        technique_summary=(
            f"[arxiv] {_abstract_lead(entry['abstract'])} "
            "(extractive abstract lead, not an LLM-generated summary)"
        ),
        supporting_evidence=" ".join(evidence_parts),
    )


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------


def _issue(message: str) -> StepIssue:
    return StepIssue(
        step="module_deep_research",
        severity="error",
        message=message,
        recoverable=True,
    )


class ArxivSearchClient:
    """arXiv-search runner implementing the `ModuleResearchRunner` protocol.

    Constructed with the structured inputs (`request`, resolved `module`,
    `options`) captured at build time; `run()` ignores the NL prompt and derives
    its own queries.
    """

    name = "arxiv"

    def __init__(
        self,
        request: ModuleDeepResearchInput,
        module: Module,
        options: ArxivSearchOptions,
        *,
        debug_dir: Path | None = None,
    ) -> None:
        self.request = request
        self.module = module
        self.options = options
        self.debug_dir = debug_dir

    # -- query planning -----------------------------------------------------

    def _plan_queries_claude(self) -> tuple[list[str], list[PlannedArxivQuery], list[_Rejection]]:
        """Run the Claude planner session and validate/render its plan.

        Raises on any failure (CLI error, timeout, unparseable output, empty
        plan, or every query rejected) so `_plan()` can degrade to the template
        planner. Writes the planner prompt (before the session) and raw output
        (after) to `debug_dir` when set.
        """
        prompt = render_arxiv_query_planner_prompt(
            self.request, self.module, self.options
        )
        if self.debug_dir is not None:
            write_arxiv_planner_prompt(self.debug_dir, prompt)

        client = ClaudeExecClient(
            ClaudeExecOptions(
                cwd=self.request.repo_path,
                allowed_tools=("Read", "Grep", "Glob", "LS"),
                max_turns=self.options.planner_max_turns,
                timeout_seconds=self.options.planner_timeout_s,
                model=self.options.planner_model,
            )
        )
        try:
            result = client.run(prompt, check=False)
        except Exception as exc:
            if self.debug_dir is not None:
                write_arxiv_planner_output(
                    self.debug_dir, f"{type(exc).__name__}: {exc}"
                )
            raise

        raw = result.final_message or result.stdout or ""
        if self.debug_dir is not None:
            write_arxiv_planner_output(
                self.debug_dir, raw or result.stderr or "(no planner output)"
            )
        if not result.ok:
            raise RuntimeError(
                f"claude planner exited {result.returncode}: "
                f"{result.stderr.strip() or '(no stderr)'}"
            )

        plan = _parse_query_plan(raw)
        survived, rendered, rejections = _validate_and_render_plan(
            plan, max_queries=self.options.max_queries
        )
        if not rendered:
            raise RuntimeError("claude planner produced no usable query")
        return rendered, survived, rejections

    def _plan(
        self,
    ) -> tuple[
        list[PlannedArxivQuery],
        list[str],
        str,
        str | None,
        list[_Rejection],
    ]:
        """Return planned/rendered queries, planner choice, fallback, and rejections."""
        if self.options.query_planner == "claude":
            try:
                rendered, survived, rejections = self._plan_queries_claude()
                return survived, rendered, "claude", None, rejections
            except Exception as exc:  # noqa: BLE001
                reason = f"arxiv claude query planner failed; used template planner: {exc}"
                _log.warning("deep_research: %s", reason)
                survived, rendered, rejections = _plan_template_with_rejections(
                    self.request, self.module, self.options
                )
                return survived, rendered, "template", reason, rejections
        survived, rendered, rejections = _plan_template_with_rejections(
            self.request, self.module, self.options
        )
        return survived, rendered, "template", None, rejections

    # -- fetch --------------------------------------------------------------

    def _fetch(
        self, query: str
    ) -> tuple[list[dict[str, str]], dict, StepIssue | None]:
        """Fetch one query with bounded retry/backoff.

        Returns `(entries, fetch_record, issue_or_None)`. Never raises for
        network/HTTP errors — a dead API surfaces as a recoverable issue.
        """
        params = urllib.parse.urlencode(
            {"search_query": query, "max_results": self.options.results_per_query}
        )
        url = f"{_ARXIV_API}?{params}"
        headers = {"User-Agent": self._user_agent()}
        record: dict = {"url": url, "status": None, "retries": 0}

        for attempt in range(_FETCH_ATTEMPTS):
            try:
                resp = _get_text(
                    url, timeout=self.options.http_timeout_s, headers=headers
                )
            except Exception as exc:  # noqa: BLE001  (URLError, socket timeout, …)
                record["status"] = f"error: {exc}"
                if attempt + 1 < _FETCH_ATTEMPTS:
                    record["retries"] += 1
                    _sleep(_REQUEST_INTERVAL_S * (attempt + 1))
                    continue
                return [], record, _issue(f"arxiv query failed ({query!r}): {exc}")

            record["status"] = resp.status
            if 200 <= resp.status < 300:
                entries = _parse_atom(resp.body)
                record["entries_returned"] = len(entries)
                return entries, record, None
            if resp.status in (429, 500, 502, 503, 504) and attempt + 1 < _FETCH_ATTEMPTS:
                record["retries"] += 1
                delay = resp.retry_after or _REQUEST_INTERVAL_S * (attempt + 1)
                _sleep(delay)
                continue
            return (
                [],
                record,
                _issue(f"arxiv query got HTTP {resp.status} ({query!r})"),
            )

        return [], record, _issue(f"arxiv query exhausted retries ({query!r})")

    def _user_agent(self) -> str:
        if self.options.mailto:
            return f"spotlights-engine (+{self.options.mailto})"
        return "spotlights-engine"

    # -- run ----------------------------------------------------------------

    def run(self, prompt: str, *, check: bool = True) -> AgentExecResult:  # noqa: ARG002
        """Ignore the NL `prompt`; plan + execute arXiv queries and return the
        findings as the wire-shape JSON in `final_message`. Never raises for
        network failures (returns empty findings + recoverable issues)."""
        planned, rendered, planner_used, fallback_reason, rejections = self._plan()
        issues: list[StepIssue] = []
        if fallback_reason is not None:
            issues.append(_issue(fallback_reason))
        if not rendered:
            issues.append(_issue("arxiv query planner produced no usable queries"))

        findings: list[AgentFinding] = []
        seen_keys: set[str] = set()
        fetch_records: list[dict] = []

        for i, query in enumerate(rendered):
            if i > 0:
                _sleep(_REQUEST_INTERVAL_S)
            entries, record, issue = self._fetch(query)
            if issue is not None:
                issues.append(issue)
            kept = 0
            for entry in entries:
                keys = {
                    f"id:{entry['arxiv_id']}",
                    f"title:{_normalize_title(entry['title'])}",
                }
                if seen_keys.intersection(keys):
                    continue
                seen_keys.update(keys)
                findings.append(_entry_to_finding(entry, len(findings) + 1))
                kept += 1
                if len(findings) >= self.options.max_findings:
                    break
            record["findings_kept"] = kept
            fetch_records.append(record)
            if len(findings) >= self.options.max_findings:
                break

        # Reassign ids so they stay dense find-0001..N in output order.
        findings = [
            f.model_copy(update={"finding_id": f"find-{idx:04d}"})
            for idx, f in enumerate(findings, start=1)
        ]

        self._write_plan_debug(
            planner_used=planner_used,
            fallback_reason=fallback_reason,
            planned=planned,
            rendered=rendered,
            rejections=rejections,
            fetches=fetch_records,
        )

        output = AgentModuleDeepResearchOutput(findings=findings, issues=issues)
        return AgentExecResult(
            command=["arxiv-search", f"queries={len(rendered)}"],
            returncode=0,
            stdout="",
            stderr="",
            final_message=output.model_dump_json(),
        )

    def _write_plan_debug(
        self,
        *,
        planner_used: str,
        fallback_reason: str | None,
        planned: list[PlannedArxivQuery],
        rendered: list[str],
        rejections: list[_Rejection],
        fetches: list[dict],
    ) -> None:
        if self.debug_dir is None:
            return
        record = {
            "planner_configured": self.options.query_planner,
            "planner_used": planner_used,
            "fallback_reason": fallback_reason,
            "planned_queries": [q.model_dump(mode="json") for q in planned],
            "rendered_queries": rendered,
            "rejected": [r.model_dump() for r in rejections],
            "fetches": fetches,
        }
        write_arxiv_plan(self.debug_dir, record)


def _normalize_title(title: str) -> str:
    return " ".join(title.lower().strip().split())


__all__ = [
    "ArxivQueryPlan",
    "ArxivSearchClient",
    "ArxivSearchOptions",
    "PlannedArxivQuery",
    "plan_queries_template",
    "render_arxiv_query_planner_prompt",
]
