"""Per-module controlled-vocabulary synthesis for brief canonicalization.

Why this exists
---------------
Brief canonicalization (`openalex_exec._CANONICALIZE_INSTRUCTION`) collapses a
free-form candidate brief onto a FIXED pattern built from a CLOSED vocabulary,
so two differently-worded briefs for the same code produce byte-identical text
and therefore the same OpenAlex queries. That closed vocabulary was hand-authored
for one domain (tensor/ML kernels). This module derives the vocabulary from the
repository itself, per module, so the canonicalizer generalizes to ANY domain
(parsers, databases, networking, graphics, string algorithms, schedulers, ...)
without hand-authoring a bucket per domain.

Determinism contract
--------------------
The synthesized vocabulary is an input to a reproducibility-critical path, so it
must itself be reproducible:

1. The synthesis prompt is run at near-zero temperature with a fixed seed.
2. `ModuleVocab.normalized()` is a deterministic post-pass (lowercase, dedupe,
   alphabetical sort, fixed slot order) that removes any residual model ordering
   variance — the same discipline the brief canonicalizer applies to itself.
3. The result is cached per module keyed by a content hash of the synthesis
   INPUT (`vocab_input_hash`); it is rebuilt only when that input changes. A
   frozen vocabulary is what lets briefs stay byte-identical across runs.

This module performs no network or LLM calls on its own: `build_module_vocab`
takes a writer object exposing `run(prompt, *, check) -> result` with
`.final_message: str | None` and `.returncode: int` (the same shape the OpenAlex
runner's query writers already use), so it is trivially testable with a stub.
"""

from __future__ import annotations

import hashlib
import json
import logging
import re
from pathlib import Path
from typing import Protocol

from pydantic import BaseModel, ConfigDict, Field

_log = logging.getLogger(__name__)


# Cardinality bounds. Closed vocabularies must stay small or they stop being
# "closed" — an over-large list reintroduces the wording variance we are trying
# to kill. These are intentionally generous ceilings, not targets.
_MAX_OBJECT_NOUNS = 16
_MAX_TECHNIQUES = 14
_MAX_GOAL_CLASSES = 8
_MAX_EXAMPLES_PER_TECHNIQUE = 4


class _Writer(Protocol):
    def run(self, prompt: str, *, check: bool = ...):  # pragma: no cover - protocol
        ...


class ModuleVocab(BaseModel):
    """The closed controlled vocabulary synthesized for one module.

    Mirrors the four slots the brief canonicalizer consumes:
      - `object_nouns`      -> the OBJECTS slot vocabulary
      - `techniques`        -> the ADJECTIVES/TECHNIQUE closed list
      - `goal_classes`      -> the ADJECTIVES/GOAL-CLASS closed list, in priority
                               order (index 0 wins ties)
      - `examples_by_technique` -> the deterministic TECHNIQUE -> named-methods
                               table (EXAMPLES slot)
    """

    model_config = ConfigDict(extra="forbid")

    object_nouns: list[str] = Field(default_factory=list)
    techniques: list[str] = Field(default_factory=list)
    # Priority order is semantic (not alphabetical): the first goal-class that
    # applies to a brief wins. We preserve the model's order here and lock it in
    # the cache, rather than sorting, so the tie-break is stable per module.
    goal_classes: list[str] = Field(default_factory=list)
    examples_by_technique: dict[str, str] = Field(default_factory=dict)

    def normalized(self) -> "ModuleVocab":
        """Return a deterministic, de-duplicated, sorted copy.

        - all terms lowercased and whitespace-collapsed
        - duplicates removed (first occurrence wins, so priority is kept)
        - `object_nouns`/`techniques` sorted alphabetically (closed sets: order
          is irrelevant, so canonicalize it)
        - `goal_classes` de-duped but NOT reordered (priority is meaningful)
        - `examples_by_technique` keyed only by a known technique, each value a
          sorted, de-duped, comma-joined method list; techniques with no example
          map to "" (the canonicalizer treats "" as "repeat the subject")
        - all slots clamped to their cardinality ceiling
        """
        objects = _dedup_sorted(self.object_nouns, limit=_MAX_OBJECT_NOUNS)
        techniques = _dedup_sorted(self.techniques, limit=_MAX_TECHNIQUES)
        goals = _dedup_keep_order(self.goal_classes, limit=_MAX_GOAL_CLASSES)

        technique_set = set(techniques)
        examples: dict[str, str] = {}
        for tech in techniques:
            raw = self.examples_by_technique.get(tech, "")
            methods = _dedup_sorted(
                _split_methods(raw), limit=_MAX_EXAMPLES_PER_TECHNIQUE
            )
            examples[tech] = ", ".join(methods)
        # Drop any example entry whose technique is not in the closed list, so the
        # table can never reference a term the canonicalizer does not know.
        for key in list(examples):
            if key not in technique_set:
                del examples[key]

        return ModuleVocab(
            object_nouns=objects,
            techniques=techniques,
            goal_classes=goals,
            examples_by_technique=examples,
        )

    def canonical_json(self) -> str:
        """Stable JSON for caching and byte-equality checks (sorted keys)."""
        return json.dumps(
            self.normalized().model_dump(), sort_keys=True, ensure_ascii=False, indent=2
        )

    def is_usable(self) -> bool:
        """A vocabulary is usable only if every slot the canonicalizer reads has
        content. An empty slot would make the canonicalizer fall back mid-brief,
        which is worse than not synthesizing at all."""
        n = self.normalized()
        return bool(n.object_nouns and n.techniques and n.goal_classes)


_WS_RE = re.compile(r"\s+")


def _clean(term: str) -> str:
    return _WS_RE.sub(" ", term.strip().lower())


def _dedup_sorted(terms: list[str], *, limit: int) -> list[str]:
    seen: list[str] = []
    for t in terms:
        c = _clean(t)
        if c and c not in seen:
            seen.append(c)
    return sorted(seen)[:limit]


def _dedup_keep_order(terms: list[str], *, limit: int) -> list[str]:
    seen: list[str] = []
    for t in terms:
        c = _clean(t)
        if c and c not in seen:
            seen.append(c)
    return seen[:limit]


def _split_methods(raw: str) -> list[str]:
    # Comma/newline are the ONLY separators: method names are frequently
    # multi-word ("packrat parsing", "predicate pushdown"), so a whitespace
    # split would shred them. Using only comma/newline also makes this a
    # fixed point — re-splitting an already comma-joined value is a no-op,
    # which is what keeps `normalized()` idempotent (so `canonical_json()`,
    # which normalizes again, cannot drift).
    return [p for p in (_clean(p) for p in re.split(r"[,\n]", raw)) if p]


# --------------------------------------------------------------------------
# Synthesis instruction
# --------------------------------------------------------------------------
#
# Deliberately domain-neutral. The examples span parsing, databases, networking,
# graphics, string algorithms, scheduling AND numeric kernels, so the model is
# not nudged toward any one ecosystem. The repository's own material supplies the
# domain; this prompt only teaches the SHAPE and the determinism discipline.
_VOCAB_SYNTHESIS_INSTRUCTION = (
    "You are building a small CLOSED CONTROLLED VOCABULARY for ONE software "
    "module, from the module description and its optimization candidates supplied "
    "below. This vocabulary is later used to normalize free-text optimization "
    "briefs into a fixed canonical form, so that two differently-worded briefs "
    "about the SAME code collapse to identical text. Your output must therefore "
    "be STABLE: run on the same material twice, it must be the same vocabulary.\n"
    "\n"
    "WHAT YOU ARE PRODUCING — exactly one JSON object, no prose, with these keys:\n"
    "  \"object_nouns\": the primary DATA STRUCTURES the module's operations read "
    "or write, each a short canonical noun phrase.\n"
    "  \"techniques\": the OPTIMIZATION TECHNIQUES that plausibly apply to this "
    "module's operations, each a short canonical phrase describing HOW a change is "
    "made (a mechanism), not what it achieves.\n"
    "  \"goal_classes\": the kinds of IMPROVEMENT the candidates pursue, each a "
    "short canonical phrase describing WHAT is gained. ORDER THEM BY PRIORITY: the "
    "most load-bearing improvement for THIS module first. Order matters here.\n"
    "  \"examples_by_technique\": a map from each technique above to a short list "
    "of ESTABLISHED, NAMED methods/APIs/algorithms that realize it IN THIS "
    "MODULE'S OWN LANGUAGE AND ECOSYSTEM (comma-separated string; \"\" if none is "
    "well-established).\n"
    "\n"
    "HOW TO START (follow in order):\n"
    "  1. Read the module description and each candidate's current approach. "
    "Identify the handful of CORE OPERATIONS the module performs and the DATA each "
    "operation consumes or produces. Those data items become \"object_nouns\".\n"
    "  2. For each candidate, decide the single MECHANISM its change uses (e.g. "
    "precompute a result, batch many small operations into one, avoid a redundant "
    "pass, stream instead of buffer, cache, use a better data structure, remove a "
    "synchronization point). Collect these as \"techniques\".\n"
    "  3. For each candidate, decide the single GOAL its change serves (e.g. less "
    "memory, lower latency, fewer round-trips, fewer allocations, higher "
    "throughput, better accuracy). Collect and PRIORITIZE these as "
    "\"goal_classes\".\n"
    "  4. For each technique, name the standard methods/APIs a practitioner in "
    "this module's ecosystem would reach for. If nothing is standard, use \"\".\n"
    "\n"
    "WHAT IS IMPORTANT:\n"
    "  - CANONICALIZE SYNONYMS. If the material says \"strip padding\", \"remove "
    "pad\", \"unpad\" — pick ONE phrase and use it everywhere. Collapsing synonyms "
    "to a single term is the whole point.\n"
    "  - PREFER ESTABLISHED, GENERIC TERMS over the material's incidental wording. "
    "Choose the name a textbook or standard library would use.\n"
    "  - KEEP EACH LIST SMALL. A closed vocabulary that lists everything is not "
    "closed. Only include a term if a candidate actually motivates it.\n"
    "  - STAY IN THIS MODULE'S DOMAIN. Derive terms from the supplied material, "
    "not from any single framework you happen to know.\n"
    "\n"
    "WHAT IS NOT IMPORTANT / NEVER INCLUDE:\n"
    "  - file paths, symbol names, class or variable names, or the host "
    "framework/library/product BRAND name;\n"
    "  - impact, severity, priority scores, effort estimates, or line numbers;\n"
    "  - speculative techniques no candidate motivates;\n"
    "  - decorative modifiers that vary run to run (\"ragged\", \"fancy\", "
    "\"robust\") — keep the bare noun/mechanism.\n"
    "\n"
    "DETERMINISM RULES (these make the output reproducible):\n"
    "  - lowercase every term; use the SINGLE most established name for each "
    "concept; never invent a novel synonym;\n"
    "  - do not reorder words inside an established phrase, change casing, or "
    "expand/contract acronyms unless the material does;\n"
    "  - \"object_nouns\" and \"techniques\" are unordered SETS — the consumer "
    "sorts them, so do not depend on order there; \"goal_classes\" IS ordered by "
    "priority.\n"
    "\n"
    "CROSS-DOMAIN EXAMPLES (shape only — DO NOT copy these terms unless the module "
    "material actually calls for them):\n"
    "  - a text-parsing module: object_nouns [token stream, parse tree, source "
    "buffer]; techniques [incremental reparse, memoized lookahead]; goal_classes "
    "[latency reduction, allocation reduction]; examples_by_technique "
    "{\"memoized lookahead\": \"packrat parsing\"}.\n"
    "  - a query engine: object_nouns [column batch, row group, join key]; "
    "techniques [vectorized execution, predicate pushdown, late materialization]; "
    "goal_classes [throughput increase, memory reduction]; examples_by_technique "
    "{\"predicate pushdown\": \"\"}.\n"
    "  - a networking module: object_nouns [packet buffer, connection table]; "
    "techniques [zero-copy transfer, batched syscalls]; goal_classes "
    "[latency reduction, syscall reduction]; examples_by_technique "
    "{\"batched syscalls\": \"sendmmsg, io_uring\"}.\n"
    "  - a numeric/array module: object_nouns [coefficient matrix, index vector]; "
    "techniques [in-place update, segmented reduction]; goal_classes "
    "[memory reduction, host-sync elimination]; examples_by_technique "
    "{\"segmented reduction\": \"scatter-add\"}.\n"
    "\n"
    "Output the JSON object and NOTHING else — no markdown fence, no commentary.\n"
    "\n"
    "--- MODULE MATERIAL ---\n"
)


def _render_input(
    *, module_name: str, module_description: str, candidate_briefs: list[str]
) -> str:
    """The synthesis INPUT block, rendered deterministically.

    Candidate briefs are sorted so input ordering cannot vary between runs (the
    hash and the model output must not depend on candidate enumeration order)."""
    lines = [f"Module: {module_name}", f"Description: {module_description or '(none)'}"]
    lines.append("Candidates (each is one optimization opportunity):")
    if candidate_briefs:
        for i, brief in enumerate(sorted(_clean(b) for b in candidate_briefs), 1):
            lines.append(f"  {i}. {brief}")
    else:
        lines.append("  (none)")
    return "\n".join(lines)


def vocab_input_hash(
    *, module_name: str, module_description: str, candidate_briefs: list[str]
) -> str:
    """Content hash of the synthesis input — the cache key. Order-independent in
    the candidate list (briefs are sorted before hashing)."""
    payload = _render_input(
        module_name=module_name,
        module_description=module_description,
        candidate_briefs=candidate_briefs,
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


def _parse_vocab(text: str) -> ModuleVocab | None:
    """Parse the model's JSON reply, tolerating a stray markdown fence."""
    s = text.strip()
    if s.startswith("```"):
        s = re.sub(r"^```[a-zA-Z]*\n?", "", s)
        s = re.sub(r"\n?```$", "", s).strip()
    try:
        data = json.loads(s)
    except (ValueError, TypeError):
        return None
    if not isinstance(data, dict):
        return None
    try:
        return ModuleVocab.model_validate(data)
    except Exception:  # noqa: BLE001 - any schema mismatch -> treat as no vocab
        return None


def _rerank_goal_classes(vocab: ModuleVocab, candidate_briefs: list[str]) -> ModuleVocab:
    """Replace the model's goal priority order with a deterministic one.

    `goal_classes` is the only slot the canonicalizer treats as ORDERED (the
    first applicable goal wins). Trusting the synthesis model's order makes that
    tie-break non-reproducible across a cache rebuild. Re-rank instead by how many
    candidate briefs motivate each goal (a goal's significant word appearing in a
    brief), highest first, with an alphabetical tie-break. This keeps the semantic
    intent (the most load-bearing goal still wins) while being fully deterministic
    — unlike a plain alphabetical sort, which would discard priority entirely."""
    goals = vocab.goal_classes
    if len(goals) <= 1:
        return vocab
    briefs = [b.lower() for b in candidate_briefs]

    def motivation(goal: str) -> int:
        words = {w for w in re.findall(r"[a-z0-9]+", goal) if len(w) >= 4}
        if not words:
            return 0
        return sum(1 for b in briefs if any(w in b for w in words))

    ordered = sorted(goals, key=lambda g: (-motivation(g), g))
    if ordered == goals:
        return vocab
    return vocab.model_copy(update={"goal_classes": ordered})


def build_module_vocab(
    writer: _Writer,
    *,
    module_name: str,
    module_description: str,
    candidate_briefs: list[str],
) -> ModuleVocab | None:
    """Synthesize (do not cache) the controlled vocabulary for one module.

    Returns a normalized `ModuleVocab`, or None if synthesis failed or produced
    an unusable (empty-slot) vocabulary. Best-effort: callers fall back to the
    built-in vocabulary so this can never make a run worse than the static path.
    """
    prompt = _VOCAB_SYNTHESIS_INSTRUCTION + _render_input(
        module_name=module_name,
        module_description=module_description,
        candidate_briefs=candidate_briefs,
    )
    try:
        result = writer.run(prompt, check=False)
    except Exception as exc:  # noqa: BLE001 - surfaced as a soft miss below
        _log.warning("vocab synthesis failed (%s) — using built-in vocabulary", exc)
        return None
    text = getattr(result, "final_message", None) or ""
    if getattr(result, "returncode", 1) != 0 or not text.strip():
        _log.warning("vocab synthesis empty (rc=%s) — using built-in vocabulary",
                     getattr(result, "returncode", None))
        return None
    parsed = _parse_vocab(text)
    if parsed is None:
        _log.warning("vocab synthesis unparseable — using built-in vocabulary")
        return None
    vocab = _rerank_goal_classes(parsed.normalized(), candidate_briefs)
    if not vocab.is_usable():
        _log.warning("vocab synthesis produced empty slots — using built-in vocabulary")
        return None
    return vocab


def load_or_build_vocab(
    writer: _Writer,
    *,
    cache_dir: Path,
    module_slug: str,
    module_name: str,
    module_description: str,
    candidate_briefs: list[str],
) -> ModuleVocab | None:
    """Return the cached vocabulary when the input hash matches, else synthesize
    and cache. A corrupt or stale cache entry is ignored and rebuilt.

    Cache file: `<cache_dir>/<module_slug>.json`, shape
    `{"input_hash": <hex>, "vocab": <ModuleVocab>}`.
    """
    want_hash = vocab_input_hash(
        module_name=module_name,
        module_description=module_description,
        candidate_briefs=candidate_briefs,
    )
    cache_file = cache_dir / f"{module_slug}.json"
    if cache_file.is_file():
        try:
            blob = json.loads(cache_file.read_text(encoding="utf-8"))
            if isinstance(blob, dict) and blob.get("input_hash") == want_hash:
                vocab = ModuleVocab.model_validate(blob["vocab"]).normalized()
                if vocab.is_usable():
                    _log.info("vocab cache hit for %s (%s)", module_slug, want_hash)
                    return vocab
        except Exception:  # noqa: BLE001 - any corruption -> rebuild
            _log.warning("vocab cache unreadable for %s — rebuilding", module_slug)

    vocab = build_module_vocab(
        writer,
        module_name=module_name,
        module_description=module_description,
        candidate_briefs=candidate_briefs,
    )
    if vocab is None:
        return None
    try:
        cache_dir.mkdir(parents=True, exist_ok=True)
        cache_file.write_text(
            json.dumps(
                {"input_hash": want_hash, "vocab": vocab.model_dump()},
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )
    except OSError as exc:
        _log.warning("vocab cache write failed for %s (%s)", module_slug, exc)
    return vocab


__all__ = [
    "ModuleVocab",
    "build_module_vocab",
    "load_or_build_vocab",
    "vocab_input_hash",
]
