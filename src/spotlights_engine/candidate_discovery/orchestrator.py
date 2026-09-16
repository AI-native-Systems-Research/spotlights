"""Stage 1 orchestrator: bootstrap + alternating review loop.

Holds cross-iteration state (raw and post-drop previous candidates, the
high-water-mark id) and binds agent invocation, §6.1 mutation guard,
schema parse, content drops (§6.4–§6.6), id integrity (§6.7), persistence
(§6.8), and telemetry. The §6.2 single retry lives here too — schema-parse
failures (timeout / bad JSON / nonzero exit) trigger one retry with the
strict-mode reminder appended; a second failure raises.
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass
from pathlib import Path

from pydantic import ValidationError

from spotlights_engine.candidate_discovery import layout, prompts
from spotlights_engine.candidate_discovery.agent_schema import AgentCandidates
from spotlights_engine.candidate_discovery.agents import (
    AgentInvocation,
    AgentRunner,
    ClaudeRunner,
    CodexRunner,
    _SchemaParseError,
)
from spotlights_engine.candidate_discovery.api import (
    DiscoveryConfig,
    DiscoveryResult,
    DiscoveryTruncation,
    IterationTelemetry,
)
from spotlights_engine.candidate_discovery.errors import (
    DiscoveryAgentFailureError,
    DiscoveryMutationError,
    DiscoverySetupError,
    DiscoveryValidationError,
)
from spotlights_engine.candidate_discovery.repo_guard import RepoGuard
from spotlights_engine.candidate_discovery.telemetry import (
    TelemetryBuilder,
    render_diff_markdown,
)
from spotlights_engine.candidate_discovery.validation import Validator
from spotlights_engine.schemas.candidate import Candidate, Candidates
from spotlights_engine.schemas.pipeline import CandidateDiscoveryInput
from spotlights_engine.schemas.project import Module
from spotlights_engine.utils.id_helpers import parse_id, slug_for
from spotlights_engine.utils.schema_compat import primary_file

_log = logging.getLogger(__name__)


# Initial high-water-mark counter (zero) before any candidate is seen. The
# agent works in the bare `cand-NNNN` id space; the orchestrator tracks the
# counter and renders the bare form back to the review prompt.
_MIN_SEEN_COUNTER = 0


def _bare_id(counter: int) -> str:
    """The agent-local bare form `cand-NNNN` for a counter."""
    return f"cand-{counter:04d}"

# Per plan §17: Linux MAX_ARG_STRLEN is 131_072 bytes per argument. Claude
# inlines the schema via `--json-schema <text>`; if a future schema addition
# pushes the serialized JSON past this ceiling the subprocess will fail with
# an opaque "argument list too long" error. Sanity-check at setup time with
# headroom for environment growth.
_MAX_SCHEMA_BYTES = 120_000


@dataclass
class _IterOutcome:
    raw: Candidates
    candidates: Candidates
    telemetry: IterationTelemetry
    agent_invocation: AgentInvocation


class Orchestrator:
    def __init__(
        self,
        *,
        input: CandidateDiscoveryInput,
        config: DiscoveryConfig,
        module: Module,
    ) -> None:
        self._input = input
        self._config = config
        self._module = module
        # Module id segment (D3): the agent emits bare `cand-NNNN`; the
        # orchestrator prefixes each promoted candidate to `cand-<segment>-NNNN`
        # before any schema object is built. Standalone callers may omit it, in
        # which case the module slug is the segment.
        self._segment = config.id_segment or slug_for(input.module_qualified_name)
        self._validator = Validator(
            repo_path=config.repo_path,
            module_path=module.path,
            submodule_paths=[s.path for s in module.submodules],
        )
        self._repo_guard = RepoGuard(config.repo_path)
        self._telemetry_builder = TelemetryBuilder()

        self._prev_raw: Candidates | None = None
        self._prev_post_drop: Candidates | None = None
        self._prev_post_drop_ids: set[str] = set()
        # Every candidate that has survived some earlier iteration, keyed by id
        # (latest surviving version wins). Candidates whose id is here but not in
        # `_prev_post_drop_ids` were dropped by a later pass; they are fed back
        # into the review prompt so a reviewer can argue to re-add them, and the
        # id-integrity check exempts their re-add from strict monotonicity.
        self._seen_by_id: dict[str, Candidate] = {}
        # High-water mark in the agent-local *bare* counter space. The review
        # loop and prompt speak bare `cand-NNNN`; schema objects carry the
        # prefixed form. `parse_id` recovers the counter from a prefixed id.
        self._max_seen_counter: int = _MIN_SEEN_COUNTER

        self._iterations: list[IterationTelemetry] = []
        self._iterations_fh = None
        self._last_iter_dir: Path | None = None
        # `_last_iter_dir` is assigned before the attempt runs, so after a
        # failure it points at a directory with no `candidates.json`. The final
        # artifact must come from the last iteration that actually produced one.
        self._last_good_iter_dir: Path | None = None
        self._truncation: DiscoveryTruncation | None = None

    def run(self) -> DiscoveryResult:
        run_start = time.monotonic()
        # Construct runners *before* minting the run dir so a missing CLI
        # raises before the directory exists (avoids a false "resume" on next call).
        claude = ClaudeRunner(self._config)
        codex = CodexRunner(self._config)
        review_agents: list[AgentRunner] = [codex, claude]

        self._mint_run_dir()
        try:
            with self._open_iterations_jsonl():
                boot_prompt = prompts.render_bootstrap(
                    self._input.module_qualified_name,
                    self._module,
                    repo_context_markdown=self._config.repo_context_markdown,
                    spotlight_context=self._input.context,
                )
                # A bootstrap failure leaves nothing to salvage, so it stays
                # fatal — outside the try below.
                boot = self._run_iteration(n=0, agent=claude, prompt=boot_prompt)
                self._record(boot)

                try:
                    self._run_review_loop(review_agents)
                except DiscoveryAgentFailureError as e:
                    # An agent that dies of a rate limit in a *refinement* pass
                    # does not invalidate what the earlier passes found. Keep
                    # those candidates and record the truncation; the old
                    # behaviour discarded 88 of 214 candidates (41%) across the
                    # IOCR run, including 11 valid ones from a module whose only
                    # fault was a 429 in its last iteration.
                    if self._prev_post_drop is None or not self._prev_post_drop.candidates:
                        raise
                    self._truncation = DiscoveryTruncation(
                        iteration=int(e.context.get("iteration") or 0),
                        agent=str(e.context.get("agent") or "unknown"),
                        error=type(e).__name__,
                        cause=str(e),
                        completed_iterations=len(self._iterations),
                    )
                    _log.warning(
                        "[%s] discovery: iteration %d (%s) failed — keeping %d "
                        "candidates from %d completed iterations (%s)",
                        self._input.module_qualified_name,
                        self._truncation.iteration,
                        self._truncation.agent,
                        len(self._prev_post_drop.candidates),
                        self._truncation.completed_iterations,
                        self._truncation.cause,
                    )

                self._copy_final()
        finally:
            duration = time.monotonic() - run_start

        return self._finalize(total_duration_s=duration)

    def _run_review_loop(self, review_agents: list[AgentRunner]) -> None:
        for n in range(1, self._config.num_review_iterations + 1):
            agent = review_agents[(n - 1) % 2]
            # The review prompt speaks the agent-local bare id space:
            # render the prior candidates with their bare ids and pass
            # the bare high-water mark, so the agent contract stays
            # `cand-NNNN` (D3 option A).
            prev_json = self._to_bare_json(self._prev_post_drop)  # type: ignore[arg-type]
            removed_json = self._removed_pool_bare_json()
            review_prompt = prompts.render_review(
                self._input.module_qualified_name,
                self._module,
                prev_json,
                _bare_id(self._max_seen_counter),
                removed_candidates_json=removed_json,
                repo_context_markdown=self._config.repo_context_markdown,
                spotlight_context=self._input.context,
            )
            out = self._run_iteration(n=n, agent=agent, prompt=review_prompt)
            self._record(out)

    def _mint_run_dir(self) -> None:
        # Hand the agents the discovery-only schema (no `state`,
        # proposal lists, etc.) so codex's strict structured-output stays
        # valid. The orchestrator promotes parsed payloads to full `Candidate`
        # objects via `AgentCandidates.to_candidates()`.
        schema_text = json.dumps(AgentCandidates.model_json_schema(), indent=2)
        schema_bytes = len(schema_text.encode("utf-8"))
        if schema_bytes >= _MAX_SCHEMA_BYTES:
            raise DiscoverySetupError(
                f"Candidates schema ({schema_bytes} bytes) exceeds claude "
                f"--json-schema argv ceiling ({_MAX_SCHEMA_BYTES} bytes)",
                schema_bytes=schema_bytes,
                limit=_MAX_SCHEMA_BYTES,
            )
        root = layout.candidate_discovery_root(self._config.artifacts_dir)
        root.mkdir(parents=True)
        layout.schema_path(self._config.artifacts_dir).write_text(
            schema_text, encoding="utf-8"
        )
        if self._config.repo_context_markdown is not None:
            layout.repo_context_path(self._config.artifacts_dir).write_text(
                self._config.repo_context_markdown, encoding="utf-8"
            )

    def _open_iterations_jsonl(self):
        path = layout.iterations_jsonl(self._config.artifacts_dir)
        path.parent.mkdir(parents=True, exist_ok=True)
        self._iterations_fh = path.open("a", buffering=1, encoding="utf-8")
        return _Closer(self._iterations_fh)

    def _run_iteration(self, n: int, agent: AgentRunner, prompt: str) -> _IterOutcome:
        iter_dir = layout.iter_dir(self._config.artifacts_dir, n, agent.name)
        iter_dir.mkdir(parents=True)
        self._last_iter_dir = iter_dir

        schema_retries = 0
        last_exc: Exception | None = None
        iter_start = time.monotonic()

        for attempt in (0, 1):
            attempt_prompt = prompt
            if attempt == 1:
                attempt_prompt = prompt + "\n\n" + prompts.STRICT_RETRY_REMINDER
            (iter_dir / "prompt.md").write_text(attempt_prompt, encoding="utf-8")

            try:
                with self._repo_guard.observe():
                    inv = agent.invoke(
                        prompt=attempt_prompt,
                        iter_dir=iter_dir,
                        schema_path=layout.schema_path(self._config.artifacts_dir),
                    )
                payload_json = agent.parse_last_message(iter_dir)
                try:
                    agent_parsed = AgentCandidates.model_validate_json(payload_json)
                except ValidationError as e:
                    raise _SchemaParseError(f"schema validation failed: {e}") from e
                # Promote bare agent ids to the prefixed schema form here, at the
                # construction boundary (D3): the rest of the loop — validator,
                # integrity check, telemetry, persistence — operates on final
                # `cand-<segment>-NNNN` ids.
                parsed = agent_parsed.to_candidates(segment=self._segment)

                self._check_qualified_name(parsed, n=n, agent=agent.name)
                survivors, drops = self._validator.run(parsed)
                self._check_id_integrity(parsed, survivors, n=n, agent=agent.name)
                normalized = self._normalize_and_persist(
                    n=n, agent=agent.name, survivors=survivors, iter_dir=iter_dir
                )

                iteration_duration = time.monotonic() - iter_start
                telemetry = self._telemetry_builder.build(
                    n=n,
                    agent=agent.name,
                    inv=inv,
                    survivors=normalized,
                    drops=drops,
                    prev=self._prev_post_drop,
                    schema_retries=schema_retries,
                    iteration_duration_s=iteration_duration,
                )

                if n >= 1:
                    (iter_dir / "diff_from_prev.md").write_text(
                        render_diff_markdown(self._prev_post_drop, normalized),
                        encoding="utf-8",
                    )

                self._last_good_iter_dir = iter_dir
                return _IterOutcome(
                    raw=parsed,
                    candidates=normalized,
                    telemetry=telemetry,
                    agent_invocation=inv,
                )
            except DiscoveryMutationError as e:
                # Plan §6: raised iterations carry iteration/agent in context.
                # RepoGuard raises without that context (it has no notion of n/agent),
                # so enrich here before letting it propagate. Mutation is fatal
                # (not retryable), so we don't enter the schema-retry path.
                e.context.setdefault("iteration", n)
                e.context.setdefault("agent", agent.name)
                raise
            except _SchemaParseError as e:
                last_exc = e
                if attempt == 1:
                    break
                schema_retries += 1
                continue

        raise DiscoveryAgentFailureError(
            "schema parse failed twice",
            iteration=n,
            agent=agent.name,
            cause=str(last_exc) if last_exc else None,
        ) from last_exc

    def _check_qualified_name(self, parsed: Candidates, n: int, agent: str) -> None:
        expected = self._input.module_qualified_name
        if parsed.module_qualified_name != expected:
            raise DiscoveryValidationError(
                "qualified-name mismatch",
                iteration=n,
                agent=agent,
                expected=expected,
                got=parsed.module_qualified_name,
            )

    def _check_id_integrity(
        self,
        parsed: Candidates,
        survivors: list[Candidate],
        n: int,
        agent: str,
    ) -> None:
        raw_ids = [c.id for c in parsed.candidates]
        if len(set(raw_ids)) != len(raw_ids):
            raise DiscoveryValidationError(
                "duplicate id within iteration",
                iteration=n,
                agent=agent,
                ids=raw_ids,
            )

        prev_by_id = (
            {c.id: c for c in self._prev_raw.candidates}
            if self._prev_raw is not None
            else {}
        )
        for c in parsed.candidates:
            if c.id in prev_by_id:
                if primary_file(c) != primary_file(prev_by_id[c.id]):
                    raise DiscoveryValidationError(
                        "carry-over id changed file",
                        iteration=n,
                        agent=agent,
                        id=c.id,
                        prev_file=primary_file(prev_by_id[c.id]),
                        new_file=primary_file(c),
                    )
            elif c.id in self._seen_by_id:
                # Re-add of a previously dropped candidate (not in the immediately
                # previous raw set, but seen in some earlier iteration). It must
                # keep its original file — an id reused for a different file is a
                # collision, not a re-add.
                seen = self._seen_by_id[c.id]
                if primary_file(c) != primary_file(seen):
                    raise DiscoveryValidationError(
                        "re-added id changed file",
                        iteration=n,
                        agent=agent,
                        id=c.id,
                        prev_file=primary_file(seen),
                        new_file=primary_file(c),
                    )

        survivor_ids = {c.id for c in survivors}
        # Re-added ids (seen in an earlier iteration) keep their original
        # counter, so they are exempt from strict monotonicity; only genuinely
        # new ids must exceed the high-water mark.
        truly_new = survivor_ids - self._prev_post_drop_ids - set(self._seen_by_id)
        for nid in truly_new:
            # Monotonicity is on the per-(module, session) counter, which is the
            # same whether ids are bare or prefixed (uniqueness across modules
            # comes from the slug, not the number). Compare counters, not the
            # full id strings, so a slug containing digits can't perturb order.
            if parse_id(nid)[2] <= self._max_seen_counter:
                raise DiscoveryValidationError(
                    "newly minted id not strictly greater than max_seen_id",
                    iteration=n,
                    agent=agent,
                    id=nid,
                    max_seen_id=_bare_id(self._max_seen_counter),
                )

    def _normalize_and_persist(
        self,
        n: int,
        agent: str,
        survivors: list[Candidate],
        iter_dir: Path,
    ) -> Candidates:
        # An empty `survivors` is valid per the architecture: the manager will
        # mark the module run `SKIPPED` if discovery returns zero candidates.
        normalized = Candidates(
            module_qualified_name=self._input.module_qualified_name,
            candidates=survivors,
        )

        (iter_dir / "candidates.json").write_text(
            normalized.model_dump_json(indent=2), encoding="utf-8"
        )
        return normalized

    def _removed_pool_bare_json(self) -> str | None:
        """Render previously-dropped candidates for the review prompt.

        A candidate is "dropped" when it survived some earlier iteration (so it
        is in `_seen_by_id`) but is absent from the latest survivor set
        (`_prev_post_drop_ids`). Rendered in the agent-local bare id space, like
        `prev_json`, so the agent can re-emit it verbatim to re-add it. Returns
        `None` when nothing has been dropped.
        """
        dropped = [
            c for cid, c in self._seen_by_id.items()
            if cid not in self._prev_post_drop_ids
        ]
        if not dropped:
            return None
        pool = Candidates(
            module_qualified_name=self._input.module_qualified_name,
            candidates=dropped,
        )
        return self._to_bare_json(pool)

    def _to_bare_json(self, candidates: Candidates) -> str:
        """Render `candidates` with their bare `cand-NNNN` ids for the agent.

        The loop carries prefixed schema ids, but the review prompt must speak
        the agent-local bare form (D3 option A). Rewrite each id back to its
        counter; everything else is unchanged.
        """
        bare = [
            c.model_copy(update={"id": _bare_id(parse_id(c.id)[2])})
            for c in candidates.candidates
        ]
        return candidates.model_copy(update={"candidates": bare}).model_dump_json(
            indent=2
        )

    def _record(self, outcome: _IterOutcome) -> None:
        self._iterations.append(outcome.telemetry)
        assert self._iterations_fh is not None
        self._iterations_fh.write(outcome.telemetry.model_dump_json() + "\n")
        self._iterations_fh.flush()

        self._prev_raw = outcome.raw
        self._prev_post_drop = outcome.candidates
        self._prev_post_drop_ids = {c.id for c in outcome.candidates.candidates}
        for c in outcome.candidates.candidates:
            # Remember the latest surviving version of each candidate so a later
            # iteration can be shown (and re-add) anything a subsequent pass drops.
            self._seen_by_id[c.id] = c
            counter = parse_id(c.id)[2]
            if counter > self._max_seen_counter:
                self._max_seen_counter = counter

        _log.info(
            "[%s] discovery: iteration %d (%s) — %d candidates so far",
            self._input.module_qualified_name,
            outcome.telemetry.n,
            outcome.telemetry.agent,
            len(outcome.candidates.candidates),
        )

    def _copy_final(self) -> None:
        assert self._last_good_iter_dir is not None
        src = self._last_good_iter_dir / "candidates.json"
        dst = layout.final_artifact(self._config.artifacts_dir)
        dst.write_bytes(src.read_bytes())

    def _finalize(self, total_duration_s: float) -> DiscoveryResult:
        assert self._prev_post_drop is not None
        cost_reports = [t.cost_usd for t in self._iterations if t.cost_usd is not None]
        total_cost = sum(cost_reports) if cost_reports else None
        return DiscoveryResult(
            candidates=self._prev_post_drop,
            iterations=list(self._iterations),
            total_duration_s=total_duration_s,
            total_cost_usd=total_cost,
            truncated_by=self._truncation,
        )


class _Closer:
    def __init__(self, fh) -> None:
        self._fh = fh

    def __enter__(self) -> None:
        return None

    def __exit__(self, exc_type, exc, tb) -> None:
        self._fh.close()


__all__ = ["Orchestrator"]
