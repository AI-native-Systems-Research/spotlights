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
import time
from dataclasses import dataclass
from pathlib import Path

from pydantic import ValidationError

from spotlights_engine.candidate_discovery import layout, prompts
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
    IterationTelemetry,
)
from spotlights_engine.candidate_discovery.errors import (
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


_MIN_SEEN_ID = "cand-0000"

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
    def __init__(self, config: DiscoveryConfig) -> None:
        self._config = config
        self._validator = Validator(config)
        self._repo_guard = RepoGuard(config.repo_path)
        self._telemetry_builder = TelemetryBuilder()

        self._prev_raw: Candidates | None = None
        self._prev_post_drop: Candidates | None = None
        self._prev_post_drop_ids: set[str] = set()
        self._max_seen_id: str = _MIN_SEEN_ID

        self._iterations: list[IterationTelemetry] = []
        self._iterations_fh = None
        self._last_iter_dir: Path | None = None

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
                    self._config.module_qualified_name,
                    self._config.module,
                    repo_context_markdown=self._config.repo_context_markdown,
                )
                boot = self._run_iteration(n=0, agent=claude, prompt=boot_prompt)
                self._record(boot)

                for n in range(1, self._config.num_review_iterations + 1):
                    agent = review_agents[(n - 1) % 2]
                    prev_json = self._prev_post_drop.model_dump_json(indent=2)  # type: ignore[union-attr]
                    review_prompt = prompts.render_review(
                        self._config.module_qualified_name,
                        self._config.module,
                        prev_json,
                        self._max_seen_id,
                        repo_context_markdown=self._config.repo_context_markdown,
                    )
                    out = self._run_iteration(n=n, agent=agent, prompt=review_prompt)
                    self._record(out)

                self._copy_final()
        finally:
            duration = time.monotonic() - run_start

        return self._finalize(total_duration_s=duration)

    def _mint_run_dir(self) -> None:
        schema_text = json.dumps(Candidates.model_json_schema(), indent=2)
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
                    parsed = Candidates.model_validate_json(payload_json)
                except ValidationError as e:
                    raise _SchemaParseError(f"schema validation failed: {e}") from e

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

        raise DiscoveryValidationError(
            "schema parse failed twice",
            iteration=n,
            agent=agent.name,
            cause=str(last_exc) if last_exc else None,
        ) from last_exc

    def _check_qualified_name(self, parsed: Candidates, n: int, agent: str) -> None:
        expected = self._config.module_qualified_name
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

        if self._prev_raw is not None:
            prev_by_id = {c.id: c for c in self._prev_raw.candidates}
            for c in parsed.candidates:
                if c.id in prev_by_id and c.file != prev_by_id[c.id].file:
                    raise DiscoveryValidationError(
                        "carry-over id changed file",
                        iteration=n,
                        agent=agent,
                        id=c.id,
                        prev_file=prev_by_id[c.id].file,
                        new_file=c.file,
                    )

        survivor_ids = {c.id for c in survivors}
        new_ids = survivor_ids - self._prev_post_drop_ids
        for nid in new_ids:
            if nid <= self._max_seen_id:
                raise DiscoveryValidationError(
                    "newly minted id not strictly greater than max_seen_id",
                    iteration=n,
                    agent=agent,
                    id=nid,
                    max_seen_id=self._max_seen_id,
                )

    def _normalize_and_persist(
        self,
        n: int,
        agent: str,
        survivors: list[Candidate],
        iter_dir: Path,
    ) -> Candidates:
        try:
            normalized = Candidates(
                module_qualified_name=self._config.module_qualified_name,
                candidates=survivors,
            )
        except ValidationError as e:
            raise DiscoveryValidationError(
                "post-drop list empty",
                iteration=n,
                agent=agent,
                cause=str(e),
            ) from e

        (iter_dir / "candidates.json").write_text(
            normalized.model_dump_json(indent=2), encoding="utf-8"
        )
        return normalized

    def _record(self, outcome: _IterOutcome) -> None:
        self._iterations.append(outcome.telemetry)
        assert self._iterations_fh is not None
        self._iterations_fh.write(outcome.telemetry.model_dump_json() + "\n")
        self._iterations_fh.flush()

        self._prev_raw = outcome.raw
        self._prev_post_drop = outcome.candidates
        self._prev_post_drop_ids = {c.id for c in outcome.candidates.candidates}
        for c in outcome.candidates.candidates:
            if c.id > self._max_seen_id:
                self._max_seen_id = c.id

    def _copy_final(self) -> None:
        assert self._last_iter_dir is not None
        src = self._last_iter_dir / "candidates.json"
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
        )


class _Closer:
    def __init__(self, fh) -> None:
        self._fh = fh

    def __enter__(self) -> None:
        return None

    def __exit__(self, exc_type, exc, tb) -> None:
        self._fh.close()


__all__ = ["Orchestrator"]
