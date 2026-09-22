"""Public CLI for `spotlights-engine`.

Thin shim over `spotlights_manager.run_with_telemetry`. Architectural inputs
(`--repo`, `--objective`, `--hint`, `--max-findings-per-module`,
`--no-deep-research`) bind to
`SpotlightsManagerInput` / `SpotlightContext`; runtime/infra knobs
(`--output-folder`, `--artifacts-dir`, parallelism, debug caps, `--no-resume`)
bind to `SpotlightsManagerConfig`. Library users construct those types
directly.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import logging
import os
import re
import sys
from pathlib import Path

from pydantic import ValidationError

from spotlights_engine.agent_proposals import AgentProposalsConfig
from spotlights_engine.candidate_discovery import DiscoveryConfig
from spotlights_engine.defaults import (
    DEFAULT_ARTIFACTS as _DEFAULT_ARTIFACTS,
)
from spotlights_engine.defaults import (
    DEFAULT_OUTPUT as _DEFAULT_OUTPUT,
)
from spotlights_engine.defaults import (
    DEFAULT_REPO as _DEFAULT_REPO,
)
from spotlights_engine.model_config import (
    MODELS_ENV_VAR,
    ModelConfig,
    load_model_config,
    models_path,
)
from spotlights_engine.module_deep_research import (
    CodexExecOptions,
)
from spotlights_engine.module_knowledge import (
    KnowledgeBase,
    KnowledgeRecord,
    RetrieveRequest,
)
from spotlights_engine.modules_extractor import (
    ExtractorConfig,
)
from spotlights_engine.proposal_from_finding_creator import (
    ProposalFromFindingConfig,
)
from spotlights_engine.schemas.common import SpotlightContext
from spotlights_engine.schemas.pipeline import SpotlightsManagerInput
from spotlights_engine.spotlights_manager import (
    ModuleFilter,
    SpotlightsManagerConfig,
    SpotlightsManagerResult,
    run_with_telemetry,
)

_DEFAULT_OBJECTIVE = "reduce hot-path latency on common workloads"


def _build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="spotlights-engine",
        description=(
            "Propose evidence-backed, high-leverage code changes for a target "
            "repo. Runs structural extraction, candidate discovery, deep "
            "research, and proposal generation; writes Markdown results."
        ),
    )

    p.add_argument(
        "--repo",
        type=Path,
        default=_DEFAULT_REPO,
        help=f"Path to the target repo (default: {_DEFAULT_REPO}).",
    )
    p.add_argument(
        "--repo-url",
        default=None,
        help="Canonical target repo URL for the public run manifest.",
    )
    p.add_argument(
        "--include",
        action="append",
        default=None,
        nargs="+",
        metavar="QN",
        help=(
            "Restrict to one or more slash-form qualified names (e.g. "
            "v1/kv_offload). A parent name selects itself plus every module "
            "beneath it (e.g. v1/worker matches v1/worker and v1/worker/gpu). "
            "Repeat the flag or pass multiple values after one flag. "
            "Default: all modules."
        ),
    )
    p.add_argument(
        "--objective",
        default=_DEFAULT_OBJECTIVE,
        help=f"Spotlight objective (default: {_DEFAULT_OBJECTIVE!r}).",
    )
    p.add_argument(
        "--hint",
        action="append",
        default=[],
        help=(
            "Workload hint (repeatable). Threaded into "
            "SpotlightContext.workload_hints."
        ),
    )
    p.add_argument(
        "--output-folder",
        type=Path,
        default=_DEFAULT_OUTPUT,
        help=(
            "Where index.md and per-module pages land "
            f"(default: {_DEFAULT_OUTPUT})."
        ),
    )
    p.add_argument(
        "--artifacts-dir",
        type=Path,
        default=_DEFAULT_ARTIFACTS,
        help=(
            "Where checkpoints and raw transcripts land (resume key) "
            f"(default: {_DEFAULT_ARTIFACTS})."
        ),
    )

    p.add_argument(
        "--max-parallel",
        type=int,
        default=1,
        help="Modules processed concurrently (default: 1).",
    )
    p.add_argument(
        "--max-parallel-pairs",
        type=int,
        default=None,
        help=(
            "Within-step parallelism for step 4 (proposal_from_finding_creator). "
            "Default: ProposalFromFindingConfig default (5)."
        ),
    )
    p.add_argument(
        "--max-parallel-candidates",
        type=int,
        default=None,
        help=(
            "Within-step parallelism for step 5 (agent_proposals). "
            "Default: AgentProposalsConfig default (5)."
        ),
    )
    p.add_argument(
        "--max-findings-per-module",
        type=int,
        default=None,
        help=(
            "Cap on findings produced by step 3 per module. "
            "Default: SpotlightsManagerInput default (30)."
        ),
    )
    review = p.add_mutually_exclusive_group()
    review.add_argument(
        "--review-iterations",
        type=int,
        default=None,
        metavar="N",
        help=(
            "Number of candidate-discovery review iterations (step 2). "
            "0 disables the review session (bootstrap pass only). "
            "Default: DiscoveryConfig default (3)."
        ),
    )
    review.add_argument(
        "--no-review",
        dest="review_iterations",
        action="store_const",
        const=0,
        help="Disable the candidate-discovery review session (alias for --review-iterations 0).",
    )

    p.add_argument(
        "--claude-model",
        default=None,
        metavar="ID",
        help=(
            "Model id passed to `claude --model` for every step. Overrides "
            f"models.yaml (or ${MODELS_ENV_VAR}) for this run. Omit to use the "
            "file; leave the file blank to inherit ~/.claude/settings.json."
        ),
    )
    p.add_argument(
        "--codex-model",
        default=None,
        metavar="ID",
        help=(
            "Model id passed to Codex for every step. Overrides models.yaml "
            f"(or ${MODELS_ENV_VAR}) for this run. Omit to use the file."
        ),
    )

    p.add_argument(
        "--no-candidate-hotspots",
        dest="include_candidate_hotspots",
        action="store_false",
        help=(
            "Do not surface step-2 candidate_discovery hot spots in the "
            "step-3 (module_deep_research) prompt. Default: they are included."
        ),
    )

    p.add_argument(
        "--per-candidate-deep-research",
        dest="per_candidate_deep_research",
        action="store_true",
        help=(
            "Run step 3 (module_deep_research) once per candidate hot spot "
            "instead of once per module. Each research pass is scoped to a "
            "single candidate; findings from all passes are merged, deduped, "
            "and capped together. Applies to every runner (Codex/Claude/"
            "OpenAlex). Default: off — one module-wide pass."
        ),
    )

    p.add_argument(
        "--enable-claude-search",
        dest="enable_claude_search",
        action="store_true",
        help=(
            "Also run the Claude runner in step 3 (module_deep_research). "
            "Default: off — step 3 runs Codex only."
        ),
    )

    p.add_argument(
        "--enable-openalex",
        dest="enable_openalex",
        action="store_true",
        help=(
            "Replace Codex with the OpenAlex runner in step 3 "
            "(module_deep_research): relevance-ranked academic-paper research "
            "via the free OpenAlex REST API instead of the code-aware Codex "
            "survey. Default: off. No API key required; set OPENALEX_API_KEY "
            "for premium credits and OPENALEX_MAILTO for the polite pool."
        ),
    )

    p.add_argument(
        "--openalex-model",
        default=None,
        metavar="ID",
        help=(
            "Model id passed to the OpenAlex runner in step 3. In --openalex-query-mode "
            "codex this is the Codex query-writer's model; otherwise a no-op. Omit "
            "to use the SDK default. No-op unless --enable-openalex is set."
        ),
    )

    p.add_argument(
        "--openalex-query-mode",
        dest="openalex_query_mode",
        choices=("regex", "codex"),
        default="codex",
        help=(
            "How the OpenAlex runner turns the step-3 prompt into an OpenAlex "
            "search query. 'regex' — deterministic parse of the module's "
            "qualified name + objective, no LLM. 'codex' (default) — a Codex "
            "query-writer distills the prompt into keywords, falling back to "
            "the regex derivation if it yields nothing. No-op unless "
            "--enable-openalex is set."
        ),
    )

    p.add_argument(
        "--no-deep-research",
        dest="enable_deep_research",
        action="store_false",
        help=(
            "Skip step 3 (module_deep_research) entirely: no research CLI "
            "session, no findings, step 4 short-circuits to zero proposals. "
            "Steps 1, 2 and 5 still run. The step-3-only knobs "
            "(--enable-claude-search, --no-candidate-hotspots, "
            "--max-findings-per-module) become no-ops but still take part in "
            "the resume fingerprint, so a run must be resumed with the same "
            "flags it was started with. Default: deep research is enabled."
        ),
    )

    p.add_argument(
        "--no-proposals-from-findings",
        dest="enable_proposals_from_findings",
        action="store_false",
        help=(
            "Skip step 4 (proposal_from_finding_creator) pairing: step 3 still "
            "runs and its findings are kept, but no per-(candidate, finding) "
            "Claude session fires. Every candidate advances with zero "
            "finding-derived proposals. Steps 1, 2, 3 and 5 still run. This is "
            "the cost lever for step 4 (one Claude session per pair). Included "
            "in the resume fingerprint, so resume with the same flag it started "
            "with. Default: proposals-from-findings enabled."
        ),
    )

    p.add_argument(
        "--no-resume",
        dest="resume",
        action="store_false",
        help="Refuse to start over an existing run dir.",
    )

    p.add_argument(
        "--debug-first-n-pairs",
        type=int,
        default=None,
        help="Debug-only: cap step 4 to the first N (candidate, finding) pairs.",
    )
    p.add_argument(
        "--debug-first-n-candidates",
        type=int,
        default=None,
        help="Debug-only: cap step 5 to the first N candidates.",
    )

    p.add_argument(
        "--dry-run",
        action="store_true",
        help=(
            "Resolve and print the module scope, objective, and parallelism "
            "that would run, then exit without invoking any agent (no cost)."
        ),
    )
    p.add_argument(
        "--max-cost",
        type=float,
        default=None,
        metavar="USD",
        help=(
            "Advisory cost ceiling in USD. Recorded for enforcement at step "
            "boundaries; must be > 0."
        ),
    )

    verbosity = p.add_mutually_exclusive_group()
    verbosity.add_argument(
        "--quiet",
        "-q",
        action="store_true",
        help="Only show warnings and errors on stderr.",
    )
    verbosity.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Show DEBUG-level progress (per-pair / per-candidate completions).",
    )
    p.add_argument(
        "--log-file",
        type=Path,
        default=None,
        help="Tee progress logs to this file at the same level as stderr.",
    )

    return p


def _build_knowledge_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="spotlights-engine knowledge",
        description=(
            "Local module-knowledge archive/retrieve/wiki commands. These commands "
            "are deterministic and do not invoke agents."
        ),
    )
    p.add_argument(
        "--root",
        type=Path,
        default=Path(".spotlights/knowledge"),
        help="Knowledge root directory (default: .spotlights/knowledge).",
    )
    sub = p.add_subparsers(dest="knowledge_command", required=True)

    archive = sub.add_parser("archive", help="Archive one record or a list of records from JSON.")
    archive.add_argument(
        "--record-json",
        required=True,
        help="Path to a KnowledgeRecord JSON object/list, or '-' for stdin.",
    )
    archive.add_argument(
        "--render-wiki",
        action="store_true",
        help="Render the generated wiki after archiving.",
    )

    retrieve = sub.add_parser("retrieve", help="Retrieve ranked local knowledge records.")
    retrieve.add_argument("query", help="Query text.")
    retrieve.add_argument("--top-k", type=int, default=10)
    retrieve.add_argument("--source-type", action="append", default=[])
    retrieve.add_argument("--tag", action="append", default=[])
    retrieve.add_argument("--json", action="store_true", help="Print machine-readable JSON.")
    retrieve.add_argument(
        "--render-query-page",
        action="store_true",
        help="Write a generated wiki page for this retrieval.",
    )
    retrieve.add_argument("--query-id", default=None, help="Optional stable wiki query page id.")

    wiki = sub.add_parser("wiki", help="Render or verify generated knowledge wiki pages.")
    wiki_sub = wiki.add_subparsers(dest="wiki_command", required=True)
    wiki_sub.add_parser("render", help="Render wiki pages from archive/records.jsonl.")
    verify = wiki_sub.add_parser("verify", help="Verify generated wiki pages.")
    verify.add_argument(
        "--no-strict",
        dest="strict",
        action="store_false",
        help="Skip strict record-page coverage checks.",
    )
    verify.set_defaults(strict=True)
    return p


_LOG_FORMAT = "%(asctime)s %(levelname)-5s %(message)s"
_LOG_DATEFMT = "%H:%M:%S"
_HANDLER_TAG = "_spotlights_cli_handler"


def _configure_logging(args: argparse.Namespace) -> None:
    """Install stderr (and optional file) handlers on the spotlights_engine
    root logger. Idempotent — repeated calls do not duplicate handlers."""
    if args.quiet:
        level = logging.WARNING
    elif args.verbose:
        level = logging.DEBUG
    else:
        level = logging.INFO

    root = logging.getLogger("spotlights_engine")
    root.setLevel(level)

    # Drop any handlers a previous in-process call to main() installed.
    for h in list(root.handlers):
        if getattr(h, _HANDLER_TAG, False):
            root.removeHandler(h)
            h.close()

    formatter = logging.Formatter(_LOG_FORMAT, datefmt=_LOG_DATEFMT)

    stderr_handler = logging.StreamHandler(sys.stderr)
    stderr_handler.setFormatter(formatter)
    stderr_handler.setLevel(level)
    setattr(stderr_handler, _HANDLER_TAG, True)
    root.addHandler(stderr_handler)

    if args.log_file is not None:
        args.log_file.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(args.log_file, encoding="utf-8")
        file_handler.setFormatter(formatter)
        file_handler.setLevel(level)
        setattr(file_handler, _HANDLER_TAG, True)
        root.addHandler(file_handler)


def _flatten_include(raw: list[list[str]] | None) -> list[str]:
    if not raw:
        return []
    flat: list[str] = []
    for group in raw:
        flat.extend(group)
    return flat


def _build_input(args: argparse.Namespace) -> SpotlightsManagerInput:
    context_kwargs = {
        "objective": args.objective,
        "workload_hints": list(args.hint),
    }
    input_kwargs: dict = {
        "repo_path": args.repo,
        "context": SpotlightContext(**context_kwargs),
    }
    if args.repo_url:
        input_kwargs["repo_url"] = args.repo_url
    if args.max_findings_per_module is not None:
        input_kwargs["max_findings_per_module"] = args.max_findings_per_module
    input_kwargs["include_candidate_hotspots"] = args.include_candidate_hotspots
    input_kwargs["per_candidate_deep_research"] = args.per_candidate_deep_research
    input_kwargs["enable_claude_search"] = args.enable_claude_search
    input_kwargs["enable_deep_research"] = args.enable_deep_research
    input_kwargs["enable_proposals_from_findings"] = (
        args.enable_proposals_from_findings
    )
    input_kwargs["enable_openalex"] = args.enable_openalex
    input_kwargs["openalex_query_mode"] = args.openalex_query_mode
    if args.openalex_model is not None:
        input_kwargs["openalex_model"] = args.openalex_model
    return SpotlightsManagerInput(**input_kwargs)


#: What `DiscoveryConfig.codex_model` accepts. Validating against it here means a
#: bad Codex id fails at the flag instead of inside pydantic after `--dry-run`
#: has already reported the run as fine.
_CODEX_MODEL_RE = re.compile(r"^[\w.\-/\[\]]+$")

#: Claude ids are not constrained by any field downstream, and real ones contain
#: characters the Codex pattern rejects — Bedrock's `...-v1:0`, OpenRouter's
#: `...:online`. So only reject what could not be an argv value at all.
_CLAUDE_MODEL_RE = re.compile(r"^\S+$")


def _validate_model_id(
    source: str, value: str, *, from_flag: bool, pattern: re.Pattern[str], expected: str
) -> None:
    """Reject a model id that would fail deeper in the stack, or is unusable.

    A flag error goes through argparse (usage text, exit 2). A file value raises
    instead: printing argparse usage for a typo in `models.yaml` would point the
    reader at the wrong thing, so the message names the file.
    """
    if pattern.match(value):
        return
    detail = f"{value!r} is not a valid model id ({expected})"
    if from_flag:
        _build_argparser().error(f"{source}: {detail}")
    raise ValueError(f"{source}: {detail}")


def _codex_explicitly_blank(args: argparse.Namespace) -> bool:
    """True when the user passed `--codex-model ""` (as opposed to omitting it).

    The distinction matters only for step 2, the one step with a built-in Codex
    default that a blank value would otherwise not override.
    """
    return args.codex_model is not None and not args.codex_model.strip()


def _resolve_models(args: argparse.Namespace) -> tuple[str | None, str | None]:
    """The run's `(claude, codex)` model ids, or `None` to inherit the CLI default.

    Precedence: `--claude-model` / `--codex-model` > `SPOTLIGHTS_MODELS_FILE` >
    the bundled `models.yaml` > the agent CLI's own config. Resolved once, here,
    so every step config carries an explicit value and nothing deeper in the
    stack reads a config file.

    A flag passed as an empty string (`--codex-model ""`) is an explicit request
    to inherit, and beats a value in the file — otherwise a pinned bundled file
    would leave no way to ask for the CLI's own default from the command line.
    An omitted flag is `None` and falls through to the file.
    """
    # Only read the file when a flag has not already answered. A stale
    # SPOTLIGHTS_MODELS_FILE is an error, and it should not abort a run that
    # specified both models on the command line and would have ignored the file.
    needs_file = args.claude_model is None or args.codex_model is None
    file_cfg = load_model_config() if needs_file else ModelConfig()

    if args.claude_model is None:
        claude = file_cfg.claude
    else:
        claude = args.claude_model.strip() or None
    if args.codex_model is None:
        codex = file_cfg.codex
    else:
        codex = args.codex_model.strip() or None

    file_path = models_path()
    if claude:
        from_flag = args.claude_model is not None
        _validate_model_id(
            "--claude-model" if from_flag else f"{file_path} (claude)",
            claude,
            from_flag=from_flag,
            pattern=_CLAUDE_MODEL_RE,
            expected="no whitespace",
        )
    if codex:
        from_flag = args.codex_model is not None
        _validate_model_id(
            "--codex-model" if from_flag else f"{file_path} (codex)",
            codex,
            from_flag=from_flag,
            pattern=_CODEX_MODEL_RE,
            expected="expected letters, digits, and any of . - _ / [ ]",
        )
    return claude, codex


def _build_config(args: argparse.Namespace) -> SpotlightsManagerConfig:
    claude_model, codex_model = _resolve_models(args)

    proposal_kwargs: dict = {}
    if args.max_parallel_pairs is not None:
        proposal_kwargs["max_parallel_pairs"] = args.max_parallel_pairs
    if args.debug_first_n_pairs is not None:
        proposal_kwargs["debug_first_n_pairs"] = args.debug_first_n_pairs
    if claude_model:
        proposal_kwargs["claude_model"] = claude_model
    proposal_cfg = (
        ProposalFromFindingConfig(**proposal_kwargs) if proposal_kwargs else None
    )

    agent_kwargs: dict = {}
    if args.max_parallel_candidates is not None:
        agent_kwargs["max_parallel_candidates"] = args.max_parallel_candidates
    if args.debug_first_n_candidates is not None:
        agent_kwargs["debug_first_n_candidates"] = args.debug_first_n_candidates
    if claude_model:
        agent_kwargs["claude_model"] = claude_model
    if codex_model:
        agent_kwargs["codex_model"] = codex_model
    agent_cfg = AgentProposalsConfig(**agent_kwargs) if agent_kwargs else None

    discovery_kwargs: dict = {}
    if args.review_iterations is not None:
        discovery_kwargs["num_review_iterations"] = args.review_iterations
    if claude_model:
        discovery_kwargs["claude_model"] = claude_model
    if codex_model:
        discovery_kwargs["codex_model"] = codex_model
    elif _codex_explicitly_blank(args):
        # `DiscoveryConfig.codex_model` defaults to `gpt-5.5` (kept for resume
        # compatibility), so "no value" cannot mean inherit here the way it does
        # for every other step. An *explicitly* empty flag is a deliberate ask,
        # and has to override that default or `--codex-model ""` would silently
        # do nothing to step 2.
        discovery_kwargs["codex_model"] = None
    discovery_cfg = DiscoveryConfig(**discovery_kwargs) if discovery_kwargs else None

    # Step 1 always has a config object, so stamp the model straight onto it.
    # SPOTLIGHTS_SOURCE_ROOT_MAX_TURNS raises the source-root discovery turn cap
    # for chattier models (e.g. local Qwen/Qwen3.8-27B) that over-explore and hit the
    # default 15-turn ceiling before emitting the module list. Unset = default.
    # SPOTLIGHTS_EXTRACTOR_MAX_TURNS raises the enrich/assign (Stage-3A) turn cap
    # for the same reason: chatty models blow past the default 60 and abort with
    # `terminal_reason=max_turns`. Unset = default.
    _extractor_kwargs: dict = {"claude_model": claude_model}
    _srmt = os.environ.get("SPOTLIGHTS_SOURCE_ROOT_MAX_TURNS", "").strip()
    if _srmt:
        _extractor_kwargs["source_root_max_turns"] = int(_srmt)
    _emt = os.environ.get("SPOTLIGHTS_EXTRACTOR_MAX_TURNS", "").strip()
    if _emt:
        _extractor_kwargs["max_turns"] = int(_emt)
    # SPOTLIGHTS_ENRICH_TIMEOUT_S / SPOTLIGHTS_EXTRACTOR_TIMEOUT_S raise the
    # per-shard enrich and whole-repo stage deadlines: a slow local proxy
    # (Qwen/Qwen3.8-27B over IBM LiteLLM) generates far slower than hosted Claude and
    # trips the defaults. Unset = default.
    _ets = os.environ.get("SPOTLIGHTS_ENRICH_TIMEOUT_S", "").strip()
    if _ets:
        _extractor_kwargs["enrich_timeout_s"] = int(_ets)
    _tts = os.environ.get("SPOTLIGHTS_EXTRACTOR_TIMEOUT_S", "").strip()
    if _tts:
        _extractor_kwargs["timeout_s"] = int(_tts)
    # SPOTLIGHTS_MAX_PARALLEL_ENRICH_SHARDS cuts Stage-3 fan-out (local proxy
    # refuses connections under the default 5). SPOTLIGHTS_ENRICH_API_RETRIES
    # adds retries for transient API failures incl. ConnectionRefused. Unset =
    # default.
    _mpes = os.environ.get("SPOTLIGHTS_MAX_PARALLEL_ENRICH_SHARDS", "").strip()
    if _mpes:
        _extractor_kwargs["max_parallel_enrich_shards"] = int(_mpes)
    _ear = os.environ.get("SPOTLIGHTS_ENRICH_API_RETRIES", "").strip()
    if _ear:
        _extractor_kwargs["enrich_api_retries"] = int(_ear)
    extractor_cfg = ExtractorConfig(**_extractor_kwargs)

    # Step 3 has no config object of its own: its Codex model rides on the
    # `CodexExecOptions` the manager copies per module, the Claude model on a
    # dedicated manager field.
    deep_research_cfg = CodexExecOptions(model=codex_model) if codex_model else None

    include = _flatten_include(args.include)
    return SpotlightsManagerConfig(
        artifacts_dir=args.artifacts_dir,
        output_folder=args.output_folder,
        max_parallel_sessions=args.max_parallel,
        module_filter=ModuleFilter(include=include) if include else None,
        extractor=extractor_cfg,
        discovery=discovery_cfg,
        deep_research=deep_research_cfg,
        models=ModelConfig(claude=claude_model, codex=codex_model),
        proposal_from_finding=proposal_cfg,
        agent_proposals=agent_cfg,
        resume=args.resume,
    )


def _write_result_json(result: SpotlightsManagerResult, output_folder: Path) -> Path:
    """Persist the full SpotlightsManagerResult as JSON next to index.md.

    `extractor_invocation` is a dataclass (not a pydantic model), so we
    serialize it via `dataclasses.asdict` and splice it into the dump.
    """
    if not hasattr(result, "model_dump"):
        raise TypeError("result must provide pydantic model_dump()")
    payload = result.model_dump(mode="json", exclude={"extractor_invocation"})
    payload["extractor_invocation"] = dataclasses.asdict(result.extractor_invocation)
    path = output_folder / "result.json"
    path.write_text(json.dumps(payload, indent=2, default=str) + "\n", encoding="utf-8")
    return path


def _print_summary(
    result: SpotlightsManagerResult,
    *,
    deep_research_enabled: bool = True,
    proposals_from_findings_enabled: bool = True,
) -> None:
    """Render the §2 stdout shape from a completed run.

    `deep_research_enabled=False` (i.e. `--no-deep-research`) makes the [3/5]
    line read "disabled" instead of "0 findings", which would otherwise be
    indistinguishable from "research ran and found nothing".
    """
    qns = list(result.module_runs.keys())
    if qns:
        kept_label = f"{len(qns)} module{'s' if len(qns) != 1 else ''} kept"
        kept_label = f"{kept_label} ({', '.join(qns)})"
    else:
        kept_label = "0 modules kept"
    print(f"[1/5] modules_extractor … {kept_label}")

    for qn in qns:
        run = result.module_runs[qn]
        n_cands = len(run.candidates.candidates) if run.candidates else 0
        n_findings = len(run.findings)
        n_proposals = (
            sum(
                1
                for c in run.candidates.candidates
                for p in c.proposals
                if p.source == "research_finding"
            )
            if run.candidates
            else 0
        )
        n_agent_proposals = (
            sum(
                1
                for c in run.candidates.candidates
                for p in c.proposals
                if p.source == "agent_knowledge"
            )
            if run.candidates
            else 0
        )
        print(f"[2/5] candidate_discovery ({qn}) … {n_cands} candidates")
        if deep_research_enabled:
            print(f"[3/5] module_deep_research ({qn}) … {n_findings} findings")
        else:
            print(f"[3/5] module_deep_research ({qn}) … disabled")
        if proposals_from_findings_enabled:
            print(
                f"[4/5] proposal_from_finding_creator ({qn}) … "
                f"{n_proposals} proposals attached"
            )
        else:
            print(f"[4/5] proposal_from_finding_creator ({qn}) … disabled")
        print(
            f"[5/5] agent_proposals ({qn}) … "
            f"{n_agent_proposals} agent proposals attached"
        )

    if result.renderer_result is not None:
        print(f"results: {result.renderer_result.index_path}")
    else:
        print("results: (renderer skipped)")
        for issue in result.manager_issues:
            print(f"  manager-issue [{issue.step}/{issue.severity}]: {issue.message}")


def _read_json_argument(value: str) -> object:
    if value == "-":
        return json.loads(sys.stdin.read())
    return json.loads(Path(value).read_text(encoding="utf-8"))


def _knowledge_archive(args: argparse.Namespace) -> int:
    kb = KnowledgeBase.open(args.root)
    try:
        payload = _read_json_argument(args.record_json)
        raw_records = payload if isinstance(payload, list) else [payload]
        records = [KnowledgeRecord.model_validate(raw) for raw in raw_records]
    except (OSError, json.JSONDecodeError, ValidationError, TypeError) as exc:
        print(f"invalid knowledge record JSON: {exc}", file=sys.stderr)
        return 2

    result = kb.archive_many(records)
    print(f"archive: {result.records_written} records at {result.path}")
    print(f"inserted: {result.inserted}")
    print(f"updated: {result.updated}")
    if args.render_wiki:
        render = kb.render_wiki()
        print(f"wiki: {render.wiki_dir} ({render.pages_written} pages)")
    return 0


def _knowledge_retrieve(args: argparse.Namespace) -> int:
    kb = KnowledgeBase.open(args.root)
    try:
        request = RetrieveRequest(
            query=args.query,
            top_k=args.top_k,
            source_types=list(args.source_type),
            tags=list(args.tag),
        )
    except ValidationError as exc:
        print(f"invalid retrieve request: {exc}", file=sys.stderr)
        return 2

    results = kb.retrieve(request)
    if args.render_query_page:
        query_page = kb.render_retrieval_wiki(args.query, results, query_id=args.query_id)
        print(f"query_page: {query_page}")

    if args.json:
        print(json.dumps([item.model_dump(mode="json") for item in results], indent=2))
    else:
        print(f"results: {len(results)}")
        for idx, item in enumerate(results, start=1):
            record = item.record
            terms = ", ".join(item.matched_terms) or "(none)"
            print(
                f"{idx}. [{record.source_type}] {record.title} "
                f"(score={item.score:.4f}, id={record.record_id})"
            )
            print(f"   matched: {terms}")
    return 0


def _knowledge_wiki(args: argparse.Namespace) -> int:
    kb = KnowledgeBase.open(args.root)
    if args.wiki_command == "render":
        result = kb.render_wiki()
        print(f"wiki: {result.wiki_dir}")
        print(f"pages: {result.pages_written}")
        print(f"source_hash: {result.source_hash}")
        return 0
    if args.wiki_command == "verify":
        report = kb.verify_wiki(strict=args.strict)
        print(f"checked_pages: {report.checked_pages}")
        for issue in report.issues:
            print(f"{issue.severity}: {issue.page_path}: {issue.message}")
        return 0 if report.ok else 1
    return 2


def _knowledge_main(argv: list[str]) -> int:
    args = _build_knowledge_argparser().parse_args(argv)
    args.root = args.root.resolve()
    if args.knowledge_command == "archive":
        return _knowledge_archive(args)
    if args.knowledge_command == "retrieve":
        return _knowledge_retrieve(args)
    if args.knowledge_command == "wiki":
        return _knowledge_wiki(args)
    return 2


def main(argv: list[str] | None = None) -> int:
    raw_argv = list(sys.argv[1:] if argv is None else argv)
    if raw_argv and raw_argv[0] == "init":
        from spotlights_engine.init_skills import main as init_main

        return init_main(raw_argv[1:])
    if raw_argv and raw_argv[0] == "knowledge":
        return _knowledge_main(raw_argv[1:])
    if raw_argv and raw_argv[0] == "prep-evolve":
        from spotlights_engine.prep_evolve.cli import main as prep_main

        return prep_main(raw_argv[1:])
    if raw_argv and raw_argv[0] == "apply":
        from spotlights_engine.one_shot_apply.cli import main as apply_main

        return apply_main(raw_argv[1:])
    if raw_argv and raw_argv[0] == "doctor":
        from spotlights_engine.doctor import main as doctor_main

        return doctor_main(raw_argv[1:])

    args = _build_argparser().parse_args(argv)

    # Resolve to absolute up-front: codex runs subprocesses with `-C <repo_path>`,
    # so any relative path baked into a config (schema, last_message, artifacts)
    # would resolve under the target repo, not this project's CWD.
    args.artifacts_dir = args.artifacts_dir.resolve()
    args.output_folder = args.output_folder.resolve()
    if args.repo is not None:
        args.repo = args.repo.resolve()

    args.artifacts_dir.mkdir(parents=True, exist_ok=True)
    args.output_folder.mkdir(parents=True, exist_ok=True)

    if args.max_cost is not None and args.max_cost <= 0:
        _build_argparser().error("--max-cost must be > 0")

    # Warn (never error) so wrapper scripts that always pass step-3 knobs keep
    # working. Checked before the --dry-run return so a dry run still reports it.
    if not args.enable_deep_research:
        ignored = []
        if args.enable_claude_search:
            ignored.append("--enable-claude-search")
        if args.enable_openalex:
            ignored.append("--enable-openalex")
        if args.openalex_query_mode != "codex":
            ignored.append("--openalex-query-mode")
        if args.per_candidate_deep_research:
            ignored.append("--per-candidate-deep-research")
        if not args.include_candidate_hotspots:
            ignored.append("--no-candidate-hotspots")
        if args.max_findings_per_module is not None:
            ignored.append("--max-findings-per-module")
        if ignored:
            # Printed, not logged: `spotlights_engine/__init__.py` installs a
            # NullHandler on the package logger, so `logging.lastResort` never
            # fires and a pre-`_configure_logging` warning would be swallowed.
            verb = "is" if len(ignored) == 1 else "are"
            print(
                f"warning: --no-deep-research: {', '.join(ignored)} {verb} "
                f"step-3-only and {verb} ignored "
                f"(but still recorded in the resume fingerprint)",
                file=sys.stderr,
            )

    if args.dry_run:
        include = _flatten_include(args.include)
        scope = ", ".join(include) if include else "(all modules)"
        # Resolve models here as well: a dry run that reports "fine" and is
        # followed by a real run that dies on a bad id or an unparseable
        # models file is worse than no dry run at all.
        try:
            dry_claude, dry_codex = _resolve_models(args)
        except (OSError, ValueError) as exc:
            print(f"spotlights-engine: {exc}", file=sys.stderr)
            return 2
        print("dry-run: no agents invoked, no cost incurred.")
        print(f"  repo:        {args.repo}")
        print(f"  objective:   {args.objective!r}")
        print(f"  scope:       {scope}")
        print(f"  max-parallel:{args.max_parallel}")
        if args.max_cost is not None:
            print(f"  max-cost:    ${args.max_cost:.2f}")
        # Report what the engine will actually pass, not what the config says.
        # Step 2 has a built-in Codex default, so `(CLI default)` here would be
        # the same misreport doctor and the run manifest were fixed to avoid.
        # An explicit `--codex-model ""` overrides step 2's built-in default in
        # the real run, so the dry run must not claim otherwise.
        if _codex_explicitly_blank(args):
            codex_label = "(CLI default)"
        elif dry_codex:
            codex_label = dry_codex
        else:
            default_codex = DiscoveryConfig().codex_model
            codex_label = (
                f"{default_codex} (step 2 default; steps 3+5 inherit)"
                if default_codex
                else "(CLI default)"
            )
        print(f"  claude-model:{dry_claude or '(CLI default)'}")
        print(f"  codex-model: {codex_label}")
        if not args.enable_deep_research:
            print("  deep-research: DISABLED (step 3 skipped, step 4 empty)")
        if not args.enable_proposals_from_findings:
            print(
                "  proposals-from-findings: DISABLED "
                "(step 4 pairing skipped, step 3 findings kept)"
            )
        return 0

    _configure_logging(args)

    inp = _build_input(args)
    try:
        cfg = _build_config(args)
    except (OSError, ValueError) as exc:
        # A malformed or unreadable models file must not be a traceback.
        print(f"spotlights-engine: {exc}", file=sys.stderr)
        return 2

    result = run_with_telemetry(inp, config=cfg)
    _print_summary(
        result,
        deep_research_enabled=inp.enable_deep_research,
        proposals_from_findings_enabled=inp.enable_proposals_from_findings,
    )
    if isinstance(result, SpotlightsManagerResult):
        json_path = _write_result_json(result, args.output_folder)
        print(f"result json: {json_path}")

    any_unrecoverable = any(
        any(not iss.recoverable for iss in run.issues)
        for run in result.module_runs.values()
    )
    return 1 if any_unrecoverable else 0


if __name__ == "__main__":
    sys.exit(main())
