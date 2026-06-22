"""Public CLI for `spotlights-engine`.

Thin shim over `spotlights_manager.run_with_telemetry`. Architectural inputs
(`--repo`, `--objective`, `--hint`, `--max-findings-per-module`) bind to
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
from spotlights_engine.module_deep_research import AntigravityExecOptions, CodexExecOptions
from spotlights_engine.module_knowledge import KnowledgeBase, KnowledgeRecord, RetrieveRequest
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
        "--include",
        action="append",
        default=None,
        nargs="+",
        metavar="QN",
        help=(
            "Restrict to one or more slash-form qualified names (e.g. "
            "v1/kv_offload). A parent name expands to all leaves beneath "
            "it (e.g. v1/worker matches v1/worker/gpu). Repeat the flag "
            "or pass multiple values after one flag. Default: all modules."
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

    provider = p.add_argument_group(
        "provider/runtime overrides",
        "Optional generic agent-provider settings; defaults use each CLI's local config.",
    )
    provider.add_argument(
        "--codex-profile",
        default=None,
        help="Codex config profile for Codex-backed steps (e.g. a local gateway profile).",
    )
    provider.add_argument(
        "--codex-model",
        default=None,
        help=(
            "Codex model override for Codex-backed steps. "
            "If omitted, profile/default config wins."
        ),
    )
    provider.add_argument(
        "--claude-model",
        default=None,
        help="Claude model passed to Claude Code subprocesses.",
    )
    provider.add_argument(
        "--claude-base-url",
        default=None,
        help="Claude-compatible API base URL exposed to Claude Code as ANTHROPIC_BASE_URL.",
    )
    provider.add_argument(
        "--claude-auth-token-env",
        default=None,
        help=(
            "Name of an existing environment variable whose value should be copied "
            "to ANTHROPIC_AUTH_TOKEN for Claude subprocesses."
        ),
    )
    provider.add_argument(
        "--claude-unset-env",
        action="append",
        default=[],
        metavar="NAME[,NAME...]",
        help="Environment variable(s) to remove before launching Claude; repeatable.",
    )
    provider.add_argument(
        "--claude-disable-experimental-betas",
        action="store_true",
        help="Set CLAUDE_CODE_DISABLE_EXPERIMENTAL_BETAS=1 for Claude subprocesses.",
    )
    provider.add_argument(
        "--antigravity-base-url",
        default=None,
        help="Gemini-compatible base URL for the Antigravity SDK runner.",
    )
    provider.add_argument(
        "--antigravity-model",
        default=None,
        help="Model name for the Antigravity SDK runner.",
    )
    provider.add_argument(
        "--antigravity-api-key-env",
        default=None,
        help="Environment variable containing the Antigravity/Gemini-compatible API key.",
    )
    provider.add_argument(
        "--antigravity-timeout-seconds",
        type=int,
        default=None,
        help=(
            "Timeout for the Antigravity SDK runner. "
            "Default: runner/provider default."
        ),
    )
    provider.add_argument(
        "--antigravity-auth-mechanism",
        choices=("bearer", "x-goog-api-key"),
        default="bearer",
        help="How to pass the Antigravity API key to the configured endpoint (default: bearer).",
    )
    provider.add_argument(
        "--no-antigravity-sse-normalizer",
        dest="antigravity_sse_normalizer",
        action="store_false",
        help="Disable the local SSE normalizer used for some Gemini-compatible gateways.",
    )
    provider.set_defaults(antigravity_sse_normalizer=True)
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


def _flatten_csv(raw: list[str]) -> list[str]:
    names: list[str] = []
    for item in raw:
        names.extend(part.strip() for part in item.split(",") if part.strip())
    return names


def _apply_claude_env_directives(args: argparse.Namespace) -> None:
    """Set parent-process directives consumed by Claude subprocess helpers."""
    import os

    if args.claude_model:
        os.environ["SPOTLIGHTS_CLAUDE_MODEL"] = args.claude_model
    if args.claude_base_url:
        os.environ["SPOTLIGHTS_CLAUDE_BASE_URL"] = args.claude_base_url
    if args.claude_auth_token_env:
        os.environ["SPOTLIGHTS_CLAUDE_AUTH_TOKEN_ENV"] = args.claude_auth_token_env
    unset_names = _flatten_csv(args.claude_unset_env)
    if unset_names:
        os.environ["SPOTLIGHTS_CLAUDE_UNSET_ENV"] = ",".join(unset_names)
    if args.claude_disable_experimental_betas:
        os.environ["SPOTLIGHTS_CLAUDE_DISABLE_EXPERIMENTAL_BETAS"] = "1"


def _build_input(args: argparse.Namespace) -> SpotlightsManagerInput:
    context_kwargs = {
        "objective": args.objective,
        "workload_hints": list(args.hint),
    }
    input_kwargs: dict = {
        "repo_path": args.repo,
        "context": SpotlightContext(**context_kwargs),
    }
    if args.max_findings_per_module is not None:
        input_kwargs["max_findings_per_module"] = args.max_findings_per_module
    return SpotlightsManagerInput(**input_kwargs)


def _build_config(args: argparse.Namespace) -> SpotlightsManagerConfig:
    proposal_cfg: ProposalFromFindingConfig | None = None
    if args.max_parallel_pairs is not None or args.debug_first_n_pairs is not None:
        kwargs: dict = {}
        if args.max_parallel_pairs is not None:
            kwargs["max_parallel_pairs"] = args.max_parallel_pairs
        if args.debug_first_n_pairs is not None:
            kwargs["debug_first_n_pairs"] = args.debug_first_n_pairs
        proposal_cfg = ProposalFromFindingConfig(**kwargs)

    agent_cfg: AgentProposalsConfig | None = None
    if (
        args.max_parallel_candidates is not None
        or args.debug_first_n_candidates is not None
    ):
        kwargs = {}
        if args.max_parallel_candidates is not None:
            kwargs["max_parallel_candidates"] = args.max_parallel_candidates
        if args.debug_first_n_candidates is not None:
            kwargs["debug_first_n_candidates"] = args.debug_first_n_candidates
        agent_cfg = AgentProposalsConfig(**kwargs)

    include = _flatten_include(args.include)

    discovery_cfg: DiscoveryConfig | None = None
    if args.codex_profile is not None or args.codex_model is not None:
        discovery_kwargs: dict = {}
        if args.codex_profile is not None:
            discovery_kwargs["codex_profile"] = args.codex_profile
            # Let a profile own model/provider selection unless the caller
            # explicitly supplies a model override as well.
            discovery_kwargs["codex_model"] = None
        if args.codex_model is not None:
            discovery_kwargs["codex_model"] = args.codex_model
        discovery_cfg = DiscoveryConfig(**discovery_kwargs)

    deep_research_cfg: CodexExecOptions | None = None
    if args.codex_profile is not None or args.codex_model is not None:
        deep_research_cfg = CodexExecOptions(
            profile=args.codex_profile,
            model=args.codex_model,
        )

    antigravity_cfg: AntigravityExecOptions | None = None
    if (
        args.antigravity_base_url is not None
        or args.antigravity_model is not None
        or args.antigravity_api_key_env is not None
        or args.antigravity_timeout_seconds is not None
    ):
        antigravity_cfg = AntigravityExecOptions(
            model=args.antigravity_model,
            antigravity_base_url=(
                args.antigravity_base_url.rstrip("/")
                if args.antigravity_base_url is not None
                else None
            ),
            antigravity_api_key_env=args.antigravity_api_key_env,
            api_key_auth_mechanism=args.antigravity_auth_mechanism,
            timeout_seconds=args.antigravity_timeout_seconds,
            normalize_sse_bytes_repr=args.antigravity_sse_normalizer,
        )

    if args.codex_profile is not None and agent_cfg is None:
        agent_cfg = AgentProposalsConfig(codex_profile=args.codex_profile)
    elif args.codex_profile is not None:
        agent_cfg = agent_cfg.model_copy(update={"codex_profile": args.codex_profile})
    if args.codex_model is not None and agent_cfg is None:
        agent_cfg = AgentProposalsConfig(codex_model=args.codex_model)
    elif args.codex_model is not None:
        agent_cfg = agent_cfg.model_copy(update={"codex_model": args.codex_model})

    return SpotlightsManagerConfig(
        artifacts_dir=args.artifacts_dir,
        output_folder=args.output_folder,
        max_parallel_sessions=args.max_parallel,
        module_filter=ModuleFilter(include=include) if include else None,
        discovery=discovery_cfg,
        deep_research=deep_research_cfg,
        deep_research_antigravity=antigravity_cfg,
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


def _print_summary(result: SpotlightsManagerResult) -> None:
    """Render the §2 stdout shape from a completed run."""
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
            sum(len(c.deep_research_proposals) for c in run.candidates.candidates)
            if run.candidates
            else 0
        )
        n_agent_proposals = (
            sum(len(c.agent_proposals) for c in run.candidates.candidates)
            if run.candidates
            else 0
        )
        print(f"[2/5] candidate_discovery ({qn}) … {n_cands} candidates")
        print(f"[3/5] module_deep_research ({qn}) … {n_findings} findings")
        print(
            f"[4/5] proposal_from_finding_creator ({qn}) … "
            f"{n_proposals} proposals attached"
        )
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

    _apply_claude_env_directives(args)
    _configure_logging(args)

    inp = _build_input(args)
    cfg = _build_config(args)

    result = run_with_telemetry(inp, config=cfg)
    _print_summary(result)
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
