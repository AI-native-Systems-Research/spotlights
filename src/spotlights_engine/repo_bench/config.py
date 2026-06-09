"""Repo-specific config for the bench pipeline.

Loads a TOML file describing the patterns the pipeline should use to
filter / analyze / extract for a given target repo. Default: the
bundled `configs/vllm.toml`.

Schema kept intentionally flat — every section is "list of regex
patterns + their tag," and the modules consuming it just iterate.

Repo-agnostic baselines (generic perf nouns like `throughput`,
`latency`) are merged with each config's `extra` lists so a config
file only needs to add repo-specific patterns.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any

if sys.version_info >= (3, 11):
    import tomllib
else:  # pragma: no cover
    import tomli as tomllib  # type: ignore[no-redef]

from pydantic import BaseModel, ConfigDict, Field


# ── Schema ────────────────────────────────────────────────────────────


class _Pattern(BaseModel):
    """A regex + the tag it normalizes to. Used in workload analyzer."""
    model_config = ConfigDict(extra="forbid")
    regex: str = Field(min_length=1)
    tag: str = Field(min_length=1)


class _ModelPattern(BaseModel):
    """Same shape as `_Pattern` but the field is named `family` for
    readability (model patterns map to a canonical family name)."""
    model_config = ConfigDict(extra="forbid")
    regex: str = Field(min_length=1)
    family: str = Field(min_length=1)


class _FilePattern(BaseModel):
    model_config = ConfigDict(extra="forbid")
    regex: str = Field(min_length=1)
    category: str = Field(min_length=1)


class FilterConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    perf_nouns_extra: list[str] = Field(
        default_factory=list,
        description="Repo-specific perf nouns added to the generic baseline "
        "(throughput, latency, speedup, etc).",
    )
    chore_tags_extra: list[str] = Field(
        default_factory=list,
        description="Repo-specific chore tags. Generic baseline includes "
        "bugfix, fix, ci, test, doc, refactor, chore, etc.",
    )
    perf_labels: list[str] = Field(
        default_factory=list,
        description="GitHub labels that mark a PR as perf-relevant even "
        "without a numeric claim. Used by the LabelsCarryPerfSignal rule.",
    )


class WorkloadConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    models: list[_ModelPattern] = Field(default_factory=list)
    features: list[_Pattern] = Field(default_factory=list)
    hardware: list[_Pattern] = Field(default_factory=list)
    file_categories: list[_FilePattern] = Field(default_factory=list)


class WorkloadCommandsConfig(BaseModel):
    """LLM extractor prompt customization (opt-in path)."""
    model_config = ConfigDict(extra="forbid")
    serve_command_keyword: str = Field(
        default="vllm serve",
        description="Token the LLM looks for as a serve-command marker. "
        "E.g. `vllm serve` for vLLM, `pg_bench` for Postgres.",
    )
    bench_command_keyword: str = Field(
        default="vllm bench",
        description="Token marking the benchmark driver command.",
    )


class RepoBenchConfig(BaseModel):
    """Top-level bench config bound to one target repo."""
    model_config = ConfigDict(extra="forbid")
    name: str = Field(min_length=1)
    repo: str = Field(min_length=1, description="GitHub `owner/name`.")
    description: str = ""
    filter: FilterConfig = Field(default_factory=FilterConfig)
    workload: WorkloadConfig = Field(default_factory=WorkloadConfig)
    workload_commands: WorkloadCommandsConfig = Field(
        default_factory=WorkloadCommandsConfig,
    )


# ── Loader ────────────────────────────────────────────────────────────


_CONFIGS_DIR = Path(__file__).parent / "configs"


@lru_cache(maxsize=8)
def load_config(name_or_path: str = "vllm") -> RepoBenchConfig:
    """Load a config by short name (looking under `configs/`) or path.

    Names match `configs/<name>.toml`. Paths are absolute or relative
    to cwd. Cached because repeat reads in the same process are
    common (CLI dispatches once, then internals re-resolve).
    """
    path: Path
    candidate = Path(name_or_path)
    if candidate.suffix == ".toml" and candidate.exists():
        path = candidate
    else:
        # Treat as short name; look under bundled configs.
        path = _CONFIGS_DIR / f"{name_or_path}.toml"
        if not path.exists():
            available = sorted(p.stem for p in _CONFIGS_DIR.glob("*.toml"))
            raise FileNotFoundError(
                f"config not found: {name_or_path!r}. Tried {path}. "
                f"Bundled configs: {available}."
            )

    with path.open("rb") as f:
        data: dict[str, Any] = tomllib.load(f)
    return RepoBenchConfig.model_validate(data)


# ── Compiled accessors (used by heuristics + workloads) ──────────────


@dataclass(frozen=True)
class CompiledFilterPatterns:
    """Pre-compiled regex pieces for filter heuristics."""
    perf_nouns: str       # alternation, ready to inline into bigger regex
    chore_tags: str       # alternation, anchored at title start by caller
    perf_labels: tuple[str, ...] = ()  # lowercased labels for set membership


@dataclass(frozen=True)
class CompiledWorkloadPatterns:
    """Pre-compiled (regex, tag/family) pairs for workload analyzer.

    Tuples instead of dicts so iteration is deterministic.
    """
    models: tuple[tuple[str, str], ...]    # (regex, family)
    features: tuple[tuple[str, str], ...]  # (regex, tag)
    hardware: tuple[tuple[str, str], ...]
    file_categories: tuple[tuple[str, str], ...]


# ── Generic baselines (every config inherits these) ──────────────────

# Perf nouns that work for any code repo.
_BASELINE_PERF_NOUNS = (
    "throughput", "latency", "speedup", "speed[- ]?up", "perf",
    "performance", "regression", "improvement", "faster", "slower",
)

# Chore tag prefixes seen across many repos (in addition to repo-specific).
_BASELINE_CHORE_TAGS = (
    "bug\\s*fix", "bugfix", "fix", "ci", "ci/?build", "build",
    "test", "tests", "doc", "docs",
    "chore", "refactor", "cleanup", "style", "lint", "typo",
    "nit", "misc", "deps", "dep",
)


def compile_filter_patterns(cfg: RepoBenchConfig) -> CompiledFilterPatterns:
    nouns = list(_BASELINE_PERF_NOUNS) + list(cfg.filter.perf_nouns_extra)
    tags = list(_BASELINE_CHORE_TAGS) + list(cfg.filter.chore_tags_extra)
    labels = tuple(lbl.lower() for lbl in cfg.filter.perf_labels)
    return CompiledFilterPatterns(
        perf_nouns="|".join(nouns),
        chore_tags="|".join(tags),
        perf_labels=labels,
    )


def compile_workload_patterns(cfg: RepoBenchConfig) -> CompiledWorkloadPatterns:
    return CompiledWorkloadPatterns(
        models=tuple((p.regex, p.family) for p in cfg.workload.models),
        features=tuple((p.regex, p.tag) for p in cfg.workload.features),
        hardware=tuple((p.regex, p.tag) for p in cfg.workload.hardware),
        file_categories=tuple(
            (p.regex, p.category) for p in cfg.workload.file_categories
        ),
    )


__all__ = [
    "CompiledFilterPatterns",
    "CompiledWorkloadPatterns",
    "RepoBenchConfig",
    "FilterConfig",
    "WorkloadConfig",
    "WorkloadCommandsConfig",
    "compile_filter_patterns",
    "compile_workload_patterns",
    "load_config",
]
