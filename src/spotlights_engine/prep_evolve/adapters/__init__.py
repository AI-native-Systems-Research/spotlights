"""Adapter registry, keyed by the canonical evolver name.

The CLI normalizes the `agentic-strategy-evolution` alias to `nous` and expands
`all` before any lookup here (plan §2), so this registry only knows the three
canonical keys.
"""

from __future__ import annotations

from collections.abc import Callable

from spotlights_engine.prep_evolve.adapters.base import Adapter, GeneratedFile
from spotlights_engine.prep_evolve.adapters.coral import CoralAdapter
from spotlights_engine.prep_evolve.adapters.nous import NousAdapter
from spotlights_engine.prep_evolve.adapters.skydiscover import SkydiscoverAdapter

# Factory functions so each call gets a fresh adapter with run-specific options
# (e.g. --model). The signature is (model) -> Adapter.
ADAPTERS: dict[str, Callable[[str | None], Adapter]] = {
    "skydiscover": lambda model: SkydiscoverAdapter(model=model),
    "coral": lambda model: CoralAdapter(model=model),
    "nous": lambda model: NousAdapter(model=model),
}

# Display order for `--evolver all`.
ALL_EVOLVERS = ["skydiscover", "coral", "nous"]

# CLI alias -> registry key.
EVOLVER_ALIASES = {"agentic-strategy-evolution": "nous"}


def normalize_evolver(value: str) -> list[str]:
    """Resolve a `--evolver` value to a list of registry keys.

    `all` expands to every adapter; the `agentic-strategy-evolution` alias
    collapses to `nous`. Raises `KeyError` for an unknown value (the CLI maps
    that to a clean error).
    """
    if value == "all":
        return list(ALL_EVOLVERS)
    key = EVOLVER_ALIASES.get(value, value)
    if key not in ADAPTERS:
        raise KeyError(value)
    return [key]


def build_adapter(key: str, model: str | None = None) -> Adapter:
    return ADAPTERS[key](model)


__all__ = [
    "ADAPTERS",
    "ALL_EVOLVERS",
    "Adapter",
    "EVOLVER_ALIASES",
    "GeneratedFile",
    "build_adapter",
    "normalize_evolver",
]
