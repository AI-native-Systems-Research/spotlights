"""Error hierarchy for `prep-evolve`.

The CLI maps each subclass to a clean stderr message + nonzero exit. All
failures that should abort *before* any bundle is written derive from
`PrepEvolveError`, so the dispatch layer can catch the base type once.
"""

from __future__ import annotations


class PrepEvolveError(Exception):
    """Base class for all prep-evolve failures."""


class SelectionError(PrepEvolveError):
    """The requested module run or candidate id does not exist in the result."""


class RepoResolutionError(PrepEvolveError):
    """The target repo path could not be resolved or is not a directory."""


class StalenessError(PrepEvolveError):
    """The live repo no longer matches the recorded candidate well enough to
    place an evolve target confidently (line range/symbol drift)."""


class ScopeError(PrepEvolveError):
    """The requested `--scope` is incompatible with the chosen evolver."""


class UnsupportedEvolverError(PrepEvolveError):
    """An unknown `--evolver` value, or an evolver whose `supports()` rejected
    the spec."""


class BundleExistsError(PrepEvolveError):
    """The bundle directory already exists and `--force` was not given (or the
    prior manifest is missing so ownership cannot be determined)."""


__all__ = [
    "BundleExistsError",
    "PrepEvolveError",
    "RepoResolutionError",
    "ScopeError",
    "SelectionError",
    "StalenessError",
    "UnsupportedEvolverError",
]
