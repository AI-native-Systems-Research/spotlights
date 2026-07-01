"""§6.4–§6.6 content-drop validation pipeline.

Each rule may drop offending candidates (incrementing a counter) but never
raises. ID-integrity (§6.7), qualified-name (§6.3), and post-drop emptiness
(§6.8) live on the orchestrator because they *raise* and require cross-
iteration state.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path

from spotlights_engine.schemas.candidate import Candidate, Candidates
from spotlights_engine.utils.schema_compat import primary_file, primary_span


@dataclass
class DropCounters:
    dropped_outside_module: int = 0
    dropped_in_submodule: int = 0
    dropped_missing_file: int = 0
    dropped_invalid_ranges: int = 0


@dataclass
class _LineCountCache:
    cache: dict[tuple[str, int], int] = field(default_factory=dict)

    def get(self, path: Path) -> int:
        st = path.stat()
        key = (str(path), st.st_mtime_ns)
        if key not in self.cache:
            with path.open("rb") as fh:
                # Count newlines, then +1 if file is non-empty and doesn't end
                # with a newline (so a single-line file with no trailing \n
                # still reports line_count >= 1).
                data = fh.read()
            if not data:
                self.cache[key] = 0
            else:
                count = data.count(b"\n")
                if not data.endswith(b"\n"):
                    count += 1
                self.cache[key] = count
        return self.cache[key]


class Validator:
    """§6.4 → §6.5 → §6.6 in order. Returns survivors plus drop counters."""

    def __init__(
        self,
        *,
        repo_path: Path,
        module_path: str,
        submodule_paths: Sequence[str] = (),
    ) -> None:
        self._repo_path = repo_path
        self._module_path = module_path
        # Sub-modules of the target module are audited as separate discovery
        # runs, so candidates from files that live inside a sub-module must be
        # dropped here even though they are under `module_path`. Empty for leaf
        # modules → no exclusion.
        self._submodule_paths = list(submodule_paths)

    def run(self, parsed: Candidates) -> tuple[list[Candidate], DropCounters]:
        counters = DropCounters()
        line_cache = _LineCountCache()

        repo_root = self._repo_path.resolve(strict=False)
        module_root = (self._repo_path / self._module_path).resolve(strict=False)
        submodule_roots = [
            (self._repo_path / p).resolve(strict=False)
            for p in self._submodule_paths
        ]

        kept_after_containment = self._drop_outside_module(
            parsed.candidates, repo_root, module_root, submodule_roots, counters
        )
        kept_after_existence = self._drop_missing_files(
            kept_after_containment, repo_root, counters
        )
        kept_after_ranges = self._drop_invalid_ranges(
            kept_after_existence, repo_root, line_cache, counters
        )
        return kept_after_ranges, counters

    def _drop_outside_module(
        self,
        candidates: list[Candidate],
        repo_root: Path,
        module_root: Path,
        submodule_roots: list[Path],
        counters: DropCounters,
    ) -> list[Candidate]:
        survivors: list[Candidate] = []
        for c in candidates:
            file_path = Path(primary_file(c))
            if file_path.is_absolute():
                counters.dropped_outside_module += 1
                continue
            resolved = (repo_root / file_path).resolve(strict=False)
            if not _is_within(resolved, module_root):
                counters.dropped_outside_module += 1
                continue
            if any(_is_within(resolved, sub) for sub in submodule_roots):
                # Under the module, but inside one of its sub-modules — that
                # sub-module is a separate discovery target.
                counters.dropped_in_submodule += 1
                continue
            survivors.append(c)
        return survivors

    def _drop_missing_files(
        self,
        candidates: list[Candidate],
        repo_root: Path,
        counters: DropCounters,
    ) -> list[Candidate]:
        survivors: list[Candidate] = []
        for c in candidates:
            resolved = (repo_root / primary_file(c)).resolve(strict=False)
            if not resolved.is_file():
                counters.dropped_missing_file += 1
                continue
            survivors.append(c)
        return survivors

    def _drop_invalid_ranges(
        self,
        candidates: list[Candidate],
        repo_root: Path,
        line_cache: _LineCountCache,
        counters: DropCounters,
    ) -> list[Candidate]:
        survivors: list[Candidate] = []
        for c in candidates:
            resolved = (repo_root / primary_file(c)).resolve(strict=False)
            line_count = line_cache.get(resolved)
            if primary_span(c).line_end > line_count:
                counters.dropped_invalid_ranges += 1
                continue
            survivors.append(c)
        return survivors


def _is_within(child: Path, parent: Path) -> bool:
    try:
        child.relative_to(parent)
    except ValueError:
        return False
    return True


__all__ = ["DropCounters", "Validator"]
