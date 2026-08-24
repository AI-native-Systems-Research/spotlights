"""The one_shot_apply error hierarchy."""

from __future__ import annotations

import pytest

from spotlights_engine.one_shot_apply.errors import (
    ArtifactWriteError,
    NotAGitRepoError,
    OneShotApplyError,
    WorktreeError,
)


@pytest.mark.parametrize("exc_type", [NotAGitRepoError, WorktreeError, ArtifactWriteError])
def test_subclasses_share_the_base(exc_type: type[Exception]) -> None:
    assert issubclass(exc_type, OneShotApplyError)


@pytest.mark.parametrize(
    "name",
    ["ArtifactWriteError", "ClaudeUnavailableError", "NotAGitRepoError", "WorktreeError"],
)
def test_every_error_is_reachable_from_the_package_root(name: str) -> None:
    """The package root is the documented import surface.

    A caller that wants to catch one of these should not have to reach into
    `one_shot_apply.errors` for some of them and the package for the rest — an
    error missing from `__init__` is invisible to anyone reading the public API.
    """
    import spotlights_engine.one_shot_apply as pkg

    assert hasattr(pkg, name)
    assert name in pkg.__all__


def test_base_is_catchable_as_exception() -> None:
    with pytest.raises(OneShotApplyError) as excinfo:
        raise NotAGitRepoError("not a git checkout: /tmp/x")
    assert "not a git checkout" in str(excinfo.value)
