"""The one_shot_fix error hierarchy."""

from __future__ import annotations

import pytest

from spotlights_engine.one_shot_fix.errors import (
    NotAGitRepoError,
    OneShotFixError,
    WorktreeError,
)


@pytest.mark.parametrize("exc_type", [NotAGitRepoError, WorktreeError])
def test_subclasses_share_the_base(exc_type: type[Exception]) -> None:
    assert issubclass(exc_type, OneShotFixError)


def test_base_is_catchable_as_exception() -> None:
    with pytest.raises(OneShotFixError) as excinfo:
        raise NotAGitRepoError("not a git checkout: /tmp/x")
    assert "not a git checkout" in str(excinfo.value)
