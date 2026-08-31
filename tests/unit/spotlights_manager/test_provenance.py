"""Engine provenance: where `spotlights_commit_sha` finds the engine's commit.

A clone answers from git. A `uv tool install git+…` has no checkout to ask, so
the commit has to come from the installer's PEP 610 record instead.
"""

from __future__ import annotations

import json
from importlib import metadata
from pathlib import Path

import pytest

from spotlights_engine.spotlights_manager import provenance as P

INSTALL_SHA = "dbb942bfee2fd22bb3db79c68eefb33870c8d9e6"


class _FakeDistribution:
    """Just enough of `importlib.metadata.Distribution` for `read_text`."""

    def __init__(self, files: dict[str, str]) -> None:
        self._files = files

    def read_text(self, filename: str) -> str | None:
        return self._files.get(filename)


def _no_git(monkeypatch: pytest.MonkeyPatch) -> None:
    """Stand in for an installed tree: every git query comes back empty."""
    monkeypatch.setattr(P, "_git_output", lambda args, cwd: "")


def _distribution(
    monkeypatch: pytest.MonkeyPatch, files: dict[str, str] | None
) -> None:
    def fake(name: str) -> _FakeDistribution:
        assert name == P._DIST_NAME
        if files is None:
            raise metadata.PackageNotFoundError(name)
        return _FakeDistribution(files)

    monkeypatch.setattr(P.metadata, "distribution", fake)


def test_git_checkout_wins_over_install_metadata(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An editable install or clone is the live tree — its HEAD is the truth,
    even when stale install metadata also records a commit."""
    monkeypatch.setattr(P, "_git_output", lambda args, cwd: "69a6d3d")
    _distribution(
        monkeypatch,
        {"direct_url.json": json.dumps({"vcs_info": {"vcs": "git", "commit_id": INSTALL_SHA}})},
    )

    assert P.spotlights_commit_sha() == "69a6d3d"


def test_falls_back_to_the_commit_the_tool_was_installed_from(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """`uv tool install git+https://…/spotlights.git` pins the resolved commit
    in the dist-info; the installed tree is not a checkout, so git cannot see
    it and the manifests would otherwise report ""."""
    _no_git(monkeypatch)
    _distribution(
        monkeypatch,
        {
            "direct_url.json": json.dumps(
                {
                    "url": "https://github.com/AI-native-Systems-Research/spotlights.git",
                    "vcs_info": {
                        "vcs": "git",
                        "commit_id": INSTALL_SHA,
                        "requested_revision": "main",
                    },
                }
            )
        },
    )

    assert P.spotlights_commit_sha() == INSTALL_SHA


def test_local_directory_install_records_no_commit(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """`uv tool install .` writes `dir_info`, which carries no commit — there is
    nothing to report and nothing to invent."""
    _no_git(monkeypatch)
    _distribution(
        monkeypatch,
        {"direct_url.json": json.dumps({"url": "file:///clone", "dir_info": {}})},
    )

    assert P.spotlights_commit_sha() == ""


@pytest.mark.parametrize(
    "files",
    [
        pytest.param(None, id="distribution-not-installed"),
        pytest.param({}, id="no-direct-url-json"),
        pytest.param({"direct_url.json": "{not json"}, id="unparseable"),
        pytest.param(
            {"direct_url.json": json.dumps({"vcs_info": []})},
            id="vcs-info-not-a-mapping",
        ),
        pytest.param(
            {"direct_url.json": json.dumps({"vcs_info": {"vcs": "hg", "commit_id": "abc"}})},
            id="non-git-vcs",
        ),
        pytest.param(
            {"direct_url.json": json.dumps({"vcs_info": {"vcs": "git"}})},
            id="git-without-commit-id",
        ),
    ],
)
def test_degrades_to_empty_rather_than_raising(
    monkeypatch: pytest.MonkeyPatch, files: dict[str, str] | None
) -> None:
    """Provenance is descriptive, never load-bearing: a missing or malformed
    record is noted downstream, it does not fail the run."""
    _no_git(monkeypatch)
    _distribution(monkeypatch, files)

    assert P.spotlights_commit_sha() == ""


def test_collect_provenance_carries_the_installed_commit(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """The fallback has to reach the manifests, which read `collect_provenance`
    rather than calling `spotlights_commit_sha` themselves."""
    _no_git(monkeypatch)
    _distribution(
        monkeypatch,
        {"direct_url.json": json.dumps({"vcs_info": {"vcs": "git", "commit_id": INSTALL_SHA}})},
    )

    prov = P.collect_provenance(repo_path=tmp_path)

    assert prov["spotlights_commit_sha"] == INSTALL_SHA
    assert prov["target_commit_sha"] == ""
