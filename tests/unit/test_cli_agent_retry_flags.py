"""The retry flags must reach step 5, and must not disturb a run that ignores them.

Two properties, and the second is the one that breaks silently: the engine's
`--resume` fingerprint hashes each step's config, so three new keys with default
values would make every half-finished run directory on disk fail to resume for a
feature it never used. `persistence._RETRY_FIELDS` keeps them out of the hash
while the retry is off; this file is what notices if that stops working.

See docs/handoff/rate-limit-failures.md for why the retry exists at all.
"""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from spotlights_engine.cli import _build_argparser, _build_config
from spotlights_engine.model_config import MODELS_ENV_VAR
from spotlights_engine.spotlights_manager.persistence import build_config_fingerprint


@pytest.fixture
def no_models_file(tmp_path, monkeypatch):
    empty = tmp_path / "empty.yaml"
    empty.write_text("", encoding="utf-8")
    monkeypatch.setenv(MODELS_ENV_VAR, str(empty))


def _config(argv: list[str]):
    return _build_config(_build_argparser().parse_args(argv))


def _agent_hash(cfg) -> str:
    return build_config_fingerprint(
        module_filter=None,
        extractor_cfg=cfg.extractor,
        discovery_cfg=cfg.discovery,
        deep_research_cfg=cfg.deep_research,
        proposal_from_finding_cfg=cfg.proposal_from_finding,
        agent_proposals_cfg=cfg.agent_proposals,
    )["agent_proposals_hash"]


def test_no_flags_leaves_step_five_config_absent(no_models_file):
    """Unset flags must not conjure an `AgentProposalsConfig` into existence."""
    assert _config(["--repo", "."]).agent_proposals is None


def test_flags_reach_the_policy(no_models_file):
    cfg = _config(
        [
            "--repo",
            ".",
            "--agent-retry-attempts",
            "4",
            "--agent-retry-base-s",
            "10",
            "--agent-retry-max-s",
            "90",
        ]
    )

    assert cfg.agent_proposals is not None
    policy = cfg.agent_proposals.retry_policy
    assert (policy.attempts, policy.base_s, policy.max_s) == (4, 10.0, 90.0)
    assert policy.enabled


def test_retry_off_hashes_exactly_as_before_the_feature(no_models_file):
    """A default run and an explicitly-off run must both resume old run dirs."""
    baseline = _agent_hash(_config(["--repo", "."]))
    explicitly_off = _agent_hash(
        _config(["--repo", ".", "--agent-retry-attempts", "1"])
    )

    assert explicitly_off == baseline
    # Setting the delays without enabling the retry changes nothing either: they
    # are unreachable while `attempts == 1`, so they must not invalidate a resume.
    assert (
        _agent_hash(
            _config(["--repo", ".", "--agent-retry-base-s", "7", "--agent-retry-max-s", "9"])
        )
        == baseline
    )


def test_turning_the_retry_on_does_invalidate_resume(no_models_file):
    """The opposite guarantee: retrying changes how a call is made, so the

    fingerprint has to move. A resumed run must not silently inherit a policy
    the earlier half of it never ran under.
    """
    baseline = _agent_hash(_config(["--repo", "."]))

    assert _agent_hash(_config(["--repo", ".", "--agent-retry-attempts", "2"])) != baseline


def test_an_incoherent_attempt_count_is_rejected_at_the_boundary(no_models_file):
    with pytest.raises(ValidationError):
        _config(["--repo", ".", "--agent-retry-attempts", "0"])
