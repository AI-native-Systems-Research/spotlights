"""`--claude-model` / `--codex-model` must reach every step's config.

The pipeline has five steps plus `apply`, each with its own config object. A new
step added without model plumbing would silently fall back to the CLI default
and nothing else would notice — so this file asserts the fan-out explicitly.
"""

from __future__ import annotations

import pytest

from spotlights_engine.cli import _build_argparser, _build_config
from spotlights_engine.model_config import MODELS_ENV_VAR

CLAUDE = "aws/claude-opus-4-8"
CODEX = "gpt-5.5-test"


@pytest.fixture
def no_models_file(tmp_path, monkeypatch):
    """Point the loader at a nonexistent file so only flags are in play."""
    monkeypatch.setenv(MODELS_ENV_VAR, str(tmp_path / "absent.yaml"))


def _config(argv: list[str]):
    return _build_config(_build_argparser().parse_args(argv))


def test_flags_reach_every_step_config(no_models_file):
    cfg = _config(["--claude-model", CLAUDE, "--codex-model", CODEX])

    # Claude runs in steps 1, 2, 4 and 5.
    assert cfg.extractor.claude_model == CLAUDE
    assert cfg.discovery is not None and cfg.discovery.claude_model == CLAUDE
    assert (
        cfg.proposal_from_finding is not None and cfg.proposal_from_finding.claude_model == CLAUDE
    )
    assert cfg.agent_proposals is not None
    assert cfg.agent_proposals.claude_model == CLAUDE

    # Codex runs in steps 2, 3 and 5.
    assert cfg.discovery.codex_model == CODEX
    assert cfg.deep_research is not None and cfg.deep_research.model == CODEX
    assert cfg.agent_proposals.codex_model == CODEX

    # Step 3's Claude runner has no config object of its own; it reads this.
    assert cfg.models is not None
    assert (cfg.models.claude, cfg.models.codex) == (CLAUDE, CODEX)


def test_no_flags_and_no_file_leaves_everything_inheriting(no_models_file):
    """The pre-existing behaviour: the engine passes no model anywhere."""
    cfg = _config([])

    assert cfg.extractor.claude_model is None
    assert cfg.models is not None
    assert (cfg.models.claude, cfg.models.codex) == (None, None)
    # Untouched step configs stay None rather than being conjured into existence.
    assert cfg.discovery is None
    assert cfg.proposal_from_finding is None
    assert cfg.agent_proposals is None
    assert cfg.deep_research is None


def test_flag_overrides_the_file(tmp_path, monkeypatch):
    (tmp_path / "m.yaml").write_text("claude: from-file\ncodex: also-from-file\n", encoding="utf-8")
    monkeypatch.setenv(MODELS_ENV_VAR, str(tmp_path / "m.yaml"))

    cfg = _config(["--claude-model", CLAUDE])
    assert cfg.models is not None
    # Flag wins for claude; codex falls through to the file.
    assert (cfg.models.claude, cfg.models.codex) == (CLAUDE, "also-from-file")


def test_file_alone_populates_the_step_configs(tmp_path, monkeypatch):
    (tmp_path / "m.yaml").write_text("claude: file-claude\n", encoding="utf-8")
    monkeypatch.setenv(MODELS_ENV_VAR, str(tmp_path / "m.yaml"))

    cfg = _config([])
    assert cfg.extractor.claude_model == "file-claude"
    assert cfg.discovery is not None and cfg.discovery.claude_model == "file-claude"


def test_review_iterations_still_works_alongside_a_model(no_models_file):
    """`--review-iterations` shares DiscoveryConfig with the model flags."""
    cfg = _config(["--no-review", "--claude-model", CLAUDE])
    assert cfg.discovery is not None
    assert cfg.discovery.num_review_iterations == 0
    assert cfg.discovery.claude_model == CLAUDE


def test_explicit_empty_flag_means_inherit_and_beats_the_file(tmp_path, monkeypatch):
    """`--codex-model ""` must be able to override a pinned file.

    The bundled file pins Codex, so without this there is no command-line way to
    ask for the CLI's own default.
    """
    (tmp_path / "m.yaml").write_text("codex: pinned-in-file\n", encoding="utf-8")
    monkeypatch.setenv(MODELS_ENV_VAR, str(tmp_path / "m.yaml"))

    assert _config([]).models.codex == "pinned-in-file"
    assert _config(["--codex-model", ""]).models.codex is None
    # Whitespace is the same request.
    assert _config(["--codex-model", "   "]).models.codex is None


def test_invalid_model_id_exits_rather_than_failing_deeper(no_models_file, capsys):
    """A `:` fails DiscoveryConfig's pattern; catch it at the flag instead."""
    with pytest.raises(SystemExit):
        _config(["--codex-model", "bad:id"])
    assert "not a valid model id" in capsys.readouterr().err


def test_context_window_tag_is_a_valid_model_id(no_models_file):
    """`[1m]` suffixes are real ids and must not be rejected."""
    cfg = _config(["--claude-model", "aws/claude-opus-4-8[1m]"])
    assert cfg.models.claude == "aws/claude-opus-4-8[1m]"


def test_malformed_models_file_is_a_value_error_not_a_traceback(tmp_path, monkeypatch):
    (tmp_path / "m.yaml").write_text("claude: [unclosed\n", encoding="utf-8")
    monkeypatch.setenv(MODELS_ENV_VAR, str(tmp_path / "m.yaml"))

    with pytest.raises(ValueError, match="not valid YAML"):
        _config([])


def test_default_config_fingerprint_is_unchanged_by_this_feature(no_models_file):
    """Adding model fields must not invalidate existing run dirs.

    `config_fingerprint` hashes each step config, so a new key — even one whose
    value is null — changes the hash and makes every pre-existing run dir fail
    `--resume` with `ResumeMismatchError`. These are the hashes a clean v0.1.0
    checkout produces for a default run; they are pinned here because the
    breakage is silent and only shows up on someone's half-finished run.
    """
    from spotlights_engine.spotlights_manager.persistence import (
        build_config_fingerprint,
    )

    cfg = _config(["--repo", "."])
    fp = build_config_fingerprint(
        module_filter=None,
        extractor_cfg=cfg.extractor,
        discovery_cfg=cfg.discovery,
        deep_research_cfg=cfg.deep_research,
        proposal_from_finding_cfg=cfg.proposal_from_finding,
        agent_proposals_cfg=cfg.agent_proposals,
    )

    assert fp["extractor_hash"] == (
        "02f082f810708f73dd4dc7331eeeb73223ab2fd13c343cef5ec616265089444d"
    )
    assert fp["discovery_hash"] == (
        "4e752640482669c0b18a2daba11e75735bae261de4277be45ba35de474722db2"
    )
    assert fp["deep_research_hash"] == (
        "92e3780386e13d0c3f284f7d220c0110064ee3ada277a05d1436e4cd9f26f97e"
    )
    assert fp["proposal_from_finding_hash"] == (
        "7da44bf7293a6aec9ba332e0bfd8f3bb1394cdb38ddbb240fe12e0f7eb9a4dd8"
    )
    assert fp["agent_proposals_hash"] == (
        "29092cf24868d69d4e6e6c90f17323f6bf853fadb2acb6b455dcdf07b8b4bce0"
    )


def test_setting_a_model_does_change_the_fingerprint(no_models_file):
    """The flip side: a pinned model is a real config change and must register.

    Otherwise a run could resume across a model switch and silently mix two
    models' output in one result.
    """
    from spotlights_engine.spotlights_manager.persistence import (
        hash_pydantic_excluding,
    )

    default = _config(["--repo", "."]).extractor
    pinned = _config(["--repo", ".", "--claude-model", CLAUDE]).extractor
    exclude = {"artifacts_dir"}
    assert hash_pydantic_excluding(default, exclude=exclude) != (
        hash_pydantic_excluding(pinned, exclude=exclude)
    )


def test_bad_model_id_in_the_file_names_the_file_not_the_flag(tmp_path, monkeypatch):
    """Argparse usage text would point the reader at the wrong thing."""
    bad = tmp_path / "m.yaml"
    bad.write_text("codex: bad:id\n", encoding="utf-8")
    monkeypatch.setenv(MODELS_ENV_VAR, str(bad))

    with pytest.raises(ValueError, match=str(bad)):
        _config([])


def test_codex_context_window_tag_survives_the_downstream_pattern(no_models_file):
    """`DiscoveryConfig.codex_model` carries its own regex; the two must agree.

    A CLI regex wider than the pydantic one means `--dry-run` passes and the real
    run dies.
    """
    cfg = _config(["--codex-model", "gpt-5.5[1m]"])
    assert cfg.discovery is not None
    assert cfg.discovery.codex_model == "gpt-5.5[1m]"
