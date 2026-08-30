"""Unit tests for `spotlights_engine.model_config`."""

from __future__ import annotations

from pathlib import Path

import pytest

from spotlights_engine.model_config import (
    MODELS_ENV_VAR,
    ModelConfig,
    load_model_config,
    models_path,
)


def _write(path: Path, text: str) -> Path:
    path.write_text(text, encoding="utf-8")
    return path


def test_explicit_path_beats_env_var(tmp_path, monkeypatch):
    explicit = _write(tmp_path / "explicit.yaml", "claude: from-explicit\n")
    env = _write(tmp_path / "env.yaml", "claude: from-env\n")
    monkeypatch.setenv(MODELS_ENV_VAR, str(env))

    assert load_model_config(explicit).claude == "from-explicit"


def test_env_var_beats_bundled(tmp_path, monkeypatch):
    env = _write(tmp_path / "env.yaml", "claude: from-env\ncodex: also-env\n")
    monkeypatch.setenv(MODELS_ENV_VAR, str(env))

    cfg = load_model_config()
    assert (cfg.claude, cfg.codex) == ("from-env", "also-env")


def test_bundled_file_ships_blank_so_behaviour_is_unchanged(monkeypatch):
    """The shipped file pins nothing.

    Blank on both keys is what makes this feature a pure addition: no step
    changes model, and the resume `config_fingerprint` of every existing run dir
    still matches. Pinning `codex` here would silently move steps 3 and 5 off
    `~/.codex/config.toml` and invalidate those run dirs.
    """
    monkeypatch.delenv(MODELS_ENV_VAR, raising=False)

    cfg = load_model_config()
    assert cfg == ModelConfig(claude=None, codex=None)


def test_empty_file_is_all_inherit(tmp_path):
    """`yaml.safe_load` returns None for an empty document."""
    assert load_model_config(_write(tmp_path / "e.yaml", "")) == ModelConfig()


def test_comments_only_file_is_all_inherit(tmp_path):
    path = _write(tmp_path / "c.yaml", "# just a comment\n# and another\n")
    assert load_model_config(path) == ModelConfig()


@pytest.mark.parametrize(
    "body",
    [
        "claude:\ncodex:\n",  # key present, no value
        'claude: ""\ncodex: ""\n',  # explicit empty string
        "claude: null\ncodex: null\n",  # explicit null
        "claude: '   '\ncodex: '\t'\n",  # whitespace only
        "{}\n",  # empty mapping, keys absent
    ],
)
def test_blank_forms_all_mean_inherit(tmp_path, body):
    """Every way a user might leave a value blank resolves to None.

    Callers only ever test truthiness, so `""` must never reach an argv builder
    and produce `--model ""`.
    """
    assert load_model_config(_write(tmp_path / "b.yaml", body)) == ModelConfig()


def test_values_are_stripped(tmp_path):
    path = _write(tmp_path / "s.yaml", 'claude: "  padded  "\n')
    assert load_model_config(path).claude == "padded"


def test_unknown_key_is_rejected(tmp_path):
    """`extra="forbid"` turns a typo into an error instead of a silent no-op."""
    path = _write(tmp_path / "t.yaml", "cluade: oops\n")
    with pytest.raises(Exception, match="cluade|extra"):
        load_model_config(path)


def test_non_mapping_document_is_rejected(tmp_path):
    path = _write(tmp_path / "l.yaml", "- claude\n- codex\n")
    with pytest.raises(ValueError, match="must contain a YAML mapping"):
        load_model_config(path)


def test_models_path_reports_the_file_without_reading_it(tmp_path, monkeypatch):
    """`doctor` needs to name the source even when the file is absent."""
    missing = tmp_path / "gone.yaml"
    monkeypatch.setenv(MODELS_ENV_VAR, str(missing))
    assert models_path() == missing

    explicit = tmp_path / "given.yaml"
    assert models_path(explicit) == explicit


def test_malformed_yaml_raises_value_error_not_yaml_error(tmp_path):
    """`yaml.YAMLError` is not a `ValueError`, so the loader must translate it.

    Every caller guards config loading with `(OSError, ValueError)`. A raw
    YAMLError would escape all of them and surface as a traceback.
    """
    path = _write(tmp_path / "bad.yaml", "claude: [unclosed\n")
    with pytest.raises(ValueError, match="not valid YAML"):
        load_model_config(path)


def test_error_messages_name_the_offending_file(tmp_path):
    """Both failure modes must say *which* file is wrong."""
    bad_yaml = _write(tmp_path / "a.yaml", "claude: [unclosed\n")
    with pytest.raises(ValueError, match=str(bad_yaml)):
        load_model_config(bad_yaml)

    bad_key = _write(tmp_path / "b.yaml", "cluade: x\n")
    with pytest.raises(ValueError, match=str(bad_key)):
        load_model_config(bad_key)


def test_env_var_path_is_tilde_expanded(monkeypatch, tmp_path):
    """The env var is typed by hand, so `~` has to work."""
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv(MODELS_ENV_VAR, "~/mine.yaml")
    assert models_path() == tmp_path / "mine.yaml"


def test_typod_env_var_path_is_an_error_not_a_silent_inherit(monkeypatch, tmp_path):
    """A path someone named and got wrong must not quietly mean "inherit".

    An absent bundled file is a missing convenience; an absent file the env var
    explicitly points at is a typo, and running on CLI defaults while the
    operator believes their pin is live is the failure mode worth being loud
    about. `doctor` already fails on it — the run path must agree.
    """
    monkeypatch.setenv(MODELS_ENV_VAR, str(tmp_path / "typo.yaml"))
    with pytest.raises(ValueError, match="does not point to a readable file"):
        load_model_config()


def test_explicit_missing_path_still_returns_inherit(tmp_path):
    """An explicit `path=` argument keeps the lenient behaviour.

    Callers passing a path have already decided it is optional; only the env var
    carries the "somebody typed this" signal.
    """
    assert load_model_config(tmp_path / "absent.yaml") == ModelConfig()


def test_absent_bundled_file_is_not_an_error(monkeypatch, tmp_path):
    import spotlights_engine.model_config as mc

    monkeypatch.delenv(MODELS_ENV_VAR, raising=False)
    monkeypatch.setattr(mc, "_BUNDLED_MODELS_PATH", tmp_path / "gone.yaml")
    assert load_model_config() == ModelConfig()
