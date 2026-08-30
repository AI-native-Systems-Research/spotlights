"""Global model selection for the agent CLIs the engine drives.

One model id per CLI, applied to every step of the pipeline. The engine passes
the resolved id as `--model` to `claude` / `codex`; an unset id means the engine
passes nothing and the CLI falls back to its own configuration
(`~/.claude/settings.json`, `~/.codex/config.toml`).

Loading mirrors `costing.rates`: an explicit path beats the
`SPOTLIGHTS_MODELS_FILE` env var, which beats the bundled `models.yaml`.

The file is optional at every layer — a missing file, a missing key, a blank
value, or an explicit `null` all mean "inherit". Nothing here raises on absence;
only malformed YAML or a wrong value type does.
"""

from __future__ import annotations

import os
from pathlib import Path

import yaml
from pydantic import BaseModel, ConfigDict, field_validator
from pydantic import ValidationError as PydanticValidationError

MODELS_ENV_VAR = "SPOTLIGHTS_MODELS_FILE"

_BUNDLED_MODELS_PATH = Path(__file__).parent / "models.yaml"


class ModelConfig(BaseModel):
    """Resolved model id per agent CLI. `None` means "inherit the CLI default".

    Blank strings normalize to `None` so callers only ever test truthiness —
    a YAML `claude:` with no value and a `claude: ""` are the same request.
    """

    model_config = ConfigDict(extra="forbid")

    claude: str | None = None
    codex: str | None = None

    @field_validator("claude", "codex", mode="before")
    @classmethod
    def _blank_is_inherit(cls, value: object) -> object:
        if isinstance(value, str) and not value.strip():
            return None
        return value.strip() if isinstance(value, str) else value


def models_path(path: Path | None = None) -> Path:
    """The file `load_model_config` would read, without reading it.

    Exposed so `doctor` can report *where* the effective models came from.
    `~` is expanded, since the env var is something a person types by hand.
    """
    if path is not None:
        return path.expanduser()
    env_path = os.environ.get(MODELS_ENV_VAR)
    return Path(env_path).expanduser() if env_path else _BUNDLED_MODELS_PATH


def load_model_config(path: Path | None = None) -> ModelConfig:
    """Load the global model selection.

    Precedence: explicit `path` > `SPOTLIGHTS_MODELS_FILE` env var > the bundled
    `models.yaml`. A missing bundled file, or a file that parses to nothing,
    yields an all-inherit config rather than an error — the file is a
    convenience, not a requirement. A missing file that `SPOTLIGHTS_MODELS_FILE`
    explicitly names *is* an error: that is a typo, not an absent convenience.

    Note this is only the *file* layer. `--claude-model` / `--codex-model`
    override whatever this returns, and are applied by the caller.
    """
    resolved = models_path(path)
    if not resolved.is_file():
        # An absent *bundled* file is fine — the file is optional. But a path
        # somebody explicitly named and got wrong is not: silently running on CLI
        # defaults while they believe their pin is live is the worst outcome
        # available, and `doctor` already fails on it. Keep the run path honest
        # too, rather than having the two disagree.
        if path is None and os.environ.get(MODELS_ENV_VAR):
            raise ValueError(
                f"{MODELS_ENV_VAR}={resolved} does not point to a readable file"
            )
        return ModelConfig()
    try:
        payload = yaml.safe_load(resolved.read_text(encoding="utf-8"))
    except yaml.YAMLError as exc:
        # `yaml.YAMLError` is NOT a `ValueError` (unlike `json`'s
        # `JSONDecodeError`, which is). Callers guard config loading with
        # `(OSError, ValueError)`, so a raw YAMLError would escape every one of
        # them and surface as a traceback. Translate at the boundary.
        raise ValueError(f"{resolved} is not valid YAML: {exc}") from exc
    if payload is None:
        return ModelConfig()
    if not isinstance(payload, dict):
        raise ValueError(
            f"{resolved} must contain a YAML mapping of cli -> model id, "
            f"got {type(payload).__name__}"
        )
    try:
        return ModelConfig.model_validate(payload)
    except PydanticValidationError as exc:
        # Pydantic's error *is* a ValueError, so this is not about the type —
        # it is to name the offending file, which the bare error does not.
        raise ValueError(f"{resolved}: {exc}") from exc


__all__ = ["MODELS_ENV_VAR", "ModelConfig", "load_model_config", "models_path"]
