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
    """
    if path is not None:
        return path
    env_path = os.environ.get(MODELS_ENV_VAR)
    return Path(env_path) if env_path else _BUNDLED_MODELS_PATH


def load_model_config(path: Path | None = None) -> ModelConfig:
    """Load the global model selection.

    Precedence: explicit `path` > `SPOTLIGHTS_MODELS_FILE` env var > the bundled
    `models.yaml`. A resolved path that does not exist, or a file that parses to
    nothing, yields an all-inherit config rather than an error — the file is a
    convenience, not a requirement.

    Note this is only the *file* layer. `--claude-model` / `--codex-model`
    override whatever this returns, and are applied by the caller.
    """
    resolved = models_path(path)
    if not resolved.is_file():
        return ModelConfig()
    payload = yaml.safe_load(resolved.read_text(encoding="utf-8"))
    if payload is None:
        return ModelConfig()
    if not isinstance(payload, dict):
        raise ValueError(
            f"{resolved} must contain a YAML mapping of cli -> model id, "
            f"got {type(payload).__name__}"
        )
    return ModelConfig.model_validate(payload)


__all__ = ["MODELS_ENV_VAR", "ModelConfig", "load_model_config", "models_path"]
