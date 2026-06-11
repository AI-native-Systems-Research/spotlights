"""Configuration objects for local CLI-backed agents and optional proxies."""

from __future__ import annotations

import os
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

IBM_LITELLM_PROXY_URL = "https://ete-litellm.ai-models.vpc-int.res.ibm.com"
DROP_EXACT = frozenset(
    {
        "OPENAI_BASE_URL",
        "OPENAI_API_BASE",
        "ANTHROPIC_BASE_URL",
        "GOOGLE_GEMINI_BASE_URL",
        "HTTP_PROXY",
        "HTTPS_PROXY",
        "ALL_PROXY",
        "VIRTUAL_ENV",
    }
)

CODEX_IBM_MODELS = {
    "gpt4o": "Azure/gpt-4o",
    "gpt41mini": "Azure/gpt-4.1-mini",
    "gpt41nano": "Azure/gpt-4.1-nano",
    "gpt5mini": "Azure/gpt-5-mini-2025-08-07",
    "gpt52": "Azure/gpt-5.2-2025-12-11",
    "gpt53codex": "azure/gpt-5.3-codex",
    "gpt53chat": "azure/gpt-5.3-chat",
    "gpt54": "azure/gpt-5.4",
    "gpt55": "azure/gpt-5.5",
    "gpt54pro": "azure/gpt-5.4-pro",
}

GEMINI_IBM_MODELS = {
    "gemini25flash": "gcp/gemini-2.5-flash",
    "gemini25pro": "gcp/gemini-2.5-pro",
    "gemini25flashlite": "GCP/gemini-2.5-flash-lite",
    "gemini3propreview": "gcp/gemini-3-pro-preview",
    "gemini31propreview": "gcp/gemini-3.1-pro-preview",
    "gemini3flashpreview": "gcp/gemini-3-flash-preview",
    "gemini31flashlitepreview": "gcp/gemini-3.1-flash-lite-preview",
    "gemini35flash": "gcp/gemini-3.5-flash",
}

CLAUDE_IBM_MODELS = {
    # Claude Code's Anthropic-compatible route expects Claude model IDs, not
    # Codex/OpenAI public model-group names like `aws/claude-opus-4-7`.
    "opus48": "claude-opus-4-8",
    "opus47": "claude-opus-4-7",
    "aws_opus48": "aws/claude-opus-4-8",
    "aws_opus47": "aws/claude-opus-4-7",
    "opus47us": "aws/us.claude-opus-4-7",
    "opus46": "claude-opus-4-6",
    "opus45": "claude-opus-4-5",
    "haiku45": "claude-haiku-4-5",
    "sonnet45": "claude-sonnet-4-5",
    "sonnet46": "claude-sonnet-4-6",
}


@dataclass(frozen=True)
class LiteLlmProxyConfig:
    """Optional proxy metadata. This is not an agent by itself."""

    base_url: str = IBM_LITELLM_PROXY_URL
    api_key_env: str = "LITELLM_API_KEY"


@dataclass(frozen=True)
class CodexCliConfig:
    bin: str = "codex"
    profile: str | None = None
    model: str | None = None
    approval: str = "never"
    sandbox_read: str = "read-only"
    sandbox_write: str = "workspace-write"
    json_events: bool = True
    output_last_message: Path | None = None
    keep_env: frozenset[str] = frozenset({"LITELLM_API_KEY"})

    @classmethod
    def ibm_litellm(cls, *, profile: str = "gpt53codex") -> CodexCliConfig:
        if profile not in CODEX_IBM_MODELS:
            raise ValueError(f"not a Codex IBM LiteLLM profile: {profile}")
        return cls(profile=profile)


@dataclass(frozen=True)
class ClaudeCliConfig:
    bin: str = "claude"
    model: str | None = None
    permission_read: str = "plan"
    permission_network: str = "acceptEdits"
    permission_write: str = "acceptEdits"
    network_allowed_tools: tuple[str, ...] = ("Bash",)
    anthropic_base_url: str | None = None
    anthropic_auth_token_env: str | None = None
    anthropic_api_key_env: str | None = None
    disable_experimental_betas: bool = False
    unset_env: frozenset[str] = frozenset()
    keep_env: frozenset[str] = frozenset(
        {
            "ANTHROPIC_BASE_URL",
            "ANTHROPIC_AUTH_TOKEN",
            "ANTHROPIC_API_KEY",
            "CLAUDE_CODE_DISABLE_EXPERIMENTAL_BETAS",
        }
    )

    @classmethod
    def ibm_litellm(
        cls,
        *,
        model: str = CLAUDE_IBM_MODELS["opus47"],
        proxy: LiteLlmProxyConfig | None = None,
    ) -> ClaudeCliConfig:
        proxy = proxy or LiteLlmProxyConfig()
        return cls(
            model=model,
            anthropic_base_url=proxy.base_url,
            anthropic_auth_token_env=proxy.api_key_env,
            disable_experimental_betas=True,
            unset_env=frozenset({"ANTHROPIC_API_KEY"}),
        )


@dataclass(frozen=True)
class GeminiCliConfig:
    bin: str = "gemini"
    model: str | None = None
    approval_read: str = "plan"
    approval_network: str = "yolo"
    approval_write: str = "auto_edit"
    gemini_base_url: str | None = None
    gemini_api_key_env: str | None = None
    api_key_auth_mechanism: Literal["x-goog-api-key", "bearer"] = "bearer"
    skip_trust: bool = True
    settings_path: Path | None = None
    keep_env: frozenset[str] = frozenset(
        {
            "GEMINI_API_KEY",
            "GOOGLE_GEMINI_BASE_URL",
            "GEMINI_API_KEY_AUTH_MECHANISM",
            "GEMINI_CLI_SYSTEM_SETTINGS_PATH",
        }
    )

    @classmethod
    def ibm_litellm(
        cls,
        *,
        model: str = GEMINI_IBM_MODELS["gemini31propreview"],
        proxy: LiteLlmProxyConfig | None = None,
    ) -> GeminiCliConfig:
        """Return IBM LiteLLM gateway env mapping for Gemini CLI.

        Gemini CLI expects the proxy root URL, not the OpenAI-compatible `/v1`
        URL. For IBM this means `https://ete-litellm.ai-models.vpc-int.res.ibm.com`.
        """
        proxy = proxy or LiteLlmProxyConfig()
        return cls(
            model=model,
            gemini_base_url=proxy.base_url,
            gemini_api_key_env=proxy.api_key_env,
            api_key_auth_mechanism="bearer",
        )

    def with_settings_path(self, settings_path: Path | None) -> GeminiCliConfig:
        return GeminiCliConfig(
            bin=self.bin,
            model=self.model,
            approval_read=self.approval_read,
            approval_network=self.approval_network,
            approval_write=self.approval_write,
            gemini_base_url=self.gemini_base_url,
            gemini_api_key_env=self.gemini_api_key_env,
            api_key_auth_mechanism=self.api_key_auth_mechanism,
            skip_trust=self.skip_trust,
            settings_path=settings_path,
            keep_env=self.keep_env,
        )


def base_env(
    *, keep: set[str] | frozenset[str], inject: Mapping[str, str] | None = None
) -> dict[str, str]:
    kept = set(keep)
    env = dict(os.environ)
    for key in list(env):
        if key in kept:
            continue
        if key in DROP_EXACT:
            env.pop(key, None)
    if inject:
        env.update(dict(inject))
    return env


@dataclass(frozen=True)
class CliAgentsConfig:
    """Top-level configuration for the three CLI adapters."""

    codex: CodexCliConfig
    claude: ClaudeCliConfig
    gemini: GeminiCliConfig

    @classmethod
    def ibm_litellm_from_env(
        cls,
        *,
        gemini_settings_path: Path | None = None,
    ) -> CliAgentsConfig:
        """Build the default IBM LiteLLM-backed configuration from environment variables."""
        proxy = LiteLlmProxyConfig(
            base_url=os.getenv("CLI_AGENTS_LITELLM_BASE_URL", IBM_LITELLM_PROXY_URL),
            api_key_env=os.getenv("CLI_AGENTS_LITELLM_KEY_ENV", "LITELLM_API_KEY"),
        )
        return cls(
            codex=CodexCliConfig.ibm_litellm(
                profile=os.getenv("CLI_AGENTS_CODEX_PROFILE", "gpt55"),
            ),
            claude=ClaudeCliConfig.ibm_litellm(
                model=os.getenv("CLI_AGENTS_CLAUDE_MODEL", CLAUDE_IBM_MODELS["opus47"]),
                proxy=proxy,
            ),
            gemini=GeminiCliConfig.ibm_litellm(
                model=os.getenv("CLI_AGENTS_GEMINI_MODEL", GEMINI_IBM_MODELS["gemini31propreview"]),
                proxy=LiteLlmProxyConfig(
                    base_url=os.getenv("CLI_AGENTS_GEMINI_BASE_URL", proxy.base_url),
                    api_key_env=os.getenv("CLI_AGENTS_GEMINI_KEY_ENV", proxy.api_key_env),
                ),
            ).with_settings_path(gemini_settings_path),
        )
