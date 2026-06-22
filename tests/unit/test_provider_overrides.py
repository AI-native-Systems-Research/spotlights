from __future__ import annotations

from pathlib import Path

from spotlights_engine import cli
from spotlights_engine.claude_env import build_claude_env, claude_model_args
from spotlights_engine.module_deep_research.claude_exec import ClaudeExecClient, ClaudeExecOptions


def test_claude_env_allows_explicit_gateway_without_inherited_leaks() -> None:
    env = build_claude_env(
        base_env={
            "ANTHROPIC_API_KEY": "official-key-that-should-not-win",
            "ANTHROPIC_BASE_URL": "https://stale.example.com",
            "ANTHROPIC_AUTH_TOKEN": "stale-token",
            "LITELLM_API_KEY": "gateway-token",
            "SPOTLIGHTS_CLAUDE_BASE_URL": "https://proxy.example.com",
            "SPOTLIGHTS_CLAUDE_AUTH_TOKEN_ENV": "LITELLM_API_KEY",
            "SPOTLIGHTS_CLAUDE_DISABLE_EXPERIMENTAL_BETAS": "1",
            "SPOTLIGHTS_CLAUDE_UNSET_ENV": "SHOULD_DROP",
            "SPOTLIGHTS_CLAUDE_MODEL": "claude-opus-x",
            "SHOULD_DROP": "yes",
        }
    )

    assert env["ANTHROPIC_BASE_URL"] == "https://proxy.example.com"
    assert env["ANTHROPIC_AUTH_TOKEN"] == "gateway-token"
    assert env["CLAUDE_CODE_DISABLE_EXPERIMENTAL_BETAS"] == "1"
    assert "ANTHROPIC_API_KEY" not in env
    assert "SHOULD_DROP" not in env
    assert "SPOTLIGHTS_CLAUDE_BASE_URL" not in env


def test_claude_model_env_is_used_when_no_explicit_model(monkeypatch) -> None:
    monkeypatch.setenv("SPOTLIGHTS_CLAUDE_MODEL", "claude-opus-x")

    assert claude_model_args() == ["--model", "claude-opus-x"]
    cmd = ClaudeExecClient(ClaudeExecOptions(claude_bin="claude")).build_command()

    assert "--model" in cmd
    assert cmd[cmd.index("--model") + 1] == "claude-opus-x"


def test_cli_provider_flags_build_generic_runtime_config(tmp_path: Path) -> None:
    args = cli._build_argparser().parse_args(
        [
            "--repo",
            str(tmp_path / "target"),
            "--artifacts-dir",
            str(tmp_path / "artifacts"),
            "--output-folder",
            str(tmp_path / "out"),
            "--codex-profile",
            "gpt55",
            "--antigravity-base-url",
            "https://proxy.example.com/",
            "--antigravity-model",
            "gcp/gemini-flash",
            "--antigravity-api-key-env",
            "LITELLM_API_KEY",
            "--claude-model",
            "claude-opus-x",
            "--claude-base-url",
            "https://claude-proxy.example.com",
            "--claude-auth-token-env",
            "LITELLM_API_KEY",
            "--claude-unset-env",
            "ANTHROPIC_API_KEY,FOO",
            "--claude-disable-experimental-betas",
        ]
    )

    cfg = cli._build_config(args)

    assert cfg.discovery is not None
    assert cfg.discovery.codex_profile == "gpt55"
    assert cfg.discovery.codex_model is None
    assert cfg.deep_research is not None
    assert cfg.deep_research.profile == "gpt55"
    assert cfg.agent_proposals is not None
    assert cfg.agent_proposals.codex_profile == "gpt55"
    assert cfg.deep_research_antigravity is not None
    assert cfg.deep_research_antigravity.antigravity_base_url == "https://proxy.example.com"
    assert cfg.deep_research_antigravity.model == "gcp/gemini-flash"
    assert cfg.deep_research_antigravity.antigravity_api_key_env == "LITELLM_API_KEY"

    cli._apply_claude_env_directives(args)
    import os

    assert os.environ["SPOTLIGHTS_CLAUDE_MODEL"] == "claude-opus-x"
    assert os.environ["SPOTLIGHTS_CLAUDE_BASE_URL"] == "https://claude-proxy.example.com"
    assert os.environ["SPOTLIGHTS_CLAUDE_AUTH_TOKEN_ENV"] == "LITELLM_API_KEY"
    assert os.environ["SPOTLIGHTS_CLAUDE_UNSET_ENV"] == "ANTHROPIC_API_KEY,FOO"
    assert os.environ["SPOTLIGHTS_CLAUDE_DISABLE_EXPERIMENTAL_BETAS"] == "1"


def test_discovery_codex_runner_profile_does_not_force_model(tmp_path: Path, monkeypatch) -> None:
    import shutil

    from spotlights_engine.candidate_discovery.agents import CodexRunner
    from spotlights_engine.candidate_discovery.api import DiscoveryConfig

    monkeypatch.setattr(shutil, "which", lambda name: f"/fake/bin/{name}")

    repo = tmp_path / "repo"
    repo.mkdir()
    schema = tmp_path / "schema.json"
    schema.write_text("{}", encoding="utf-8")
    iter_dir = tmp_path / "iter"
    iter_dir.mkdir()

    runner = CodexRunner(
        DiscoveryConfig(
            repo_path=repo,
            artifacts_dir=tmp_path,
            codex_profile="gpt55",
            codex_model=None,
        )
    )

    argv = runner._build_argv(schema, iter_dir)

    assert argv[1:3] == ["--profile", "gpt55"]
    assert not any(arg.startswith('model="') for arg in argv)


def test_discovery_codex_runner_does_not_receive_claude_gateway_env(
    tmp_path: Path, monkeypatch
) -> None:
    import shutil

    from spotlights_engine.candidate_discovery.agents import CodexRunner
    from spotlights_engine.candidate_discovery.api import DiscoveryConfig

    monkeypatch.setattr(shutil, "which", lambda name: f"/fake/bin/{name}")
    monkeypatch.setenv("SPOTLIGHTS_CLAUDE_BASE_URL", "https://claude-proxy.example.com")
    monkeypatch.setenv("SPOTLIGHTS_CLAUDE_AUTH_TOKEN", "secret")
    monkeypatch.setenv("ANTHROPIC_BASE_URL", "https://stale.example.com")
    monkeypatch.setenv("ANTHROPIC_AUTH_TOKEN", "stale")
    repo = tmp_path / "repo"
    repo.mkdir()

    runner = CodexRunner(DiscoveryConfig(repo_path=repo, artifacts_dir=tmp_path))
    env = runner._build_env()

    assert "ANTHROPIC_BASE_URL" not in env
    assert "ANTHROPIC_AUTH_TOKEN" not in env
