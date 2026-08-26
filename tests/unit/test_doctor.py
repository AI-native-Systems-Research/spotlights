from spotlights_engine import doctor
from spotlights_engine.doctor import ProbeOutcome

# A rate table with both real key shapes: a model-keyed Claude row and the
# codex CLI-family fallback row (codex reports no resolvable model id).
RATES = {
    "anthropic:aws/claude-opus-4-8": object(),
    "openai:codex": object(),
}


def test_probe_cli_missing_on_path(monkeypatch):
    monkeypatch.setattr(doctor.shutil, "which", lambda _n: None)
    res = doctor.probe_cli("claude", rates=RATES)
    assert res.ok is False
    assert "not found" in res.detail.lower()


def test_probe_cli_process_fails_is_unauthenticated(monkeypatch):
    monkeypatch.setattr(doctor.shutil, "which", lambda _n: "/bin/" + _n)
    monkeypatch.setattr(
        doctor,
        "_probe_model",
        lambda name, **_kw: ProbeOutcome(ok=False, model=None, error="exit 1 — not logged in"),
    )
    res = doctor.probe_cli("claude", rates=RATES)
    assert res.ok is False
    assert "not logged in" in res.detail


def test_probe_cli_priced_model_ok(monkeypatch):
    monkeypatch.setattr(doctor.shutil, "which", lambda _n: "/bin/" + _n)
    monkeypatch.setattr(
        doctor,
        "_probe_model",
        lambda name, **_kw: ProbeOutcome(ok=True, model="aws/claude-opus-4-8", error=""),
    )
    res = doctor.probe_cli("claude", rates=RATES)
    assert res.ok is True
    assert "anthropic:aws/claude-opus-4-8" in res.detail


def test_probe_cli_unpriced_model_fails(monkeypatch):
    monkeypatch.setattr(doctor.shutil, "which", lambda _n: "/bin/" + _n)
    monkeypatch.setattr(
        doctor,
        "_probe_model",
        lambda name, **_kw: ProbeOutcome(ok=True, model="aws/claude-opus-4-7", error=""),
    )
    res = doctor.probe_cli("claude", rates=RATES)
    assert res.ok is False
    assert "aws/claude-opus-4-7" in res.detail
    assert doctor.RATES_ENV_VAR in res.detail
    assert "docs/cost-and-manifest.md" in res.detail


def test_probe_cli_codex_family_fallback_ok(monkeypatch):
    """Codex reports no model id → rate key falls back to the CLI family."""
    monkeypatch.setattr(doctor.shutil, "which", lambda _n: "/bin/" + _n)
    monkeypatch.setattr(
        doctor,
        "_probe_model",
        lambda name, **_kw: ProbeOutcome(ok=True, model=None, error=""),
    )
    res = doctor.probe_cli("codex", rates=RATES)
    assert res.ok is True
    assert "openai:codex" in res.detail


def test_probe_cli_strips_context_window_tag(monkeypatch):
    monkeypatch.setattr(doctor.shutil, "which", lambda _n: "/bin/" + _n)
    monkeypatch.setattr(
        doctor,
        "_probe_model",
        lambda name, **_kw: ProbeOutcome(ok=True, model="aws/claude-opus-4-8[1m]", error=""),
    )
    res = doctor.probe_cli("claude", rates=RATES)
    assert res.ok is True


def test_run_checks_returns_all(monkeypatch):
    monkeypatch.setattr(doctor.shutil, "which", lambda _n: "/bin/" + _n)
    monkeypatch.setattr(
        doctor,
        "_probe_model",
        lambda name, **_kw: ProbeOutcome(ok=True, model=None, error=""),
    )
    results = doctor.run_checks()
    names = {r.name for r in results}
    assert {"claude", "codex", "rates"} <= names


def test_main_exit_code_fail(monkeypatch, capsys):
    monkeypatch.setattr(doctor.shutil, "which", lambda _n: None)
    code = doctor.main([])
    out = capsys.readouterr().out
    assert code == 1
    assert "claude" in out


def test_main_exit_code_ok(monkeypatch):
    monkeypatch.setattr(doctor.shutil, "which", lambda _n: "/bin/" + _n)

    def fake_probe(name, **_kw):
        model = "aws/claude-opus-4-8" if name == "claude" else None
        return ProbeOutcome(ok=True, model=model, error="")

    monkeypatch.setattr(doctor, "_probe_model", fake_probe)
    # Uses the bundled rate table, which has both required keys.
    code = doctor.main([])
    assert code == 0


def test_run_checks_includes_the_models_check(monkeypatch):
    monkeypatch.setattr(doctor.shutil, "which", lambda _n: "/bin/" + _n)
    monkeypatch.setattr(
        doctor,
        "_probe_model",
        lambda name, **_kw: ProbeOutcome(ok=True, model=None, error=""),
    )
    names = {r.name for r in doctor.run_checks()}
    assert "models" in names


def test_probe_uses_the_configured_model_not_the_cli_default(monkeypatch, tmp_path):
    """The probe must exercise what a run would ask for.

    Without this, doctor reports the CLI's own default — so it can fail on an
    unpriced model that no run would ever request, or pass on one the run will
    never use.
    """
    models = tmp_path / "m.yaml"
    models.write_text("claude: aws/claude-opus-4-8\ncodex: gpt-5.5\n", encoding="utf-8")
    monkeypatch.setenv("SPOTLIGHTS_MODELS_FILE", str(models))
    monkeypatch.setattr(doctor.shutil, "which", lambda _n: "/bin/" + _n)

    seen: dict[str, str | None] = {}

    def fake_probe(name, *, cwd=None, model=None):
        seen[name] = model
        return ProbeOutcome(ok=True, model=None, error="")

    monkeypatch.setattr(doctor, "_probe_model", fake_probe)
    doctor.run_checks()

    assert seen == {"claude": "aws/claude-opus-4-8", "codex": "gpt-5.5"}


def test_models_check_names_the_env_file_as_its_source(monkeypatch, tmp_path):
    models = tmp_path / "m.yaml"
    models.write_text("claude: pinned\n", encoding="utf-8")
    monkeypatch.setenv("SPOTLIGHTS_MODELS_FILE", str(models))

    res = doctor.check_models()
    assert res.ok is True
    assert str(models) in res.detail
    assert "claude=pinned" in res.detail
    # A blank value reads as inherit, not as an empty string.
    assert "codex=CLI default" in res.detail


def test_models_check_ok_when_the_file_is_absent(monkeypatch, tmp_path):
    """The file is optional, so its absence is not a failed check."""
    monkeypatch.setenv("SPOTLIGHTS_MODELS_FILE", str(tmp_path / "gone.yaml"))

    res = doctor.check_models()
    assert res.ok is True
    assert "own default" in res.detail


def test_models_check_fails_on_a_malformed_file(monkeypatch, tmp_path):
    bad = tmp_path / "bad.yaml"
    bad.write_text("- not\n- a mapping\n", encoding="utf-8")
    monkeypatch.setenv("SPOTLIGHTS_MODELS_FILE", str(bad))

    res = doctor.check_models()
    assert res.ok is False
    assert str(bad) in res.detail
