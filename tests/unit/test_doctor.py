from pathlib import Path

from spotlights_engine import doctor


def test_check_cli_missing(monkeypatch):
    monkeypatch.setattr(doctor.shutil, "which", lambda _n: None)
    res = doctor.check_cli("claude")
    assert res.name == "claude"
    assert res.ok is False
    assert "not found" in res.detail.lower()


def test_check_cli_present(monkeypatch):
    monkeypatch.setattr(doctor.shutil, "which", lambda _n: "/usr/local/bin/" + _n)
    res = doctor.check_cli("codex")
    assert res.ok is True
    assert "/usr/local/bin/codex" in res.detail


def test_check_repo(tmp_path):
    assert doctor.check_repo(tmp_path).ok is True
    assert doctor.check_repo(tmp_path / "nope").ok is False


def test_run_checks_returns_all(tmp_path):
    results = doctor.run_checks(tmp_path)
    names = {r.name for r in results}
    assert {"claude", "codex", "repo", "rates"} <= names


def test_main_exit_code_fail(monkeypatch, tmp_path, capsys):
    monkeypatch.setattr(doctor.shutil, "which", lambda _n: None)
    code = doctor.main(["--repo", str(tmp_path)])
    out = capsys.readouterr().out
    assert code == 1
    assert "claude" in out


def test_main_exit_code_ok(monkeypatch, tmp_path):
    monkeypatch.setattr(doctor.shutil, "which", lambda _n: "/bin/" + _n)
    code = doctor.main(["--repo", str(tmp_path)])
    assert code == 0
