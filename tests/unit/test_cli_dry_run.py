from spotlights_engine import cli


def test_dry_run_prints_scope_and_exits_zero(capsys, tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir()
    code = cli.main(
        [
            "--repo",
            str(repo),
            "--include",
            "v1/kv_offload",
            "--objective",
            "reduce latency",
            "--output-folder",
            str(tmp_path / "out"),
            "--artifacts-dir",
            str(tmp_path / "art"),
            "--dry-run",
        ]
    )
    out = capsys.readouterr().out
    assert code == 0
    assert "dry-run" in out.lower()
    assert "v1/kv_offload" in out
    assert "reduce latency" in out


def test_max_cost_must_be_positive(tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir()
    try:
        cli.main(
            [
                "--repo",
                str(repo),
                "--max-cost",
                "0",
                "--dry-run",
                "--output-folder",
                str(tmp_path / "out"),
                "--artifacts-dir",
                str(tmp_path / "art"),
            ]
        )
    except SystemExit as exc:
        assert exc.code == 2  # argparse error exit
    else:
        raise AssertionError("expected SystemExit for non-positive --max-cost")
