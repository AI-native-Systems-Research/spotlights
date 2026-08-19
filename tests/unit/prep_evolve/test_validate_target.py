"""Live-target validation tests (plan §10.2)."""

from __future__ import annotations

from pathlib import Path

import pytest

from spotlights_engine.prep_evolve.errors import StalenessError
from spotlights_engine.prep_evolve.resolve import (
    load_result,
    resolve_candidate,
    resolve_candidates,
    resolve_module_run,
)
from spotlights_engine.prep_evolve.validate_target import (
    capture_revision,
    validate_candidate_target,
    validate_scope_file,
)

from . import _fixtures as fx


def _candidate(tmp_path: Path):
    loaded = load_result(fx.write_result(tmp_path))
    run = resolve_module_run(loaded, "v1/attention")
    candidates = resolve_candidates(run, "v1/attention")
    return resolve_candidate(candidates, "cand-v1_attention-0002")


def test_valid_candidate(tmp_path: Path) -> None:
    repo = fx.make_repo(tmp_path)
    cand = _candidate(tmp_path)
    validated = validate_candidate_target(repo, cand)
    assert validated.line_start == fx.CAND_START
    assert validated.line_end == fx.CAND_END
    assert len(validated.source_excerpt_sha256) == 64


def test_excerpt_hash_stable(tmp_path: Path) -> None:
    repo = fx.make_repo(tmp_path)
    cand = _candidate(tmp_path)
    a = validate_candidate_target(repo, cand)
    b = validate_candidate_target(repo, cand)
    assert a.source_excerpt_sha256 == b.source_excerpt_sha256


def test_path_escape_rejected(tmp_path: Path) -> None:
    repo = fx.make_repo(tmp_path)
    cand = _candidate(tmp_path)
    cand.locations[0].file = "../outside.py"
    with pytest.raises(StalenessError):
        validate_candidate_target(repo, cand)


def test_absolute_candidate_file_rejected(tmp_path: Path) -> None:
    repo = fx.make_repo(tmp_path)
    cand = _candidate(tmp_path)
    cand.locations[0].file = str((repo / fx.CAND_FILE).resolve())
    with pytest.raises(StalenessError):
        validate_candidate_target(repo, cand)


def test_relative_repo_path_validates(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    fx.make_repo(tmp_path)
    cand = _candidate(tmp_path)
    monkeypatch.chdir(tmp_path)
    validated = validate_candidate_target(Path("repo"), cand)
    assert validated.line_start == fx.CAND_START


def test_line_range_out_of_bounds(tmp_path: Path) -> None:
    repo = fx.make_repo(tmp_path)
    cand = _candidate(tmp_path)
    cand.locations[0].spans[0].line_end = 100000
    with pytest.raises(StalenessError):
        validate_candidate_target(repo, cand)


def test_staleness_symbol_missing(tmp_path: Path) -> None:
    repo = fx.make_repo(tmp_path)
    # Overwrite the candidate file so the recorded symbol is gone.
    target = repo / fx.CAND_FILE
    target.write_text("\n".join(f"# line {i}" for i in range(1, 12)) + "\n")
    cand = _candidate(tmp_path)
    with pytest.raises(StalenessError) as exc:
        validate_candidate_target(repo, cand)
    assert "stale" in str(exc.value)


def test_qualified_symbol_class_far_from_method(tmp_path: Path) -> None:
    """A `Class.method` symbol whose recorded range covers only the method body
    must validate even when the class declaration is far above the range. The
    container (class) token lives at the `class` line, not near the method, so
    requiring it inside the ±window is a false-positive staleness error."""
    repo = fx.make_repo(tmp_path)
    target = repo / fx.CAND_FILE
    body = ["class LRUCachePolicy:"]  # line 1: class decl, far from the method
    body += [f"    # filler {i}" for i in range(2, 40)]  # lines 2..39
    body += [
        "    def evict(self, n):  # line 40",  # recorded range starts here
        "        candidates = []",
        "        return candidates",  # line 42
    ]
    target.write_text("\n".join(body) + "\n", encoding="utf-8")

    cand = _candidate(tmp_path)
    cand.locations[0].spans[0].symbol = "LRUCachePolicy.evict"
    cand.locations[0].spans[0].line_start = 40
    cand.locations[0].spans[0].line_end = 42

    validated = validate_candidate_target(repo, cand)
    assert validated.line_start == 40
    assert validated.line_end == 42


def test_qualified_symbol_renamed_container_still_stale(tmp_path: Path) -> None:
    """If the container class is genuinely gone from the file, staleness must
    still fire (the relaxation is file-wide, not unconditional)."""
    repo = fx.make_repo(tmp_path)
    target = repo / fx.CAND_FILE
    body = [f"    # filler {i}" for i in range(1, 40)]
    body += [
        "    def evict(self, n):  # line 40",
        "        return []",
    ]
    target.write_text("\n".join(body) + "\n", encoding="utf-8")

    cand = _candidate(tmp_path)
    cand.locations[0].spans[0].symbol = "LRUCachePolicy.evict"
    cand.locations[0].spans[0].line_start = 40
    cand.locations[0].spans[0].line_end = 41

    with pytest.raises(StalenessError) as exc:
        validate_candidate_target(repo, cand)
    assert "LRUCachePolicy" in str(exc.value)


def test_annotated_symbol_parenthetical_prose(tmp_path: Path) -> None:
    """Regression (IOCR run): a symbol carrying a parenthetical prose
    annotation must not require the prose verbatim. Tokens like `policy)` and
    `(ROI` never appear in code; an annotation word that *does* land in the
    range (`extraction` via `roi_extraction`) is enough evidence."""
    repo = fx.make_repo(tmp_path)
    target = repo / fx.CAND_FILE
    body = ["class STRInference:"]  # line 1: container, far above the range
    body += [f"    # step {i}" for i in range(2, 41)]  # lines 2..40
    body += ["    def do_predict(self, img):"]  # line 41: outside the ±window
    body += [f"        # step {i}" for i in range(42, 53)]  # lines 42..52
    body += [
        "        roi_info = RoiInfo()",  # line 53: recorded range starts
        "        if runtime_config['roi_extraction']['is_enabled']:",
        "            roi_info = self.__extract_region_of_interest(img)",
        "        return roi_info",  # line 56: recorded range ends
    ]
    target.write_text("\n".join(body) + "\n", encoding="utf-8")

    cand = _candidate(tmp_path)
    cand.locations[0].spans[0].symbol = (
        "STRInference.do_predict (ROI extraction and detection-reuse policy)"
    )
    cand.locations[0].spans[0].line_start = 53
    cand.locations[0].spans[0].line_end = 56

    validated = validate_candidate_target(repo, cand)
    assert validated.line_start == 53


def test_annotated_symbol_trailing_prose(tmp_path: Path) -> None:
    """Regression (IOCR run): trailing free-text annotation ("self.detectors
    registry") must not be required as code. `registry` appears nowhere; the
    `detectors` token inside the range carries the match."""
    repo = fx.make_repo(tmp_path)
    target = repo / fx.CAND_FILE
    body = [
        "from typing import Dict",
        "",
        "class AngleDetectorManager:",  # line 3: container, outside the window
        "    def __init__(self, config=None):",  # line 4: outside the window
    ]
    body += [f"        # step {i}" for i in range(5, 13)]  # lines 5..12
    body += [
        "        self.detectors: Dict[str, object] = {",  # line 13: range start
        '            "classic": SkewDetector(),',
        '            "dl": AutoAlignPipeline(config) if config else None,',
        "        }",  # line 16: range end
    ]
    target.write_text("\n".join(body) + "\n", encoding="utf-8")

    cand = _candidate(tmp_path)
    cand.locations[0].spans[0].symbol = "AngleDetectorManager.__init__ self.detectors registry"
    cand.locations[0].spans[0].line_start = 13
    cand.locations[0].spans[0].line_end = 16

    validated = validate_candidate_target(repo, cand)
    assert validated.line_start == 13


def test_mangled_private_method(tmp_path: Path) -> None:
    """Regression (IOCR run): `Detector._Detector__combine_channels` records
    the *mangled* private name, which never appears in source; the gate must
    match the demangled `__combine_channels` at the range."""
    repo = fx.make_repo(tmp_path)
    target = repo / fx.CAND_FILE
    body = [
        "class Detector:",  # line 1: container, outside the window
        "    def load_model(self, weights):",
        "        self.model = weights",
    ]
    body += [f"    # step {i}" for i in range(4, 13)]  # lines 4..12
    body += [
        "    def __combine_channels(self, pred_tensor):",  # line 13: range start
        "        merged = pred_tensor.min(axis=-1)",
        "        return merged",  # line 15: range end
    ]
    target.write_text("\n".join(body) + "\n", encoding="utf-8")

    cand = _candidate(tmp_path)
    cand.locations[0].spans[0].symbol = "Detector._Detector__combine_channels"
    cand.locations[0].spans[0].line_start = 13
    cand.locations[0].spans[0].line_end = 15

    validated = validate_candidate_target(repo, cand)
    assert validated.line_start == 13


def test_region_prefix_slash_compound(tmp_path: Path) -> None:
    """Regression (IOCR run): a `region:`-prefixed slash compound must not
    require the literal `region:` fragment (or every listed member) — any of
    the named symbols inside the range is enough."""
    repo = fx.make_repo(tmp_path)
    target = repo / fx.CAND_FILE
    body = [
        "DSIZE = (1024, 1024)",
        "",
        "def rotate_image(img, angle):",  # line 3: range start
        "    return _largest_axis_rect(img, angle)",
        "def _largest_axis_rect(img, angle):",
        "    return img",  # line 6: range end
    ]
    target.write_text("\n".join(body) + "\n", encoding="utf-8")

    cand = _candidate(tmp_path)
    cand.locations[0].spans[0].symbol = (
        "region: DSIZE / RotatePadFitTransform / rotate_image / _largest_axis_rect"
    )
    cand.locations[0].spans[0].line_start = 3
    cand.locations[0].spans[0].line_end = 6

    validated = validate_candidate_target(repo, cand)
    assert validated.line_start == 3


def test_annotated_symbol_still_stale_when_code_gone(tmp_path: Path) -> None:
    """The relaxed gate must still catch genuine staleness: when none of an
    annotated symbol's components appear near the range, it fails."""
    repo = fx.make_repo(tmp_path)
    target = repo / fx.CAND_FILE
    target.write_text("\n".join(f"VALUE_{i} = {i}" for i in range(1, 13)) + "\n")

    cand = _candidate(tmp_path)
    cand.locations[0].spans[0].symbol = (
        "STRInference.do_predict (ROI extraction and detection-reuse policy)"
    )

    with pytest.raises(StalenessError) as exc:
        validate_candidate_target(repo, cand)
    assert "stale" in str(exc.value)


def test_missing_file(tmp_path: Path) -> None:
    repo = fx.make_repo(tmp_path)
    (repo / fx.CAND_FILE).unlink()
    cand = _candidate(tmp_path)
    with pytest.raises(StalenessError):
        validate_candidate_target(repo, cand)


def test_scope_file_validation(tmp_path: Path) -> None:
    repo = fx.make_repo(tmp_path)
    validate_scope_file(repo, fx.MAIN_FILE_EXTRA)
    with pytest.raises(StalenessError):
        validate_scope_file(repo, "does/not/exist.py")
    with pytest.raises(StalenessError):
        validate_scope_file(repo, "../escape.py")


def test_git_revision_capture(tmp_path: Path) -> None:
    repo = fx.make_repo(tmp_path, git=True)
    rev = capture_revision(repo, "2026-06-16T00:00:00+00:00")
    assert rev.git_commit is not None
    assert rev.dirty is False
    assert rev.captured_at == "2026-06-16T00:00:00+00:00"


def test_git_revision_non_repo(tmp_path: Path) -> None:
    repo = fx.make_repo(tmp_path, git=False)
    rev = capture_revision(repo, "t")
    assert rev.git_commit is None
    assert rev.dirty is None
