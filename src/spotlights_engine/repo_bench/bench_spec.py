"""Build + render the observability bench spec.

Two artifacts per run, side-by-side in the run directory:
  - `<run_dir>/bench_spec.json` (canonical)
  - `<run_dir>/OBSERVABILITY_BENCH_SPEC.md` (rendered; the file the
    observability bench module reads)

The MD is generated from the JSON; humans should not hand-edit it.
Same idiom as `signal_pipeline/findings.py` — JSON is truth, MD is a
read-time render.

The spec is a cross-module contract. The repo-bench module
defines what's in it; the observability bench module is the consumer.
Each run produces a fresh copy in its own run dir; there is no stable
docs/ location, because the spec is per-run.
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from pathlib import Path

from spotlights_engine.repo_bench.schemas import (
    BenchSpec,
    BenchSpecConfig,
    SnapshotPin,
)
from spotlights_engine.repo_bench.storage import (
    atomic_write_text,
    write_json,
)

log = logging.getLogger(__name__)

_TEMPLATE_PATH = (
    Path(__file__).parent / "templates" / "bench_spec.md.template"
)

BENCH_SPEC_MD_NAME = "OBSERVABILITY_BENCH_SPEC.md"
BENCH_SPEC_JSON_NAME = "bench_spec.json"


def write_spec(
    *,
    snapshot: SnapshotPin,
    run_dir: Path,
    config_notes: str = "",
    workload_summary_md: str | None = None,
    workload_commands_md: str | None = None,
) -> tuple[Path, Path]:
    """Build the BenchSpec, write JSON + MD into `run_dir`.

    Args:
        snapshot: the SnapshotPin produced by `pick_snapshot`.
        run_dir: directory owned by the orchestrator; both files land here.
        config_notes: free-form extra context for the runner; rendered
            verbatim in §1.
        workload_summary_md: optional pre-rendered workload-signals MD
            (from `workloads.analyze`). Inlined into the spec MD as a
            "Workload signals" section. If None, the section is omitted.

    Returns: (json_path, md_path).
    """
    spec = BenchSpec(
        window_id=snapshot.window_id,
        view_id=snapshot.view_id,
        snapshot=snapshot,
        config=BenchSpecConfig(notes=config_notes),
        written_at=datetime.now(timezone.utc),
    )

    json_path = run_dir / BENCH_SPEC_JSON_NAME
    write_json(json_path, spec)

    md_path = run_dir / BENCH_SPEC_MD_NAME
    atomic_write_text(md_path, render_md(
        spec,
        workload_summary_md=workload_summary_md,
        workload_commands_md=workload_commands_md,
    ))

    log.info(
        "bench_spec: wrote %s and %s (SHA %s)",
        json_path, md_path, snapshot.snapshot_sha,
    )
    return json_path, md_path


def render_md(
    spec: BenchSpec,
    *,
    workload_summary_md: str | None = None,
    workload_commands_md: str | None = None,
) -> str:
    """Render the bench spec MD from its canonical JSON form.

    Pure function — same input → same output. No side effects.
    """
    template = _TEMPLATE_PATH.read_text(encoding="utf-8")
    sn = spec.snapshot
    snapshot_sha_short = sn.snapshot_sha[:7]
    base = template.format(
        window_id=spec.window_id,
        view_id=spec.view_id,
        written_at=spec.written_at.isoformat(),
        snapshot_sha=sn.snapshot_sha,
        snapshot_sha_short=snapshot_sha_short,
        snapshot_pr_number=sn.snapshot_pr_number,
        snapshot_merged_at=sn.snapshot_merged_at.isoformat(),
        rationale=sn.rationale,
        config_notes=spec.config.notes or "(no extra notes)",
        output_naming_template=spec.output_naming_template,
        n_view=sn.n_view,
    )
    sections: list[str] = [base]
    if workload_commands_md:
        sections.append(_workload_commands_section(workload_commands_md))
    if workload_summary_md:
        sections.append(_workload_signals_section(workload_summary_md))
    return "\n\n".join(sections)


def _workload_signals_section(summary_md: str) -> str:
    """Wrap the workload analyzer's summary MD as a spec section."""
    body = summary_md.replace("# Workload analysis", "## Workload signals from filtered PRs", 1)
    return (
        "## 7. Workload signals (auto-derived from filtered PRs)\n\n"
        "These tables describe what models / features / hardware appear "
        "across the filtered view. Use them as background — pick a "
        "workload from §6 (recommended workloads) where possible.\n\n"
        + body
    )


def _workload_commands_section(portfolio_md: str) -> str:
    """Wrap the runnable workload portfolio as a spec section."""
    return (
        "## 6. Recommended workloads (extracted from PR bodies)\n\n"
        "Each entry below is a runnable benchmark configuration extracted "
        "from filtered PRs' bodies, clustered by (model family + flags) "
        "and ranked by PR coverage. Pick whichever your hardware "
        "supports.\n\n"
        + portfolio_md
    )


def load_spec(json_path: Path) -> BenchSpec:
    """Read a previously-written BenchSpec JSON. Validates on load."""
    import json
    raw = json.loads(json_path.read_text(encoding="utf-8"))
    return BenchSpec.model_validate(raw)


__all__ = [
    "BENCH_SPEC_JSON_NAME",
    "BENCH_SPEC_MD_NAME",
    "load_spec",
    "render_md",
    "write_spec",
]
