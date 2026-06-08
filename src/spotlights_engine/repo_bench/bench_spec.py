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
    reference_bundle_name: str,
    run_dir: Path,
    config_notes: str = "",
    workload_summary_md: str | None = None,
) -> tuple[Path, Path]:
    """Build the BenchSpec, write JSON + MD into `run_dir`.

    Args:
        snapshot: the SnapshotPin produced by `pick_snapshot`.
        reference_bundle_name: the prior bundle whose config the
            observability bench module should match exactly. E.g.
            `20260525T202105Z_util0.4_mem16_lru`.
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
        config=BenchSpecConfig(
            reference_bundle_name=reference_bundle_name,
            notes=config_notes,
        ),
        written_at=datetime.now(timezone.utc),
    )

    json_path = run_dir / BENCH_SPEC_JSON_NAME
    write_json(json_path, spec)

    md_path = run_dir / BENCH_SPEC_MD_NAME
    atomic_write_text(md_path, render_md(spec, workload_summary_md=workload_summary_md))

    log.info(
        "bench_spec: wrote %s and %s (SHA %s)",
        json_path, md_path, snapshot.snapshot_sha,
    )
    return json_path, md_path


def render_md(spec: BenchSpec, *, workload_summary_md: str | None = None) -> str:
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
        reference_bundle_name=spec.config.reference_bundle_name,
        reference_bundle_name_minus_timestamp=_strip_leading_timestamp(
            spec.config.reference_bundle_name
        ),
        config_notes=spec.config.notes or "(no extra notes)",
        output_naming_template=spec.output_naming_template,
        n_view=sn.n_view,
    )
    if workload_summary_md:
        return base + "\n\n" + _workload_section(workload_summary_md)
    return base


def _workload_section(summary_md: str) -> str:
    """Wrap the workload analyzer's summary MD in a spec section."""
    # Demote the summary's `# Workload analysis ...` h1 to h2 so it
    # nests under the spec's section structure.
    body = summary_md.replace("# Workload analysis", "## Workload signals from filtered PRs", 1)
    return (
        "## 6. Workload signals (auto-derived from filtered PRs)\n\n"
        "These tables describe what models / features / hardware appear "
        "across the filtered view. Use them to choose a workload your "
        "bench can run that exercises representative perf landscapes. "
        "If the signals don't match what your hardware supports, ping "
        "back rather than substituting silently.\n\n"
        + body
    )


def _strip_leading_timestamp(name: str) -> str:
    """`20260525T202105Z_util0.4_mem16_lru` → `util0.4_mem16_lru`.

    Best-effort. If the leading segment isn't a timestamp, return the
    name unchanged.
    """
    if not name:
        return name
    head, _, tail = name.partition("_")
    # ISO compact timestamp: digits + 'T' + digits + 'Z'.
    if (
        len(head) >= 8
        and head.endswith("Z")
        and head[:8].isdigit()
        and "T" in head
    ):
        return tail or name
    return name


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
