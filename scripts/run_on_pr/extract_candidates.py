#!/usr/bin/env python3
"""extract_candidates.py — explode the engine's nested candidates into flat records.

Reads the engine's `result.json` (a dump of `SpotlightsManagerResult`) and emits
the flat candidate list that `overlap.py` consumes. The engine's candidates are
**nested** (`schemas/candidate.py`): each `Candidate` carries
`locations: [CodeLocation]`, each `CodeLocation` has a `file` plus
`spans: [CodeSpan]`, and the line range lives on each span. `overlap.py` reads a
**flat** `file` + `line_start`/`line_end` per record, so every `(location, span)`
pair is exploded into one flat record carrying the parent candidate's `id`,
`module_qualified_name`, `estimated_impact`, and rank.

    {
      "id": "cand-vllm_v1_kv_offload-0001",
      "module_qualified_name": "vllm/v1/kv_offload",
      "file": "vllm/v1/kv_offload/cpu/manager.py",
      "line_start": 19, "line_end": 22,
      "symbol": "_CACHE_POLICIES", "kind": "plugin_seam",
      "estimated_impact": "high", "origin": "code_agent",
      "rank": 1
    }

`rank` is the 1-based position of the parent candidate in `report.candidates`
(engine output order), so it is meaningful and rank-stable; every flat record
exploded from the same candidate shares its `id` and `rank`. The script is
deterministic and preserves `report.candidates` order.

Candidate source (design "Two candidate sources" note): prefer the top-level
`report.candidates` — already the cross-module flattened list, each entry
carrying its `module_qualified_name`. This script also tolerates older/runtime
shapes: a top-level `candidates` list (no `report` wrapper), and as a last
resort the per-module `module_runs[<qn>].candidates.candidates` lists.

Usage:
    extract_candidates.py <result.json>            # writes JSON to stdout
    extract_candidates.py <result.json> -o out.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def _iter_candidates(data: dict) -> list[dict]:
    """Return the engine's candidate objects in rank-stable order.

    Preference order (design):
      1. report.candidates  — current schema, already cross-module flattened.
      2. candidates         — older flat dump with no `report` wrapper.
      3. module_runs[*].candidates.candidates — last-resort reconstruction,
         concatenated in module_runs iteration order.
    """
    report = data.get("report")
    if isinstance(report, dict) and isinstance(report.get("candidates"), list):
        return list(report["candidates"])

    if isinstance(data.get("candidates"), list):
        return list(data["candidates"])

    out: list[dict] = []
    module_runs = data.get("module_runs")
    if isinstance(module_runs, dict):
        for run in module_runs.values():
            if not isinstance(run, dict):
                continue
            wrapper = run.get("candidates")
            if isinstance(wrapper, dict) and isinstance(wrapper.get("candidates"), list):
                out.extend(wrapper["candidates"])
    return out


def explode(candidates: list[dict]) -> list[dict]:
    """Explode each candidate's `locations[].spans[]` into flat records.

    Falls back to the legacy flat shape (top-level `file` + `line_start`/
    `line_end` on the candidate itself) when a candidate carries no `locations`,
    so older `result.json` dumps still produce usable records.
    """
    flat: list[dict] = []
    for rank, c in enumerate(candidates, start=1):
        cid = c.get("id")
        mqn = c.get("module_qualified_name")
        impact = c.get("estimated_impact")
        origin = c.get("origin")

        locations = c.get("locations")
        if isinstance(locations, list):
            # Current nested schema. Off-schema empties (`locations: []` or a
            # location with `spans: []`) are violations of CodeLocation/Candidate
            # min_length=1, so valid engine output always yields >=1 record here;
            # an empty list simply emits nothing rather than an all-null record.
            for loc in locations:
                if not isinstance(loc, dict):
                    continue
                cfile = loc.get("file")
                spans = loc.get("spans") or []
                for span in spans:
                    if not isinstance(span, dict):
                        continue
                    flat.append({
                        "id": cid,
                        "module_qualified_name": mqn,
                        "file": cfile,
                        "line_start": span.get("line_start"),
                        "line_end": span.get("line_end"),
                        "symbol": span.get("symbol"),
                        "kind": span.get("kind"),
                        "estimated_impact": impact,
                        "origin": origin,
                        "rank": rank,
                    })
        else:
            # Legacy flat candidate (no `locations` key at all): carry the
            # top-level fields. Skip a record with no usable file rather than
            # emitting an all-null junk row.
            if c.get("file") is None:
                continue
            flat.append({
                "id": cid,
                "module_qualified_name": mqn,
                "file": c.get("file"),
                "line_start": c.get("line_start"),
                "line_end": c.get("line_end"),
                "symbol": c.get("symbol"),
                "kind": c.get("kind"),
                "estimated_impact": impact,
                "origin": origin,
                "rank": rank,
            })
    return flat


def main() -> None:
    ap = argparse.ArgumentParser(description="Explode engine candidates to flat records.")
    ap.add_argument("result_json", help="Path to the engine's result.json")
    ap.add_argument("-o", "--output", default=None, help="Write JSON here (default: stdout)")
    args = ap.parse_args()

    data = json.loads(Path(args.result_json).read_text(encoding="utf-8"))
    flat = explode(_iter_candidates(data))

    text = json.dumps(flat, indent=2) + "\n"
    if args.output:
        Path(args.output).write_text(text, encoding="utf-8")
    else:
        sys.stdout.write(text)


if __name__ == "__main__":
    main()
