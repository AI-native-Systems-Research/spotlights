#!/usr/bin/env python3
"""overlap.py — deterministic candidate ∩ ground-truth matching (step 4, §4/§6).

Inputs (paths as args):
    overlap.py --candidates candidates.json --ground-truth ground_truth.json
               [--addition-tolerance 3]

`candidates.json` is the MERGED candidate list for one PR. It may be either:
  * a list of candidate objects, or
  * a {"candidates": [...]} wrapper, or
  * a list of {"module_qualified_name","candidates":[...]} objects (per-folder
    Candidates objects concatenated) — flattened here.
Each candidate must carry at least: file, and a line range. The canonical
range fields are flat line_start/line_end, but the proposer agent sometimes
emits start_line/end_line or nests the range under `anchor`
({symbol,start_line,end_line}); candidate_range() reads whichever is present so
a real localization is never silently dropped to a null range. Optional:
id, symbol, estimated_impact, subfolder.

`ground_truth.json` is step 2's output (changed_source_files, subfolders,
changed_ranges, new_files).

Match definitions (§6):
  folder_hit  candidate's parent dir ∈ ground-truth subfolders
  file_hit    candidate.file ∈ changed_source_files
  line_hit    file_hit AND [line_start,line_end] overlaps a base-side changed
              range for that file. Overlap test:
                  cand.start <= range.end && range.start <= cand.end
              Ranges flagged addition_only are widened by ±tolerance first.

Per-PR roll-up:
  pr_line_hit, pr_file_hit, pr_folder_hit (any candidate),
  matched_pairs (range -> matching candidates, with impact + rank),
  missed_ranges (changed ranges no candidate covered),
  new_file_only (every changed source file is a new file → unmeasurable, §6).

Output: match.json on stdout.
"""

import argparse
import json
import os
import sys


def load_candidates(path):
    with open(path) as f:
        data = json.load(f)
    out = []
    if isinstance(data, dict) and "candidates" in data:
        out = list(data["candidates"])
    elif isinstance(data, list):
        for item in data:
            if isinstance(item, dict) and "candidates" in item:
                mod = item.get("module_qualified_name")
                for c in item["candidates"]:
                    c = dict(c)
                    c.setdefault("module_qualified_name", mod)
                    out.append(c)
            else:
                out.append(item)
    else:
        raise ValueError("candidates.json: unrecognized shape")
    return out


def overlaps(c_start, c_end, r_start, r_end):
    return c_start <= r_end and r_start <= c_end


def candidate_range(c):
    """Extract (start, end) from a candidate, tolerant of shape drift.

    The canonical schema (design/bootstrap.md) is flat line_start/line_end, but
    the proposer agent sometimes nests the range under `anchor`
    ({symbol,start_line,end_line}) or uses start_line/end_line at top level.
    Read whichever is present so a valid localization is never dropped to a
    null range (which would silently disable the line-overlap check). Returns
    (None, None) if no usable integer range is found.
    """
    anchor = c.get("anchor")
    if not isinstance(anchor, dict):
        anchor = {}
    start = (c.get("line_start") if c.get("line_start") is not None
             else c.get("start_line") if c.get("start_line") is not None
             else anchor.get("start_line") if anchor.get("start_line") is not None
             else anchor.get("line_start"))
    end = (c.get("line_end") if c.get("line_end") is not None
           else c.get("end_line") if c.get("end_line") is not None
           else anchor.get("end_line") if anchor.get("end_line") is not None
           else anchor.get("line_end"))
    try:
        return int(start), int(end)
    except (TypeError, ValueError):
        return None, None


def candidate_symbol(c):
    """Symbol may live top-level or under `anchor` (same shape drift).

    `anchor` is sometimes a free-text string rather than a {symbol,...} dict;
    in that case there is no nested symbol, so fall back to top-level only.
    """
    anchor = c.get("anchor")
    anchor = anchor if isinstance(anchor, dict) else {}
    return c.get("symbol") or c.get("title") or anchor.get("symbol")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--candidates", required=True)
    ap.add_argument("--ground-truth", required=True)
    ap.add_argument("--addition-tolerance", type=int, default=3,
                    help="±lines widening applied only to addition_only ranges (D7)")
    args = ap.parse_args()

    candidates = load_candidates(args.candidates)
    with open(args.ground_truth) as f:
        gt = json.load(f)

    changed_files = set(gt.get("changed_source_files", []))
    subfolders = set(gt.get("subfolders", []))
    changed_ranges = gt.get("changed_ranges", {})
    new_files = set(gt.get("new_files", []))
    tol = args.addition_tolerance

    # Rank candidates by their order in the merged file (engine output order).
    verdicts = []
    for rank, c in enumerate(candidates, start=1):
        cfile = c.get("file")
        cstart, cend = candidate_range(c)
        cdir = (os.path.dirname(cfile) or ".") if cfile else None

        folder_hit = cdir in subfolders if cdir is not None else False
        file_hit = cfile in changed_files
        line_hit = False
        matched_ranges = []
        if file_hit and cstart is not None:
            for ri, r in enumerate(changed_ranges.get(cfile, [])):
                rs, re_ = r["start"], r["end"]
                if r.get("addition_only"):
                    rs, re_ = rs - tol, re_ + tol
                if overlaps(cstart, cend, rs, re_):
                    line_hit = True
                    matched_ranges.append(ri)
        verdicts.append({
            "id": c.get("id"),
            "rank": rank,
            "file": cfile,
            "line_start": cstart,
            "line_end": cend,
            "symbol": candidate_symbol(c),
            "subfolder": c.get("module_qualified_name") or cdir,
            "estimated_impact": c.get("estimated_impact"),
            "folder_hit": folder_hit,
            "file_hit": file_hit,
            "line_hit": line_hit,
            "matched_range_indices": matched_ranges,
        })

    # Per-range: which candidates hit it (for "found, ranked #k" reporting).
    matched_pairs = []
    missed_ranges = []
    for cfile, ranges in changed_ranges.items():
        for ri, r in enumerate(ranges):
            hitters = [v for v in verdicts
                       if v["file"] == cfile and ri in v["matched_range_indices"]]
            entry = {
                "file": cfile,
                "range": {"start": r["start"], "end": r["end"],
                          "addition_only": r.get("addition_only", False),
                          "new_file": r.get("new_file", False)},
            }
            if hitters:
                entry["candidates"] = [
                    {"id": h["id"], "rank": h["rank"],
                     "estimated_impact": h["estimated_impact"],
                     "line_start": h["line_start"], "line_end": h["line_end"]}
                    for h in hitters
                ]
                matched_pairs.append(entry)
            else:
                missed_ranges.append(entry)

    pr_line_hit = any(v["line_hit"] for v in verdicts)
    pr_file_hit = any(v["file_hit"] for v in verdicts)
    pr_folder_hit = any(v["folder_hit"] for v in verdicts)

    # new_file_only: there ARE changed source files and every one is a new file
    # (§6) — structurally unrecallable by base-side line overlap.
    new_file_only = bool(changed_files) and changed_files.issubset(new_files)

    out = {
        "pr_line_hit": pr_line_hit,
        "pr_file_hit": pr_file_hit,
        "pr_folder_hit": pr_folder_hit,
        "new_file_only": new_file_only,
        "addition_tolerance": tol,
        "num_candidates": len(candidates),
        "num_changed_ranges": sum(len(v) for v in changed_ranges.values()),
        "candidate_verdicts": verdicts,
        "matched_pairs": matched_pairs,
        "missed_ranges": missed_ranges,
    }
    json.dump(out, sys.stdout, indent=2)
    sys.stdout.write("\n")


if __name__ == "__main__":
    main()
