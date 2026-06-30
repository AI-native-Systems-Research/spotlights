#!/usr/bin/env python3
"""diff_ranges.py — parse a unified `git diff` into base-side changed ranges.

Reads a unified diff (the output of `git diff <base> <head> -- <paths>`, i.e.
the TWO-DOT form, §5 step 2) on stdin and emits the deterministic core of
`ground_truth.json` on stdout:

    {
      "changed_source_files": [...],
      "subfolders": [...],
      "changed_ranges": { "<file>": [ {start,end,addition_only,new_file}, ... ] },
      "addition_only_ranges": [ {file,start,end}, ... ],
      "new_files": [...],          # changed SOURCE files the PR creates
      "excluded_files": [...]      # changed non-source files (transparency)
    }

Coordinate frame (critical): ranges are BASE-SIDE — the `-a,b` side of each
`@@ -a,b +c,d @@` hunk — because candidates are generated against the pre-PR
checkout. See design/check_PRs.md §5 step 2.

Rules (per spec):
  * modified/deleted hunk (b > 0): base-side range = [a, a+b-1].
  * pure addition (b == 0): zero-width anchor [max(1,a), max(1,a)], flagged
    addition_only (git emits a==0 for a top-of-file insertion; clamp to 1).
  * a file whose old path is /dev/null is a NEW file: it has no base-side
    existence, so it is recallable only if it also appears as an in-place edit
    elsewhere (it cannot). new_file ranges are still emitted (they are
    addition_only at a==0) but the file is listed under new_files.

Source vs non-source classification (§5 step 2.2) lives HERE — the single
documented place for this knob. Config-like files are included only when no
ordinary source files survive the filter.

Usage:
    git diff <base> <head> -- <paths> | diff_ranges.py [--include p1 p2 ...]

--include, if given, restricts the reported subfolder set AND the
changed_source_files to those under one of the include prefixes (the diff
itself should already be path-restricted; this is a belt-and-suspenders knob).
"""

import argparse
import json
import os
import re
import sys

HUNK_RE = re.compile(r"^@@ -(\d+)(?:,(\d+))? \+(\d+)(?:,(\d+))? @@")

# --- Source-file filter (the documented knob, §5 step 2.2) --------------------
# A changed file is always non-source if any of these match. Config-like files
# are handled separately below because they are included only as a fallback.
_ALWAYS_NONSOURCE_BASENAME_RE = re.compile(
    r"(_test\.go$|\.spec\.[^.]+$|\.test\.[^.]+$|"
    r"\.md$|\.markdown$|\.rst$|\.txt$|"
    r"\.lock$|^go\.sum$|^go\.mod$|"
    r"\.png$|\.jpe?g$|\.gif$|\.svg$|\.pdf$)",
    re.IGNORECASE,
)
_CONFIG_BASENAME_RE = re.compile(
    r"(\.ya?ml$|\.toml$|\.ini$|\.cfg$|\.json$)",
    re.IGNORECASE,
)
_NONSOURCE_DIR_SEGMENTS = {
    "test", "tests", "testdata", "docs", "doc", "vendor",
    "node_modules", "third_party", "examples", "example",
    "__pycache__", ".git",
}
_GENERATED_BASENAME_RE = re.compile(r"(\.pb\.go$|\.gen\.go$|_generated\.|\.min\.)", re.IGNORECASE)


def is_always_excluded(path: str) -> bool:
    """Return True if `path` is never treated as source for this harness."""
    base = os.path.basename(path)
    if _ALWAYS_NONSOURCE_BASENAME_RE.search(base):
        return True
    if _GENERATED_BASENAME_RE.search(base):
        return True
    parts = path.split("/")
    for seg in parts[:-1]:
        if seg.lower() in _NONSOURCE_DIR_SEGMENTS:
            return True
    return False


def is_config_like(path: str) -> bool:
    """Return True for config-like files, included only as a fallback."""
    return bool(_CONFIG_BASENAME_RE.search(os.path.basename(path)))


def is_source(path: str) -> bool:
    """Return True if `path` is an ordinary source file we should audit."""
    if is_always_excluded(path) or is_config_like(path):
        return False
    return True


def under_include(path: str, include):
    if not include:
        return True
    for inc in include:
        inc = inc.rstrip("/")
        if path == inc or path.startswith(inc + "/"):
            return True
    return False


def parse_diff(text):
    """Parse unified diff text.

    Returns a dict keyed by the path in the auditor's coordinate frame, with a
    list of (start, end, addition_only, new_file) base-side ranges, plus the
    new-file set. Non-new files are keyed by the old/base-side path, which
    matters for renames: the auditor runs on the pre-PR checkout, where only the
    old path exists.
    """
    files = {}          # path -> list of range dicts
    new_files = set()
    cur_path = None
    cur_old_path = None
    cur_is_new = False

    lines = text.splitlines()
    i = 0
    n = len(lines)

    def _strip_prefix(p):
        # strip a leading a/ or b/ git prefix
        return p[2:] if (p.startswith("a/") or p.startswith("b/")) else p

    while i < n:
        line = lines[i]
        if line.startswith("diff --git "):
            cur_path = None
            cur_old_path = None
            cur_is_new = False
            # path is resolved from the +++ / --- lines that follow
        elif line.startswith("--- "):
            old = line[4:].strip()
            cur_is_new = old == "/dev/null"
            cur_old_path = None if cur_is_new else _strip_prefix(old)
        elif line.startswith("+++ "):
            new = line[4:].strip()
            if new == "/dev/null":
                # Deleted file: anchor the range on the BASE (old) path — that
                # file still exists in the pre-PR checkout, so a candidate can
                # legitimately overlap a line the PR later removed.
                cur_path = cur_old_path
            elif cur_is_new:
                # Brand-new file: there is no base-side path, but we still keep
                # the new path so the evaluator can flag new_file_only PRs.
                cur_path = _strip_prefix(new)
            else:
                # Modified or renamed file: anchor on the BASE path. For
                # ordinary edits this equals the new path; for renames this is
                # the only path the blind auditor could have seen.
                cur_path = cur_old_path or _strip_prefix(new)
            if cur_path is not None:
                files.setdefault(cur_path, [])
                if cur_is_new:
                    new_files.add(cur_path)
        elif line.startswith("@@"):
            m = HUNK_RE.match(line)
            if m and cur_path is not None:
                a = int(m.group(1))
                b = int(m.group(2)) if m.group(2) is not None else 1
                if b == 0:
                    anchor = max(1, a)
                    files[cur_path].append({
                        "start": anchor, "end": anchor,
                        "addition_only": True, "new_file": cur_is_new,
                    })
                else:
                    files[cur_path].append({
                        "start": a, "end": a + b - 1,
                        "addition_only": False, "new_file": cur_is_new,
                    })
        i += 1
    return files, new_files


def main():
    ap = argparse.ArgumentParser(description="Parse git diff into base-side ranges.")
    ap.add_argument("--include", nargs="*", default=None,
                    help="restrict to files under these path prefixes")
    args = ap.parse_args()

    text = sys.stdin.read()
    files, new_files = parse_diff(text)

    ordinary_source_files = []
    config_fallback_files = []
    excluded_files = []
    range_by_path = {}
    addition_only_ranges = []

    for path in sorted(files.keys()):
        if not under_include(path, args.include):
            continue
        ranges = files[path]
        if not ranges:
            # file appeared in diff with no hunks (e.g. pure rename/mode) — skip
            continue
        range_by_path[path] = ranges
        if is_always_excluded(path):
            excluded_files.append(path)
        elif is_config_like(path):
            config_fallback_files.append(path)
        else:
            ordinary_source_files.append(path)

    changed_source_files = ordinary_source_files or config_fallback_files
    excluded_files.extend(p for p in config_fallback_files if ordinary_source_files)
    changed_ranges = {path: range_by_path[path] for path in changed_source_files}

    for path in changed_source_files:
        for r in changed_ranges[path]:
            if r["addition_only"]:
                addition_only_ranges.append({"file": path, "start": r["start"], "end": r["end"]})

    subfolders = sorted({os.path.dirname(p) or "." for p in changed_source_files})
    new_source_files = sorted(f for f in new_files if f in changed_ranges)

    # Deterministic status: no surviving source files → no_source_changes (§5
    # step 2.4). A genuine git failure is the agent's responsibility to mark
    # "error"; this helper only sees a diff that parsed.
    status = "ok" if changed_source_files else "no_source_changes"

    out = {
        "status": status,
        "changed_source_files": changed_source_files,
        "subfolders": subfolders,
        "changed_ranges": changed_ranges,
        "addition_only_ranges": addition_only_ranges,
        "new_files": new_source_files,
        "excluded_files": sorted(excluded_files),
    }
    json.dump(out, sys.stdout, indent=2)
    sys.stdout.write("\n")


if __name__ == "__main__":
    main()
