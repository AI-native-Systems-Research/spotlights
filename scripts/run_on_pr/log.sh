#!/usr/bin/env bash
# log.sh — append one deterministic progress line to a check-prs log (D4, §5.0).
#
# Usage: log.sh <logfile> <pr-id> <step> <state> <detail...>
#   <logfile>  path to the progress.log to append to (created if absent)
#   <pr-id>    PR slug, or "-" for run-level lines
#   <step>     step name, e.g. pr-checkout, pr-diff-scope, fan-out, aggregate
#   <state>    START | OK | ERROR
#   <detail>   free-text one-line detail (remaining args joined with spaces)
#
# Line format (sortable; timestamp from `date`, not the model):
#   ISO8601  <pr-id>  <step>  <state>  <detail>
#
# Logging is append-only and NEVER fatal: any failure here exits 0 so a
# broken log write cannot abort the calling step.

set -u

logfile="${1:-}"
pr_id="${2:--}"
step="${3:--}"
state="${4:--}"
shift 4 2>/dev/null || true
detail="$*"

# Without a logfile there is nothing to do — succeed silently.
[ -z "$logfile" ] && exit 0

ts="$(date -u +%Y-%m-%dT%H:%M:%SZ 2>/dev/null)" || ts="????-??-??T??:??:??Z"

# Ensure the parent dir exists; ignore failure (never fatal).
dir="$(dirname "$logfile" 2>/dev/null)"
[ -n "$dir" ] && mkdir -p "$dir" 2>/dev/null || true

# Match the design's two-space field separator; `>>` is atomic enough for short
# lines.
printf '%s  %s  %s  %s  %s\n' "$ts" "$pr_id" "$step" "$state" "$detail" >> "$logfile" 2>/dev/null || true

exit 0
