#!/usr/bin/env bash
# E2: run 5 replications of the E1 config sequentially, with a cooldown
# between runs so codex's Azure quota has time to reset. Outputs land under
# runs/colpali/11-repl-01/ … 15-repl-05/.

set -u

REPO="/Users/idanfr/Projects/spotlights/colpali"
BASE_OUT="/Users/idanfr/Projects/spotlights/spotlights-engine/runs/colpali"
COOLDOWN_SECONDS=300

OBJECTIVE="find the regions that most limit retrieval latency and throughput, and propose concrete optimizations"
CUTOFF="2026-01-01"

for i in 1 2 3 4 5; do
  n=$(printf "%02d" "$i")
  # 11..15 per naming convention (E1 is 10)
  slot=$((10 + i))
  out_dir="${BASE_OUT}/${slot}-repl-${n}"
  echo ""
  echo "=================================================================="
  echo "[$(date '+%H:%M:%S')] E2 replication ${i}/5 → ${slot}-repl-${n}"
  echo "=================================================================="

  spotlights-engine \
    --repo "${REPO}" \
    --objective "${OBJECTIVE}" \
    --dr-source-cutoff-date "${CUTOFF}" \
    --max-findings-per-module 50 \
    --output-folder "${out_dir}/output" \
    --artifacts-dir "${out_dir}/artifacts" \
    --max-parallel 2

  ec=$?
  echo "[$(date '+%H:%M:%S')] replication ${i}/5 exited with code ${ec}"

  if [ "$i" -lt 5 ]; then
    echo "[$(date '+%H:%M:%S')] cooldown ${COOLDOWN_SECONDS}s before next replication"
    sleep "${COOLDOWN_SECONDS}"
  fi
done

echo ""
echo "[$(date '+%H:%M:%S')] all 5 replications finished"
