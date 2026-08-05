#!/bin/bash
# Launch an interactive shell on a normal-queue compute node with the engine env preloaded.
# Useful for probing / debugging without waiting for a batch job to schedule.
#
# CCC kills interactive jobs after 6h. Do quick investigation here; use run-engine.lsf for real runs.
#
# Usage:
#   ~/spotlights-engine/scripts/ccc/interactive-engine.sh          # 4 cores, 16 GB, 4h
#   CORES=2 MEM=8192 HOURS=2 ~/spotlights-engine/scripts/ccc/interactive-engine.sh

set -eo pipefail

CORES="${CORES:-4}"
MEM="${MEM:-16384}"
HOURS="${HOURS:-4}"

echo "Submitting interactive job: $CORES cores, ${MEM} MB, ${HOURS}h"
echo "Once it lands you'll get a shell with engine venv + CLIs on PATH."

exec bsub -Is -n "$CORES" -M "$MEM" -hl -W $((HOURS*60)) -R "span[hosts=1]" \
  bash -l -c "source /u/idanfr/spotlights-engine/scripts/ccc/env.sh && cd /u/idanfr/spotlights-engine && exec bash"
