#!/usr/bin/env bash
# Kill all vLLM-related processes (active and zombie) running under the current user.
set -euo pipefail

# Source 1: processes whose cmdline contains vLLM server entrypoints (python only).
# Narrow pattern targets actual vLLM server processes — avoids killing the
# validation runner whose --source-tree path also contains "vllm".
CMDLINE_PIDS=$(pgrep -u "$USER" -f "vllm\.entrypoints|vllm\.engine|serve\.py" 2>/dev/null \
    | xargs -r ps -o pid,args --no-headers -p 2>/dev/null \
    | grep -v "spotlights_engine.validation" \
    | awk 'tolower($2) ~ /python/ {print $1}' \
    || true)

# Source 2: processes currently holding GPU compute (e.g. EngineCore spawned via
# multiprocessing — its cmdline is "python -c from multiprocessing.spawn import ..."
# so pgrep -f misses it entirely).
GPU_PIDS=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null \
    | tr -d ' ' \
    | xargs -r ps -o pid,user --no-headers -p 2>/dev/null \
    | awk -v u="$USER" '$2 == u {print $1}' \
    || true)

PIDS=$(printf '%s\n%s\n' "$CMDLINE_PIDS" "$GPU_PIDS" | sort -un | grep -v '^$' || true)

if [[ -z "$PIDS" ]]; then
    echo "No vLLM processes found for user $USER."
    exit 0
fi

echo "vLLM processes found for user $USER:"

ZOMBIE_PARENTS=""
ACTIVE_PIDS=""

for PID in $PIDS; do
    STATE=$(ps -o stat= -p "$PID" 2>/dev/null | tr -d ' ' || true)
    CMD=$(ps -o comm= -p "$PID" 2>/dev/null || true)
    if [[ "$STATE" == Z* ]]; then
        PPID=$(ps -o ppid= -p "$PID" 2>/dev/null | tr -d ' ' || true)
        echo "  PID $PID [$CMD] zombie — will kill parent PID $PPID"
        ZOMBIE_PARENTS="$ZOMBIE_PARENTS $PPID"
    else
        echo "  PID $PID [$CMD] ($STATE) — killing"
        ACTIVE_PIDS="$ACTIVE_PIDS $PID"
    fi
done

# Kill active processes
if [[ -n "$ACTIVE_PIDS" ]]; then
    # shellcheck disable=SC2086
    kill -9 $ACTIVE_PIDS 2>/dev/null || true
fi

# Reap zombies by killing their parents (deduplicated, skip init/systemd PID 1)
if [[ -n "$ZOMBIE_PARENTS" ]]; then
    for PPID in $(echo "$ZOMBIE_PARENTS" | tr ' ' '\n' | sort -u | grep -v '^$\|^1$'); do
        PSTATE=$(ps -o stat= -p "$PPID" 2>/dev/null | tr -d ' ' || true)
        PCMD=$(ps -o comm= -p "$PPID" 2>/dev/null || true)
        echo "  Killing zombie parent PID $PPID [$PCMD] ($PSTATE)"
        kill -9 "$PPID" 2>/dev/null || true
    done
fi

# Wait for all killed processes to fully exit
ALL_KILLED=$(printf '%s\n%s\n' "$ACTIVE_PIDS" "$ZOMBIE_PARENTS" | sort -un | grep -v '^$' || true)
if [[ -n "$ALL_KILLED" ]]; then
    echo "Waiting for processes to exit..."
    TIMEOUT=30
    ELAPSED=0
    while [[ $ELAPSED -lt $TIMEOUT ]]; do
        STILL_ALIVE=""
        for PID in $ALL_KILLED; do
            if kill -0 "$PID" 2>/dev/null; then
                STILL_ALIVE="$STILL_ALIVE $PID"
            fi
        done
        if [[ -z "$STILL_ALIVE" ]]; then
            break
        fi
        sleep 1
        ELAPSED=$((ELAPSED + 1))
    done
    if [[ -n "$STILL_ALIVE" ]]; then
        echo "WARNING: processes still alive after ${TIMEOUT}s:$STILL_ALIVE"
    fi
fi

# Wait for the vLLM port to be released
VLLM_PORT="${EVOLVE_SERVER_PORT:-8000}"
echo "Waiting for port $VLLM_PORT to be released..."
TIMEOUT=15
ELAPSED=0
while [[ $ELAPSED -lt $TIMEOUT ]]; do
    if ! ss -tlnp 2>/dev/null | grep -q ":${VLLM_PORT} "; then
        break
    fi
    sleep 1
    ELAPSED=$((ELAPSED + 1))
done
if ss -tlnp 2>/dev/null | grep -q ":${VLLM_PORT} "; then
    echo "WARNING: port $VLLM_PORT still in use after ${TIMEOUT}s"
fi

echo "Done."