#!/bin/bash
#BSUB -gpu "num=1:mode=exclusive_process:gmodel=NVIDIAA100_SXM4_80GB:gmem=75G"
#BSUB -R "rusage[ngpus=1, cpu=8, mem=32GB]"
#BSUB -U infusion
#BSUB -J spotlights-validation-baseline
#BSUB -o validation_baseline.stdout
#BSUB -e validation_baseline.stderr

set -euo pipefail

# === Baseline validation ===
# Runs on the ORIGINAL vLLM 0.18.0 without the evolved policy change.
# Benchmarks: LRU and ARC only (the evolved entry is skipped).

usage() {
    cat >&2 <<EOF
Usage: $0 [--root-dir DIR] [--artifacts-dir DIR] [--venv DIR] [--env FILE] [--nvme-offload-path DIR] [--hf-cache-dir DIR]

Precedence (highest first): CLI flag > .env file > calling shell environment.

  --root-dir            ROOT_DIR (required) -- root for paths in validation_plan.json (e.g. parent of vllm/ and kv-offload-lab/)
  --artifacts-dir       ARTIFACTS_DIR -- defaults to the directory containing this script
  --venv                VENV (required) -- venv to activate (e.g. \$ROOT_DIR/vllm/.venv)
  --env                 ENV_FILE -- defaults to \$ARTIFACTS_DIR/.env (silently skipped if missing)
  --nvme-offload-path   VLLM_NVME_OFFLOAD_PATH (required) -- host-side NVMe scratch dir for KV offload
  --hf-cache-dir        HF_CACHE_DIR (required) -- HF cache dir; HF_HOME / HF_HUB_CACHE / TRANSFORMERS_CACHE default to this
EOF
}

# --- parse args into *_ARG so flags can override .env after sourcing ---
ROOT_DIR_ARG=""
ARTIFACTS_DIR_ARG=""
VENV_ARG=""
ENV_FILE_ARG=""
NVME_OFFLOAD_PATH_ARG=""
HF_CACHE_DIR_ARG=""

while [[ $# -gt 0 ]]; do
    case "$1" in
        --root-dir) ROOT_DIR_ARG="$2"; shift 2 ;;
        --root-dir=*) ROOT_DIR_ARG="${1#--root-dir=}"; shift ;;
        --artifacts-dir) ARTIFACTS_DIR_ARG="$2"; shift 2 ;;
        --artifacts-dir=*) ARTIFACTS_DIR_ARG="${1#--artifacts-dir=}"; shift ;;
        --venv) VENV_ARG="$2"; shift 2 ;;
        --venv=*) VENV_ARG="${1#--venv=}"; shift ;;
        --env) ENV_FILE_ARG="$2"; shift 2 ;;
        --env=*) ENV_FILE_ARG="${1#--env=}"; shift ;;
        --nvme-offload-path) NVME_OFFLOAD_PATH_ARG="$2"; shift 2 ;;
        --nvme-offload-path=*) NVME_OFFLOAD_PATH_ARG="${1#--nvme-offload-path=}"; shift ;;
        --hf-cache-dir) HF_CACHE_DIR_ARG="$2"; shift 2 ;;
        --hf-cache-dir=*) HF_CACHE_DIR_ARG="${1#--hf-cache-dir=}"; shift ;;
        -h|--help) usage; exit 0 ;;
        *) echo "Unknown argument: $1" >&2; usage; exit 1 ;;
    esac
done

# --- resolve ARTIFACTS_DIR and ENV_FILE so we can source .env first ---
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" 2>/dev/null && pwd)" || SCRIPT_DIR=""
DEFAULT_ARTIFACTS_DIR=""
if [[ -n "$SCRIPT_DIR" && -f "$SCRIPT_DIR/run_baseline.sh" ]]; then
    DEFAULT_ARTIFACTS_DIR="$SCRIPT_DIR"
elif [[ -n "${LS_SUBCWD:-}" && -f "$LS_SUBCWD/run_baseline.sh" ]]; then
    DEFAULT_ARTIFACTS_DIR="$LS_SUBCWD"
elif [[ -n "${LS_SUBCWD:-}" && -f "$LS_SUBCWD/artifacts/kvoffload/run_baseline.sh" ]]; then
    DEFAULT_ARTIFACTS_DIR="$LS_SUBCWD/artifacts/kvoffload"
fi
ARTIFACTS_DIR="${ARTIFACTS_DIR_ARG:-${ARTIFACTS_DIR:-$DEFAULT_ARTIFACTS_DIR}}"
if [[ -z "$ARTIFACTS_DIR" ]]; then
    echo "ARTIFACTS_DIR could not be determined. Pass --artifacts-dir or set ARTIFACTS_DIR." >&2
    exit 1
fi

ENV_FILE_DEFAULTED=0
if [[ -n "$ENV_FILE_ARG" ]]; then
    ENV_FILE="$ENV_FILE_ARG"
elif [[ -z "${ENV_FILE:-}" ]]; then
    ENV_FILE="$ARTIFACTS_DIR/.env"
    ENV_FILE_DEFAULTED=1
fi

# --- redirect stdout/stderr so files are visible during the run ---
mkdir -p "$ARTIFACTS_DIR"
exec > "$ARTIFACTS_DIR/validation_baseline.stdout" 2> "$ARTIFACTS_DIR/validation_baseline.stderr"

# --- load env file (sourced before flags are applied so flags win) ---
if [[ -f "$ENV_FILE" ]]; then
    set -a; source "$ENV_FILE"; set +a
elif [[ "$ENV_FILE_DEFAULTED" -eq 0 ]]; then
    echo ".env not found: $ENV_FILE" >&2; exit 1
fi

# --- apply flag values last so CLI > .env > calling shell ---
ROOT_DIR="${ROOT_DIR_ARG:-${ROOT_DIR:-}}"
VENV="${VENV_ARG:-${VENV:-}}"
NVME_OFFLOAD_PATH="${NVME_OFFLOAD_PATH_ARG:-${VLLM_NVME_OFFLOAD_PATH:-}}"
HF_CACHE_DIR="${HF_CACHE_DIR_ARG:-${HF_CACHE_DIR:-}}"

# --- validate required inputs ---
[[ -n "$ROOT_DIR" ]] || { echo "ROOT_DIR not set: pass --root-dir, set in $ENV_FILE, or export ROOT_DIR" >&2; exit 1; }
[[ -d "$ROOT_DIR" ]] || { echo "ROOT_DIR is not a directory: $ROOT_DIR" >&2; exit 1; }
[[ -n "$VENV" ]]     || { echo "VENV not set: pass --venv, set in $ENV_FILE, or export VENV" >&2; exit 1; }
[[ -n "$NVME_OFFLOAD_PATH" ]] || { echo "VLLM_NVME_OFFLOAD_PATH not set: pass --nvme-offload-path, set in $ENV_FILE, or export VLLM_NVME_OFFLOAD_PATH" >&2; exit 1; }
[[ -n "$HF_CACHE_DIR" ]] || { echo "HF_CACHE_DIR not set: pass --hf-cache-dir, set in $ENV_FILE, or export HF_CACHE_DIR" >&2; exit 1; }

# --- activate venv ---
# shellcheck disable=SC1090,SC1091
source "$VENV/bin/activate"

# --- environment ---
export ROOT_DIR
export HF_TOKEN="${HF_TOKEN:?HF_TOKEN must be set (in .env or environment)}"
export HF_HOME="${HF_HOME:-$HF_CACHE_DIR}"
export HF_HUB_CACHE="${HF_HUB_CACHE:-$HF_CACHE_DIR}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-$HF_CACHE_DIR}"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export VLLM_NVME_OFFLOAD_PATH="$NVME_OFFLOAD_PATH"
export VLLM_LOGGING_LEVEL="${VLLM_LOGGING_LEVEL:-INFO}"
export VLLM_WORKER_MULTIPROC_METHOD="${VLLM_WORKER_MULTIPROC_METHOD:-spawn}"

# --- paths ---
LOGS_DIR="$ARTIFACTS_DIR/logs/baseline"
mkdir -p "$LOGS_DIR"

echo "Job started:    $(date)"
echo "Host:           $(hostname)"
echo "ROOT_DIR:       $ROOT_DIR"
echo "ARTIFACTS_DIR:  $ARTIFACTS_DIR"
echo "VENV:           $VENV"
echo "ENV_FILE:       $ENV_FILE"
echo "NVME path:      $VLLM_NVME_OFFLOAD_PATH"
echo "GPU:            $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null || echo 'n/a')"
echo "Mode:           BASELINE (LRU + ARC; evolved policy skipped)"

# --- generate plan with evolved entry skipped ---
# Skip benchmark-multi-turn-kv-offload-lab (the evolved policy entry) by
# setting skipped=true and adding a skip_reason.
PLAN_SRC="$ARTIFACTS_DIR/validation_plan.json"
PLAN_BASELINE="$ARTIFACTS_DIR/validation_plan_baseline.json"
jq '(.entries[] | select(.harness_entry.id == "benchmark-multi-turn-kv-offload-lab")) |=
    (.skipped = true | .skip_reason = "baseline run: evolved policy not applied")' \
    "$PLAN_SRC" > "$PLAN_BASELINE"

echo "Plan:           $PLAN_BASELINE (evolved entry skipped)"

# --- run ---
python -m spotlights_validation.cli run \
    --plan        "$PLAN_BASELINE" \
    --source-tree "$ROOT_DIR/vllm" \
    --change-ref  "kv-offload-lab@HEAD" \
    --logs-dir    "$LOGS_DIR" \
    --out         "$ARTIFACTS_DIR/validation_result_baseline.json"

echo "Job finished: $(date)"
