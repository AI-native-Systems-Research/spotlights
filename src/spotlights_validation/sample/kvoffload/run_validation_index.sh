#!/bin/bash
#BSUB -gpu "num=1:mode=exclusive_process:gmodel=NVIDIAA100_SXM4_80GB:gmem=75G"
#BSUB -R "rusage[ngpus=1, cpu=8, mem=32GB]"
#BSUB -U infusion
#BSUB -J spotlights-validation
#BSUB -o validation_job.stdout
#BSUB -e validation_job.stderr

set -euo pipefail

# --- defaults / env-var fallbacks ---
ROOT_DIR="${ROOT_DIR:-}"
ARTIFACTS_DIR="${ARTIFACTS_DIR:-}"
VENV="${VENV:-}"
ENV_FILE="${ENV_FILE:-}"
NVME_OFFLOAD_PATH="${VLLM_NVME_OFFLOAD_PATH:-}"
HF_CACHE_DIR="${HF_CACHE_DIR:-}"
INDEXES="${INDEXES:-17}"

usage() {
    cat >&2 <<EOF
Usage: $0 [--root-dir DIR] [--artifacts-dir DIR] [--venv DIR] [--env FILE] [--nvme-offload-path DIR] [--hf-cache-dir DIR] [--indexes LIST]

Each flag can also be supplied via environment variable:
  --root-dir            ROOT_DIR (required) -- root for paths in validation_plan.json
  --artifacts-dir       ARTIFACTS_DIR -- defaults to the directory containing this script
  --venv                VENV (required) -- venv to activate
  --env                 ENV_FILE -- defaults to \$ARTIFACTS_DIR/.env (silently skipped if missing)
  --nvme-offload-path   VLLM_NVME_OFFLOAD_PATH (required)
  --hf-cache-dir        HF_CACHE_DIR (required) -- HF cache dir; HF_HOME / HF_HUB_CACHE / TRANSFORMERS_CACHE default to this
  --indexes             INDEXES (default: 17) -- comma-separated entry indexes to run
EOF
}

# --- parse args ---
while [[ $# -gt 0 ]]; do
    case "$1" in
        --root-dir) ROOT_DIR="$2"; shift 2 ;;
        --root-dir=*) ROOT_DIR="${1#--root-dir=}"; shift ;;
        --artifacts-dir) ARTIFACTS_DIR="$2"; shift 2 ;;
        --artifacts-dir=*) ARTIFACTS_DIR="${1#--artifacts-dir=}"; shift ;;
        --venv) VENV="$2"; shift 2 ;;
        --venv=*) VENV="${1#--venv=}"; shift ;;
        --env) ENV_FILE="$2"; shift 2 ;;
        --env=*) ENV_FILE="${1#--env=}"; shift ;;
        --nvme-offload-path) NVME_OFFLOAD_PATH="$2"; shift 2 ;;
        --nvme-offload-path=*) NVME_OFFLOAD_PATH="${1#--nvme-offload-path=}"; shift ;;
        --hf-cache-dir) HF_CACHE_DIR="$2"; shift 2 ;;
        --hf-cache-dir=*) HF_CACHE_DIR="${1#--hf-cache-dir=}"; shift ;;
        --indexes) INDEXES="$2"; shift 2 ;;
        --indexes=*) INDEXES="${1#--indexes=}"; shift ;;
        -h|--help) usage; exit 0 ;;
        *) echo "Unknown argument: $1" >&2; usage; exit 1 ;;
    esac
done

# --- resolve defaults ---
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
ARTIFACTS_DIR="${ARTIFACTS_DIR:-$SCRIPT_DIR}"
ENV_FILE_DEFAULTED=0
if [[ -z "$ENV_FILE" ]]; then
    ENV_FILE="$ARTIFACTS_DIR/.env"
    ENV_FILE_DEFAULTED=1
fi

# --- validate required inputs ---
[[ -n "$ROOT_DIR" ]] || { echo "ROOT_DIR not set: pass --root-dir or export ROOT_DIR" >&2; exit 1; }
[[ -d "$ROOT_DIR" ]] || { echo "ROOT_DIR is not a directory: $ROOT_DIR" >&2; exit 1; }
[[ -n "$VENV" ]]     || { echo "VENV not set: pass --venv or export VENV" >&2; exit 1; }
[[ -n "$NVME_OFFLOAD_PATH" ]] || { echo "VLLM_NVME_OFFLOAD_PATH not set: pass --nvme-offload-path or export VLLM_NVME_OFFLOAD_PATH" >&2; exit 1; }
[[ -n "$HF_CACHE_DIR" ]] || { echo "HF_CACHE_DIR not set: pass --hf-cache-dir or export HF_CACHE_DIR" >&2; exit 1; }

# --- redirect stdout/stderr so files are visible during the run ---
mkdir -p "$ARTIFACTS_DIR"
exec > "$ARTIFACTS_DIR/validation_job.stdout" 2> "$ARTIFACTS_DIR/validation_job.stderr"

# --- load env file ---
if [[ -f "$ENV_FILE" ]]; then
    set -a; source "$ENV_FILE"; set +a
elif [[ "$ENV_FILE_DEFAULTED" -eq 0 ]]; then
    echo ".env not found: $ENV_FILE" >&2; exit 1
fi

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
LOGS_DIR="$ARTIFACTS_DIR/logs"
mkdir -p "$LOGS_DIR"

echo "Job started:    $(date)"
echo "Host:           $(hostname)"
echo "ROOT_DIR:       $ROOT_DIR"
echo "ARTIFACTS_DIR:  $ARTIFACTS_DIR"
echo "VENV:           $VENV"
echo "ENV_FILE:       $ENV_FILE"
echo "NVME path:      $VLLM_NVME_OFFLOAD_PATH"
echo "INDEXES:        $INDEXES"
echo "GPU:            $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null || echo 'n/a')"

# --- run ---
python -m spotlights_validation.cli run \
    --plan        "$ARTIFACTS_DIR/validation_plan.json" \
    --source-tree "$ROOT_DIR/vllm" \
    --change-ref  "kv-offload-lab@HEAD" \
    --indexes     "$INDEXES" \
    --logs-dir    "$LOGS_DIR" \
    --out         "$ARTIFACTS_DIR/validation_result.json"

echo "Job finished: $(date)"
