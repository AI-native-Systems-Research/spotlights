#!/usr/bin/env bash
# Start the NixL prefill/decode/proxy server stack and run stress tests
# (test_edge_cases.py). Mirrors the server setup from run_accuracy_test.sh but
# runs the edge-cases stress tests instead of the accuracy suite.
set -xe

# Clean up any leftover vLLM processes from prior entries
pkill -f "vllm serve" 2>/dev/null || true
sleep 2

GIT_ROOT=$(git rev-parse --show-toplevel)

MODEL=${MODEL_NAMES:-"Qwen/Qwen3-0.6B"}
GPU_MEMORY_UTILIZATION=${GPU_MEMORY_UTILIZATION:-0.2}
PREFILL_PORT=${PREFILL_PORT:-8100}
DECODE_PORT=${DECODE_PORT:-8200}
PROXY_PORT=${PROXY_PORT:-8192}
BLOCK_SIZE=${BLOCK_SIZE:-128}

KV_CONFIG='{"kv_connector":"NixlConnector","kv_role":"kv_both"}'

trap 'kill $(jobs -pr) 2>/dev/null; wait' SIGINT SIGTERM EXIT

wait_for_server() {
  local port=$1
  timeout 1200 bash -c "
    until curl -s localhost:${port}/v1/completions > /dev/null; do
      sleep 1
    done" && return 0 || return 1
}

# Start prefill instance
echo "Starting prefill instance on port $PREFILL_PORT"
VLLM_KV_CACHE_LAYOUT='HND' \
UCX_NET_DEVICES=all \
VLLM_NIXL_SIDE_CHANNEL_PORT=5559 \
vllm serve "$MODEL" \
  --port "$PREFILL_PORT" \
  --enforce-eager \
  --block-size "$BLOCK_SIZE" \
  --gpu-memory-utilization "$GPU_MEMORY_UTILIZATION" \
  --tensor-parallel-size 1 \
  --kv-transfer-config "$KV_CONFIG" &

# Start decode instance
echo "Starting decode instance on port $DECODE_PORT"
VLLM_KV_CACHE_LAYOUT='HND' \
UCX_NET_DEVICES=all \
VLLM_NIXL_SIDE_CHANNEL_PORT=5659 \
vllm serve "$MODEL" \
  --port "$DECODE_PORT" \
  --enforce-eager \
  --block-size "$BLOCK_SIZE" \
  --gpu-memory-utilization "$GPU_MEMORY_UTILIZATION" \
  --tensor-parallel-size 1 \
  --kv-transfer-config "$KV_CONFIG" &

echo "Waiting for prefill instance..."
wait_for_server "$PREFILL_PORT"
echo "Waiting for decode instance..."
wait_for_server "$DECODE_PORT"

# Start proxy
echo "Starting proxy server on port $PROXY_PORT"
python3 "${GIT_ROOT}/tests/v1/kv_connector/nixl_integration/toy_proxy_server.py" \
  --port "$PROXY_PORT" \
  --prefiller-hosts localhost \
  --prefiller-ports "$PREFILL_PORT" \
  --decoder-hosts localhost \
  --decoder-ports "$DECODE_PORT" &

sleep 5

# Run stress tests with required env vars
echo "Running stress tests (test_edge_cases.py -k memory)"
PREFILL_PORT=$PREFILL_PORT \
DECODE_PORT=$DECODE_PORT \
PROXY_PORT=$PROXY_PORT \
python3 -m pytest -v "${GIT_ROOT}/tests/v1/kv_connector/nixl_integration/test_edge_cases.py" \
  -k memory "$@"
