#!/usr/bin/env bash
set -euo pipefail

# Configuration - override these with environment variables
MODEL="${MODEL:-nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-NVFP4}"
PORT="${PORT:-8989}"
GPU_UTIL="${GPU_UTIL:-0.9}"
TP_SIZE="${TP_SIZE:-2}"
CONTAINER_NAME="${CONTAINER_NAME:-vllm-nemotron}"

if [ -z "${HUGGING_FACE_HUB_TOKEN:-}" ]; then
  echo "Error: HUGGING_FACE_HUB_TOKEN is not set."
  echo "Export it before running: export HUGGING_FACE_HUB_TOKEN=hf_..."
  exit 1
fi

# Stop existing container if running
if docker ps -a --format '{{.Names}}' | grep -q "^${CONTAINER_NAME}$"; then
  echo "==> Stopping existing ${CONTAINER_NAME}..."
  docker stop "$CONTAINER_NAME" 2>/dev/null || true
  docker rm "$CONTAINER_NAME" 2>/dev/null || true
fi

echo "==> Starting vLLM with Nemotron tool parser..."
echo "    Model: ${MODEL}"
echo "    Port:  ${PORT}"
echo "    GPUs:  ${TP_SIZE} (tensor parallel)"

docker run -d \
  --gpus all \
  --restart unless-stopped \
  --name "$CONTAINER_NAME" \
  -v ~/.cache/huggingface:/root/.cache/huggingface \
  --shm-size=16g \
  -p "${PORT}:8000" \
  -e "HUGGING_FACE_HUB_TOKEN=${HUGGING_FACE_HUB_TOKEN}" \
  -e VLLM_NVFP4_GEMM_BACKEND=flashinfer-cutlass \
  -e VLLM_ALLOW_LONG_MAX_MODEL_LEN=1 \
  vllm-nemotron:latest \
  --model "$MODEL" \
  --dtype auto \
  --kv-cache-dtype fp8 \
  --trust-remote-code \
  --gpu-memory-utilization "$GPU_UTIL" \
  --tensor-parallel-size "$TP_SIZE" \
  --enable-auto-tool-choice \
  --tool-call-parser nemotron

echo ""
echo "vLLM starting in background. Monitor with:"
echo "  docker logs -f ${CONTAINER_NAME}"
echo ""
echo "Wait for 'Uvicorn running on http://0.0.0.0:8000' before starting the sandbox."
