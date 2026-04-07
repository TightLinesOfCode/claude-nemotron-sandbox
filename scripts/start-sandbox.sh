#!/usr/bin/env bash
set -euo pipefail

MODEL="${MODEL:-nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-NVFP4}"
WORKSPACE="${1:-.}"
SANDBOX_NAME="${SANDBOX_NAME:-claude-nemotron}"

# Resolve workspace to absolute path
WORKSPACE="$(cd "$WORKSPACE" && pwd)"

echo "==> Checking vLLM is running..."
if ! curl -s http://localhost:8989/v1/models > /dev/null 2>&1; then
  echo "Error: vLLM is not responding on port 8989."
  echo "Start it first: ./scripts/start-vllm.sh"
  exit 1
fi

echo "==> Creating sandbox '${SANDBOX_NAME}'..."
# Remove existing sandbox if present
sbx rm "$SANDBOX_NAME" 2>/dev/null || true

sbx create claude "$WORKSPACE" \
  --name "$SANDBOX_NAME" \
  --template localhost:5000/claude-sandbox-local:latest

echo "==> Launching Claude Code in sandbox..."
sbx run "$SANDBOX_NAME" -- --model "$MODEL"
