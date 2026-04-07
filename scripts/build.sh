#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

echo "==> Building vLLM image with Nemotron tool parser..."
docker build \
  -t vllm-nemotron:latest \
  -f "$PROJECT_DIR/Dockerfile.vllm-nemotron" \
  "$PROJECT_DIR"

echo "==> Building Claude sandbox template..."
docker build \
  -t claude-sandbox-local:latest \
  -f "$PROJECT_DIR/Dockerfile.claude-sandbox" \
  "$PROJECT_DIR"

echo "==> Starting local registry (if not running)..."
if ! docker ps --format '{{.Names}}' | grep -q '^registry$'; then
  docker run -d --restart unless-stopped -p 5000:5000 --name registry registry:2 2>/dev/null \
    || docker start registry
fi

echo "==> Pushing sandbox template to local registry..."
docker tag claude-sandbox-local:latest localhost:5000/claude-sandbox-local:latest
docker push localhost:5000/claude-sandbox-local:latest

echo ""
echo "Done. Images built:"
echo "  vllm-nemotron:latest                        - vLLM with Nemotron tool parser"
echo "  localhost:5000/claude-sandbox-local:latest   - Claude sandbox template"
