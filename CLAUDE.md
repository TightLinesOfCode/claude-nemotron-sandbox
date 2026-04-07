# Claude Code + Nemotron Sandbox

This project sets up Claude Code running inside a Docker sandbox (`sbx`) powered by a local NVIDIA Nemotron model served via vLLM.

## Architecture

```
┌─────────────────────────────────┐
│  sbx sandbox (Docker container) │
│  ┌───────────────────────────┐  │
│  │  Claude Code CLI          │  │
│  │  --model nemotron         │  │
│  │  ANTHROPIC_BASE_URL=      │  │
│  │   http://host.docker.     │  │
│  │   internal:8989           │  │
│  └───────────┬───────────────┘  │
│              │ API calls         │
│  ┌───────────▼───────────────┐  │
│  │  sbx proxy (3128)         │  │
│  └───────────┬───────────────┘  │
└──────────────┼──────────────────┘
               │
┌──────────────▼──────────────────┐
│  Host machine                    │
│  ┌───────────────────────────┐  │
│  │  vLLM (port 8989)         │  │
│  │  vllm-nemotron:latest     │  │
│  │  --tool-call-parser       │  │
│  │   nemotron                │  │
│  │  Model: Nemotron-3-Super  │  │
│  └───────────────────────────┘  │
└─────────────────────────────────┘
```

## Prerequisites

- Linux with NVIDIA GPUs (tested with 2x GPU, tensor parallel)
- Docker with NVIDIA Container Toolkit (`nvidia-ctk`)
- Docker Sandboxes CLI (`sbx`) installed and logged in (`sbx login`)
- User in the `docker` group (no sudo needed)
- Hugging Face account with access to the Nemotron model
- ~80GB+ GPU VRAM (for Nemotron-3-Super-120B)

## Setup Steps

When helping a user set up this project, follow these steps in order:

### 1. Build Docker Images

```bash
./scripts/build.sh
```

This builds:
- `vllm-nemotron:latest` — vLLM with the custom Nemotron tool parser
- `localhost:5000/claude-sandbox-local:latest` — Claude Code sandbox template with env vars

### 2. Start vLLM

```bash
export HUGGING_FACE_HUB_TOKEN=hf_your_token_here
./scripts/start-vllm.sh
```

Wait for the model to load (watch `docker logs -f vllm-nemotron`). It's ready when you see `Uvicorn running on http://0.0.0.0:8000`.

### 3. Start the Sandbox

```bash
./scripts/start-sandbox.sh /path/to/your/project
```

## Key Details

### Why a custom tool parser?

Nemotron models emit tool calls in a custom XML format:
```xml
<tool_call>
<function=function_name>
<parameter=param_name>
value
</parameter>
</function>
</tool_call>
```

None of vLLM's built-in parsers (hermes, pythonic, llama3_json, etc.) handle this format. Without the parser, tool calls appear as plain text and Claude Code cannot execute tools.

### Why a custom sandbox template?

`sbx` does not support passing environment variables into sandboxes. It launches the Claude Code binary directly (not through a shell), so shell profile files like `/etc/sandbox-persistent.sh` are not sourced. The only reliable way to inject `ANTHROPIC_BASE_URL` and `ANTHROPIC_API_KEY` is to bake them into the container image via `ENV` directives.

The template must be pushed to a local Docker registry because `sbx` uses its own internal containerd and cannot access images from the host Docker daemon.

### Important: Do not use `sbx secret` for anthropic

If you set an anthropic secret via `sbx secret set`, it will conflict with the `ANTHROPIC_API_KEY` environment variable in the template image, causing an "Auth conflict" error. Only use one method.

### Important: Never use sudo with sbx

Running `sbx` with sudo creates root-owned state files that break subsequent non-sudo runs. If this happens, clean up with:
```bash
sudo rm -rf ~/.local/state/sandboxes/
sudo rm -rf ~/.config/com.docker.sandboxes/
sudo rm -rf /tmp/gvisor
sbx login
```

## Customization

### Different model

Set the `MODEL` environment variable:
```bash
MODEL=nvidia/other-model ./scripts/start-vllm.sh
MODEL=nvidia/other-model ./scripts/start-sandbox.sh /path/to/project
```

### Different port

```bash
PORT=9000 ./scripts/start-vllm.sh
```

Then update `ANTHROPIC_BASE_URL` in `Dockerfile.claude-sandbox` to match and rebuild.

### GPU configuration

Edit `scripts/start-vllm.sh` or set:
```bash
TP_SIZE=4 GPU_UTIL=0.85 ./scripts/start-vllm.sh
```

## Troubleshooting

| Symptom | Cause | Fix |
|---------|-------|-----|
| `sbx` permission denied errors | Root-owned state from sudo usage | Delete `~/.local/state/sandboxes/` and `~/.config/com.docker.sandboxes/`, re-login |
| "database already in use" | Stale daemon process | `pkill -9 sandboxd`, delete state dirs |
| Tool calls appear as text | Wrong tool parser | Ensure vLLM started with `--tool-call-parser nemotron` |
| "Auth conflict" error | Both sbx secret and ENV set | Remove the sbx secret: `sbx secret rm <sandbox> anthropic` |
| Sandbox can't reach vLLM | vLLM not running or wrong port | Check `curl http://localhost:8989/v1/models` |
| "pull access denied" for template | Image not in local registry | Run `./scripts/build.sh` to push to localhost:5000 |
