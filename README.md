# claude-nemotron-sandbox

Run Claude Code as an autonomous coding agent inside a Docker sandbox, powered by a local NVIDIA Nemotron model via vLLM. No Anthropic API key required.

## What This Does

- Serves Nemotron-3-Super-120B locally with vLLM and full tool-calling support
- Runs Claude Code in an isolated Docker sandbox (`sbx`) pointed at the local model
- The sandbox has read/write access only to your project directory

## Requirements

- Linux with NVIDIA GPUs (~80GB+ VRAM for Nemotron-3-Super-120B)
- [Docker](https://docs.docker.com/engine/install/) with [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html)
- [Docker Sandboxes (`sbx`)](https://docs.docker.com/sandbox/) CLI
- [Hugging Face](https://huggingface.co/) token with access to the Nemotron model
- Your user in the `docker` group (`sudo usermod -aG docker $USER`)

## Quick Start

```bash
# 1. Clone this repo
git clone https://github.com/yourusername/claude-nemotron-sandbox.git
cd claude-nemotron-sandbox

# 2. Log in to Docker Hub (required by sbx)
sbx login

# 3. Build all images
./scripts/build.sh

# 4. Start vLLM
export HUGGING_FACE_HUB_TOKEN=hf_your_token_here
./scripts/start-vllm.sh

# 5. Wait for vLLM to finish loading (watch for "Uvicorn running on http://0.0.0.0:8000")
docker logs -f vllm-nemotron

# 6. Start the sandbox with your project directory
./scripts/start-sandbox.sh /path/to/your/project
```

## How It Works

Claude Code inside the sandbox connects to your local vLLM instance via `ANTHROPIC_BASE_URL`. vLLM serves the Nemotron model with a custom tool parser that translates Nemotron's XML tool-call format into the structured `tool_use` blocks that Claude Code expects.

The sandbox mounts your project directory as the workspace. Files created inside the sandbox appear on your host, and vice versa. The sandbox is isolated from the rest of your filesystem.

## Project Structure

```
├── CLAUDE.md                      # Instructions for Claude Code to help with setup
├── Dockerfile.claude-sandbox      # Sandbox template with env vars for local endpoint
├── Dockerfile.vllm-nemotron       # vLLM image with custom Nemotron tool parser
├── nemotron_tool_parser.py        # Custom vLLM tool parser for Nemotron XML format
├── README.md
└── scripts/
    ├── build.sh                   # Build all Docker images
    ├── start-sandbox.sh           # Create and launch the sandbox
    └── start-vllm.sh             # Start vLLM with the Nemotron model
```

## Configuration

| Variable | Default | Description |
|----------|---------|-------------|
| `MODEL` | `nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-NVFP4` | Model to serve |
| `PORT` | `8989` | Host port for vLLM |
| `TP_SIZE` | `2` | Tensor parallel size (number of GPUs) |
| `GPU_UTIL` | `0.9` | GPU memory utilization |
| `SANDBOX_NAME` | `claude-nemotron` | Name for the sbx sandbox |
| `HUGGING_FACE_HUB_TOKEN` | (required) | HF token for model download |

## Troubleshooting

See [CLAUDE.md](CLAUDE.md) for detailed troubleshooting, architecture diagrams, and guidance.

**Common issues:**
- Never use `sudo` with `sbx` — it creates root-owned files that break future runs
- Do not set `sbx secret` for anthropic — it conflicts with the env var in the template
- If `sbx` breaks, clean up with: `rm -rf ~/.local/state/sandboxes/ ~/.config/com.docker.sandboxes/`

## Lessons Learned

### sbx doesn't support environment variable passthrough

`sbx` has no `--env` or `-e` flag. It launches the Claude Code binary directly (not through a shell), so env vars set in shell profiles or `/etc/sandbox-persistent.sh` are never sourced. The only reliable way to inject `ANTHROPIC_BASE_URL` and `ANTHROPIC_API_KEY` is to bake them into the container image with `ENV` directives in a custom Dockerfile.

Additionally, `sbx` uses its own internal containerd — not the host Docker daemon — so custom images must be pushed to a local registry (`localhost:5000`) before `sbx` can pull them via the `--template` flag.

### sbx proxies all traffic through an internal gateway

Every sandbox routes HTTP/HTTPS through a proxy at `gateway.docker.internal:3128`. This proxy intercepts API requests, injects credentials (managed via `sbx secret set`), and forwards to Anthropic's servers. By setting `ANTHROPIC_BASE_URL` to point at the local vLLM instance, Claude Code sends requests to vLLM instead. The proxy still sits in the path but passes the traffic through since it's plain HTTP to a local address.

### Nemotron's tool-call format has no built-in vLLM parser

Nemotron emits tool calls in a custom XML format that no built-in vLLM parser handles:

```xml
<tool_call>
<function=function_name>
<parameter=key>value</parameter>
</function>
</tool_call>
```

The commonly-used `hermes` parser expects JSON inside `<tool_call>` tags. Without a matching parser, vLLM returns tool calls as plain text — Claude Code sees them as regular chat output and never executes any tools. The custom `nemotron_tool_parser.py` regex-extracts the function name and parameters from this XML format, converts them into structured `tool_call` objects, and vLLM then translates those into Anthropic `tool_use` blocks on the `/v1/messages` endpoint.

### Never use sudo with sbx

Running any `sbx` command with `sudo` creates root-owned state files under `~/.config/com.docker.sandboxes/` and `~/.local/state/sandboxes/`. Subsequent non-sudo runs fail with "permission denied" or "database already in use" errors. If this happens, the fix is to delete both directories and re-run `sbx login`.

## License

Apache-2.0
