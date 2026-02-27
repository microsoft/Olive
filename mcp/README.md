# Olive MCP Server

MCP server for Microsoft Olive model optimization. Provides tools for model optimization, quantization, fine-tuning, and benchmarking through the [Model Context Protocol](https://modelcontextprotocol.io/).

## Features

| Tool | Description |
|------|-------------|
| `list_supported_configs` | List all supported devices, providers, precisions, and algorithms |
| `optimize` | End-to-end optimization with automatic pass scheduling |
| `quantize` | Model quantization (RTN, GPTQ, AWQ, HQQ, and more) |
| `finetune` | LoRA / QLoRA fine-tuning |
| `capture_onnx_graph` | Capture ONNX graph via PyTorch Exporter or Model Builder |
| `benchmark` | Model evaluation using lm-eval tasks |
| `diffusion_lora` | Train LoRA adapters for diffusion models (SD 1.5, SDXL, Flux) |
| `tune_session_params` | Auto-tune ONNX Runtime session parameters |

Each tool runs in an **isolated Python environment** (managed by uv) with the appropriate dependencies, so different onnxruntime variants (CPU, CUDA, DirectML, OpenVINO, etc.) never conflict.

## Prerequisites

- Python 3.10 - 3.12
- [uv](https://docs.astral.sh/uv/) (recommended) or pip

## Installation

```bash
git clone https://github.com/xiaoyu-work/olive-mcp.git
cd olive-mcp
uv sync
```

## Configuration

All MCP clients use the same server config — only the config file location differs.

**Server definition:**

```json
{
  "command": "uv",
  "args": ["run", "--directory", "/path/to/olive-mcp", "python", "-m", "olive_mcp"]
}
```

> Replace `/path/to/olive-mcp` with your actual project path.

| Client | Config file | Key |
|--------|------------|-----|
| **VS Code (Copilot)** | `.vscode/mcp.json` | `servers.olive` |
| **Claude Desktop** | `%APPDATA%\Claude\claude_desktop_config.json` (Win) / `~/Library/Application Support/Claude/claude_desktop_config.json` (Mac) | `mcpServers.olive` |
| **Claude Code** | `.mcp.json` in project root | `mcpServers.olive` |
| **Cursor** | `.cursor/mcp.json` | `mcpServers.olive` |
| **Windsurf** | `~/.codeium/windsurf/mcp_config.json` | `mcpServers.olive` |

<details>
<summary>VS Code example (.vscode/mcp.json)</summary>

```json
{
  "servers": {
    "olive": {
      "type": "stdio",
      "command": "uv",
      "args": ["run", "--directory", "/path/to/olive-mcp", "python", "-m", "olive_mcp"]
    }
  }
}
```
</details>

<details>
<summary>Claude Desktop / Claude Code / Cursor / Windsurf example</summary>

```json
{
  "mcpServers": {
    "olive": {
      "command": "uv",
      "args": ["run", "--directory", "/path/to/olive-mcp", "python", "-m", "olive_mcp"]
    }
  }
}
```
</details>

## Usage with VS Code Copilot

1. Open **Copilot Chat** panel (`Ctrl+Alt+I`) and switch to **Agent** mode
2. Click the **Tools** icon to verify the Olive MCP tools are listed
3. Ask Copilot, for example: *"Optimize microsoft/Phi-3-mini-4k-instruct for CPU with int4"*
4. Copilot will ask for your confirmation before calling each MCP tool

## Example Prompts

```
Optimize microsoft/Phi-3-mini-4k-instruct

Quantize microsoft/Phi-3-mini-4k-instruct

Fine-tune microsoft/Phi-3-mini-4k-instruct on nampdn-ai/tiny-codes

Capture ONNX graph from microsoft/Phi-3-mini-4k-instruct

Benchmark microsoft/Phi-3-mini-4k-instruct

Train a LoRA for runwayml/stable-diffusion-v1-5 with dataset linoyts/Tuxemon
```

## Output

All optimization outputs are saved to `~/.olive-mcp/outputs/` with timestamped directories.
