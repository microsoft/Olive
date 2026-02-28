# Olive Config MCP Server

MCP server for advanced Olive config file generation and validation. Helps users explore available passes, understand their parameters, and build valid workflow configs.

## When to use this vs the main Olive MCP

| | **Olive MCP** (`mcp/`) | **Config MCP** (`mcp/config-mcp/`) |
|---|---|---|
| **Audience** | Most users | Advanced users |
| **Approach** | High-level commands (optimize, quantize, finetune) | Low-level config file authoring |
| **Use case** | "Quantize my model for CPU" | "Chain OnnxConversion → GraphSurgeries → OnnxQuantization with custom parameters" |

## Features

| Tool | Description |
|------|-------------|
| `detect_hardware` | Auto-detect CPU, RAM, GPU, and disk space for smart config defaults |
| `explore_passes` | List all 59+ passes with filtering, or get full parameter schema for a specific pass |
| `run_config` | Validate or run an Olive workflow config (validate first, run on confirmation) |
| `get_job_status` | Check progress of a running config job (long-poll) |

Each run executes in an **isolated Python environment** (managed by uv) with the correct onnxruntime variant for the target hardware.

## Prerequisites

- Python 3.10+
- [uv](https://docs.astral.sh/uv/)
- [olive-ai](https://pypi.org/project/olive-ai/) (installed automatically as dependency)

## Installation

```bash
cd Olive/mcp/config-mcp
uv sync
```

## Configuration

**Server definition:**

```json
{
  "command": "uv",
  "args": ["run", "--directory", "/path/to/Olive/mcp/config-mcp", "python", "-m", "olive_config_mcp"]
}
```

> Replace `/path/to/Olive/mcp/config-mcp` with your actual project path.

<details>
<summary>VS Code example (.vscode/mcp.json)</summary>

```json
{
  "servers": {
    "olive-config": {
      "type": "stdio",
      "command": "uv",
      "args": ["run", "--directory", "/path/to/Olive/mcp/config-mcp", "python", "-m", "olive_config_mcp"]
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
    "olive-config": {
      "command": "uv",
      "args": ["run", "--directory", "/path/to/Olive/mcp/config-mcp", "python", "-m", "olive_config_mcp"]
    }
  }
}
```
</details>

## Example Prompts

```
What passes are available for int4 quantization?

Show me the parameters for OnnxQuantization

Create a config that converts a HuggingFace model to ONNX and quantizes to int4

Build an optimization pipeline for microsoft/Phi-4-mini-instruct targeting CPU

Help me write an Olive config with GraphSurgeries and OnnxFloatToFloat16
```

## Output

Config run outputs are saved to `~/.olive-mcp/config-outputs/` with timestamped directories.
