# Olive MCP Server: Optimizing AI Models Through Natural Conversation

*Author: Xiaoyu Zhang*
*Created: 2026-04-07*

What if optimizing a machine learning model were as simple as asking *"Make my model smaller and faster"*? With the Olive MCP Server, it is.

The [Model Context Protocol (MCP)](https://modelcontextprotocol.io/) is an open standard that lets AI assistants use external tools. We built an MCP server for Olive that exposes model optimization, quantization, fine-tuning, and benchmarking as tools that any MCP-compatible AI client — VS Code Copilot, Claude Desktop, Cursor, and others — can call on your behalf.

Instead of memorizing CLI flags or writing JSON configs, you describe what you want in plain language. The AI assistant figures out the right Olive commands, runs them in the background, and reports results back to you.

---

## Why MCP for Olive?

Olive is a powerful model optimization toolkit with 40+ optimization passes, multiple quantization algorithms, and a rich CLI. But that power comes with complexity:

- Which precision should I use for my hardware?
- What's the difference between `optimize` and `quantize`?
- Do I need GPTQ or RTN? What about AWQ?
- Which execution provider matches my GPU?

The MCP server removes this barrier. The AI assistant handles tool selection, parameter configuration, and hardware adaptation — you just describe your goal.

---

## Quick Start

### Prerequisites

- Python 3.10+
- [uv](https://docs.astral.sh/uv/) (fast Python package manager)

### Installation

```bash
git clone https://github.com/microsoft/Olive.git
cd Olive/mcp
uv sync
```

### Connect to Your AI Client

Add the Olive MCP server to your client's configuration. The server definition is the same for all clients:

```json
{
  "command": "uv",
  "args": ["run", "--directory", "/path/to/Olive/mcp", "python", "-m", "olive_mcp"]
}
```

<details>
<summary>VS Code Copilot (.vscode/mcp.json)</summary>

```json
{
  "servers": {
    "olive": {
      "type": "stdio",
      "command": "uv",
      "args": ["run", "--directory", "/path/to/Olive/mcp", "python", "-m", "olive_mcp"]
    }
  }
}
```
</details>

<details>
<summary>Claude Desktop / Claude Code / Cursor / Windsurf</summary>

```json
{
  "mcpServers": {
    "olive": {
      "command": "uv",
      "args": ["run", "--directory", "/path/to/Olive/mcp", "python", "-m", "olive_mcp"]
    }
  }
}
```
</details>

---

## Usage with VS Code Copilot

VS Code Copilot is one of the easiest ways to use the Olive MCP server. Here's a step-by-step walkthrough:

### 1. Add the MCP Server

Create or edit `.vscode/mcp.json` in your project:

```json
{
  "servers": {
    "olive": {
      "type": "stdio",
      "command": "uv",
      "args": ["run", "--directory", "/path/to/Olive/mcp", "python", "-m", "olive_mcp"]
    }
  }
}
```

> Replace `/path/to/Olive/mcp` with your actual path to the cloned Olive repository's `mcp` directory.

### 2. Open Copilot Chat

Press `Ctrl+Alt+I` (Windows/Linux) or `Cmd+Alt+I` (Mac) to open the Copilot Chat panel.

### 3. Switch to Agent Mode

At the top of the Copilot Chat panel, switch from **Ask** or **Edit** mode to **Agent** mode. This enables Copilot to call external MCP tools.

### 4. Verify the Olive Tools

Click the **Tools** icon (🔧) in the chat input area to see the list of available tools. You should see all the Olive tools listed: `optimize`, `quantize`, `finetune`, `benchmark`, `detect_hardware`, and more.

### 5. Start Chatting

Simply type what you want to do. Copilot will select the appropriate Olive tools and ask for your confirmation before calling each one:

```
You: Optimize microsoft/Phi-4-mini-instruct for my machine

Copilot: I'll start by detecting your hardware...
         [Calls detect_hardware — Click "Allow" to confirm]

         You have an NVIDIA RTX 4090 with 24GB VRAM. Here are your options:
         1. **Smallest model (int4)** — ~2GB, fastest inference
         2. **Balanced (int8)** — ~4GB, good quality/speed tradeoff
         3. **Best quality (fp16)** — ~8GB, highest accuracy

You: Go with int4

Copilot: Starting int4 optimization with GPTQ...
         [Calls optimize — Click "Allow" to confirm]
         [Polls get_job_status and streams progress]

         ✅ Done!Model saved to ~/.olive-mcp/outputs/optimize-phi-4-mini-...
         Size: 7.6GB → 2.1GB (72% reduction)
```

### Tips for VS Code Copilot

- **Auto-allow tools**: If you trust the Olive tools, you can click the dropdown on the "Allow" button and select "Always allow" to skip confirmation for future calls.
- **Check progress**: If optimization is running, just ask *"What's the status?"* — Copilot will poll `get_job_status` for you.

---

## Available Tools

The MCP server exposes 9 tools and 4 guided prompts:

| Tool | What It Does |
|------|-------------|
| `optimize` | End-to-end optimization — ONNX export + quantization + graph optimizations |
| `quantize` | Weight-only quantization with RTN, GPTQ, AWQ, or HQQ |
| `finetune` | LoRA / QLoRA fine-tuning for text models and diffusion models |
| `capture_onnx_graph` | Export PyTorch models to ONNX format |
| `benchmark` | Evaluate model accuracy using lm-evaluation-harness |
| `detect_hardware` | Auto-detect CPU, GPU, RAM, and disk for smart defaults |
| `get_job_status` | Long-poll for real-time progress with phase detection |
| `cancel_job` | Cancel a running background job |
| `manage_outputs` | Browse and clean up previous optimization results |

---

## Example Conversations

Here are some things you can say to your AI assistant once the Olive MCP server is connected:

### Optimize a model

> **You:** Optimize Phi-4-mini for my machine
>
> The assistant calls `detect_hardware` to check your setup, then offers plain-language options like *"Make it as small as possible (int4)"* or *"Balance size and quality (int8)"*. One click, and it runs.

### Quantize with a specific algorithm

> **You:** Quantize microsoft/Phi-3-mini-4k-instruct to int4 using GPTQ

> The assistant calls `quantize(model_name_or_path="microsoft/Phi-3-mini-4k-instruct", algorithm="gptq", precision="int4")` and streams progress in real time.

### Fine-tune on your data

> **You:** Fine-tune Phi-4-mini on the nampdn-ai/tiny-codes dataset

> The assistant selects LoRA or QLoRA based on your available GPU memory, sets up training, and reports results.

### Train a Stable Diffusion LoRA

> **You:** Train a LoRA for stable-diffusion-v1-5 with dataset linoyts/Tuxemon

> The assistant detects this is a diffusion model and automatically routes to the specialized diffusion LoRA pipeline with DreamBooth support.

### Explore available passes

> **You:** What optimization passes are available for int4 quantization on CPU?

> The assistant calls `explore_passes` — an internal tool backed by `olive_config.json` — and returns a filtered list with descriptions for each pass.

---

## How It Works Under the Hood

The architecture has four layers, each handling a specific concern:

```
┌─────────────────────────────────────────────────┐
│  AI Client (VS Code Copilot, Claude, Cursor)    │
│  ↕ MCP Protocol (stdio)                        │
├─────────────────────────────────────────────────┤
│  Olive MCP Server                               │
│  tools.py — tool definitions & validation       │
│  jobs.py  — async job lifecycle & progress       │
├─────────────────────────────────────────────────┤
│  Isolated Virtual Environments (uv)             │
│  packages.py — resolve deps from olive_config   │
│  venv.py     — create/cache/reuse venvs         │
├─────────────────────────────────────────────────┤
│  Worker Process (runs inside venv)               │
│  worker.py — dispatches olive CLI commands       │
│  olive-ai[cpu,gpu,...] + extra dependencies      │
└─────────────────────────────────────────────────┘
```

### Isolated Environments

Different optimization tasks require different packages. Quantizing with AWQ needs `autoawq`, fine-tuning with QLoRA needs `bitsandbytes`, and GPU inference requires `onnxruntime-gpu` instead of `onnxruntime`. Installing all of these in one environment leads to conflicts.

The MCP server solves this by creating **isolated virtual environments** on-the-fly for each task. When you ask to quantize a model with AWQ, the server:

1. **Resolves dependencies** — reads `olive_config.json` to determine which packages the AWQ pass needs (e.g., `autoawq`, `datasets`)
2. **Creates or reuses a cached venv** — hashes the package list to a 12-character key; if a venv with that key already exists, it's reused instantly
3. **Runs the worker** — launches a subprocess inside the venv that executes the actual Olive command
4. **Cleans up** — venvs not used for 14 days are automatically deleted

This means `onnxruntime` (CPU) and `onnxruntime-gpu` (CUDA) never conflict, and you don't have to manage any environments manually.

### Async Job Pattern

Model optimization can take minutes to hours. The MCP server runs all heavy work in the background:

1. You ask to optimize a model
2. The tool returns immediately with a `job_id`
3. The assistant polls `get_job_status(job_id)` — this call **long-polls** for up to 30 seconds, waiting for new output before returning
4. Phase detection extracts structured progress from Olive's raw logs: *downloading → loading → quantizing → saving*
5. When the job completes, results include pass summaries, file sizes, and output paths

### Smart Error Recovery

When a job fails, the server doesn't just show a traceback. It pattern-matches the error against known failure modes and returns **actionable suggestions**:

| Error Pattern | Suggestion |
|---------------|-----------|
| Out of memory | "Try int4 precision or a smaller model" |
| CPU + fp16 | "CPU does not support fp16. Use int8 or int4" |
| 401 / 403 / gated | "This model requires a HuggingFace token" |
| Disk full | "Free up disk space or change output directory" |
| CUDA out of memory | "Reduce batch size or use QLoRA instead of LoRA" |

---

## Guided Prompts

For users who aren't sure where to start, the server provides four **guided prompts** — pre-built conversation starters that walk you through a complete workflow:

| Prompt | What It Does |
|--------|-------------|
| `optimize-model` | Detects hardware → asks your use case → recommends model + settings → runs optimization |
| `quantize-model` | Detects hardware → picks precision and algorithm → quantizes → shows size reduction |
| `finetune-model` | Checks GPU memory → asks about your data → picks LoRA or QLoRA → trains → suggests next steps |
| `compare-models` | Lists past runs → summarizes each → recommends the best one → suggests benchmarking |

These prompts are available in MCP clients that support them (e.g., Claude Desktop).

---

## Try It Now

```bash
git clone https://github.com/microsoft/Olive.git
cd Olive/mcp && uv sync
```

Add the server to your AI client, open a chat, and say:

> *"Optimize microsoft/Phi-4-mini-instruct for my machine"*

That's it. The AI handles the rest.

---

## Related Resources

- [Olive GitHub Repository](https://github.com/microsoft/Olive)
- [Model Context Protocol Specification](https://modelcontextprotocol.io/)
