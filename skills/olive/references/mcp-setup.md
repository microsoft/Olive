# Olive MCP Server Setup

The Olive skill supplies orchestration instructions; the Olive MCP server supplies the executable tools.

## Prerequisites

- Python 3.10 or later
- [uv](https://docs.astral.sh/uv/)
- Network access when downloading remote models or creating a new isolated environment

## Install the server

```shell
git clone https://github.com/microsoft/Olive.git
cd Olive/mcp
uv sync
```

Use this server definition, replacing the directory with the absolute path to the cloned `Olive/mcp`
directory:

```json
{
  "command": "uv",
  "args": [
    "run",
    "--directory",
    "/absolute/path/to/Olive/mcp",
    "python",
    "-m",
    "olive_mcp"
  ]
}
```

In JSON on Windows, escape backslashes:

```json
"C:\\absolute\\path\\to\\Olive\\mcp"
```

## Configure the AI client

| Client | Configuration file | Container key |
| --- | --- | --- |
| VS Code with Copilot | `.vscode/mcp.json` | `servers.olive` |
| Claude Desktop | `%APPDATA%\Claude\claude_desktop_config.json` on Windows | `mcpServers.olive` |
| Claude Code | `.mcp.json` in the project root | `mcpServers.olive` |
| Cursor | `.cursor/mcp.json` | `mcpServers.olive` |
| Windsurf | `~/.codeium/windsurf/mcp_config.json` | `mcpServers.olive` |

VS Code example:

```json
{
  "servers": {
    "olive": {
      "type": "stdio",
      "command": "uv",
      "args": [
        "run",
        "--directory",
        "/absolute/path/to/Olive/mcp",
        "python",
        "-m",
        "olive_mcp"
      ]
    }
  }
}
```

Other MCP clients:

```json
{
  "mcpServers": {
    "olive": {
      "command": "uv",
      "args": [
        "run",
        "--directory",
        "/absolute/path/to/Olive/mcp",
        "python",
        "-m",
        "olive_mcp"
      ]
    }
  }
}
```

Restart or reload the AI client after changing its MCP configuration. Verify that these tools are visible:

`detect_hardware`, `optimize`, `quantize`, `finetune`, `capture_onnx_graph`, `benchmark`,
`get_job_status`, `cancel_job`, and `manage_outputs`.

The server creates isolated Python environments under `~/.olive-mcp/venvs/` and saves results under
`~/.olive-mcp/outputs/`. A Hugging Face token passed to a model operation is forwarded to the worker through
an environment variable and is not persisted in job metadata or output.
