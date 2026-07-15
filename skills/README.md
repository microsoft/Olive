# Olive Agent Skills

This directory contains portable [Agent Skills](https://agentskills.io/) for AI assistants.

| Skill | Purpose |
| --- | --- |
| [`olive`](olive/SKILL.md) | Optimize, quantize, fine-tune, convert, and benchmark models with Microsoft Olive. |

## Install

With GitHub CLI 2.90.0 or later:

```shell
gh skill install microsoft/Olive olive
```

For a manual installation, copy the complete `skills/olive` directory to one of the locations supported by
your AI assistant:

- Personal GitHub Copilot skill: `~/.copilot/skills/olive`
- Project GitHub Copilot skill: `.github/skills/olive`
- Cross-agent project skill: `.agents/skills/olive`
- Claude project skill: `.claude/skills/olive`

The skill orchestrates the tools exposed by the Olive MCP server. Configure the server by following
[`mcp/README.md`](../mcp/README.md) or the bundled
[`olive/references/mcp-setup.md`](olive/references/mcp-setup.md).
