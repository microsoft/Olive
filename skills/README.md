# Olive Agent Skills

This directory contains portable [Agent Skills](https://agentskills.io/) for AI assistants.

| Skill | Purpose |
| --- | --- |
| [`olive`](olive/SKILL.md) | Use the native Olive CLI and YAML/JSON workflows to optimize AI models. |

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

The skill requires the `olive` command from the `olive-ai` Python package. It does not require an MCP
server.
