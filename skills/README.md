# Olive Agent Skills

This directory contains portable [Agent Skills](https://agentskills.io/) for AI assistants.

| Skill | Purpose |
| --- | --- |
| [`olive`](olive/SKILL.md) | Use the native Olive CLI and YAML/JSON workflows to optimize AI models. |
| [`copy-fork-pr`](dev/copy-fork-pr/SKILL.md) | Copy a fork PR branch by number, push a destination-owned branch, and open an independent draft PR. |
| [`olive-pr-review`](dev/olive-pr-review/README.md) | Experimental private H01/H03 contributor review; provisional guidance, no approval or publication. |

## Install

With GitHub CLI 2.90.0 or later:

```shell
gh skill install microsoft/Olive olive
gh skill install microsoft/Olive copy-fork-pr
```

For a manual installation, copy the complete skill directory:

| Source | Personal Copilot | Project Copilot | Cross-agent project | Claude project |
| --- | --- | --- | --- | --- |
| `skills/olive` | `~/.copilot/skills/olive` | `.github/skills/olive` | `.agents/skills/olive` | `.claude/skills/olive` |
| `skills/dev/copy-fork-pr` | `~/.copilot/skills/copy-fork-pr` | `.github/skills/copy-fork-pr` | `.agents/skills/copy-fork-pr` | `.claude/skills/copy-fork-pr` |

The `olive` skill requires the `olive` command from the `olive-ai` Python package. The `copy-fork-pr` skill
requires Git and GitHub CLI with write access to the destination repository. Neither requires an MCP server.

The experimental `olive-pr-review` Skill needs approved Git/file reads and GitHub reads for PR inputs,
not Olive runtime dependencies or GitHub write access. See its [README](dev/olive-pr-review/README.md)
for trusted-directory registration, explicit loading, and private-output instructions.
