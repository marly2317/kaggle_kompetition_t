# Claude Code setup (this project)

Guidelines from [multica-ai/andrej-karpathy-skills](https://github.com/multica-ai/andrej-karpathy-skills), vendored into this repo for **Cursor** and **Claude Code**.

## What applies automatically

| Mechanism | File | Effect |
|-----------|------|--------|
| Project instructions | `CLAUDE.md` | Claude Code reads this in the project root (Karpathy + Titanic rules) |
| Plugin skill | `skills/karpathy-guidelines/SKILL.md` | Loaded when the plugin is installed (see below) |
| Examples | `docs/agent-guidelines/EXAMPLES.md` | Reference when behavior drifts (over-abstraction, drive-by edits) |

`CLAUDE.md` is the main per-project hook. Keep it in sync with `.cursor/rules/karpathy-guidelines.mdc` when you update upstream.

## Option A: Use this repo only (simplest)

Open the project folder in Claude Code. No extra install if `CLAUDE.md` is present.

Verify: ask Claude to follow "Simplicity First" on a small change — it should prefer minimal diffs and ask before assuming.

## Option B: Install the bundled plugin (recommended for Claude Code)

From the **project root** in Claude Code:

```text
/plugin marketplace add .
/plugin install andrej-karpathy-skills@karpathy-skills
```

This uses `.claude-plugin/` and `skills/karpathy-guidelines/` in this repository.

## Option C: Install from GitHub (all projects)

If you want the skill globally without vendoring:

```text
/plugin marketplace add multica-ai/andrej-karpathy-skills
/plugin install andrej-karpathy-skills@karpathy-skills
```

You still want `CLAUDE.md` in this repo for **Titanic-specific** rules (`docs/notes.md`, fold preprocessing, etc.).

## Cursor vs Claude Code

| Tool | Use |
|------|-----|
| **Cursor** | `.cursor/rules/karpathy-guidelines.mdc`, `titanic-ml-pipeline.mdc` — see `CURSOR.md` |
| **Claude Code** | `CLAUDE.md` + optional `.claude-plugin/` — this file |

Same principles; different wiring. Do not remove `CLAUDE.md` when using the plugin — it carries project rules the plugin does not include.

## Updating from upstream

When pulling changes from [andrej-karpathy-skills](https://github.com/multica-ai/andrej-karpathy-skills), update:

- `CLAUDE.md` (Karpathy section only — keep `## Project: Titanic ML pipeline`)
- `.claude-plugin/plugin.json`, `marketplace.json`
- `skills/karpathy-guidelines/SKILL.md`
- `.cursor/rules/karpathy-guidelines.mdc`
- `docs/agent-guidelines/EXAMPLES.md`
