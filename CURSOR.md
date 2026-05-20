# Agent guidelines: setup & reuse

This repo ships a single set of Karpathy-style behavioral guidelines that work automatically in **Cursor**, **Claude Code**, and **GitHub Copilot**. Same principles, three entry points — each tool reads the file it knows.

Source: [multica-ai/andrej-karpathy-skills](https://github.com/multica-ai/andrej-karpathy-skills).

## What each tool reads

| Tool | Entry point | Trigger |
|------|-------------|---------|
| **Cursor** | `.cursor/rules/karpathy-guidelines.mdc` | `alwaysApply: true` — loaded on every request |
| **Cursor** (project rules) | `.cursor/rules/titanic-ml-pipeline.mdc` | `globs: src/**/*.py,configs/**/*.yaml` |
| **Claude Code** (per-project) | `CLAUDE.md` | read at session start in project root |
| **Claude Code** (plugin) | `skills/karpathy-guidelines/SKILL.md` via `.claude-plugin/` | loaded when the plugin is installed |
| **GitHub Copilot** (Chat + Coding Agent) | `.github/copilot-instructions.md` | read automatically in repos that contain it |
| All tools (reference) | `EXAMPLES.md` | worked anti-pattern examples linked from the principle files |

The four core principles are duplicated **verbatim** across `CLAUDE.md`, `.cursor/rules/karpathy-guidelines.mdc`, `.github/copilot-instructions.md`, and `skills/karpathy-guidelines/SKILL.md`. This is intentional: tools do not follow cross-file references reliably. Each file declares a sync header at the top listing the others — edit one, update all four.

## Verify it is working

- **Cursor:** Open the folder → **Settings → Rules** — both `karpathy-guidelines` and `titanic-ml-pipeline` should appear.
- **Claude Code:** Start a session in the repo root. Ask the agent to follow "Simplicity First" on a small change — it should prefer minimal diffs and surface assumptions before acting.
- **GitHub Copilot Chat:** Open Copilot Chat in VS Code; it picks up `.github/copilot-instructions.md` automatically. You can confirm via *Configure Code Generation Instructions* in the Copilot settings.
- **Claude Code plugin (optional):** From the repo root run `/plugin marketplace add .` then `/plugin install andrej-karpathy-skills@karpathy-skills`. The plugin loads `skills/karpathy-guidelines/SKILL.md`.

## Reusing this in a new project

The system is split into **portable** (carry to every project) and **project-specific** (rewrite per project).

**Portable — copy as-is:**

```
.github/copilot-instructions.md
.claude-plugin/marketplace.json
.claude-plugin/plugin.json
.cursor/rules/karpathy-guidelines.mdc
skills/karpathy-guidelines/SKILL.md
EXAMPLES.md
CURSOR.md
CLAUDE.md   (top half — the four principles)
```

**Project-specific — replace in the new project:**

```
CLAUDE.md                              ## Project: <name>  section
.github/copilot-instructions.md        ## Project: <name>  section
.cursor/rules/<project>-rules.mdc      Cursor project rules with appropriate globs
docs/notes.md                          experiment / decision log
README.md                              project handover
```

After copying, the only manual work is replacing the `## Project: ...` sections in `CLAUDE.md` and `.github/copilot-instructions.md` and writing a `.cursor/rules/<project>-rules.mdc` for editor-side hints.

## Syncing with upstream

When pulling updates from [multica-ai/andrej-karpathy-skills](https://github.com/multica-ai/andrej-karpathy-skills), refresh these four files (the `## Project: ...` section in `CLAUDE.md` and `.github/copilot-instructions.md` stays local):

- `CLAUDE.md` (principles section only)
- `.cursor/rules/karpathy-guidelines.mdc`
- `skills/karpathy-guidelines/SKILL.md`
- `.github/copilot-instructions.md` (principles section only)
- `EXAMPLES.md`

Do not overwrite project-specific Cursor rules (`titanic-ml-pipeline.mdc` in this repo) or the project section in `CLAUDE.md` / `.github/copilot-instructions.md` unless you intend to.
