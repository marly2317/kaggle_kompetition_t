# Using agent guidelines in Cursor (this project)

Based on [andrej-karpathy-skills](https://github.com/multica-ai/andrej-karpathy-skills).

## What is wired in this repo

| Path | Role |
|------|------|
| `.cursor/rules/karpathy-guidelines.mdc` | Always-on Karpathy behavior (`alwaysApply: true`) |
| `.cursor/rules/titanic-ml-pipeline.mdc` | Project rules for `src/`, `configs/`, `tests/` |
| `AGENTS.md` | Repo map, commands, handover |
| `CLAUDE.md` | Full guidelines + Titanic-specific rules |
| `docs/agent-guidelines/EXAMPLES.md` | Wrong vs right examples (read when unsure) |
| `skills/karpathy-guidelines/SKILL.md` | Optional: copy to `~/.cursor/skills/` for all projects |

## Confirm in Cursor

1. Open this folder in Cursor.
2. **Settings → Rules** — you should see `karpathy-guidelines` and `titanic-ml-pipeline`.
3. Reload window if rules do not appear after pull.

## Claude Code vs Cursor

- **Cursor** uses `.cursor/rules/` (see above).
- **Claude Code** uses root `CLAUDE.md` + optional `.claude-plugin/` — see `docs/agent-guidelines/CLAUDE_CODE.md`.

## Keeping in sync with upstream

When updating from [multica-ai/andrej-karpathy-skills](https://github.com/multica-ai/andrej-karpathy-skills), sync at least:

- `CLAUDE.md`
- `.cursor/rules/karpathy-guidelines.mdc`
- `docs/agent-guidelines/EXAMPLES.md`
- `skills/karpathy-guidelines/SKILL.md`

Do not overwrite `titanic-ml-pipeline.mdc` or the project section in `CLAUDE.md` unless you intend to.
