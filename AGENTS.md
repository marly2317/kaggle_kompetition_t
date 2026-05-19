# AGENTS.md

Instructions for coding agents working in this repository.

## Behavior

Follow **[andrej-karpathy-skills](https://github.com/multica-ai/andrej-karpathy-skills)** (vendored into this repo):

| File | When to read |
|------|----------------|
| `CLAUDE.md` | Full 4 principles + Titanic project rules |
| `.cursor/rules/karpathy-guidelines.mdc` | Always on in Cursor |
| `.cursor/rules/titanic-ml-pipeline.mdc` | When editing `src/`, `configs/`, `tests/` |
| `docs/agent-guidelines/EXAMPLES.md` | Wrong vs right patterns (over-abstraction, drive-by refactors, vague goals) |
| `CURSOR.md` | How rules are wired in Cursor |
| `docs/agent-guidelines/CLAUDE_CODE.md` | Claude Code: `CLAUDE.md` + `.claude-plugin/` install |
| `.claude-plugin/` | Plugin manifest + marketplace for Claude Code |
| `skills/karpathy-guidelines/SKILL.md` | Skill used by the plugin; optional copy to `~/.cursor/skills/` |

## Repo map

| Path | Role |
|------|------|
| `src/main.py` | CLI: `fit`, `submit`, `all`, `tune` |
| `src/config.py` | Load/merge YAML configs |
| `src/data.py` | Raw CSV load |
| `src/features.py` | Feature engineering + fold preprocessing |
| `src/train_functions.py` | Stratified K-fold train, OOF, threshold, reports |
| `src/inference.py` | Fold ensemble + submission |
| `src/metrics.py` | `accuracy`, `roc_auc`, `recall`, `precision`, `f1` |
| `src/tuning.py` | Optuna search |
| `configs/project.yaml` | Shared defaults |
| `configs/experiments/*.yaml` | Per-experiment overrides |
| `docs/notes.md` | Experiment log — read before changing features |
| `tests/` | Pytest smoke and unit tests |

## Commands

```bash
python -m venv .venv
# Windows: .\.venv\Scripts\Activate.ps1
pip install -e ".[dev]"

python -m src.main fit --experiment 001_best_solution
python -m src.main submit --experiment 001_best_solution
python -m src.main all --experiment 001_best_solution
python -m pytest tests/ -q
```

Data files (not in git): `data/raw/train.csv`, `test.csv`, `gender_submission.csv`.

## Active baseline

- Experiment: `001_best_solution`
- See `README.md` and `docs/notes.md` for current metrics and rejected ideas.

## When changing code

1. State assumptions; prefer the smallest diff.
2. Keep preprocessing inside each CV fold (`prepare_fold_data`).
3. Add tests only when they cover real pipeline behavior.
4. Do not commit unless the user asks; never add `Co-authored-by: Cursor` unless requested.
