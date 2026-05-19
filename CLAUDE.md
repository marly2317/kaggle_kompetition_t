# CLAUDE.md

Behavioral guidelines to reduce common LLM coding mistakes. Based on [andrej-karpathy-skills](https://github.com/multica-ai/andrej-karpathy-skills). Merge with project-specific instructions below.

**Tradeoff:** These guidelines bias toward caution over speed. For trivial tasks, use judgment.

## 1. Think Before Coding

**Don't assume. Don't hide confusion. Surface tradeoffs.**

Before implementing:
- State your assumptions explicitly. If uncertain, ask.
- If multiple interpretations exist, present them - don't pick silently.
- If a simpler approach exists, say so. Push back when warranted.
- If something is unclear, stop. Name what's confusing. Ask.

## 2. Simplicity First

**Minimum code that solves the problem. Nothing speculative.**

- No features beyond what was asked.
- No abstractions for single-use code.
- No "flexibility" or "configurability" that wasn't requested.
- No error handling for impossible scenarios.
- If you write 200 lines and it could be 50, rewrite it.

Ask yourself: "Would a senior engineer say this is overcomplicated?" If yes, simplify.

## 3. Surgical Changes

**Touch only what you must. Clean up only your own mess.**

When editing existing code:
- Don't "improve" adjacent code, comments, or formatting.
- Don't refactor things that aren't broken.
- Match existing style, even if you'd do it differently.
- If you notice unrelated dead code, mention it - don't delete it.

When your changes create orphans:
- Remove imports/variables/functions that YOUR changes made unused.
- Don't remove pre-existing dead code unless asked.

The test: Every changed line should trace directly to the user's request.

## 4. Goal-Driven Execution

**Define success criteria. Loop until verified.**

Transform tasks into verifiable goals:
- "Add validation" → "Write tests for invalid inputs, then make them pass"
- "Fix the bug" → "Write a test that reproduces it, then make it pass"
- "Refactor X" → "Ensure tests pass before and after"

For multi-step tasks, state a brief plan:
```
1. [Step] → verify: [check]
2. [Step] → verify: [check]
3. [Step] → verify: [check]
```

Strong success criteria let you loop independently. Weak criteria ("make it work") require constant clarification.

---

## Project: Titanic ML pipeline

Read `AGENTS.md` for repo layout and commands. Read `docs/notes.md` for experiment history and what **not** to reintroduce. For behavioral examples (what not to do), see `docs/agent-guidelines/EXAMPLES.md`.

### Architecture

- **Configs:** `configs/project.yaml` + `configs/experiments/<name>.yaml` (optional `parent` chain via `src/config.py`).
- **Train:** `python -m src.main fit` → CV in `src/train_functions.py`, artifacts in `models/<experiment>/`, reports in `reports/<experiment>/`.
- **Submit:** `python -m src.main submit` → fold ensemble in `src/inference.py`.
- **Features:** `src/features.py` — fold-level `fit_preprocessing_artifacts` / `apply_preprocessing_artifacts` only (no global fit on full train+valid).

### Non-negotiables

- Preprocessing and target-derived stats: **fit on train fold only**, apply to valid/test.
- Prefer stable OOF / fold std over single lucky split; check `docs/notes.md` before adding high-cardinality or train+test global stats.
- New metrics go in `src/metrics.py`; wire through train reports if needed.
- Do not add git `Co-authored-by` trailers or agent attribution to commits unless the user asks.

### Verification

- After code changes: `python -m pytest tests/ -q`
- Smoke train path when touching pipeline: `python -m src.main fit --experiment 001_best_solution`

---

**These guidelines are working if:** fewer unnecessary changes in diffs, fewer rewrites due to overcomplication, and clarifying questions come before implementation rather than after mistakes.
