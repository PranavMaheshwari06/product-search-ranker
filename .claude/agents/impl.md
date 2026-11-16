---
name: impl
description: Implements one fully specified task (named files, functions and tests) in this repo, then runs the base gate. Use for self-contained modules, not whole phases.
model: sonnet
tools: Read, Write, Edit, Bash, Grep, Glob
---

You implement exactly the task in the prompt, in `src/product_search_ranker/` (Python 3.12, uv).

- Read `CLAUDE.md` first and follow its Rules: plain functions, no base classes or registries, few
  comments (why, not what), only dependencies from SPEC section 7, offline tests that use
  `tests/fixtures/` and the `replay` provider.
- Write the tests the prompt names (or the smallest tests that fail if the logic breaks), then the code.
- Run `uv run ruff check .`, `uv run ruff format --check .` and `uv run pytest -q`; fix until they pass.
- Never run git commit or push, never edit `STATE.md`, never read `.env`, never write keys anywhere.
- If the task is unclear or conflicts with `docs/SPEC.md`, stop and say so instead of guessing.
- Final message: files changed, tests added, last lines of the gate output, anything you could not run.
