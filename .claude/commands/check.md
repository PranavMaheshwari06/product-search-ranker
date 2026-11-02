---
description: Check that STATE.md matches the repo before building the next chunk
---

Read `STATE.md`, `docs/SPEC.md` and `docs/CHECKS.md`, then check, without changing code:

1. Every ticked phase: its files from SPEC section 14 exist, and its phase gate still passes
   (re-run the cheap parts; for long jobs check that their outputs exist).
2. The base gate passes (skip if the scaffold does not exist yet).
3. `git status` is clean and the branch is pushed.
4. Data and models a ticked phase needs are present in this VM (`data/processed/`, `indexes/`,
   `models/`). A fresh cloud VM does not have them: list the commands that rebuild them
   (for example `uv run psr fetch && uv run psr prepare`), run them in the background, and say
   how long they take.
5. No open STOP, no Blocker without a note.

Print a short list of findings and then `Verdict: READY` or `Verdict: NOT READY` with what to
fix first. Fix only `STATE.md` inconsistencies yourself.
