---
description: Build the next unfinished chunk from STATE.md, then stop
---

1. Read `CLAUDE.md`, `STATE.md`, `docs/SPEC.md`, `docs/CHECKS.md`. Find the first chunk in the
   chunk table whose phases are not all checked. If an earlier phase is unchecked or a STOP is
   open, say so and stop.
2. Run the base gate from `docs/CHECKS.md` (skip in chunk 1 before the scaffold exists). If it
   fails, fix that first and log it in `STATE.md`.
3. For each phase of the chunk, in order:
   - Build what SPEC section 14 lists for the phase, following the SPEC sections it points to.
     When SPEC is unclear or wrong, choose the simplest option that keeps the SPEC's intent,
     write a Decision log line, and carry on.
   - Start long jobs in the background (CLAUDE.md) while writing the next piece of code.
   - Run the base gate and the phase gate. Fix until they pass. Anything that cannot run here
     goes to UNVERIFIED with how to check it.
   - Update `STATE.md`: tick the phase, set "Next", add result rows from `reports/*.json`.
   - `git add -A && git commit -m "<phase>: <what>" && git push`.
4. Stop at the end of the chunk, or at a STOP line. Print a summary: phases done, gate output
   (last lines), numbers written to reports, new Decision log, UNVERIFIED and Blocker lines,
   which model `STATE.md` names for the next chunk.
