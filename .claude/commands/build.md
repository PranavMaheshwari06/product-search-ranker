---
description: Build the next unfinished chunk from STATE.md, then stop
---

1. Read `CLAUDE.md`, `STATE.md`, `docs/SPEC.md`, `docs/CHECKS.md`. Find the first chunk in the
   chunk table whose phases are not all checked. If an earlier phase is unchecked or a STOP is
   open, say so and stop.
2. Run the base gate from `docs/CHECKS.md` (skip in chunk 1 before the scaffold exists). If it
   fails, fix that first and log it in `STATE.md`.
3. Plan (main session, `agent-skills:planning-and-task-breakdown`): for each phase of the chunk,
   list the tasks from SPEC section 14, mark which go to `impl` (self-contained, fully specified)
   and which stay in the main session, and name the reviewers from the routing table in
   `docs/CHECKS.md`. Give the plan and **stop for the user's OK before any code**.
4. For each phase of the chunk, in order:
   - Build it with `agent-skills:incremental-implementation` (and
     `agent-skills:test-driven-development` when behaviour changes). Follow the SPEC sections the
     phase points to. When SPEC is unclear or wrong, choose the simplest option that keeps its
     intent, write a Decision log line, and carry on.
   - Start long jobs in the background (CLAUDE.md) while writing the next piece of code. Before a
     GPU job, tell the user and wait for a go (the job checks free VRAM itself).
   - Review: `ecc:python-reviewer` and `ecc:silent-failure-hunter` on the changed files, plus the
     phase reviewers from the routing table; run them with a Sonnet model. Fix real findings.
   - Run the base gate and the phase gate through the `gatecheck` subagent; fix until they pass.
     Anything that cannot run here goes to UNVERIFIED with how to check it.
   - Update `STATE.md`: tick the phase, set "Next", add result rows from `reports/*.json`.
   - `git add -A && git commit -m "<phase>: <what>" && git push` (main session only).
5. Stop at the end of the chunk, or at a STOP line. Print a summary: phases done, gate output
   (last lines), numbers written to reports, new Decision log, UNVERIFIED and Blocker lines,
   which model `STATE.md` names for the next chunk.
