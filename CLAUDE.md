Design: `docs/SPEC.md`. Progress: `STATE.md` (read it first, update it before every commit).
Gates: `docs/CHECKS.md`. Workflow: `/check`, then `/build` (one chunk per session).

Rules
- Build only the next unfinished chunk in `STATE.md`, phase by phase. Stop at every STOP line and ask.
- Python 3.12, src layout (`src/product_search_ranker/`), uv. Plain functions and modules; a class
  only when it holds state. No base classes, factories or plugin registries. Few comments: say
  why, not what.
- Dependencies: only those in SPEC section 7. Adding one needs a Decision log line in `STATE.md`.
- Tests are offline: they use `tests/fixtures/` and the `replay` LLM provider, never the network.
- Numbers: README, design notes and `docs/decision_memo.md` cite only values in `reports/*.json`.
  Never invent or round up a result. A step that does not help is reported as such.
- Wording in repo files: no "production", "real-time", "state-of-the-art", no A/B-test claims
  (there are no users). Offline evaluation on a public dataset, said plainly.
- Data: ESCI, Apache-2.0. Keep the licence notice and the paper citation in README and
  `docs/data_card.md`. `data/`, `indexes/`, `models/`, `mlruns/`, `logs/` are git-ignored;
  `reports/` and `tests/fixtures/` are committed.
- Long jobs (downloads, embeddings, training, LLM runs): run in the background with
  `nohup ... > logs/<job>.log 2>&1 &`, poll the log, and make the job resumable (skip finished work).
- Secrets: never write keys to files or logs. In cloud sessions the Gemini key is added by the
  network proxy; locally it comes from `.env` (git-ignored).
- Anything written but not run (missing service, credentials, time) goes to UNVERIFIED in `STATE.md`.
- Commit per phase with a short imperative message, then `git push`.
