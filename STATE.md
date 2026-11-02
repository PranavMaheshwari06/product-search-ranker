# Build state

Read this first in every session. Update it before every commit.

Current stage: **C** (cloud sessions: code, tests, real-data runs, Docker, `terraform validate`)
Next: **chunk 1 (P0)**

## Chunks (one cloud session per chunk; `/build` does the first unfinished one, then stops)
Sonnet: well-specified, self-contained work. Opus: work that has to hold many earlier modules
in mind at once. Pick the newest model of that tier.

| Chunk | Phases | Model | Done when |
|---|---|---|---|
| 1 | P0 | Sonnet | scaffold, CI, real data fetched and prepared, counts match SPEC 4.2, fixture committed |
| 2 | P1 | Opus | metrics + stats tested by hand examples; BM25 row in `reports/metrics.json` (real test split) |
| 3 | P2, P3 | Opus | dense, hybrid, ltr rows with CIs; leakage test; MLflow runs |
| 4 | P4, P5 | Opus | real Gemini judge run, `reports/judge.json`, query-understanding slice table. **STOP** after: user reads `reports/judge.json` |
| 5 | P6 | Sonnet | Docker image builds; Airflow DAG test passes in Compose; `terraform validate` ok |
| 6 | P7, P8 | Sonnet | API + UI in Docker; README, design notes, decision memo from reports; UNVERIFIED complete |

Between chunks: new session, `/check`, then `/build`.

## Stage C
- [ ] P0 Scaffold + data
- [ ] P1 Metrics, statistics, BM25, eval runner
- [ ] P2 Dense retrieval + hybrid (RRF)
- [ ] P3 Learning-to-rank + MLflow
- [ ] P4 LLM client + query understanding
- [ ] P5 LLM judge vs human labels
- [ ] P6 Airflow DAG, Docker, Compose, Terraform (AWS)
- [ ] P7 API, UI, Lambda adapter, metrics endpoint
- [ ] P8 README, design notes, decision memo, UNVERIFIED complete

## Stage L (VS Code, personal laptop)
- [ ] L0 Clone, `uv sync`, gates green, work through UNVERIFIED
- [ ] L1 Only if the cloud Gemini key failed: real judge + query-understanding runs (user reviews)
- [ ] L2 `docker compose up`, one full DAG run, screenshot of the run history
- [ ] L3 AWS: **STOP**, user sets the budget; `terraform apply`, publish, live URL, 2-min video
- [ ] L4 Final README numbers; user reviews the repo before it goes public

## Results so far
Filled from `reports/metrics.json` after each run (system, locale, NDCG@10, 95% CI).

## Decision log
One line each: date, decision, reason.

## UNVERIFIED
Things written but not run. Each line: file, what is unverified, how to check. Delete the line
when it is confirmed or fixed.

## Blockers
Anything that stopped progress twice in a row, with what was tried.
