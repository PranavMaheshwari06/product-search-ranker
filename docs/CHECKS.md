# Gates

Every phase: the base gate, then the phase gate. A phase is done only when both pass; paste the
last lines of the output into the commit summary.

Base gate:
```bash
uv sync
uv run ruff check .
uv run ruff format --check .
uv run pytest -q
```

| Phase | Phase gate |
|---|---|
| P0 | `uv run psr --fixtures prepare` runs; `uv run psr fetch && uv run psr prepare` on real data; counts in `docs/data_card.md` within 0.5% of SPEC 4.2; `tests/fixtures/esci_sample.parquet` < 2 MB, 4 labels, 2 locales; test: no query in two splits |
| P1 | tests: NDCG@10 and MRR on 3 hand-computed queries, IDCG=0 exclusion, bootstrap CI contains the true mean on a synthetic case; `uv run psr eval --systems bm25 --split test` writes `reports/metrics.json` with us, es, all |
| P2 | `logs/embed.log` shows completion (or the Decision-log subset); `uv run psr eval --systems dense,hybrid` adds rows with CIs |
| P3 | leakage tests pass; `mlruns/` has the grid runs; `uv run psr eval --systems ltr` adds the row and `comparisons` (ltr vs bm25, ltr vs hybrid); `reports/ltr_importance.png` exists |
| P4 | replay tests for `client.py` and `query_understanding.py`; `uv run psr llm-ping` prints model and tokens; `reports/qu.json` has the slice comparison |
| P5 | kappa equals `sklearn` on a test case; `uv run psr judge && uv run psr agreement` write `reports/judge.json` and `reports/judge_confusion.png`; n_missing reported; **STOP** |
| P6 | gate tests (pass, fail, first run); `docker compose -f deploy/compose.yaml build`; `docker compose -f deploy/compose.yaml run --rm airflow airflow dags test psr_daily 2026-01-01` succeeds with `PSR_DATA_MODE=fixtures`; `terraform -chdir=deploy/terraform/aws fmt -check`, `init -backend=false`, `validate` |
| P7 | API tests (TestClient, fixture demo index): `/search` returns k items, `/explain` returns one value per feature, `/metrics` has the metric names; `docker run` of the API image answers `curl localhost:8000/search?q=usb+cable` |
| P8 | `grep` of README and `docs/` for numbers: each one appears in a `reports/*.json`; no banned words (CLAUDE.md); UNVERIFIED list complete |

## Routing per phase
Main session: plans, hard code, commits, STOP lines. `impl` = Sonnet subagent for self-contained
tasks. Every phase also gets `ecc:python-reviewer` and `ecc:silent-failure-hunter`, and `gatecheck`
(Haiku) runs the gates.

| Phase | Main session (Opus) | Delegate to `impl` | Extra reviewer |
|---|---|---|---|
| P2 dense + hybrid | embedding job, dense and hybrid retrieval | CLI wiring, RRF function | `ecc:rag-pipeline-reviewer` |
| P3 LTR + MLflow | features, leakage test, grid | importance plot, grid runner | `ecc:mle-reviewer` |
| P4, P5 real LLM runs | model choice, runs, STOP | none | none |
| P6 Docker, Airflow, Terraform | plan | Dockerfile, DAG, terraform files | `ecc:security-reviewer` |
| P7 API, UI | plan | endpoints, UI | `ecc:fastapi-reviewer`, `ecc:security-reviewer` |
| P8 docs | numbers from `reports/*.json` | README, notes, memo drafts | `ecc:comment-analyzer` |
