# Build state

Read this first in every session. Update it before every commit.

Current stage: **C** (cloud sessions: code, tests, real-data runs, Docker, `terraform validate`)
Next: **chunk 3 (P2, P3)**

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
- [x] P0 Scaffold + data (repaired 2026-11-03: the `data/` ignore line had kept the
  `src/product_search_ranker/data/` package out of git; now `/data/`, package committed)
- [x] P1 Metrics, statistics, BM25, eval runner
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
Official test split, `psr eval --systems bm25 --split test`, 2026-11-03. No query was excluded
(every test query has a candidate with gain > 0).

| System | Locale | NDCG@10 | 95% CI | MRR | Queries |
|---|---|---|---|---|---|
| bm25 | us | 0.6997 | [0.6952, 0.7040] | 0.7567 | 8,956 |
| bm25 | es | 0.6772 | [0.6677, 0.6869] | 0.7569 | 2,417 |
| bm25 | all | 0.6949 | [0.6909, 0.6990] | 0.7568 | 11,373 |

Slices (bm25, all locales, NDCG@10): `negation` 0.5624 (1,143 queries) is the weakest;
`brand_in_query` 0.7235 (2,194) the strongest. Full table in `reports/metrics.json`.

Data (from `reports/data_stats.json`, `psr prepare` on real data, 2026-11-03). A fresh VM has no
`data/`: `uv run psr fetch && uv run psr prepare` rebuilds it in about 2 minutes (1.1 GB raw,
572 MB processed); the counts and data sha come out the same. Then `uv run psr index` builds
the 10 BM25 indexes (2 locales x 5 fields) in about 10 minutes (988 MB in `indexes/`), and
`uv run psr eval --systems bm25 --split test` takes about 2 minutes.

| Locale | Queries | Judgements |
|---|---|---|
| us | 29,844 | 601,354 |
| es | 8,049 | 218,774 |

Both match SPEC 4.2 exactly. Per-split counts are in `docs/data_card.md`.

## Decision log
One line each: date, decision, reason.

- 2026-11-03: Gains E=1.0, S=0.1, C=0.01, I=0.0 confirmed against esci-data: `ranking/train.py`
  (`esci_label2gain`) and `ranking/launch-predictions-task1.sh` (trec_eval
  `-m 'ndcg.1=0,2=0.01,3=0.1,4=1'`). Their baseline uses NDCG over the full list; ours is
  NDCG@10 (SPEC 3), so the numbers are not directly comparable.
- 2026-11-03: MRR is averaged over the same queries as NDCG (IDCG > 0); a query whose list has
  no Exact item counts as 0. RR looks at the whole candidate list, not only the top 10.
- 2026-11-03: `psr eval` writes per-query results to `data/runs/<split>/<system>.parquet`
  (git-ignored) so later evals can make paired comparisons with earlier systems. Comparisons are
  the pairs in `config.COMPARISONS`, per locale (us, es, all), each with `n_queries`.
  `metrics.json` holds the test split; other splits go to `metrics_<split>.json`. `--fixtures`
  writes to `data/fixtures_reports/`, never to `reports/`. `--candidate` is left to P6.
- 2026-11-03: BM25 uses the bm25s defaults (method `lucene`, k1 = 1.5, b = 0.75). `psr index`
  builds all five fields (`all`, `title`, `bullets`, `brand`, `desc`) now, so P3 has its
  features; the eval uses `all` only. Indexes are written to a `.tmp` dir and renamed, so an
  interrupted build is redone and a finished one is skipped.
- 2026-11-03: `prepare` stats queries now `ORDER BY`, so `reports/data_stats.json` keys come out
  in the same order on every run (values were already stable). The committed file is in the old
  order until the next `psr prepare`.

- 2026-11-03: `.gitignore` entries for `data/`, `indexes/`, `models/`, `mlruns/`, `logs/` are
  anchored with a leading `/`. The unanchored `data/` also matched the `data` source package, so
  `git add -A` silently skipped it. After a commit that adds a package, run `git ls-files src`.
- 2026-11-02: `torch` and `sentence-transformers` are not in `pyproject.toml` yet. Every wheel on
  the PyTorch CPU index is served from `download-r2.pytorch.org`, which this cloud session's
  network policy blocks (only `download.pytorch.org` is allowed), so `uv` cannot resolve them. They
  are first needed in P2 (chunk 3). The `[[tool.uv.index]]` entry for the CPU index is kept.
  Resolve in chunk 3: allow `download-r2.pytorch.org`, or decide to use the PyPI torch wheel.
- 2026-11-02: `shopping_queries_dataset_sources.csv` is fetched from `raw.githubusercontent.com`:
  the LFS media host answers 404 for it (it is a plain file, not LFS). The two parquet files come
  from the media host as SPEC 4.1 says; `data/MANIFEST.json` records the source of each.
- 2026-11-02: Validation split uses `((query_id * 2654435761) % 4294967296) // 65536 % 10 = 0`
  instead of DuckDB `hash()`, so the split does not change if DuckDB changes its hash function.
- 2026-11-02: `psr prepare` also writes `reports/data_stats.json` and renders
  `docs/data_card.md` from it, so the card's numbers come from a report. `--fixtures` writes to
  `data/fixtures_processed/`, never to `data/processed/`.
- 2026-11-02: The `fixture` is sampled from train, val and test (not only train and test) in
  proportion per locale, so the fixture has validation queries for the LTR smoke run.
- 2026-11-02: CI `ci.yml` has the lint and test steps only; the `psr --fixtures train` and
  `eval` smoke steps are added when P1 and P3 create those commands. The terraform job skips
  itself until `deploy/terraform/aws/*.tf` exists.

## UNVERIFIED
Things written but not run. Each line: file, what is unverified, how to check. Delete the line
when it is confirmed or fixed.

- `.github/workflows/ci.yml`: never run on GitHub (P1 added the fixture `prepare`, `index` and
  `eval --systems bm25` smoke steps; they pass locally). Check: read the Actions run for the
  push; the `uv sync --frozen` step also needs the CPU torch index once torch is added back.

## Blockers
Anything that stopped progress twice in a row, with what was tried.
