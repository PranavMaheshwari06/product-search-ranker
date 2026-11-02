# product-search-ranker: specification

## 1. Problem

A shop's search box gets short, messy queries ("running shoes men without laces", "funda
iphone 13 sin cordón"). Ranking the right products first is a learning-to-rank problem, and
every change to the ranker has to be measured before it ships. Human relevance labels are the
reference, but they are slow and expensive, so teams increasingly let an LLM label results
instead. Whether that LLM judge picks the same winner as humans is usually assumed, not checked.

This project builds a ranker step by step on Amazon's Shopping Queries dataset (ESCI), where
every query-product pair has a human label, and checks the LLM judge against those labels.

## 2. Goal and headline results

1. **Ranking table**: NDCG@10 with 95% bootstrap CIs on the official test split for `bm25`,
   `dense`, `hybrid`, `ltr`, `ltr_qu`, per locale (us, es) and per query slice; paired
   differences with CIs. File: `reports/metrics.json`.
2. **LLM judge study** (headline): agreement of an LLM judge with the human labels (Cohen's
   kappa, confusion matrix), and whether judge-based NDCG ranks the systems in the same order
   as human-based NDCG (Kendall tau, "same winner?", wrong-winner pairs). File:
   `reports/judge.json`.
3. **Pipeline**: a daily Airflow DAG that rebuilds, retrains, evaluates and refuses to publish
   a worse model (eval gate), and a search API with a "why ranked here" explanation, deployable
   to AWS Lambda with Terraform.

Every number in README and docs comes from these report files. Negative or null results are
reported as they are.

## 3. Task framing and metrics

- ESCI Task 1 is re-ranking: each query comes with its judged candidate products (about 20 for
  us, about 27 for es). A system scores each candidate; candidates are sorted by score, ties
  broken by `product_id` so runs are deterministic.
- Labels and gains: E (Exact) = 1.0, S (Substitute) = 0.1, C (Complement) = 0.01,
  I (Irrelevant) = 0.0. These are the KDD Cup 2022 Task 1 gains; chunk 2 checks them against
  the esci-data README / paper and writes the source into the Decision log.
- NDCG@10 per query = DCG@10 / IDCG@10 over that query's candidates, `log2(rank + 1)`
  discount. Queries with IDCG = 0 (no candidate with gain > 0) are excluded and counted
  (`n_excluded`). The mean over queries is the system score.
- MRR: reciprocal rank of the first Exact candidate (0 if none in the list).
- Recall@k (demo retrieval only, P7): share of a query's Exact products in the top k of
  full-index retrieval.
- Statistics (`evaluate/stats.py`): paired bootstrap over queries, 10,000 resamples, seed 0,
  percentile 95% CI for each system mean and for each system difference; paired sign-flip
  randomization test, 10,000 permutations, for the p-value of a difference. Always report
  `n_queries`.
- Slices (`text.py`, regexes for en and es), per query: `len_1_2`, `len_3_5`, `len_6_plus`
  (tokens), `negation` (en: without, no, not, non; es: sin, no, excepto), `number_or_unit` (digits, sizes,
  units such as oz, ml, gb, inch, cm, pack), `brand_in_query` (a candidate's brand string
  appears in the query).

## 4. Data

### 4.1 Source and fetch (`data/fetch.py`, `psr fetch`)
- Repository: github.com/amazon-science/esci-data, folder `shopping_queries_dataset/`.
  Files: `shopping_queries_dataset_examples.parquet`, `shopping_queries_dataset_products.parquet`,
  `shopping_queries_dataset_sources.csv`. Licence Apache-2.0.
- The parquet files are Git LFS. Download URL pattern:
  `https://media.githubusercontent.com/media/amazon-science/esci-data/main/shopping_queries_dataset/<file>`.
  Check the first bytes: an LFS pointer text file means the URL is wrong. Fallback if
  media.githubusercontent.com is unreachable: Hugging Face `tasksource/esci` (pre-joined, same
  rows); record which source was used in `data/MANIFEST.json` and the Decision log.
- Stream to `data/raw/` with a progress line in the log; write `data/MANIFEST.json` with URL,
  bytes, sha256, download date. `psr fetch --if-changed` skips when size and sha256 match.
- Citation (README and data card): Reddy et al., "Shopping Queries Dataset: A Large-Scale ESCI
  Benchmark for Improving Product Search", arXiv:2206.06588, 2022.

### 4.2 Prepare (`data/prepare.py`, `psr prepare`)
- DuckDB reads the parquet files directly (do not load the full products file into pandas).
- Keep `small_version == 1` and `product_locale in ('us', 'es')`. Japanese is left out: BM25
  needs a Japanese tokenizer, and two locales are enough to show cross-language behaviour.
  README states this.
- Join examples to products on `(product_id, product_locale)`.
- Splits: the official `split` column gives train and test. Validation = train queries whose
  `xxhash64(query_id) % 10 == 0` (or the DuckDB `hash()` equivalent; stable across runs), so a
  query never appears in two splits. Test it.
- Columns kept: `query_id, query, locale, product_id, label (E/S/C/I), split, product_title,
  product_brand, product_color, product_bullet_point, product_description`. Text fields: strip
  HTML tags, collapse whitespace, keep raw length.
- Output: `data/processed/{train,val,test}.parquet`, `data/processed/products_{us,es}.parquet`
  (unique products in the kept rows).
- Expected counts for `small_version` (from the esci-data README): us 29,844 queries /
  601,354 judgements; es 8,049 queries / 218,774 judgements. **STOP** if the prepared counts
  differ by more than 0.5%. Record actual split counts in `docs/data_card.md`.

### 4.3 Fixture (`data/fixture.py`, `psr make-fixture`)
- `tests/fixtures/esci_sample.parquet`: 200 queries (150 us, 50 es) sampled with seed 0 from
  train and test in proportion, all candidates of those queries, all four labels present, text
  fields truncated to 300 characters. Target size under 2 MB. Committed.
- Every command takes `--fixtures`: it reads this file instead of `data/processed/` and uses
  the `replay` LLM provider. CI and tests run only in this mode.

### 4.4 Data card (`docs/data_card.md`)
Source, licence, citation, filters, split rule, counts per split and locale, label
distribution, known issues found during prepare (empty titles, duplicate pairs, very long
bullets).

## 5. Systems

### 5.1 `bm25` (`retrieval/bm25.py`)
- `bm25s`, one index per locale over the locale's products. Document text = title + brand +
  color + bullet points (description left out of the main index: long and noisy; it gets its
  own field index for LTR features).
- Tokenizer (`text.py`): lowercase, Unicode normalise, split on non-alphanumerics, keep
  numbers, stopwords for en (bm25s built-in) and a short es list in `text.py`. No stemming.
- Scoring candidates: compute the query's scores over the index and read the candidates'
  positions. Indexes saved to `indexes/bm25_<locale>_<field>/`.

### 5.2 `dense` (`retrieval/dense.py`)
- `intfloat/multilingual-e5-small` via sentence-transformers on CPU. E5 prefixes: `"query: "`
  and `"passage: "`. Passage = title + brand + color + first 400 characters of bullets;
  `max_seq_length = 256`; normalised embeddings stored as float16 `.npy` plus an id file in
  `indexes/dense_<locale>/`.
- Run as a resumable background job (`psr embed`), saving every 20k products. Measure
  throughput on the first 2,000 products and log the time estimate. If the full product set
  would take more than 2.5 hours on the VM, embed products of val + test + a 30% query sample of
  train, and write that to the Decision log.
- Score = cosine(query, product).

### 5.3 `hybrid` (`retrieval/hybrid.py`)
Reciprocal rank fusion of the bm25 and dense ranks within each query's candidates, `k = 60`.

### 5.4 `ltr` (`rank/features.py`, `rank/ltr.py`)
- LightGBM `objective = lambdarank`, labels I=0, C=1, S=2, E=3 with
  `label_gain = [0, 0.01, 0.1, 1.0]` so training optimises the same gains as the metric;
  `eval_at = [10]`; groups = queries.
- Features per (query, product), each with a one-line docstring in `features.py`:
  `bm25_all, bm25_title, bm25_bullets, bm25_brand, bm25_desc, bm25_rank, dense_cos, dense_rank,
  rrf_score, title_jaccard, query_tokens_in_title, brand_in_query, color_in_query,
  number_match, title_len, has_description, n_bullets, query_len, n_candidates, locale_es`.
  Nothing derived from labels. Test: no feature column name contains `label` or `esci`, and
  with shuffled training labels val NDCG drops to the random-order level (fixture).
- Training: train split; early stopping on val (100 rounds); small grid on val only
  (`num_leaves` 31/63, `learning_rate` 0.05/0.1, `min_data_in_leaf` 50/200); best config
  retrained once with the same seed. Test split touched only by `psr eval`.
- MLflow file store `./mlruns`, experiment `ltr`: params, val NDCG@10, feature list, git sha,
  importance plot. Best model: `models/ltr.txt` + `models/ltr_meta.json` (features, params,
  val score, data manifest sha).
- `reports/ltr_importance.png` (gain importance).

### 5.5 `ltr_qu` (`llm/query_understanding.py`)
- Hard queries = queries in the `negation`, `number_or_unit` or `brand_in_query` slices.
- One LLM call per batch of 20 queries returns a JSON array, one object per query:
  `{"query_id", "must_not": [str], "brand": str|null, "size": str|null, "color": str|null,
  "product_type": str|null}` validated with pydantic; invalid items are retried once, then
  left empty and counted.
- Extra features: `must_not_in_title`, `qu_brand_match`, `qu_size_match`, `qu_color_match`,
  `qu_type_in_title`; neutral value (0, plus a `qu_present` flag) for queries without output.
- Coverage: all hard queries in val and test, up to 3,000 hard queries from train (seed 0).
  Cached in `data/llm_cache/qu.jsonl` (append-only, resumable).
- Reported on the hard-query slice only: `ltr` vs `ltr_qu` paired difference with CI, number
  of LLM calls, tokens, cost per 1k queries when prices are set (6.1).

## 6. LLM layer and judge

### 6.1 Client (`llm/client.py`)
- One function `complete_json(prompt, schema, provider=None) -> (obj, meta)`; providers:
  - `replay`: reads `tests/fixtures/replay/<sha256 of prompt>.json`; missing file = test error
    that names the hash. Used by tests and CI.
  - `gemini`: plain `httpx` POST to
    `https://generativelanguage.googleapis.com/v1beta/models/<model>:generateContent`,
    `generationConfig = {temperature: 0, responseMimeType: "application/json"}`. Sends
    `x-goog-api-key` only if `GEMINI_API_KEY` is set; in cloud sessions the network proxy adds
    it. Model name from config (`PSR_GEMINI_MODEL`); chunk 4 lists models
    (`GET /v1beta/models`), picks the current Flash model, and logs its exact name and the date.
  - `ollama`: `POST http://localhost:11434/api/chat` with `format: "json"` (Stage L option).
- `meta`: provider, model, input/output tokens (from `usageMetadata`), latency, retries.
- Rate limit: config requests per minute (default 10), retry 429/5xx with backoff
  (5, 15, 45 s), then fail the item, not the run.
- `--record`: save real responses as replay fixtures for the fixture queries.
- Prices: `PSR_PRICE_IN_PER_M` and `PSR_PRICE_OUT_PER_M` in config, default unset. Reports show
  tokens always and cost only when the user has set prices. Never guess a price.
- `psr llm-ping`: one tiny call, prints model and token counts. First step of chunk 4.

### 6.2 Judge (`judge/sample.py`, `judge/judge.py`)
- Sample: 150 test queries (100 us, 50 es), seed 0, stratified by query length bucket. All
  candidates of each query (about 3,300 pairs).
- One call per query: query + list of candidates (product_id, title, brand, color, bullets
  truncated to 600 characters) + the four label definitions from the ESCI paper; returns
  `[{"product_id", "label"}]`. Validation: every candidate exactly once; retry once; else the
  missing pairs are counted, not filled.
- Batch-effect check: 30 of the queries also judged one pair per call; report agreement
  between batched and single-pair labels.
- Cache: `data/llm_cache/judge.jsonl` (resumable). Labels copied to `reports/judge_labels.parquet`
  (small, committed).

### 6.3 Agreement (`judge/agreement.py`, `psr agreement`)
- Pair level: accuracy, Cohen's kappa (4-class) and Exact-vs-rest kappa, each with a 95% CI by
  query bootstrap; unit test checks kappa against `sklearn.metrics.cohen_kappa_score`;
  row-normalised confusion matrix (`reports/judge_confusion.png`); per locale.
- System level: on the judged queries, NDCG@10 of each system with human labels and with
  judge labels; Kendall tau between the two system orders (`scipy.stats.kendalltau`);
  `same_winner` (bool); for every system pair, whether the judge gets the sign of the
  difference right, counted separately for pairs where the human-label CI excludes 0.
- 20 disagreement examples (largest gain gap) with query, title, human label, judge label.
- Output `reports/judge.json`: model, date, n_queries, n_pairs, n_missing, tokens, latency
  p50/p95, cost if prices set, pair-level metrics, system-level table, batch-effect result,
  examples.

## 7. Dependencies

Runtime: `duckdb`, `pyarrow`, `pandas`, `numpy`, `scipy`, `scikit-learn`, `bm25s`,
`sentence-transformers`, `torch` (CPU wheel from `https://download.pytorch.org/whl/cpu` via a
`[[tool.uv.index]]` entry with `explicit = true`), `lightgbm`, `mlflow`, `httpx`, `pydantic`,
`matplotlib`, `fastapi`, `uvicorn`, `prometheus-client`, `mangum`, `boto3`.
Dev: `pytest`, `ruff`. CLI: stdlib `argparse`.
Airflow is not a project dependency: it lives only in `deploy/docker/Dockerfile.airflow`
(official image + this package).

Not used, on purpose: an ORM, a vector database (embeddings are `.npy`), LangChain or other
agent frameworks, React (one static HTML page), Kubernetes, Grafana, a drift library, typer/click.

## 8. Layout

```
src/product_search_ranker/
  config.py            paths, gains, seeds, locales, model names, env overrides
  cli.py               argparse subcommands (section 9)
  text.py              tokenizer, html strip, slice regexes (en, es)
  data/fetch.py  data/prepare.py  data/fixture.py
  retrieval/bm25.py  retrieval/dense.py  retrieval/hybrid.py
  rank/features.py  rank/ltr.py
  evaluate/metrics.py  evaluate/stats.py  evaluate/run.py  evaluate/gate.py
  llm/client.py  llm/query_understanding.py
  judge/sample.py  judge/judge.py  judge/agreement.py
  pipeline/publish.py
  serve/app.py  serve/demo_index.py  serve/lambda_handler.py  serve/static/index.html
dags/psr_daily.py
deploy/docker/Dockerfile.api  Dockerfile.airflow  Dockerfile.lambda
deploy/compose.yaml
deploy/terraform/aws/  main.tf variables.tf outputs.tf versions.tf
tests/  fixtures/esci_sample.parquet  fixtures/replay/*.json  test_*.py
reports/  docs/  .github/workflows/ci.yml
```

## 9. CLI (`psr`, entry point in `pyproject.toml`)

`fetch [--if-changed]`, `prepare`, `make-fixture`, `index`, `embed`, `train`,
`eval --systems a,b --split test [--candidate]`, `llm-ping`, `qu`, `judge`, `agreement`,
`gate`, `publish --target s3://bucket/prefix|local:<dir> [--dry-run]`, `build-demo`, `serve`.
Global `--fixtures`. Each command prints what it wrote.

## 10. Report formats

`reports/metrics.json` (`psr eval` merges into it, never drops other systems):
```json
{"updated": "<iso date>", "git_sha": "...", "data_sha": "...", "split": "test",
 "gains": {"E": 1.0, "S": 0.1, "C": 0.01, "I": 0.0},
 "systems": {"bm25": {"us": {"ndcg@10": 0.0, "ci95": [0.0, 0.0], "mrr": 0.0,
                             "n_queries": 0, "n_excluded": 0}, "es": {}, "all": {}}},
 "comparisons": [{"a": "ltr", "b": "bm25", "locale": "all", "delta": 0.0,
                  "ci95": [0.0, 0.0], "p_value": 0.0}],
 "slices": {"negation": {"bm25": {"ndcg@10": 0.0, "ci95": [0.0, 0.0], "n_queries": 0}}}}
```
Zeros above are placeholders for the shape only. `reports/judge.json`: section 6.3.
`reports/qu.json`: section 5.5. Plots as PNG next to them.

## 11. Pipeline (P6)

- `dags/psr_daily.py`, Airflow 3 (pin the latest stable 3.x at build time; record it).
  `schedule="@daily"`, `catchup=False`. Tasks are `BashOperator`s (standard provider) that run
  the CLI: `fetch --if-changed` -> `prepare` -> `index` -> `train` -> `eval --systems ltr
  --candidate` -> `gate` -> `publish`. `PSR_DATA_MODE=fixtures` in Compose makes the whole DAG
  run on the fixture in under a few minutes.
- `evaluate/gate.py`: `decide(candidate, published, margin) -> (ok, reason)`: fail if candidate
  NDCG@10 (all) < published NDCG@10 - margin (config, default 0.005); pass if nothing is
  published yet. `psr gate` exits non-zero on fail, so the DAG stops before publish. Unit tests
  for pass, fail, first run.
- `pipeline/publish.py`: copies `models/ltr.txt`, `models/ltr_meta.json`, the demo index and
  `reports/metrics.json` (as `published_metrics.json`) to the target; S3 via boto3; refuses to
  run without credentials unless `--dry-run`.
- `deploy/compose.yaml`: `airflow` (built from `Dockerfile.airflow`, `airflow standalone`),
  `mlflow` (`mlflow server` on the file store), `api` (built from `Dockerfile.api`). Volumes for
  `data/`, `models/`, `reports/`. Gate in chunk 5: `docker compose run --rm airflow airflow dags
  test psr_daily 2026-01-01` passes, and an import-errors check is empty.

## 12. Serving and AWS (P7)

- `psr build-demo --n-products 50000`: demo index = products of the test queries' candidates
  plus a random fill, both locales: product table, BM25 indexes, embeddings, written to
  `indexes/demo/`.
- `serve/app.py` (FastAPI): `GET /search?q=&locale=us&k=10`: BM25 top 200 + dense top 200 ->
  RRF -> top 100 -> LTR features (`n_candidates = 100`) -> top k with score. `GET /explain?q=&
  locale=&product_id=`: LightGBM `pred_contrib` per feature. `GET /health`. `GET /metrics`
  (prometheus-client: `psr_search_seconds` histogram, `psr_zero_results_total`,
  `psr_requests_total{endpoint}`). `GET /` serves `static/index.html` (search box, results,
  "why" toggle that calls `/explain`).
- `serve/lambda_handler.py`: `handler = Mangum(app)`. `Dockerfile.lambda` from
  `public.ecr.aws/lambda/python:3.12`, demo index and e5 model baked in at build time.
- `deploy/terraform/aws/` (region `eu-central-1`): S3 bucket (versioning, public access
  blocked), ECR repository, IAM role for Lambda (logs + read on the bucket), Lambda from the ECR
  image (memory 3008 MB, timeout 30 s), function URL (auth NONE, CORS GET), CloudWatch log group
  (7-day retention), `aws_budgets_budget` with `budget_limit_usd` and `budget_email` variables.
  Outputs: function URL, bucket name. Stage C runs only `terraform fmt -check`,
  `terraform init -backend=false`, `terraform validate`. Apply happens in Stage L after the user
  sets the budget.

## 13. CI (`.github/workflows/ci.yml`)

On push and pull request: `uv sync --frozen`, `ruff check`, `ruff format --check`, `pytest -q`,
then `psr --fixtures train` and `psr --fixtures eval --systems bm25,ltr` (smoke). Second job:
`terraform fmt -check` and `validate` with `hashicorp/setup-terraform`. No secrets in CI.

## 14. Phases

| Phase | Builds | Done when |
|---|---|---|
| P0 | `pyproject.toml`, `uv.lock`, package skeleton, `config.py`, `cli.py`, `text.py`, `data/*`, `.github/workflows/ci.yml`, fixture, data card | gates in CHECKS P0; real data prepared, counts within 0.5% of 4.2 |
| P1 | `evaluate/metrics.py`, `stats.py`, `run.py`, `retrieval/bm25.py` | hand-computed NDCG/MRR tests; bootstrap test on a known case; `reports/metrics.json` has bm25 on the real test split |
| P2 | `retrieval/dense.py`, `hybrid.py` | embeddings cached; dense and hybrid rows with CIs |
| P3 | `rank/features.py`, `rank/ltr.py` | leakage tests; MLflow runs; ltr row and paired comparisons; importance plot |
| P4 | `llm/client.py`, `llm/query_understanding.py` | replay tests; `psr llm-ping` works; `reports/qu.json` on the hard-query slice |
| P5 | `judge/*` | kappa test vs scikit-learn; real run; `reports/judge.json`, confusion plot; **STOP** for user review |
| P6 | `evaluate/gate.py`, `pipeline/publish.py`, `dags/`, Dockerfiles, Compose, Terraform | gate tests; `docker compose` DAG test passes; `terraform validate` ok |
| P7 | `serve/*`, `psr build-demo` | API tests with FastAPI TestClient on the fixture demo index; container answers `/search` |
| P8 | README, `docs/notes/*.md`, `docs/decision_memo.md`, data card final | every number traceable to `reports/`; UNVERIFIED list complete |

## 15. Design note template (`docs/notes/<topic>.md`)

Topics: `data-and-split`, `metrics-and-statistics`, `ltr-features`, `llm-judge`,
`pipeline-gate`, `serving-on-lambda`.

```
# <Topic>
Problem: one paragraph.
Options considered: 2-3 bullets with the trade-off.
Decision: what was built and why.
Evidence: numbers from reports/*.json (file and key named).
Limits: what this does not show.
```

## 16. Decision memo (`docs/decision_memo.md`, P8)

One page for a product manager: the question ("ship the learning-to-rank model instead of
BM25?"), recommendation, evidence table (NDCG@10 with CIs, slices where it helps or hurts,
latency), whether the LLM judge can replace human labels for the next iterations, risks, and the
online A/B test we would run next (metric, duration, guardrails), clearly marked as a proposal.

## 17. Limitations (README)

- Offline evaluation only; no real users and no online test.
- Amazon US and Spanish catalogue; Japanese left out; not a German shop.
- The judge study covers 150 queries; CIs are reported and are wide.
- LLM results depend on one model version on one date.
