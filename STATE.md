# Build state

Read this first in every session. Update it before every commit.

Current stage: **L** (local WSL2 machine; the cloud session ended, so chunk 3 onward runs here)
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
- [ ] P4 LLM client + query understanding (offline code merged 2026-11-16, tested with `replay`;
  real runs and the `ltr_qu` system (needs P3's `rank/features.py`) and `reports/qu.json` pending)
- [ ] P5 LLM judge vs human labels (offline code merged 2026-11-16; real runs pending)
- [ ] P6 Airflow DAG, Docker, Compose, Terraform (AWS)
- [ ] P7 API, UI, Lambda adapter, metrics endpoint
- [ ] P8 README, design notes, decision memo, UNVERIFIED complete

## Stage L (VS Code, personal laptop)
- [x] L0 Clone, `uv sync`, gates green, work through UNVERIFIED (2026-11-04: torch and
  sentence-transformers added, BM25 indexes rebuilt, bm25 test row reproduced)
- [ ] L1 Only if the cloud Gemini key failed: real judge + query-understanding runs (user reviews)
- [ ] L2 `docker compose up`, one full DAG run, screenshot of the run history
- [ ] L3 AWS: **STOP**, user sets the budget; `terraform apply`, publish, live URL, 2-min video
- [ ] L4 Final README numbers; user reviews the repo before it goes public

## Results so far
Filled from `reports/metrics.json` after each run (system, locale, NDCG@10, 95% CI).
Test split, 2026-11-03. No query was excluded (`n_excluded` = 0 everywhere).

| System | Locale | NDCG@10 | 95% CI | MRR | Queries |
|---|---|---|---|---|---|
| bm25 | us | 0.6997 | [0.6952, 0.704] | 0.7567 | 8,956 |
| bm25 | es | 0.6772 | [0.6677, 0.6869] | 0.7569 | 2,417 |
| bm25 | all | 0.6949 | [0.6909, 0.699] | 0.7568 | 11,373 |

bm25 by slice (all locales): negation 0.5624 [0.5476, 0.5772] (1,143 queries) is the weakest;
number_or_unit 0.6582, len_6_plus 0.6717, brand_in_query 0.7235. Full table in
`reports/metrics.json` → `slices`.

Local rebuild, 2026-11-04 (WSL2, 16 threads, 11 GB): `psr index` built the 10 indexes (399 s of
build time in the log); `psr eval --systems bm25 --split test` gave the same NDCG@10, CIs and
`n_queries` as the table above, and `reports/runs/test/bm25.parquet` is byte-identical. Only the
`updated` and `git_sha` fields of `reports/metrics.json` changed, so that file was not committed.
Pre-flight for P2 (throwaway script, not a report): multilingual-e5-small on CPU, 2,000 random
products, 31.1 passages/s with 8 torch threads, which projects to 5.80 h for 649,866 products.
`/build` decides whether to embed a subset (SPEC 5.2).
GPU re-time (RTX 4050 Laptop, 6.00 GiB, fp16, 2,000 passages, max_seq_length 256), passages/s and
peak `max_memory_allocated`: bs 64 955.5, 0.37 GiB; bs 128 1132.0, 0.51 GiB; bs 256 1110.2,
0.80 GiB; bs 512 976.3, 1.29 GiB (no OOM). Projected for 649,866 products at bs 512: 0.18 h.

Rebuild on a fresh VM: `uv run psr index` (10 BM25 indexes, about 7 min) then
`uv run psr eval --systems bm25` (a few minutes; logs in `logs/bm25.log`).

Data (from `reports/data_stats.json`, `psr prepare` on real data, 2026-11-03):

| Locale | Queries | Judgements |
|---|---|---|
| us | 29,844 | 601,354 |
| es | 8,049 | 218,774 |

Both match SPEC 4.2 exactly. Per-split counts are in `docs/data_card.md`.

## Decision log
One line each: date, decision, reason.

- 2026-11-03: Gains kept as E 1.0, S 0.1, C 0.01, I 0.0. Source: the ESCI paper (arXiv:2206.06588,
  Task 1): "we set gain values of 1.0, 0.1, 0.01, and 0.0" for E, S, C, I. Note: the esci-data
  repo script `ranking/prepare_trec_eval_files.py` maps S->2, C->3 and runs trec_eval with
  `ndcg.1=0,2=0.01,3=0.1,4=1`, i.e. S 0.01 and C 0.1, the reverse of the paper. We follow the paper.
- 2026-11-03: MRR is averaged over the same queries as NDCG (IDCG = 0 queries excluded from both),
  so `n_queries` describes both numbers.
- 2026-11-03: `psr eval` saves per-query results to `reports/runs/<split>/<system>.parquet`
  (small, committed) and recomputes `comparisons` and `slices` from every saved run, so P2/P3 can
  compare with bm25 without re-running it. Pairs are listed in `config.COMPARISONS`.
- 2026-11-03: `--fixtures` eval writes to `data/fixtures_reports/`, never to `reports/`; a split
  other than test writes `metrics_<split>.json`.
- 2026-11-03: `psr index` builds all five BM25 field indexes (all, title, bullets, brand, desc) now,
  since P3 features need them; `psr eval` builds `all` on demand if missing.
- 2026-11-03: `text.normalize` skips accent stripping for pure-ASCII strings (same output, much
  faster on ~650k product texts).

- 2026-11-03: `.gitignore` entries for `data/`, `indexes/`, `models/`, `mlruns/`, `logs/` are
  anchored with a leading `/`. The unanchored `data/` also matched the `data` source package, so
  `git add -A` silently skipped it. After a commit that adds a package, run `git ls-files src`.
- 2026-11-02 (resolved 2026-11-04): `torch` and `sentence-transformers` were not in `pyproject.toml`
  because the cloud session's network policy blocked `download-r2.pytorch.org`, where the PyTorch
  CPU index serves its wheels. On the local machine both hosts are reachable and `uv` resolves
  them, so the CPU index entry stays.
- 2026-11-04: torch + sentence-transformers added (CPU index); chunk 3 onward runs locally because
  the cloud session ended. Locked: torch 2.15.0+cpu, sentence-transformers 6.2.0, Linux only (no
  macOS marker needed). SPEC section 7 already lists both.
- 2026-11-04: torch from PyPI (CUDA build, 2.15.0+cu132) instead of the CPU index, so embeddings
  run on the local GPU; CI still runs on CPU. The wheel's CUDA 13.2 is newer than the 13.1 the
  driver (591.66) reports; fp16 kernels ran fine (same major version), so no `cuXXX` index.
- 2026-11-16: All 20 GitHub Actions runs failed in 0 s with "workflow file issue", not a torch
  problem: `hashFiles` is not allowed in a job-level `if`, so GitHub rejected `ci.yml` and started
  no runner. The terraform job is removed (it returns in P6, once `deploy/terraform/aws` exists).
  Check workflow files with `uvx --from actionlint-py actionlint .github/workflows/ci.yml`.
- 2026-11-16: torch comes from two dependency groups, `gpu` (default, local; CUDA build from the
  pytorch.org `cu132` index) and `cpu` (CI, Docker, Lambda; `cpu` index). Both lock torch 2.15.0,
  so results do not depend on the variant. `cu130` (the driver's 13.1 match) only has torch up to
  2.14.1 and `cu131` does not exist, so `cu132` stays; it ran fp16 correctly on the 13.1 driver.
  If a CUDA kernel ever fails, switch the `gpu` index to `cu130` and the floor to 2.14. Local:
  `uv sync`. CI: `uv sync --frozen --no-group gpu --group cpu` plus `UV_NO_SYNC=1` (else
  `uv run` re-syncs to the GPU group). Lock restricted to Linux (`tool.uv.environments`).
- 2026-11-16: AWS budget for L3 is at most 5 EUR in total. Design for it: Lambda and S3, no NAT
  gateway, RDS or ECS, and an AWS Budgets alert before any `terraform apply`. Still a STOP at L3.
- 2026-11-04: Dense embeddings (P2) are computed on the GPU in fp16 with batch size 512 (peak
  1.29 GB of 6.00 GB). This deviates from SPEC 5.2 ("on CPU"). The 2.5 h subset rule is applied to
  the measured GPU throughput: 0.18 h for all 649,866 products, so no subset. Serving (P7) still
  encodes queries on CPU.
- 2026-11-04: LightGBM stays on CPU (the PyPI wheel has no GPU build; data is small).
- 2026-11-04: `.claude/settings.json` (was `.claude/commands/settings.json`, where Claude Code does
  not read it as settings). Allow and deny rules unchanged; the deny list still blocks
  `terraform apply`/`destroy`, force pushes and reading `.env`.
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
  `eval` smoke steps are added when P1 and P3 create those commands. The terraform job was
  meant to skip itself until `deploy/terraform/aws/*.tf` exists; it was invalid and is removed
  (2026-11-16).

- 2026-11-04: P4/P5 offline code built on branch `p4-p5-llm-code` in the cloud (merged into `main`
  2026-11-16), in parallel with chunk 3; the user approved the change of order. The cloud proxy did not add a Gemini
  key (`GET /v1beta/models` without a key answered 403), so the real runs are Stage L1, locally,
  with `GEMINI_API_KEY` in `.env`.
- 2026-11-04: Judge prompt uses the four ESCI label definitions verbatim from Reddy et al. 2022
  (arXiv:2206.06588), read via Firecrawl because arxiv.org is blocked here. The same paper says
  "we set gain values of 1.0, 0.1, 0.01, and 0.0" for E, S, C, I, matching `config.GAINS`.
- 2026-11-04: `PSR_GEMINI_MODEL` has no default. `psr llm-ping` lists the models when it is unset
  and picks the stable `gemini-<version>-flash` with the highest version (lite and preview
  models left out), then says to put it in `.env` and log the name and date here.
- 2026-11-04: `config.load_dotenv()` reads `.env` with the stdlib (no python-dotenv, SPEC 7).
- 2026-11-04: Judge sample: per locale, query counts split across the length buckets in
  proportion (largest remainder), seed 0, capped by what exists (the fixture has 45 us + 15 es
  test queries). Candidates go into the prompt in `product_id` order. The 30 batch-effect
  queries are every k-th sampled query id. A query call is retried once (real providers only)
  when a candidate is missing, duplicated or has a bad label; after that the pair is counted in
  `n_missing`, not filled.
- 2026-11-04: Judge cache `data/llm_cache/judge.jsonl` (fixtures: `data/fixtures_llm_cache/`),
  append-only, keyed `batch:<query_id>` / `single:<query_id>:<product_id>`, so a run resumes.
  `reports/judge_labels.parquet` has `label_human`, `label_judge` (batched) and `label_single`.
- 2026-11-04: Kappa and accuracy CIs: query bootstrap (10,000 resamples, seed 0) over summed
  per-query confusion matrices. System level uses only judged queries where every candidate got
  a label (`n_dropped_incomplete` reported); it scores systems with `evaluate.run.score_system`,
  so it covers whatever systems exist when `psr agreement` runs.
- 2026-11-04: Query understanding: hard = negation, number_or_unit or brand_in_query (brands
  from the query's candidates). Batches of 20 queries from one locale, in query_id order.
  Coverage on the real data: 8,954 queries (val 1,157, test 4,797, train sample 3,000), 448
  batches. Items missing, duplicated or invalid get one retry call together; after that they
  are cached empty and counted. `qu_features` lives in `query_understanding.py` so
  `rank/features.py` (P3) only imports it: `qu_present`, `must_not_in_title`, `qu_brand_match`,
  `qu_size_match` (size with spaces removed, not inside a longer number), `qu_color_match`,
  `qu_type_in_title` (share of type words in the title); all 0 when a query has no output.
- 2026-11-04: Judge and qu cache only items whose call got a reply. A call that fails outright
  (network, quota, after the 5/15/45 s retries) writes nothing, so the next run asks again
  instead of storing an empty result.
- 2026-11-16: Gemini key checked with direct calls (the key itself never printed): model listing
  works; `generateContent` returned a valid JSON reply on `gemini-3.5-flash-lite`,
  `gemini-3.1-flash-lite`, `gemini-3-flash-preview` and `gemini-flash-lite-latest`, but 503 or a
  read timeout on `gemini-3.5-flash` to `gemini-3.8-flash` and `gemini-flash-latest`, and
  `gemini-2.5-flash` is closed to new users. `psr llm-ping` picks the highest stable Flash,
  which is the overloaded `gemini-3.8-flash`, so it stalls for about 90 s per call on its
  5/15/45 s retries. `PSR_GEMINI_MODEL` stays unset: chunk 4 re-tests the models, picks one
  (the judge needs a model that is not a lite one if quality matters) and logs name and date
  here. A first 403 ("API not used in project ... or disabled") was the API still switching on
  after it was enabled; it cleared within minutes.
- 2026-11-16: Chunk 3 embed job (P2) guards the GPU: `psr embed` reads free VRAM first
  (`torch.cuda.mem_get_info`) and stops below 3 GiB free, saying what to close and to run
  `nvidia-smi`. `/build` tells the user before it launches the ~11 min job and ends with a DONE
  line in `logs/embed.log`. The user keeps the laptop plugged in and awake; the job resumes.
- 2026-11-16: Model routing (user-approved): main session (Opus) plans, writes hard code, commits
  and handles STOP lines; `.claude/agents/impl.md` (Sonnet) builds self-contained tasks;
  `.claude/agents/gatecheck.md` (Haiku, read-only) runs gates; `ecc:*` reviewers per phase. `/build`
  now plans first and stops for the user's OK. Routing table in `docs/CHECKS.md`, rule in `CLAUDE.md`.
  P2/P3 stay mostly in the main session (coupled code, leakage risk).
- 2026-11-16: Docker Desktop with WSL integration works (Docker 29.8.2, Compose v5.5.1,
  `hello-world` ran), so L2 can start when P6 exists.

## UNVERIFIED
Things written but not run. Each line: file, what is unverified, how to check. Delete the line
when it is confirmed or fixed.

- `llm/client.py` gemini provider: a real key was checked 2026-11-16 (see Decision log), but
  `psr llm-ping` has not yet completed one call, because its auto-picked model
  (`gemini-3.8-flash`) answered 503 or timed out. Check: set `PSR_GEMINI_MODEL` in `.env` to a
  model that answers, then `uv run psr llm-ping`. The ollama provider was never called.
- `psr judge` and `psr agreement` on real data: never run. Check: `uv run psr judge` (about 150
  batched calls plus about 660 single-pair calls; at 10 requests/min about 1.5 h, resumable),
  then `uv run psr agreement` after dense, hybrid and ltr exist. P5 then has its **STOP**.
- `psr qu` on real data: never run. Check: `uv run psr qu --limit 40` first (2 calls), read
  `data/llm_cache/qu.jsonl`, then `uv run psr qu` (448+ calls, about 45 min at 10 requests/min,
  resumable).
- CI has no judge smoke step: it needs recorded replay fixtures. Check: `uv run psr --fixtures
  judge --provider gemini --record`, commit `tests/fixtures/replay/`, then add `psr --fixtures
  judge` and `psr --fixtures agreement` to `ci.yml`.

## Blockers
Anything that stopped progress twice in a row, with what was tried.
