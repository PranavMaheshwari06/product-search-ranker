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
- [ ] P4 LLM client + query understanding
- [ ] P5 LLM judge vs human labels
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
- 2026-11-05: All 20 GitHub Actions runs failed in 0 s with "workflow file issue", not a torch
  problem: `hashFiles` is not allowed in a job-level `if`, so GitHub rejected `ci.yml` and started
  no runner. The terraform job is removed (it returns in P6, once `deploy/terraform/aws` exists).
  Check workflow files with `uvx --from actionlint-py actionlint .github/workflows/ci.yml`.
- 2026-11-05: torch comes from two dependency groups, `gpu` (default, local; CUDA build from the
  pytorch.org `cu132` index) and `cpu` (CI, Docker, Lambda; `cpu` index). Both lock torch 2.15.0,
  so results do not depend on the variant. `cu130` (the driver's 13.1 match) only has torch up to
  2.14.1 and `cu131` does not exist, so `cu132` stays; it ran fp16 correctly on the 13.1 driver.
  If a CUDA kernel ever fails, switch the `gpu` index to `cu130` and the floor to 2.14. Local:
  `uv sync`. CI: `uv sync --frozen --no-group gpu --group cpu` plus `UV_NO_SYNC=1` (else
  `uv run` re-syncs to the GPU group). Lock restricted to Linux (`tool.uv.environments`).
- 2026-11-05: AWS budget for L3 is at most 5 EUR in total. Design for it: Lambda and S3, no NAT
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
  `eval` smoke steps are added when P1 and P3 create those commands. The terraform job skips
  itself until `deploy/terraform/aws/*.tf` exists.

## UNVERIFIED
Things written but not run. Each line: file, what is unverified, how to check. Delete the line
when it is confirmed or fixed.

- `.github/workflows/ci.yml`: every earlier run failed on the invalid terraform job (see Decision
  log, 2026-11-05). Fixed and actionlint-clean; the exact CI commands pass locally in a scratch CPU
  env. Not yet confirmed on a GitHub runner. Check: read the Actions run after the push.

## Blockers
Anything that stopped progress twice in a row, with what was tried.
