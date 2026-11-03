# product-search-ranker

Product search ranking on Amazon's Shopping Queries dataset (ESCI): BM25, dense, hybrid and
learning-to-rank systems measured against human relevance labels, and a check of whether an
LLM judge can stand in for those labels. Everything is an offline evaluation on a public
dataset. Design: [docs/SPEC.md](docs/SPEC.md). Progress: [STATE.md](STATE.md).

This README is a placeholder until phase P8.

## Data
Shopping Queries Dataset (ESCI), Apache-2.0, https://github.com/amazon-science/esci-data. Only
the US and Spanish locales of the reduced (`small_version`) set are used. See
[docs/data_card.md](docs/data_card.md).

> Reddy et al., "Shopping Queries Dataset: A Large-Scale ESCI Benchmark for Improving Product
> Search", arXiv:2206.06588, 2022.

## Quick start
```bash
uv sync
uv run psr fetch && uv run psr prepare     # real data, about 1.2 GB download
uv run psr --fixtures prepare              # fixture only, no download
uv run pytest -q
```
