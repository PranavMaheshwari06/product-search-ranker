"""The judged sample: test queries stratified by length bucket, with all their candidates."""

import numpy as np
import pandas as pd

from product_search_ranker import config
from product_search_ranker.text import LENGTH_SLICES, query_slices

N_QUERIES = {"us": 100, "es": 50}


def length_bucket(query: str, locale: str) -> str:
    return next(s for s in LENGTH_SLICES if s in query_slices(query, locale))


def _allocate(sizes: dict[str, int], n: int) -> dict[str, int]:
    """Split n across buckets in proportion to their sizes (largest remainder), capped by size."""
    total = sum(sizes.values())
    n = min(n, total)
    exact = {b: n * s / total for b, s in sizes.items()}
    out = {b: int(v) for b, v in exact.items()}
    for b in sorted(exact, key=lambda b: (out[b] - exact[b], b))[: n - sum(out.values())]:
        out[b] += 1
    return out


def sample(test: pd.DataFrame, n_queries: dict[str, int] = N_QUERIES) -> pd.DataFrame:
    """Rows of the sampled queries (every candidate), with a `bucket` column. Seeded."""
    queries = test[["query_id", "query", "locale"]].drop_duplicates().sort_values("query_id")
    queries["bucket"] = [
        length_bucket(q, loc) for q, loc in zip(queries["query"], queries.locale, strict=True)
    ]
    rng = np.random.default_rng(config.SEED)
    picked = []
    for locale in config.LOCALES:
        qs = queries[queries.locale == locale]
        sizes = qs.groupby("bucket").size().to_dict()
        for bucket, k in sorted(_allocate(sizes, n_queries[locale]).items()):
            ids = qs.query_id[qs.bucket == bucket].to_numpy()
            picked.extend(rng.choice(ids, size=k, replace=False).tolist())
    rows = test[test.query_id.isin(picked)]
    rows = rows.merge(queries[["query_id", "bucket"]], on="query_id")
    return rows.sort_values(["query_id", "product_id"]).reset_index(drop=True)
