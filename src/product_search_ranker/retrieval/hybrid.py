"""Reciprocal rank fusion of bm25 and dense within each query's candidates (SPEC 5.3)."""

import numpy as np
import pandas as pd

from product_search_ranker.retrieval import bm25, dense

K_RRF = 60


def ranks(df: pd.DataFrame, scores) -> np.ndarray:
    """1-based rank within each query; same order as metrics.per_query (ties by product_id)."""
    d = pd.DataFrame(
        {
            "query_id": df.query_id.to_numpy(),
            "product_id": df.product_id.to_numpy(),
            "score": np.asarray(scores),
        }
    )
    d = d.sort_values(["query_id", "score", "product_id"], ascending=[True, False, True])
    out = np.empty(len(d))
    out[d.index.to_numpy()] = d.groupby("query_id").cumcount().to_numpy() + 1
    return out


def rrf(df: pd.DataFrame, *score_arrays, k: int = K_RRF) -> np.ndarray:
    return sum(1.0 / (k + ranks(df, s)) for s in score_arrays)


def score(df: pd.DataFrame, fixtures: bool = False) -> np.ndarray:
    return rrf(df, bm25.score(df, fixtures), dense.score(df, fixtures))
