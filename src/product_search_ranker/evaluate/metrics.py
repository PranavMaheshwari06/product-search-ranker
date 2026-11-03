"""Per-query NDCG@k and reciprocal rank over each query's judged candidates (SPEC 3)."""

import numpy as np
import pandas as pd

from product_search_ranker import config


def dcg(gains, k: int = config.K) -> float:
    g = np.asarray(gains, dtype=np.float64)[:k]
    return float((g / np.log2(np.arange(2, len(g) + 2))).sum())


def ndcg(gains_in_rank_order, k: int = config.K) -> float | None:
    """None when IDCG is 0 (no candidate with gain > 0): the caller excludes and counts it."""
    idcg = dcg(sorted(gains_in_rank_order, reverse=True), k)
    if idcg == 0:
        return None
    return dcg(gains_in_rank_order, k) / idcg


def reciprocal_rank(labels_in_rank_order) -> float:
    """1 / rank of the first Exact candidate in the whole list, 0 if there is none."""
    for i, label in enumerate(labels_in_rank_order):
        if label == "E":
            return 1.0 / (i + 1)
    return 0.0


def per_query(df: pd.DataFrame, scores, k: int = config.K) -> pd.DataFrame:
    """One row per query: query_id, locale, ndcg (NaN if excluded), rr, excluded.

    Candidates are sorted by score, descending, ties broken by product_id so runs are
    deterministic. Excluded queries keep rr = NaN too, so both means use the same queries.
    """
    d = df[["query_id", "locale", "product_id", "label"]].assign(score=np.asarray(scores))
    d = d.sort_values(["query_id", "score", "product_id"], ascending=[True, False, True])
    d["gain"] = d.label.map(config.GAINS)
    rows = []
    for (qid, locale), g in d.groupby(["query_id", "locale"], sort=False):
        n = ndcg(g.gain.to_numpy(), k)
        rr = reciprocal_rank(g.label.to_numpy()) if n is not None else np.nan
        rows.append((qid, locale, np.nan if n is None else n, rr, n is None))
    return pd.DataFrame(rows, columns=["query_id", "locale", "ndcg", "rr", "excluded"])
