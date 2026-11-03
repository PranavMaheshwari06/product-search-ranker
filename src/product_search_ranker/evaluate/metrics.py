"""Per-query NDCG@k and reciprocal rank over each query's judged candidates (SPEC section 3)."""

import numpy as np
import pandas as pd

from product_search_ranker import config


def rank(rows: pd.DataFrame, score_col: str = "score") -> pd.DataFrame:
    """Sort candidates by score, highest first; ties go to the smaller product_id."""
    out = rows.sort_values(
        ["query_id", score_col, "product_id"], ascending=[True, False, True], kind="mergesort"
    )
    out = out.assign(rank=out.groupby("query_id").cumcount() + 1)
    return out.reset_index(drop=True)


def per_query(rows: pd.DataFrame, score_col: str = "score", k: int = config.K) -> pd.DataFrame:
    """One row per query: locale, ndcg (NaN when IDCG = 0, i.e. excluded) and rr.

    `rows` needs query_id, locale, product_id, label and the score column.
    """
    r = rank(rows, score_col)
    gain = r.label.map(config.GAINS).to_numpy()
    discount = 1.0 / np.log2(r["rank"].to_numpy() + 1)
    top = r["rank"].to_numpy() <= k
    r = r.assign(dcg=np.where(top, gain * discount, 0.0), gain=gain)

    # The ideal ranking: the same candidates sorted by gain.
    ideal = r.sort_values(["query_id", "gain"], ascending=[True, False], kind="mergesort")
    ideal_rank = ideal.groupby("query_id").cumcount().to_numpy() + 1
    ideal = ideal.assign(idcg=np.where(ideal_rank <= k, ideal.gain / np.log2(ideal_rank + 1), 0))

    exact_rank = r["rank"].where(r.label == "E")
    out = pd.DataFrame(
        {
            "locale": r.groupby("query_id").locale.first(),
            "dcg": r.groupby("query_id").dcg.sum(),
            "idcg": ideal.groupby("query_id").idcg.sum(),
            "first_exact": exact_rank.groupby(r.query_id).min(),
        }
    )
    out["ndcg"] = (out.dcg / out.idcg).where(out.idcg > 0)
    out["rr"] = (1.0 / out.first_exact).fillna(0.0)
    return out[["locale", "ndcg", "rr"]].reset_index()


def summarize(pq: pd.DataFrame) -> dict:
    """Mean NDCG and MRR over the included queries (IDCG > 0); excluded ones are counted."""
    kept = pq[pq.ndcg.notna()]
    return {
        "ndcg@10": float(kept.ndcg.mean()) if len(kept) else None,
        "mrr": float(kept.rr.mean()) if len(kept) else None,
        "n_queries": int(len(kept)),
        "n_excluded": int(len(pq) - len(kept)),
    }
