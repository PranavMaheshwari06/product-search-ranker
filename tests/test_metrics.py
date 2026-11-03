from math import log2

import pandas as pd
import pytest

from product_search_ranker.evaluate import metrics


def frame(rows):
    return pd.DataFrame(rows, columns=["query_id", "locale", "product_id", "label", "score"])


# Three queries worked out by hand. Gains: E 1.0, S 0.1, C 0.01, I 0.0.
HAND = frame(
    [
        # q1 ranked E, I, S, C
        (1, "us", "a", "E", 4.0),
        (1, "us", "b", "I", 3.0),
        (1, "us", "c", "S", 2.0),
        (1, "us", "d", "C", 1.0),
        # q2 ranked I, S, E
        (2, "us", "a", "I", 3.0),
        (2, "us", "b", "S", 2.0),
        (2, "us", "c", "E", 1.0),
    ]
    # q3: S first, then nine I, then the only E at rank 11 (outside the top 10).
    + [(3, "es", "p00", "S", 20.0)]
    + [(3, "es", f"p{i:02d}", "I", 20.0 - i) for i in range(1, 10)]
    + [(3, "es", "p10", "E", 1.0)]
)

Q1_NDCG = (1 + 0.1 / log2(4) + 0.01 / log2(5)) / (1 + 0.1 / log2(3) + 0.01 / log2(4))
Q2_NDCG = (0.1 / log2(3) + 1 / log2(4)) / (1 + 0.1 / log2(3))
Q3_NDCG = 0.1 / (1 + 0.1 / log2(3))


def test_ndcg_and_mrr_on_three_hand_computed_queries():
    pq = metrics.per_query(HAND, HAND.score).set_index("query_id")
    assert pq.ndcg[1] == pytest.approx(Q1_NDCG)
    assert pq.ndcg[2] == pytest.approx(Q2_NDCG)
    assert pq.ndcg[3] == pytest.approx(Q3_NDCG)
    assert pq.ndcg[1] == pytest.approx(0.987093, abs=1e-6)
    assert pq.ndcg[2] == pytest.approx(0.529674, abs=1e-6)
    assert pq.ndcg[3] == pytest.approx(0.094065, abs=1e-6)
    # MRR looks at the whole list, not only the top 10.
    assert list(pq.rr) == pytest.approx([1.0, 1 / 3, 1 / 11])
    assert not pq.excluded.any()


def test_query_without_any_gain_is_excluded_and_counted():
    df = frame([(9, "us", "a", "I", 1.0), (9, "us", "b", "I", 0.5)])
    pq = metrics.per_query(df, df.score)
    assert pq.excluded.tolist() == [True]
    assert pq.ndcg.isna().all() and pq.rr.isna().all()
    assert metrics.ndcg([0.0, 0.0]) is None


def test_ties_are_broken_by_product_id():
    df = frame([(1, "us", "b", "I", 1.0), (1, "us", "a", "E", 1.0)])
    pq = metrics.per_query(df, df.score)
    assert pq.ndcg[0] == 1.0 and pq.rr[0] == 1.0


def test_dcg_cuts_at_k():
    assert metrics.dcg([1.0] * 12, k=10) == pytest.approx(
        sum(1 / log2(i + 1) for i in range(1, 11))
    )
