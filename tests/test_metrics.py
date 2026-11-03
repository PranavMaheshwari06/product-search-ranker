import math

import numpy as np
import pandas as pd
import pytest

from product_search_ranker.evaluate import metrics, stats


def _rows(query_id, labels, scores, ids=None, locale="us"):
    ids = ids or [f"p{i}" for i in range(len(labels))]
    return pd.DataFrame(
        {
            "query_id": query_id,
            "locale": locale,
            "product_id": ids,
            "label": list(labels),
            "score": scores,
        }
    )


def _by_query(rows):
    return metrics.per_query(rows).set_index("query_id")


# Three queries computed by hand with gains E=1, S=0.1, C=0.01, I=0 and discount 1/log2(rank+1).
def test_ndcg_and_rr_on_three_hand_computed_queries():
    rows = pd.concat(
        [
            # q1 ranked S, E, I. DCG = 0.1/1 + 1/log2(3); IDCG = 1/1 + 0.1/log2(3).
            _rows(1, "SEI", [3.0, 2.0, 1.0]),
            # q2 scores tie for a and b, so product_id breaks it: c (I), a (C), b (E).
            # DCG = 0 + 0.01/log2(3) + 1/log2(4); IDCG = 1 + 0.01/log2(3).
            _rows(2, "ICE", [2.0, 1.0, 1.0], ids=["c", "a", "b"]),
            # q3 already in ideal order.
            _rows(3, "ESC", [3.0, 2.0, 1.0], locale="es"),
        ]
    )
    got = _by_query(rows)
    l3 = math.log2(3)
    assert got.ndcg[1] == pytest.approx((0.1 + 1 / l3) / (1 + 0.1 / l3))
    assert got.ndcg[1] == pytest.approx(0.687550, abs=1e-6)
    assert got.ndcg[2] == pytest.approx((0.01 / l3 + 0.5) / (1 + 0.01 / l3))
    assert got.ndcg[2] == pytest.approx(0.503135, abs=1e-6)
    assert got.ndcg[3] == pytest.approx(1.0)
    assert list(got.rr) == pytest.approx([1 / 2, 1 / 3, 1.0])

    s = metrics.summarize(got.reset_index())
    assert s["ndcg@10"] == pytest.approx(0.730228, abs=1e-6)
    assert s["mrr"] == pytest.approx((1 / 2 + 1 / 3 + 1) / 3)
    assert (s["n_queries"], s["n_excluded"]) == (3, 0)


def test_query_with_idcg_zero_is_excluded_and_counted():
    rows = pd.concat([_rows(1, "III", [3.0, 2.0, 1.0]), _rows(2, "ESI", [1.0, 2.0, 3.0])])
    got = _by_query(rows)
    assert np.isnan(got.ndcg[1])
    s = metrics.summarize(got.reset_index())
    assert (s["n_queries"], s["n_excluded"]) == (1, 1)
    assert s["mrr"] == pytest.approx(1 / 3)


def test_cutoff_at_10_and_rr_without_exact():
    # The only Exact item sits at rank 11: no gain inside the top 10, but MRR still sees it.
    labels = "I" * 10 + "E"
    got = _by_query(_rows(1, labels, list(range(11, 0, -1))))
    assert got.ndcg[1] == 0.0
    assert got.rr[1] == pytest.approx(1 / 11)
    # Substitutes only: NDCG is defined, RR is 0.
    got = _by_query(_rows(2, "IS", [2.0, 1.0]))
    assert got.ndcg[2] == pytest.approx((0.1 / math.log2(3)) / 0.1)
    assert got.rr[2] == 0.0


def test_bootstrap_ci_contains_the_true_mean():
    rng = np.random.default_rng(1)
    values = rng.normal(0.5, 0.1, size=2000)
    lo, hi = stats.bootstrap_ci(values)
    assert lo < 0.5 < hi
    # Width close to the normal-theory 2 * 1.96 * sd / sqrt(n).
    assert hi - lo == pytest.approx(2 * 1.96 * 0.1 / math.sqrt(2000), rel=0.15)
    assert stats.bootstrap_ci(values) == (lo, hi)


def test_paired_diff_and_sign_flip():
    rng = np.random.default_rng(2)
    a = rng.uniform(0, 1, size=500)
    same = stats.paired_diff(a, a)
    assert same["delta"] == 0.0 and same["p_value"] == 1.0
    better = stats.paired_diff(a + 0.05, a)
    assert better["delta"] == pytest.approx(0.05)
    assert better["ci95"][0] == pytest.approx(0.05) and better["p_value"] < 0.001
    noise = rng.normal(0, 0.1, size=500)
    null = stats.paired_diff(a + noise, a)
    assert null["ci95"][0] < 0 < null["ci95"][1]
    assert null["p_value"] > 0.05
