import json

import numpy as np
import pandas as pd

from product_search_ranker.cli import main
from product_search_ranker.evaluate import run
from product_search_ranker.retrieval import hybrid


def frame(index=None):
    return pd.DataFrame(
        {"query_id": ["q1"] * 3 + ["q2"] * 3, "product_id": list("abcdef")}, index=index
    )


def test_rrf_by_hand():
    a = [3.0, 2.0, 1.0, 1.0, 2.0, 3.0]  # ranks 1,2,3 | 3,2,1
    b = [1.0, 3.0, 2.0, 3.0, 2.0, 1.0]  # ranks 3,1,2 | 1,2,3
    got = hybrid.rrf(frame(), a, b)
    want = [
        1 / 61 + 1 / 63,
        1 / 62 + 1 / 61,
        1 / 63 + 1 / 62,
        1 / 63 + 1 / 61,
        1 / 62 + 1 / 62,
        1 / 61 + 1 / 63,
    ]
    np.testing.assert_allclose(got, want)


def test_ties_rank_by_product_id():
    assert hybrid.ranks(frame(), [1.0] * 6).tolist() == [1, 2, 3, 1, 2, 3]


def test_non_default_index_matches_reset_index():
    s = [0.1, 0.9, 0.5, 0.3, 0.3, 0.8]
    np.testing.assert_array_equal(
        hybrid.ranks(frame(index=[50, 7, 99, 3, 4, 0]), s), hybrid.ranks(frame(), s)
    )


def test_eval_reports_dense_and_hybrid(home, fake_dense):
    main(["--fixtures", "eval", "--systems", "bm25,dense,hybrid"])
    report = json.loads(run.report_path("test", True).read_text())
    for name in ("dense", "hybrid"):
        assert set(report["systems"][name]) == {"us", "es", "all"}
        assert all("ci95" in report["systems"][name][loc] for loc in ("us", "es", "all"))
    pairs = {(c["a"], c["b"]) for c in report["comparisons"] if c["locale"] == "all"}
    assert {("dense", "bm25"), ("hybrid", "bm25")} <= pairs
