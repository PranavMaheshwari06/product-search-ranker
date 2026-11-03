import json

import pandas as pd
import pytest

from product_search_ranker import config
from product_search_ranker.cli import main
from product_search_ranker.retrieval import bm25


@pytest.fixture
def indexed(home):
    main(["--fixtures", "prepare"])
    main(["--fixtures", "index", "--fields", "all"])
    return home


def test_bm25_scores_rank_a_matching_product_first(indexed):
    rows = pd.read_parquet(config.processed_dir(True) / "test.parquet")
    q = rows[rows.locale == "us"].iloc[0]
    target = rows[rows.product_id == q.product_id].iloc[[0]]
    # Querying with the product's own title must score it above zero.
    probe = target.assign(query=q.product_title, query_id=-1)
    assert bm25.score_candidates(probe, "all", fixtures=True)[0] > 0
    empty = target.assign(query="the", query_id=-2)
    assert bm25.score_candidates(empty, "all", fixtures=True)[0] == 0


def test_index_is_skipped_when_present(indexed, capsys):
    main(["--fixtures", "index", "--fields", "all"])
    assert "skip" in capsys.readouterr().out


def test_eval_writes_metrics_and_keeps_other_systems(indexed):
    path = config.metrics_path("test", fixtures=True)
    path.parent.mkdir(parents=True)
    path.write_text(json.dumps({"systems": {"other": {"all": {}}}}))

    main(["--fixtures", "eval", "--systems", "bm25", "--split", "test"])
    report = json.loads(path.read_text())
    assert set(report["systems"]) == {"other", "bm25"}
    res = report["systems"]["bm25"]
    assert set(res) == {"us", "es", "all"}
    assert res["all"]["n_queries"] == res["us"]["n_queries"] + res["es"]["n_queries"] > 0
    lo, hi = res["all"]["ci95"]
    assert 0 <= lo <= res["all"]["ndcg@10"] <= hi <= 1
    assert report["gains"] == config.GAINS and report["split"] == "test"
    assert set(report["slices"]) == {
        "len_1_2",
        "len_3_5",
        "len_6_plus",
        "negation",
        "number_or_unit",
        "brand_in_query",
    }
    assert not (config.reports_dir() / "metrics.json").exists()
