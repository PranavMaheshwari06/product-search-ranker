import json

import bm25s
import numpy as np
import pandas as pd

from product_search_ranker import config
from product_search_ranker.cli import main
from product_search_ranker.evaluate import run
from product_search_ranker.retrieval import bm25
from product_search_ranker.text import tokenize


def test_candidate_scores_match_a_full_index_retrieval(home):
    main(["--fixtures", "prepare"])
    df = pd.read_parquet(config.processed_dir(True) / "test.parquet")
    us = df[df.locale == "us"]
    query = us["query"].iloc[0]
    index = bm25.Bm25Index("us", fixtures=True)
    top_ids, top_scores = index.retriever.retrieve(
        [index.retriever.get_tokens_ids(tokenize(query, "us"))], k=5, show_progress=False
    )
    ids = (bm25.index_dir("us", fixtures=True) / "product_ids.txt").read_text().split("\n")
    got = index.scores(query, [ids[i] for i in top_ids[0]])
    np.testing.assert_allclose(got, top_scores[0], rtol=1e-6)
    assert isinstance(index.retriever, bm25s.BM25)


def test_query_with_only_stopwords_scores_zero(home):
    main(["--fixtures", "prepare"])
    index = bm25.Bm25Index("us", fixtures=True)
    pid = next(iter(index.row))
    assert index.scores("the of and", [pid]).tolist() == [0.0]


def test_eval_writes_us_es_all_and_keeps_other_systems(home):
    main(["--fixtures", "eval", "--systems", "bm25", "--split", "test"])
    path = run.report_path("test", fixtures=True)
    report = json.loads(path.read_text())
    assert set(report["systems"]["bm25"]) == {"us", "es", "all"}
    rows = report["systems"]["bm25"]
    allq = rows["all"]
    for key in ("n_queries", "n_excluded"):
        assert allq[key] == rows["us"][key] + rows["es"][key]
    lo, hi = allq["ci95"]
    assert lo <= allq["ndcg@10"] <= hi
    assert report["gains"] == config.GAINS and report["split"] == "test"
    assert "bm25" in report["slices"]["len_1_2"]
    # Fixture runs never write the committed reports/.
    assert not (config.reports_dir() / "metrics.json").exists()

    report["systems"]["other"] = {"all": {"ndcg@10": 0.5}}
    path.write_text(json.dumps(report))
    main(["--fixtures", "eval", "--systems", "bm25"])
    assert json.loads(path.read_text())["systems"]["other"] == {"all": {"ndcg@10": 0.5}}


def test_comparisons_are_paired_over_shared_queries(home, monkeypatch):
    main(["--fixtures", "eval", "--systems", "bm25"])
    runs = run.runs_dir("test", fixtures=True)
    pq = pd.read_parquet(runs / "bm25.parquet")
    pq.assign(ndcg=pq.ndcg.where(pq.excluded, 1.0)).to_parquet(runs / "ltr.parquet")
    monkeypatch.setattr(config, "COMPARISONS", (("ltr", "bm25"),))
    main(["--fixtures", "eval", "--systems", "bm25"])
    comps = json.loads(run.report_path("test", True).read_text())["comparisons"]
    allc = next(c for c in comps if c["locale"] == "all")
    kept = pq[~pq.excluded]
    assert allc["delta"] == round(float((1.0 - kept.ndcg).mean()), 4)
    assert allc["n_queries"] == len(kept)
    assert {c["locale"] for c in comps} == {"us", "es", "all"}
