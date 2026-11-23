import json
import shutil

import numpy as np
import pandas as pd
import pytest

from product_search_ranker import config
from product_search_ranker.cli import main
from product_search_ranker.evaluate import metrics
from product_search_ranker.llm import query_understanding as qu
from product_search_ranker.rank import features, ltr


@pytest.fixture
def built(home, fake_dense):
    main(["--fixtures", "prepare"])
    return {s: features.build(s, fixtures=True) for s in ("train", "val")}


def test_text_features_by_hand():
    df = pd.DataFrame(
        {
            "query_id": ["q1"],
            "query": ["red usb cable 2 pack"],
            "locale": ["us"],
            "product_id": ["p1"],
            "product_title": ["Red USB cable 2m"],
            "product_brand": ["Anker"],
            "product_color": ["Red"],
            "product_bullet_point": ["pack of 2"],
            "title_raw_len": ["16"],
            "bullet_raw_len": ["9"],
            "description_raw_len": ["0"],
        }
    )
    r = features.text_features(df).iloc[0]
    assert r.title_jaccard == 3 / 6  # {red, usb, cable} shared of 6 distinct tokens
    assert r.query_tokens_in_title == 3 / 5
    assert (r.brand_in_query, r.color_in_query) == (0.0, 1.0)
    assert r.number_match == 1.0  # "2" is in the bullets, not the title ("2m")
    assert (r.title_len, r.has_description, r.bullet_len) == (16.0, 0.0, 9.0)
    assert (r.query_len, r.n_candidates, r.locale_es) == (5.0, 1.0, 0.0)


def test_no_feature_is_derived_from_labels():
    assert len(features.FEATURES) == len(set(features.FEATURES)) == 20
    assert not any("label" in f or "esci" in f for f in ltr.QU_FEATURES)
    assert ltr.QU_FEATURES[: len(features.FEATURES)] == features.FEATURES


def test_built_features_cover_every_row_and_cache(built, home):
    train = built["train"]
    assert list(train[features.FEATURES].columns) == features.FEATURES
    assert train[features.FEATURES].drop(columns="number_match").notna().all().all()
    assert features.features_path("train", True).exists()
    assert (features.build("train", True).query_id == train.query_id).all()


def test_feature_table_does_not_depend_on_labels(built, fake_dense):
    path = config.processed_dir(True) / "train.parquet"
    df = pd.read_parquet(path)
    before = built["train"][features.FEATURES]
    df["label"] = np.random.default_rng(1).permutation(df.label.to_numpy())
    df.to_parquet(path, index=False)
    features.features_path("train", True).unlink()
    shutil.rmtree(features.features_path("train", True).with_suffix(""))
    after = features.build("train", fixtures=True)[features.FEATURES]
    pd.testing.assert_frame_equal(before, after)


def test_stale_feature_cache_is_rebuilt(built):
    path = features.features_path("val", True)
    (path.with_suffix("") / "fingerprint.json").write_text("{}")
    before = path.stat().st_mtime_ns
    features.build("val", fixtures=True)
    assert path.stat().st_mtime_ns > before


def test_shuffled_labels_score_at_random_order_level(built):
    train, val = built["train"], built["val"]
    rng = np.random.default_rng(config.SEED)
    random_level = np.mean(
        [metrics.per_query(val, rng.random(len(val))).ndcg.mean() for _ in range(50)]
    )
    params = {"num_leaves": 31, "learning_rate": 0.1, "min_data_in_leaf": 5}
    shuffled = train.assign(label=rng.permutation(train.label.to_numpy()))
    booster = ltr.fit(shuffled, val, params)
    assert abs(ltr.val_ndcg(booster, val) - random_level) < 0.1


def test_train_writes_model_meta_plot_and_eval_row(built, home):
    meta = ltr.train(fixtures=True)
    assert (config.models_dir(True) / "ltr.txt").exists()
    saved = json.loads((config.models_dir(True) / "ltr_meta.json").read_text())
    assert saved["features"] == features.FEATURES and saved["val_ndcg10"] == meta["val_ndcg10"]
    assert (config.eval_reports_dir(True) / "ltr_importance.png").exists()
    assert len(list((home / "mlruns").rglob("meta.yaml"))) >= len(ltr.GRID)

    main(["--fixtures", "eval", "--systems", "bm25,ltr", "--split", "test"])
    report = json.loads((config.eval_reports_dir(True) / "metrics.json").read_text())
    assert "ltr" in report["systems"] and report["systems"]["ltr"]["all"]["n_queries"] > 0


def _write_qu_cache(recs):
    path = qu.cache_path(True)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(r) + "\n" for r in recs))


def _qu(qid, product_type):
    return {"query_id": int(qid), "must_not": [], "brand": None, "size": None, "color": None,
            "product_type": product_type}  # fmt: skip


def test_qu_features_join_on_query_and_zero_without_output(built):
    train = built["train"]
    df = pd.read_parquet(config.processed_dir(True) / "train.parquet")
    with_out, empty = df.query_id.unique()[:2]
    word = df[df.query_id == with_out].product_title.iloc[0].split()[0]
    _write_qu_cache([
        {"query_id": int(with_out), "locale": "us", "qu": _qu(with_out, word), "meta": []},
        {"query_id": int(empty), "locale": "us", "qu": None, "meta": []},
    ])  # fmt: skip
    out = qu.add_features(train, "train", fixtures=True)
    assert list(out.columns[-len(qu.FEATURES) :]) == list(qu.FEATURES)
    assert (out.query_id == train.query_id).all()
    assert (out.qu_present == (out.query_id == with_out)).all()
    assert out[out.query_id != with_out][list(qu.FEATURES)].eq(0).all().all()
    assert out[out.query_id == with_out].qu_type_in_title.iloc[0] == 1.0


def test_train_qu_and_eval_write_ltr_qu_row_and_qu_report(built, home):
    ltr.train(fixtures=True)
    with pytest.raises(SystemExit, match="not in the qu cache"):
        ltr.train_qu(fixtures=True)
    with pytest.raises(SystemExit, match="same call"):
        main(["--fixtures", "eval", "--systems", "ltr_qu"])
    cov = qu.coverage(fixtures=True)
    _write_qu_cache(
        {"query_id": int(q), "locale": loc, "qu": _qu(q, text.split()[-1]),
         "meta": [{"model": "fake", "input_tokens": 10, "output_tokens": 5}]}
        for q, text, loc in zip(cov.query_id, cov["query"], cov.locale, strict=True)
    )  # fmt: skip
    meta = ltr.train_qu(fixtures=True)
    with qu.cache_path(True).open("a") as f:
        f.write("\n")
    with pytest.raises(SystemExit, match="another qu cache"):
        main(["--fixtures", "eval", "--systems", "ltr,ltr_qu"])
    meta = ltr.train_qu(fixtures=True)
    assert meta["features"] == ltr.QU_FEATURES
    assert (config.models_dir(True) / "ltr_qu.txt").exists()

    main(["--fixtures", "eval", "--systems", "ltr,ltr_qu", "--split", "test"])
    report = json.loads((config.eval_reports_dir(True) / "metrics.json").read_text())
    assert report["systems"]["ltr_qu"]["all"]["n_queries"] > 0
    assert {(c["a"], c["b"]) for c in report["comparisons"]} >= {("ltr_qu", "ltr")}
    rep = json.loads(qu.report_path(True).read_text())
    assert [c["locale"] for c in rep["comparisons"]] == ["us", "es", "all"]
    assert rep["hard_test_queries"] == (cov.split == "test").sum() == rep["hard_test_with_output"]
    assert rep["systems"]["ltr_qu"]["all"]["n_queries"] <= rep["hard_test_queries"]
    assert rep["models"] == ["fake"] and rep["cost_per_1k_queries_usd"] is None
