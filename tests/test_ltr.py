import json
import shutil

import numpy as np
import pandas as pd
import pytest

from product_search_ranker import config
from product_search_ranker.cli import main
from product_search_ranker.evaluate import metrics
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
    assert not any("label" in f or "esci" in f for f in features.FEATURES)


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
