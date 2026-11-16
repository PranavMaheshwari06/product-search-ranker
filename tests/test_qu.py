import json

import pandas as pd
import pytest

from product_search_ranker import config
from product_search_ranker.data.prepare import prepare
from product_search_ranker.llm import query_understanding as qu
from product_search_ranker.llm.client import prompt_hash
from product_search_ranker.text import HARD_SLICES, query_slices


def _fake(qid, text):
    words = text.split()
    neg = [w for i, w in enumerate(words[1:], 1) if words[i - 1] in ("without", "sin")]
    return {
        "query_id": int(qid),
        "must_not": neg,
        "brand": None,
        "size": None,
        "color": None,
        "product_type": words[-1],
    }


def _write(locale, items, reply):
    path = config.replay_dir() / f"{prompt_hash(qu.build_prompt(locale, items))}.json"
    path.write_text(json.dumps({"model": "replay-test", "text": json.dumps(reply)}))


@pytest.fixture
def planned(home):
    """Replay files: in the first batch the first item is invalid and comes back valid on the
    retry; in batch `gap` the first query is missing twice (cached empty); the last batch gets
    no reply at all (nothing cached, asked again next run)."""
    prepare(fixtures=True)
    config.replay_dir().mkdir(parents=True)
    cov = qu.coverage(fixtures=True)
    plan = qu.batches(cov.sort_values(["locale", "query_id"]))
    gap = next(n for n, (_, items) in enumerate(plan) if n > 0 and len(items) > 1)
    assert gap < len(plan) - 1
    for n, (locale, items) in enumerate(plan):
        reply = [_fake(q, t) for q, t in items]
        if n == 0:
            reply[0]["brand"] = 5
            _write(locale, items[:1], [_fake(*items[0])])
        if n == gap:
            reply = reply[1:]
            _write(locale, items[:1], [])
        if n < len(plan) - 1:
            _write(locale, items, reply)
    return cov, plan, gap


def test_coverage_is_hard_queries_of_val_and_test_plus_train(planned):
    cov, _, _ = planned
    assert set(cov.split) == {"train", "val", "test"}
    assert not cov.query_id.duplicated().any()
    test = pd.read_parquet(config.processed_dir(True) / "test.parquet")
    hard = set()
    for (qid, query, locale), g in test.groupby(["query_id", "query", "locale"]):
        if query_slices(query, locale, list(g.product_brand)) & set(HARD_SLICES):
            hard.add(qid)
    assert set(cov.query_id[cov.split == "test"]) == hard
    assert 0 < len(hard) < test.query_id.nunique()
    assert (cov.split == "train").sum() <= qu.MAX_TRAIN


def test_run_retries_invalid_items_counts_empty_and_resumes(planned):
    cov, plan, gap = planned
    stats = qu.run(fixtures=True, provider="replay")
    totals = stats["per_split"]
    assert sum(s["done"] for s in totals.values()) == len(cov) - len(plan[-1][1])
    assert sum(s["empty"] for s in totals.values()) == 1
    # Two batches needed a retry; the last batch never got a reply.
    assert stats["n_calls"] == (len(plan) - 1) + 2
    cache = qu.read_cache(fixtures=True)
    first = plan[0][1][0][0]
    assert cache[first]["qu"]["brand"] is None
    assert cache[plan[gap][1][0][0]]["qu"] is None
    assert not set(q for q, _ in plan[-1][1]) & set(cache)

    for p in config.replay_dir().glob("*.json"):
        p.unlink()
    again = qu.run(fixtures=True, provider="replay")
    assert again == stats


def test_parse_items_keeps_ids_seen_exactly_once():
    obj = [{"query_id": 1}, {"query_id": 2}, {"query_id": 2}, {"query_id": 3, "size": [1]}]
    ok, failed = qu.parse_items(obj, [1, 2, 3, 4])
    assert list(ok) == [1] and ok[1]["must_not"] == [] and failed == [2, 3, 4]


def test_features_neutral_without_output_and_hand_cases():
    product = {
        "product_title": "Zapatillas de running Nike con cordones, 24 oz",
        "product_brand": "Nike",
        "product_color": "Rojo",
        "product_bullet_point": "Pack de 2",
    }
    assert qu.qu_features(None, product, "es") == dict.fromkeys(qu.FEATURES, 0.0)
    q = {
        "query_id": 1,
        "must_not": ["cordones"],
        "brand": "nike",
        "size": "24oz",
        "color": "rojo",
        "product_type": "zapatillas running",
    }
    f = qu.qu_features(q, product, "es")
    assert f == {
        "qu_present": 1.0,
        "must_not_in_title": 1.0,
        "qu_brand_match": 1.0,
        "qu_size_match": 1.0,
        "qu_color_match": 1.0,
        "qu_type_in_title": 1.0,
    }
    other = {
        **q,
        "must_not": ["velcro"],
        "brand": "adidas",
        "size": "4 oz",
        "color": "azul",
        "product_type": "botas running",
    }
    f = qu.qu_features(other, product, "es")
    assert f["must_not_in_title"] == f["qu_brand_match"] == f["qu_color_match"] == 0.0
    assert f["qu_size_match"] == 0.0  # "4 oz" must not match inside "24 oz"
    assert f["qu_type_in_title"] == 0.5
