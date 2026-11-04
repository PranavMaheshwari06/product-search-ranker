import json

import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import cohen_kappa_score

from product_search_ranker import config
from product_search_ranker.data.prepare import prepare
from product_search_ranker.evaluate import run as eval_run
from product_search_ranker.judge import agreement, judge
from product_search_ranker.judge.sample import sample
from product_search_ranker.llm.client import prompt_hash

FLIP = {"E": "S", "S": "E", "C": "I", "I": "C"}


def test_kappa_matches_sklearn():
    rng = np.random.default_rng(0)
    human = rng.choice(list("ESCI"), size=500, p=[0.5, 0.25, 0.05, 0.2])
    judged = np.where(rng.random(500) < 0.7, human, rng.choice(list("ESCI"), size=500))
    assert agreement.cohen_kappa(human, judged) == pytest.approx(
        cohen_kappa_score(human, judged, labels=list("ESCI"))
    )
    m = agreement.confusion_counts(human, judged)
    to_e = lambda x: np.where(x == "E", "E", "R")  # noqa: E731
    assert float(agreement.kappa_from_counts(agreement.exact_vs_rest(m))) == pytest.approx(
        cohen_kappa_score(to_e(human), to_e(judged))
    )


def test_parse_labels_counts_duplicates_unknown_and_bad_labels_as_missing():
    obj = [
        {"product_id": "a", "label": "E"},
        {"product_id": "b", "label": "S"},
        {"product_id": "b", "label": "S"},
        {"product_id": "c", "label": "X"},
        {"product_id": "zzz", "label": "I"},
    ]
    labels, missing = judge.parse_labels(obj, ["a", "b", "c", "d"])
    assert labels == {"a": "E"} and missing == ["b", "c", "d"]
    assert judge.parse_labels({"not": "a list"}, ["a"]) == ({}, ["a"])


def test_a_query_without_any_reply_is_not_cached(home):
    prepare(fixtures=True)
    test = pd.read_parquet(config.processed_dir(True) / "test.parquet", columns=judge.COLUMNS)
    rows = test[test.query_id == test.query_id.iloc[0]]
    assert judge.judge_query(rows, provider="replay") is None


@pytest.fixture
def judged_home(home):
    """Fixture data plus replay files: the 'judge' copies human labels, flips every 4th pair and
    leaves one product out of the first query."""
    prepare(fixtures=True)
    test = pd.read_parquet(config.processed_dir(True) / "test.parquet", columns=judge.COLUMNS)
    picked = sample(test)
    config.replay_dir().mkdir(parents=True)
    queries = sorted(picked.query_id.unique())

    def write(rows, drop_first=False):
        rows = rows.sort_values("product_id")
        first = rows.iloc[0]
        prompt = judge.build_prompt(first.query, first.locale, rows)
        items = [
            {"product_id": p, "label": fake[(q, p)]}
            for q, p in zip(rows.query_id, rows.product_id, strict=True)
        ]
        path = config.replay_dir() / f"{prompt_hash(prompt)}.json"
        path.write_text(
            json.dumps({"model": "replay-test", "text": json.dumps(items[int(drop_first) :])})
        )

    fake = {}
    for qid in queries:
        rows = picked[picked.query_id == qid].sort_values("product_id")
        for i, r in enumerate(rows.itertuples()):
            fake[(qid, r.product_id)] = FLIP[r.label] if i % 4 == 3 else r.label
    for qid in queries:
        rows = picked[picked.query_id == qid].sort_values("product_id")
        write(rows, drop_first=qid == queries[0])
        if qid in judge.single_query_ids(queries):
            for i in range(len(rows)):
                write(rows.iloc[[i]])
    return picked


def test_sample_is_stratified_seeded_and_capped(judged_home):
    test = pd.read_parquet(config.processed_dir(True) / "test.parquet", columns=judge.COLUMNS)
    a, b = sample(test), sample(test)
    assert a.equals(b)
    per_locale = a.groupby("locale").query_id.nunique()
    available = test.groupby("locale").query_id.nunique()
    assert (per_locale == available.clip(upper=pd.Series({"us": 100, "es": 50}))).all()
    assert set(a.bucket) <= {"len_1_2", "len_3_5", "len_6_plus"}
    small = sample(test, {"us": 6, "es": 3})
    assert small.groupby("locale").query_id.nunique().to_dict() == {"es": 3, "us": 6}


def test_judge_and_agreement_end_to_end(judged_home, monkeypatch):
    table = judge.run(fixtures=True, provider="replay")
    assert int(table.label_judge.isna().sum()) == 1
    assert table.label_single.notna().any()
    # Resumable: a second run makes no new calls (replay files could even be gone).
    for p in config.replay_dir().glob("*.json"):
        p.unlink()
    assert judge.run(fixtures=True, provider="replay").equals(table)

    def fake_scores(name, df, fixtures):
        gain = df.label.map(config.GAINS).to_numpy()
        return gain if name == "good" else -gain

    monkeypatch.setattr(eval_run, "score_system", fake_scores)
    report = agreement.run(["good", "bad"], fixtures=True)
    out = config.eval_reports_dir(True)
    assert (out / "judge_confusion.png").exists()
    assert json.loads((out / "judge.json").read_text())["n_pairs"] == len(table)

    assert report["n_missing"] == 1 and report["n_calls"] > 0
    kept = table[table.label_judge.notna()]
    assert report["pair_level"]["all"]["kappa"]["value"] == pytest.approx(
        cohen_kappa_score(kept.label_human, kept.label_judge, labels=list("ESCI")), abs=1e-4
    )
    lo, hi = report["pair_level"]["all"]["kappa"]["ci95"]
    assert lo <= report["pair_level"]["all"]["kappa"]["value"] <= hi
    sl = report["system_level"]
    assert sl["same_winner"] and sl["kendall_tau"] == 1.0 and sl["wrong_winner_pairs"] == []
    assert sl["n_dropped_incomplete"] == 1
    assert sl["systems"]["good"]["human"] == 1.0
    assert len(report["examples"]) == agreement.N_EXAMPLES
    assert report["batch_effect"]["accuracy"]["value"] == 1.0
