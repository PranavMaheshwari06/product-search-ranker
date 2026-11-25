import io
import json

import pytest

from product_search_ranker import config
from product_search_ranker.evaluate.gate import decide
from product_search_ranker.pipeline import publish as pub


def report(ndcg):
    return {"systems": {"ltr": {"all": {"ndcg@10": ndcg}}}}


def test_gate_decisions():
    assert decide(report(0.70), None, 0.005)[0]  # first run
    assert decide(report(0.74), report(0.74), 0.005)[0]
    assert decide(report(0.736), report(0.74), 0.005)[0]  # inside the margin
    assert decide(report(0.735), report(0.74), 0.005)[0]  # exactly at the margin passes
    assert not decide(report(0.734), report(0.74), 0.005)[0]
    assert not decide({}, report(0.74), 0.005)[0]  # no candidate number
    assert not decide(report(0.74), {}, 0.005)[0]  # broken published file
    assert not decide(report(float("nan")), report(0.74), 0.005)[0]
    assert not decide(report(0.74), report(0.74), float("nan"))[0]
    assert not decide(report(0.74), None, -0.1)[0]


def test_parse_target():
    assert pub.parse_target("s3://b/a/b/") == ("s3", "b", "a/b")
    assert pub.parse_target("local:/x") == ("local", "/x", "")
    for bad in ("s3://", "local:", "local:rel/x", "/x", "gs://b"):
        with pytest.raises(SystemExit):
            pub.parse_target(bad)


@pytest.fixture
def built(home):
    """Fake model files and candidate metrics, as train + eval --candidate leave them."""
    models = config.models_dir(True)
    models.mkdir(parents=True)
    (models / "ltr.txt").write_text("model")
    (models / "ltr_meta.json").write_text("{}")
    cand = config.candidate_dir(True)
    cand.mkdir(parents=True)
    (cand / "metrics.json").write_text(json.dumps(report(0.7)))
    return home


def test_publish_local_then_read_back(built, tmp_path):
    out = tmp_path / "pub"
    assert pub.read_published(f"local:{out}") is None
    pub.publish(f"local:{out}", fixtures=True)
    assert (out / "ltr.txt").read_text() == "model"
    assert pub.read_published(f"local:{out}") == report(0.7)


def test_publish_dry_run_writes_nothing(built, tmp_path, monkeypatch):
    monkeypatch.setattr(pub, "_s3_client", lambda s3=None: pytest.fail("no client on dry run"))
    pub.publish("s3://bucket/p", fixtures=True, dry_run=True)
    pub.publish(f"local:{tmp_path / 'pub'}", fixtures=True, dry_run=True)
    assert not (tmp_path / "pub").exists()


def test_publish_refuses_without_credentials(built, monkeypatch):
    import boto3

    monkeypatch.setattr(boto3.Session, "get_credentials", lambda self: None)
    with pytest.raises(SystemExit, match="credentials"):
        pub.publish("s3://bucket/p", fixtures=True)


def test_publish_missing_model_refuses(home, tmp_path):
    with pytest.raises(SystemExit, match="missing"):
        pub.publish(f"local:{tmp_path / 'pub'}", fixtures=True)


class FakeS3:
    class exceptions:
        class NoSuchKey(Exception):
            pass

    def __init__(self):
        self.objects = {}

    def upload_file(self, src, bucket, key):
        self.objects[(bucket, key)] = open(src, "rb").read()

    def get_object(self, Bucket, Key):
        if (Bucket, Key) not in self.objects:
            raise self.exceptions.NoSuchKey
        return {"Body": io.BytesIO(self.objects[(Bucket, Key)])}


def test_publish_s3_with_fake_client(built):
    s3 = FakeS3()
    assert pub.read_published("s3://b/models", s3=s3) is None
    pub.publish("s3://b/models", fixtures=True, s3=s3)
    assert ("b", "models/ltr.txt") in s3.objects
    assert pub.read_published("s3://b/models", s3=s3) == report(0.7)


def test_candidate_eval_leaves_metrics_json_alone(home):
    from product_search_ranker.evaluate.run import evaluate, report_path

    path = evaluate(["bm25"], fixtures=True, candidate=True)
    assert path == config.candidate_dir(True) / "metrics.json"
    assert not report_path("test", True).exists()


def test_publish_refuses_stale_or_non_ltr_metrics(built):
    import os

    metrics = config.candidate_dir(True) / "metrics.json"
    model = config.models_dir(True) / "ltr.txt"
    os.utime(metrics, (model.stat().st_mtime - 10,) * 2)
    with pytest.raises(SystemExit, match="older than the model"):
        pub.files_to_publish(True)
    metrics.write_text(json.dumps({"systems": {"bm25": {}}}))
    with pytest.raises(SystemExit, match="no ltr"):
        pub.files_to_publish(True)


def test_metrics_are_published_last(built):
    demo = config.indexes_dir(True) / "demo"
    demo.mkdir(parents=True)
    (demo / "products.parquet").write_text("x")
    assert list(pub.files_to_publish(True))[-1] == pub.PUBLISHED_METRICS
