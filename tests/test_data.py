import duckdb
import httpx
import pandas as pd
import pytest

from product_search_ranker import config
from product_search_ranker.cli import main
from product_search_ranker.data import fetch as fetch_mod
from product_search_ranker.data.prepare import VAL_RULE, check_counts

from .conftest import FIXTURE


def test_fixture_shape():
    df = pd.read_parquet(FIXTURE)
    assert FIXTURE.stat().st_size < 2_000_000
    assert set(df.label) == set(config.LABELS)
    assert set(df.locale) == set(config.LOCALES)
    assert df.groupby("locale").query_id.nunique().to_dict() == {"us": 150, "es": 50}
    assert df.query_id.nunique() == 200
    assert df.product_description.str.len().max() <= 300


def test_prepare_fixtures_no_query_in_two_splits(home):
    main(["--fixtures", "prepare"])
    out = config.processed_dir(fixtures=True)
    splits = {s: pd.read_parquet(out / f"{s}.parquet") for s in config.SPLITS}
    assert all(len(df) for df in splits.values())
    seen = {}
    for name, df in splits.items():
        for qid in df.query_id.unique():
            assert seen.setdefault(qid, name) == name, f"query {qid} in {seen[qid]} and {name}"
    for loc in config.LOCALES:
        prods = pd.read_parquet(out / f"products_{loc}.parquet")
        assert not prods.duplicated(["product_id", "locale"]).any()
        assert (prods.locale == loc).all()


def test_validation_split_follows_the_hash_rule():
    df = pd.read_parquet(FIXTURE, columns=["query_id", "split"]).drop_duplicates()
    con = duckdb.connect()
    con.register("q", df)
    in_val = con.execute(f"SELECT query_id, {VAL_RULE} AS v FROM q").df().set_index("query_id").v
    got = df.set_index("query_id").split.eq("val")
    non_test = df.set_index("query_id").split.ne("test")
    assert (in_val[non_test] == got[non_test]).all()


def test_check_counts_stops_when_off_by_more_than_half_a_percent():
    ok = {"us": {"queries": 29_844, "judgements": 601_354},
          "es": {"queries": 8_049, "judgements": 218_774}}  # fmt: skip
    check_counts(ok)
    bad = {**ok, "es": {"queries": 8_049, "judgements": 218_774 * 1.01}}
    with pytest.raises(SystemExit, match="STOP"):
        check_counts(bad)


def _client(handler):
    return httpx.Client(transport=httpx.MockTransport(handler))


def test_download_rejects_an_lfs_pointer(tmp_path):
    pointer = b"version https://git-lfs.github.com/spec/v1\noid sha256:abc\nsize 10\n"
    client = _client(lambda r: httpx.Response(200, content=pointer))
    dest = tmp_path / "x.parquet"
    with pytest.raises(RuntimeError, match="LFS"):
        fetch_mod._download(client, "https://example.test/x", dest, None)
    assert not dest.exists()


def test_download_resumes_from_a_part_file(tmp_path):
    body = bytes(range(256)) * 4

    def handler(request):
        start = int(request.headers.get("range", "bytes=0-").split("=")[1].split("-")[0])
        return httpx.Response(206 if start else 200, content=body[start:])

    dest = tmp_path / "x.bin"
    (tmp_path / "x.bin.part").write_bytes(body[:100])
    fetch_mod._download(_client(handler), "https://example.test/x", dest, len(body))
    assert dest.read_bytes() == body
