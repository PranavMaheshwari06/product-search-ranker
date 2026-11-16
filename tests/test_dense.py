import numpy as np
import pandas as pd
import pytest

from product_search_ranker import config
from product_search_ranker.cli import main


@pytest.fixture
def prepared(home):
    main(["--fixtures", "prepare"])
    return home


def test_embed_writes_aligned_normalised_float16(prepared, fake_dense):
    out = fake_dense.embed("us", fixtures=True)
    emb = np.load(out / "embeddings.npy")
    ids = (out / "product_ids.txt").read_text().split("\n")
    products = pd.read_parquet(config.processed_dir(True) / "products_us.parquet")
    assert emb.dtype == np.float16 and len(emb) == len(ids) == len(products)
    assert ids == list(products.product_id)
    np.testing.assert_allclose(np.linalg.norm(emb.astype(np.float32), axis=1), 1.0, atol=1e-2)
    assert not list(out.glob("part_*"))


def test_embed_resumes_from_saved_shards(prepared, fake_dense, monkeypatch):
    monkeypatch.setattr(fake_dense, "SHARD", 50)
    monkeypatch.setattr(fake_dense, "TIMING_N", 10)
    full = np.load(fake_dense.embed("us", fixtures=True) / "embeddings.npy")
    out = fake_dense.index_dir("us", fixtures=True)
    n_shards = -(-len(full) // 50)
    (out / "embeddings.npy").unlink()
    for i in range(n_shards - 1):  # as if the job died before the last shard
        np.save(out / f"part_{i:04d}.npy", full[i * 50 : (i + 1) * 50])

    calls = []
    real = fake_dense.encode
    monkeypatch.setattr(fake_dense, "encode", lambda texts: calls.append(len(texts)) or real(texts))
    np.testing.assert_array_equal(np.load(fake_dense.embed("us", True) / "embeddings.npy"), full)
    assert calls == [len(full) - (n_shards - 1) * 50]


def test_score_is_cosine_and_ranks_a_title_match_first(prepared, fake_dense):
    products = pd.read_parquet(config.processed_dir(True) / "products_us.parquet").head(5)
    title = products.product_title.iloc[2]
    df = pd.DataFrame(
        {"query_id": 1, "query": title, "locale": "us", "product_id": products.product_id}
    )
    got = fake_dense.score(df, fixtures=True)
    assert got.argmax() == 2 and got.max() > 0.5
    assert np.all(got <= 1.0 + 1e-3) and np.all(got >= -1.0 - 1e-3)


def test_real_data_without_index_says_to_run_embed(tmp_path, monkeypatch, fake_dense):
    monkeypatch.setenv("PSR_HOME", str(tmp_path))
    with pytest.raises(SystemExit, match="psr embed"):
        fake_dense.DenseIndex("us", fixtures=False)


def test_e5_prefixes_on_passages_and_queries(prepared, fake_dense, monkeypatch):
    from product_search_ranker.evaluate.run import load_split

    seen = []
    real = fake_dense.encode
    monkeypatch.setattr(fake_dense, "encode", lambda texts: seen.extend(texts) or real(texts))
    fake_dense.embed("us", fixtures=True)
    assert seen and all(t.startswith("passage: ") for t in seen)
    seen.clear()
    df = load_split("test", fixtures=True)
    fake_dense.score(df[df.locale == "us"], fixtures=True)
    assert seen and all(t.startswith("query: ") for t in seen)


def test_embed_rejects_a_shard_with_the_wrong_row_count(prepared, fake_dense, monkeypatch):
    monkeypatch.setattr(fake_dense, "SHARD", 50)
    out = fake_dense.index_dir("us", fixtures=True)
    out.mkdir(parents=True)
    np.save(out / "part_0000.npy", np.zeros((7, 64), dtype=np.float16))
    with pytest.raises(SystemExit, match="wrong row count"):
        fake_dense.embed("us", fixtures=True)
