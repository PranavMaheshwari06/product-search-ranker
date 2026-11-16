"""Dense retrieval (SPEC 5.2): multilingual-e5-small embeddings, cosine per candidate."""

import time
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd

from product_search_ranker import config

MODEL = "intfloat/multilingual-e5-small"
SHARD = 20_000
BATCH = 512
MIN_FREE_GIB = 3.0
TIMING_N = 2_000
COLUMNS = ["product_id", "product_title", "product_brand", "product_color", "product_bullet_point"]


def index_dir(locale: str, fixtures: bool = False) -> Path:
    return config.indexes_dir(fixtures) / f"dense_{locale}"


def passage(title, brand, color, bullets) -> str:
    return "passage: " + " ".join(filter(None, (title, brand, color, (bullets or "")[:400])))


@lru_cache(maxsize=1)
def model():
    import torch
    from sentence_transformers import SentenceTransformer

    device = "cuda" if torch.cuda.is_available() else "cpu"
    m = SentenceTransformer(MODEL, device=device)
    m.max_seq_length = 256
    if device == "cuda":
        m.half()  # fp16 on the GPU (Decision log 2026-11-04); CPU stays fp32.
    return m


def encode(texts: list[str]) -> np.ndarray:
    """L2-normalised embeddings as float16. Texts already carry their e5 prefix."""
    emb = model().encode(
        texts, batch_size=BATCH, normalize_embeddings=True, convert_to_numpy=True,
        show_progress_bar=False,
    )  # fmt: skip
    emb = emb.astype(np.float16)
    if not np.isfinite(emb).all():  # fp16 can overflow; NaN scores would sort silently last
        raise SystemExit("non-finite embeddings: rerun on CPU or lower max_seq_length")
    return emb


def check_vram() -> None:
    """Stop before a long GPU job when another program holds most of the GPU memory."""
    import torch

    if not torch.cuda.is_available():
        return
    torch.cuda.empty_cache()  # the previous locale's cache is ours, not another program's
    free, total = torch.cuda.mem_get_info()
    if free < MIN_FREE_GIB * 2**30:
        raise SystemExit(
            f"only {free / 2**30:.2f} of {total / 2**30:.2f} GiB GPU memory free, need "
            f"{MIN_FREE_GIB} GiB. Close programs that use the GPU (browser tabs with video, "
            "games, other notebooks), check with `nvidia-smi`, then rerun `psr embed`: it resumes."
        )


def embed(locale: str, fixtures: bool = False) -> Path:
    """Embed every product of a locale, one saved shard per SHARD products; resumes from shards."""
    out = index_dir(locale, fixtures)
    if (out / "embeddings.npy").exists():
        print(f"exists {out}", flush=True)
        return out
    check_vram()
    path = config.processed_dir(fixtures) / f"products_{locale}.parquet"
    products = pd.read_parquet(path, columns=COLUMNS)
    out.mkdir(parents=True, exist_ok=True)
    n_shards = -(-len(products) // SHARD)
    print(f"{locale}: {len(products):,} products, {n_shards} shards of {SHARD:,}", flush=True)
    t0 = time.time()
    for i in range(n_shards):
        shard = out / f"part_{i:04d}.npy"
        part = products.iloc[i * SHARD : (i + 1) * SHARD]
        if shard.exists():
            if len(np.load(shard, mmap_mode="r")) != len(part):
                raise SystemExit(f"{shard} has the wrong row count: delete {out} and rerun")
            continue
        texts = [passage(*row) for row in part[COLUMNS[1:]].itertuples(index=False, name=None)]
        if i == 0:
            t = time.time()
            head = encode(texts[:TIMING_N])
            rate = len(head) / (time.time() - t)
            print(
                f"{locale}: {rate:.0f} passages/s on the first {len(head):,}, "
                f"estimate {len(products) / rate / 60:.1f} min for the locale",
                flush=True,
            )
            emb = np.vstack([head, encode(texts[TIMING_N:])]) if len(texts) > TIMING_N else head
        else:
            emb = encode(texts)
        with open(shard.with_suffix(".tmp"), "wb") as f:
            np.save(f, emb)
        shard.with_suffix(".tmp").rename(shard)
        print(f"{locale}: shard {i + 1}/{n_shards} ({time.time() - t0:.0f} s)", flush=True)

    parts = [np.load(out / f"part_{i:04d}.npy") for i in range(n_shards)]
    emb = np.vstack(parts)
    if len(emb) != len(products):
        raise SystemExit(f"{out}: {len(emb):,} embeddings for {len(products):,} products")
    (out / "product_ids.txt").write_text("\n".join(products.product_id))
    with open(out / "embeddings.tmp", "wb") as f:
        np.save(f, emb)
    (out / "embeddings.tmp").rename(out / "embeddings.npy")
    for i in range(n_shards):
        (out / f"part_{i:04d}.npy").unlink()
    print(f"wrote {out} ({len(emb):,} x {emb.shape[1]}, {time.time() - t0:.0f} s)", flush=True)
    return out


class DenseIndex:
    """Saved product embeddings plus the product_id -> row lookup."""

    def __init__(self, locale: str, fixtures: bool = False):
        path = index_dir(locale, fixtures)
        if not (path / "embeddings.npy").exists():
            if not fixtures:
                raise SystemExit(f"{path} is missing: run `psr embed` first (a GPU job)")
            embed(locale, fixtures)
        self.emb = np.load(path / "embeddings.npy", mmap_mode="r")
        ids = (path / "product_ids.txt").read_text().split("\n")
        self.row = {pid: i for i, pid in enumerate(ids)}


def score(df: pd.DataFrame, fixtures: bool = False) -> np.ndarray:
    """Cosine of each (query, candidate) row of `df`; result is aligned with `df`."""
    df = df.reset_index(drop=True)
    out = np.zeros(len(df), dtype=np.float64)
    for locale, part in df.groupby("locale", sort=False):
        index = DenseIndex(locale, fixtures)
        queries = part.drop_duplicates("query_id")
        q = encode([f"query: {s}" for s in queries["query"]]).astype(np.float32)
        q_row = pd.Series(np.arange(len(queries)), index=queries.query_id)
        p_rows = np.fromiter((index.row[p] for p in part.product_id), dtype=np.int64)
        p = np.asarray(index.emb[p_rows], dtype=np.float32)
        out[part.index] = np.einsum("ij,ij->i", p, q[q_row[part.query_id].to_numpy()])
    return out
