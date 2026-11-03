"""BM25 (bm25s) over each locale's products; one index per (locale, field)."""

import shutil
import time
from pathlib import Path

import bm25s
import numpy as np
import pandas as pd

from product_search_ranker import config
from product_search_ranker.text import tokenize

# The description is long and noisy, so it stays out of `all` and gets its own index for LTR.
FIELDS = {
    "all": ("product_title", "product_brand", "product_color", "product_bullet_point"),
    "title": ("product_title",),
    "bullets": ("product_bullet_point",),
    "brand": ("product_brand",),
    "desc": ("product_description",),
}


def index_dir(locale: str, field: str = "all", fixtures: bool = False) -> Path:
    return config.indexes_dir(fixtures) / f"bm25_{locale}_{field}"


def build(locale: str, field: str = "all", fixtures: bool = False) -> Path:
    """Build and save one index; skipped when it already exists."""
    out = index_dir(locale, field, fixtures)
    if (out / "product_ids.txt").exists():
        print(f"exists {out}")
        return out
    t0 = time.time()
    cols = list(FIELDS[field])
    path = config.processed_dir(fixtures) / f"products_{locale}.parquet"
    products = pd.read_parquet(path, columns=["product_id", *cols])
    # Token ids are built here so the corpus is never held as millions of token strings.
    vocab: dict[str, int] = {}
    corpus = [
        [vocab.setdefault(t, len(vocab)) for t in tokenize(" ".join(filter(None, row)), locale)]
        for row in products[cols].itertuples(index=False, name=None)
    ]
    retriever = bm25s.BM25()
    retriever.index((corpus, vocab), show_progress=False)
    tmp = out.with_name(out.name + ".tmp")
    shutil.rmtree(tmp, ignore_errors=True)
    retriever.save(tmp, show_progress=False)
    (tmp / "product_ids.txt").write_text("\n".join(products.product_id))
    tmp.rename(out)
    print(f"wrote {out} ({len(products):,} products, {time.time() - t0:.0f} s)")
    return out


class Bm25Index:
    """A loaded index plus the product_id -> row lookup."""

    def __init__(self, locale: str, field: str = "all", fixtures: bool = False):
        path = build(locale, field, fixtures)
        self.locale = locale
        self.retriever = bm25s.BM25.load(path, show_progress=False)
        ids = (path / "product_ids.txt").read_text().split("\n")
        self.row = {pid: i for i, pid in enumerate(ids)}

    def scores(self, query: str, product_ids) -> np.ndarray:
        """BM25 score of `query` for each product, read from the full-index score vector."""
        rows = np.fromiter((self.row[p] for p in product_ids), dtype=np.int64)
        tokens = tokenize(query, self.locale)
        ids = self.retriever.get_tokens_ids(tokens)
        if not ids:
            return np.zeros(len(rows), dtype=np.float32)
        return self.retriever.get_scores(ids)[rows]


def score(df: pd.DataFrame, fixtures: bool = False, field: str = "all") -> np.ndarray:
    """Score every (query, candidate) row of `df`; result is aligned with `df`."""
    df = df.reset_index(drop=True)
    out = np.zeros(len(df), dtype=np.float64)
    for locale, part in df.groupby("locale", sort=False):
        index = Bm25Index(locale, field, fixtures)
        for (_, query), g in part.groupby(["query_id", "query"], sort=False):
            out[g.index] = index.scores(query, g.product_id)
    return out
