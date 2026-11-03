"""BM25 over each locale's products, one bm25s index per (locale, field)."""

import shutil
import time
from pathlib import Path

import bm25s
import numpy as np
import pandas as pd

from product_search_ranker import config
from product_search_ranker.text import tokenize

# `all` is the main index; the single-field ones are LTR features (P3). The description is left
# out of `all` because it is long and noisy.
FIELDS = {
    "all": ("product_title", "product_brand", "product_color", "product_bullet_point"),
    "title": ("product_title",),
    "bullets": ("product_bullet_point",),
    "brand": ("product_brand",),
    "desc": ("product_description",),
}


def index_path(locale: str, field: str, fixtures: bool = False) -> Path:
    return config.indexes_dir(fixtures) / f"bm25_{locale}_{field}"


def build_index(locale: str, field: str, fixtures: bool = False) -> Path:
    """Build and save one index; skip it if it is already on disk."""
    path = index_path(locale, field, fixtures)
    if (path / "product_ids.txt").exists():
        print(f"skip {path} (exists)", flush=True)
        return path
    t0 = time.time()
    products = pd.read_parquet(
        config.processed_dir(fixtures) / f"products_{locale}.parquet",
        columns=["product_id", *FIELDS[field]],
    )
    text = products[list(FIELDS[field])].fillna("").agg(" ".join, axis=1)
    corpus = [tokenize(t, locale) for t in text]
    retriever = bm25s.BM25()
    retriever.index(corpus, show_progress=False)
    # Write to a temp dir and rename, so an interrupted build is redone, not half-loaded.
    tmp = path.with_name(path.name + ".tmp")
    shutil.rmtree(tmp, ignore_errors=True)
    retriever.save(tmp, show_progress=False)
    (tmp / "product_ids.txt").write_text("\n".join(products.product_id) + "\n")
    shutil.rmtree(path, ignore_errors=True)
    tmp.rename(path)
    print(f"wrote {path} ({len(products):,} products, {time.time() - t0:.0f} s)", flush=True)
    return path


def load_index(locale: str, field: str, fixtures: bool = False):
    path = index_path(locale, field, fixtures)
    if not (path / "product_ids.txt").exists():
        flag = "--fixtures " if fixtures else ""
        raise SystemExit(f"{path} is missing: run `psr {flag}index` first")
    retriever = bm25s.BM25.load(path, show_progress=False, verbosity=0)
    ids = (path / "product_ids.txt").read_text().split()
    return retriever, ids


def score_candidates(rows: pd.DataFrame, field: str = "all", fixtures: bool = False) -> np.ndarray:
    """BM25 score of each (query, product) row, read from the query's scores over the full index.

    `rows` needs query_id, query, locale, product_id. Returns scores aligned with `rows`.
    """
    rows = rows.reset_index(drop=True)
    out = np.zeros(len(rows), dtype=np.float32)
    for locale, part in rows.groupby("locale", sort=False):
        retriever, ids = load_index(locale, field, fixtures)
        pos = pd.Series(np.arange(len(ids)), index=ids)
        doc_idx = pos.loc[part.product_id].to_numpy()
        part = part.assign(doc_idx=doc_idx)
        for _, g in part.groupby("query_id", sort=False):
            tokens = tokenize(g["query"].iloc[0], locale)
            if tokens:
                out[g.index] = retriever.get_scores(tokens)[g.doc_idx.to_numpy()]
    return out
