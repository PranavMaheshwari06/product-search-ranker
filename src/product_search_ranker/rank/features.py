"""LTR features per (query, product) row (SPEC 5.4). Nothing here reads `label`.

Features for a split are cached in `<processed>/features_<split>.parquet`; each BM25 field and the
dense score are cached on their own first, so a crashed build resumes where it stopped.
"""

import json
import re
import shutil
from pathlib import Path

import numpy as np
import pandas as pd

from product_search_ranker import config
from product_search_ranker.retrieval import bm25, dense, hybrid
from product_search_ranker.text import TOKEN_RE, normalize, tokenize

TEXT_COLUMNS = [
    "product_title",
    "product_brand",
    "product_color",
    "product_bullet_point",
    "title_raw_len",
    "bullet_raw_len",
    "description_raw_len",
]
KEY_COLUMNS = ["query_id", "query", "locale", "product_id"]

FEATURES = [
    "bm25_all",
    "bm25_title",
    "bm25_bullets",
    "bm25_brand",
    "bm25_desc",
    "bm25_rank",
    "dense_cos",
    "dense_rank",
    "rrf_score",
    "title_jaccard",
    "query_tokens_in_title",
    "brand_in_query",
    "color_in_query",
    "number_match",
    "title_len",
    "has_description",
    "bullet_len",  # SPEC lists n_bullets; `prepare` flattens bullets, so their length stands in
    "query_len",
    "n_candidates",
    "locale_es",
]
NUMBER_RE = re.compile(r"\d+")


def features_path(split: str, fixtures: bool = False) -> Path:
    return config.processed_dir(fixtures) / f"features_{split}.parquet"


def _cached(path: Path, compute) -> np.ndarray:
    if path.exists():
        return np.load(path)
    values = np.asarray(compute(), dtype=np.float64)
    tmp = path.with_name(path.stem + ".tmp.npy")
    np.save(tmp, values)
    tmp.rename(path)
    return values


def _phrase(s: str | None) -> str:
    return " " + " ".join(TOKEN_RE.findall(normalize(s or ""))) + " "


def text_features(df: pd.DataFrame) -> pd.DataFrame:
    """Overlap and shape features computed from the query and the product's own fields."""
    queries = df[["query_id", "query", "locale"]].drop_duplicates("query_id")
    q_tok = {qid: set(tokenize(q, loc)) for qid, q, loc in queries.values}
    q_phrase = {qid: _phrase(q) for qid, q, _ in queries.values}
    rows = []
    for r in df.itertuples(index=False):
        q = q_tok[r.query_id]
        title = set(tokenize(r.product_title, r.locale))
        inter, union = len(q & title), len(q | title)
        brand, color = _phrase(r.product_brand), _phrase(r.product_color)
        nums = {t for t in q if NUMBER_RE.fullmatch(t)}
        if nums:
            body = title | set(tokenize(r.product_bullet_point, r.locale))
            num_match = len(nums & body) / len(nums)
        else:
            num_match = np.nan  # no number in the query: the feature does not apply
        rows.append(
            (
                inter / union if union else 0.0,
                inter / len(q) if q else 0.0,
                float(brand.strip() != "" and brand in q_phrase[r.query_id]),
                float(color.strip() != "" and color in q_phrase[r.query_id]),
                num_match,
            )
        )
    cols = ["title_jaccard", "query_tokens_in_title", "brand_in_query", "color_in_query"]
    out = pd.DataFrame(rows, columns=[*cols, "number_match"], index=df.index)
    out["title_len"] = df.title_raw_len.astype(float)
    out["has_description"] = (df.description_raw_len.astype(float) > 0).astype(float)
    out["bullet_len"] = df.bullet_raw_len.astype(float)
    out["query_len"] = df.query_id.map(lambda qid: len(q_tok[qid])).astype(float)
    out["n_candidates"] = df.groupby("query_id").product_id.transform("size").astype(float)
    out["locale_es"] = (df.locale == "es").astype(float)
    return out


def build(split: str, fixtures: bool = False) -> pd.DataFrame:
    """Feature table for a split, rows in the order of the split's parquet file."""
    from product_search_ranker.evaluate.run import data_sha

    path = features_path(split, fixtures)
    cache = path.with_suffix("")
    # ponytail: rebuilt BM25 indexes or embeddings with unchanged data are not detected; delete
    # features_* by hand then. Upgrade: hash the index files too.
    fp = {"data_sha": data_sha(fixtures), "features": FEATURES, "model": dense.MODEL}
    fp_path = cache / "fingerprint.json"
    if path.exists():
        if fp_path.exists() and json.loads(fp_path.read_text()) == fp:
            return pd.read_parquet(path)
        path.unlink()
        shutil.rmtree(cache, ignore_errors=True)
        print(f"{split}: cached features are stale, rebuilding", flush=True)
    df = pd.read_parquet(
        config.processed_dir(fixtures) / f"{split}.parquet",
        columns=[*KEY_COLUMNS, "label", *TEXT_COLUMNS],
    )
    cache.mkdir(parents=True, exist_ok=True)
    fp_path.write_text(json.dumps(fp))
    keys = df[KEY_COLUMNS]
    out = df[["query_id", "locale", "product_id", "label"]].copy()
    for field in bm25.FIELDS:
        out[f"bm25_{field}"] = _cached(
            cache / f"bm25_{field}.npy", lambda f=field: bm25.score(keys, fixtures, f)
        )
        print(f"{split}: bm25_{field} ready", flush=True)
    out["dense_cos"] = _cached(cache / "dense_cos.npy", lambda: dense.score(keys, fixtures))
    print(f"{split}: dense_cos ready", flush=True)
    out["bm25_rank"] = hybrid.ranks(df, out.bm25_all)
    out["dense_rank"] = hybrid.ranks(df, out.dense_cos)
    out["rrf_score"] = 1 / (hybrid.K_RRF + out.bm25_rank) + 1 / (hybrid.K_RRF + out.dense_rank)
    out = out.join(text_features(df))
    tmp = path.with_name(path.stem + ".tmp.parquet")
    out.to_parquet(tmp, index=False)
    tmp.rename(path)
    print(f"wrote {path} ({len(out):,} rows)", flush=True)
    return out
