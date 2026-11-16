"""Query understanding for hard queries: an LLM reads each query once and returns its constraints.

Hard queries are those in the negation, number_or_unit or brand_in_query slices (SPEC 5.5).
One call covers 20 queries of one locale; items that come back invalid are asked again once,
then left empty and counted. Results go to an append-only JSONL cache, so runs resume.
The `qu_*` features below are what P3's feature table adds for the `ltr_qu` system.
"""

import json
import re

import numpy as np
import pandas as pd
from pydantic import BaseModel, ValidationError

from product_search_ranker import config
from product_search_ranker.llm.client import LLMError, complete_json, cost_usd, prompt_hash
from product_search_ranker.text import HARD_SLICES, normalize, query_slices, tokenize

BATCH = 20
MAX_TRAIN = 3000
FEATURES = (
    "qu_present",
    "must_not_in_title",
    "qu_brand_match",
    "qu_size_match",
    "qu_color_match",
    "qu_type_in_title",
)


class QueryUnderstanding(BaseModel):
    query_id: int
    must_not: list[str] = []
    brand: str | None = None
    size: str | None = None
    color: str | None = None
    product_type: str | None = None


def hard_queries(split: str, fixtures: bool = False) -> pd.DataFrame:
    """query_id, query, locale of the split's queries that fall in a hard slice."""
    cols = ["query_id", "query", "locale", "product_brand"]
    df = pd.read_parquet(config.processed_dir(fixtures) / f"{split}.parquet", columns=cols)
    rows = []
    for (qid, query, locale), g in df.groupby(["query_id", "query", "locale"], sort=True):
        if query_slices(query, locale, list(g.product_brand)) & set(HARD_SLICES):
            rows.append((qid, query, locale))
    return pd.DataFrame(rows, columns=["query_id", "query", "locale"])


def coverage(fixtures: bool = False) -> pd.DataFrame:
    """All hard val and test queries, plus up to 3,000 hard train queries (seed 0)."""
    parts = [hard_queries(s, fixtures).assign(split=s) for s in ("val", "test")]
    train = hard_queries("train", fixtures)
    if len(train) > MAX_TRAIN:
        pick = np.random.default_rng(config.SEED).choice(len(train), MAX_TRAIN, replace=False)
        train = train.iloc[np.sort(pick)]
    parts.append(train.assign(split="train"))
    return pd.concat(parts, ignore_index=True)


def build_prompt(locale: str, items: list[tuple[int, str]]) -> str:
    lines = "\n".join(
        json.dumps({"query_id": int(q), "query": t}, ensure_ascii=False) for q, t in items
    )
    shop = "Spanish (amazon.es)" if locale == "es" else "English (amazon.com)"
    return f"""You read product search queries for an online shop and extract their constraints.
Shop language: {shop}.

For each query return one JSON object:
{{"query_id": <int>, "must_not": [<str>], "brand": <str or null>, "size": <str or null>,
  "color": <str or null>, "product_type": <str or null>}}
- must_not: things the query excludes ("without laces" -> ["laces"], "sin cordón" -> ["cordón"]).
- brand: the brand the query asks for, if any.
- size: a size, amount or count with its unit as written ("24 oz", "13 pulgadas", "2 pack").
- color: the colour asked for, if any.
- product_type: the kind of product, in a few words ("running shoes", "funda").
Copy words as they appear in the query; use null or [] when the query does not say.

Queries, one JSON object per line:
{lines}

Return a JSON array with exactly one object per query_id above."""


def parse_items(obj, query_ids: list[int]) -> tuple[dict[int, dict], list[int]]:
    """Valid items for ids that appear exactly once; every other id is returned as failed."""
    counts: dict[int, int] = {}
    good: dict[int, dict] = {}
    for item in obj if isinstance(obj, list) else []:
        try:
            qu = QueryUnderstanding.model_validate(item)
        except ValidationError:
            continue
        counts[qu.query_id] = counts.get(qu.query_id, 0) + 1
        good[qu.query_id] = qu.model_dump()
    ok = {q: good[q] for q in query_ids if counts.get(q) == 1}
    return ok, [q for q in query_ids if q not in ok]


def _call(locale, items, provider, model, record):
    """One call; returns (ok items, failed ids, meta or None)."""
    prompt = build_prompt(locale, items)
    try:
        obj, meta = complete_json(prompt, provider=provider, model=model, record=record)
    except LLMError as e:
        print(f"qu: call failed for {len(items)} queries: {e}", flush=True)
        return {}, [q for q, _ in items], None
    ok, failed = parse_items(obj, [q for q, _ in items])
    return ok, failed, {**meta, "call": prompt_hash(prompt)}


def understand_batch(locale, items, provider=None, model=None, record=False) -> list[dict]:
    """Cache records for one batch: a first call, then one call for the items that failed.

    Items whose last call got no reply at all (network, quota) get no record, so the next run
    asks again; items that came back invalid twice are cached empty and counted.
    """
    ok, failed, meta = _call(locale, items, provider, model, record)
    metas = [meta] if meta else []
    unanswered = set() if meta else set(failed)
    if failed:
        retry = [(q, t) for q, t in items if q in set(failed)]
        ok2, failed, meta2 = _call(locale, retry, provider, model, record)
        ok |= ok2
        metas += [meta2] if meta2 else []
        unanswered = set() if meta2 else set(failed)
    recs = [
        {"query_id": int(q), "locale": locale, "qu": ok.get(q), "meta": []}
        for q, _ in items
        if q not in unanswered
    ]
    if recs:
        recs[0]["meta"] = metas
    return recs


def cache_path(fixtures: bool = False):
    return config.llm_cache_dir(fixtures) / "qu.jsonl"


def read_cache(fixtures: bool = False) -> dict[int, dict]:
    path = cache_path(fixtures)
    if not path.exists():
        return {}
    recs = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    return {r["query_id"]: r for r in recs}


def batches(todo: pd.DataFrame) -> list[tuple[str, list[tuple[int, str]]]]:
    out = []
    for locale, g in todo.groupby("locale", sort=True):
        items = list(zip(g.query_id.astype(int), g["query"], strict=True))
        out += [(locale, items[i : i + BATCH]) for i in range(0, len(items), BATCH)]
    return out


def run(fixtures=False, provider=None, model=None, record=False, limit=None) -> dict:
    """`psr qu`: extract constraints for the covered hard queries that are not cached yet."""
    config.load_dotenv()
    cov = coverage(fixtures)
    done = read_cache(fixtures)
    todo = cov[~cov.query_id.isin(done)].sort_values(["locale", "query_id"])
    if limit is not None:
        todo = todo.head(limit)
    path = cache_path(fixtures)
    path.parent.mkdir(parents=True, exist_ok=True)
    plan = batches(todo)
    with path.open("a") as f:
        for i, (locale, items) in enumerate(plan, 1):
            for rec in understand_batch(locale, items, provider, model, record):
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
                done[rec["query_id"]] = rec
            f.flush()
            print(f"qu: batch {i}/{len(plan)} ({locale})", flush=True)
    stats = summary(cov, done)
    print(json.dumps(stats, indent=2))
    return stats


def summary(cov: pd.DataFrame, cache: dict[int, dict]) -> dict:
    """Coverage per split, empty (invalid twice) items, calls and tokens."""
    metas = [m for r in cache.values() for m in r["meta"]]
    tin = sum(m["input_tokens"] or 0 for m in metas)
    tout = sum(m["output_tokens"] or 0 for m in metas)
    per_split = {}
    for split, g in cov.groupby("split"):
        recs = [cache.get(int(q)) for q in g.query_id]
        per_split[split] = {
            "hard_queries": len(g),
            "done": sum(r is not None for r in recs),
            "empty": sum(r is not None and r["qu"] is None for r in recs),
        }
    return {
        "per_split": per_split,
        "n_calls": len(metas),
        "input_tokens": tin,
        "output_tokens": tout,
        "cost_usd": cost_usd(tin, tout),
    }


# Features. Each takes the query's extraction (None if missing) and one product row.


def _has_all(needle: str | None, haystack: str, locale: str) -> bool:
    words = tokenize(needle, locale)
    return bool(words) and set(words) <= set(tokenize(haystack, locale))


def _size_key(s: str) -> str:
    return re.sub(r"[^0-9a-z.]", "", normalize(s))


def qu_features(qu: dict | None, product: dict, locale: str) -> dict:
    """The qu_* feature values for one (query, product); all 0 when the query has no output."""
    if not qu:
        return dict.fromkeys(FEATURES, 0.0)
    title = product.get("product_title") or ""
    text = " ".join(filter(None, (title, product.get("product_bullet_point"))))
    brand = product.get("product_brand") or title
    color = " ".join(filter(None, (product.get("product_color"), title)))
    type_words = tokenize(qu["product_type"], locale)
    title_words = set(tokenize(title, locale))
    size = _size_key(qu["size"] or "")
    return {
        # 1: the query has an LLM extraction (the other features mean something).
        "qu_present": 1.0,
        # 1: a term the query excludes appears in the title.
        "must_not_in_title": float(any(_has_all(t, title, locale) for t in qu["must_not"])),
        # 1: every word of the asked brand is in the product brand (title if no brand).
        "qu_brand_match": float(_has_all(qu["brand"], brand, locale)),
        # 1: the asked size, spaces removed, appears in title or bullets, not inside a number.
        "qu_size_match": float(
            bool(size) and re.search(rf"(?<![0-9.]){re.escape(size)}", _size_key(text)) is not None
        ),
        # 1: every word of the asked colour is in the product colour or title.
        "qu_color_match": float(_has_all(qu["color"], color, locale)),
        # Share of the product-type words found in the title (0 to 1).
        "qu_type_in_title": (
            sum(w in title_words for w in type_words) / len(type_words) if type_words else 0.0
        ),
    }
