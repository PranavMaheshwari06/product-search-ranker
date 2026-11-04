"""LLM judge: label every candidate of the sampled queries with E/S/C/I (SPEC 6.2).

Replies are cached in an append-only JSONL file, so an interrupted run resumes where it stopped.
"""

import json
from typing import Literal

import pandas as pd
from pydantic import BaseModel

from product_search_ranker import config
from product_search_ranker.judge.sample import sample
from product_search_ranker.llm.client import LLMError, complete_json

BULLET_CHARS = 600
N_SINGLE_QUERIES = 30
COLUMNS = [
    "query_id",
    "query",
    "locale",
    "product_id",
    "label",
    "product_title",
    "product_brand",
    "product_color",
    "product_bullet_point",
]

# Verbatim from Reddy et al. 2022 (arXiv:2206.06588), the ESCI label definitions.
LABEL_DEFINITIONS = """\
- E (Exact): the item is relevant for the query, and satisfies all the query specifications \
(e.g., a water bottle matching all attributes of a query "plastic water bottle 24oz", such as \
material and size)
- S (Substitute): the item is somewhat relevant, i.e., it fails to fulfill some aspects of the \
query but the item can be used as a functional substitute (e.g., fleece for a "sweater" query)
- C (Complement): the item does not fulfill the query, but could be used in combination with an \
exact item (e.g., track pants for "running shoes" query)
- I (Irrelevant): the item is irrelevant, or it fails to fulfill a central aspect of the query \
(e.g., socks for a "telescope" query, or a wheat flour bread for a "gluten-free bread" query)"""

LANGUAGE = {"us": "English (amazon.com)", "es": "Spanish (amazon.es)"}


class JudgeItem(BaseModel):
    product_id: str
    label: Literal["E", "S", "C", "I"]


def _candidate(row) -> dict:
    return {
        "product_id": row.product_id,
        "title": row.product_title,
        "brand": row.product_brand,
        "color": row.product_color,
        "bullets": (row.product_bullet_point or "")[:BULLET_CHARS],
    }


def build_prompt(query: str, locale: str, rows: pd.DataFrame) -> str:
    """One prompt for a query and its candidates, in product_id order (deterministic hash)."""
    cands = [_candidate(r) for r in rows.sort_values("product_id").itertuples()]
    lines = "\n".join(json.dumps(c, ensure_ascii=False) for c in cands)
    return f"""You are judging product search results for an online shop.
Shop language: {LANGUAGE[locale]}.
Query: {json.dumps(query, ensure_ascii=False)}

Label how each product relates to the query, using exactly one of these labels:
{LABEL_DEFINITIONS}

Products, one JSON object per line:
{lines}

Return a JSON array with one object per product, every product_id exactly once:
[{{"product_id": "...", "label": "E" | "S" | "C" | "I"}}]"""


def parse_labels(obj, product_ids: list[str]) -> tuple[dict[str, str], list[str]]:
    """Labels for product_ids that appear exactly once with a valid label; the rest are missing."""
    items = obj if isinstance(obj, list) else []
    counts: dict[str, int] = {}
    labels = {}
    for item in items:
        try:
            it = JudgeItem.model_validate(item)
        except Exception:
            continue
        counts[it.product_id] = counts.get(it.product_id, 0) + 1
        labels[it.product_id] = it.label
    ok = {p: labels[p] for p in product_ids if counts.get(p) == 1}
    return ok, [p for p in product_ids if p not in ok]


def judge_query(rows: pd.DataFrame, provider=None, model=None, record=False) -> dict:
    """Judge one query's candidates in one call; one retry if any candidate is missing."""
    first = rows.iloc[0]
    prompt = build_prompt(first.query, first.locale, rows)
    ids = sorted(rows.product_id)
    labels, missing, metas = {}, ids, []
    for attempt in range(2):
        try:
            obj, meta = complete_json(prompt, provider=provider, model=model, record=record)
        except LLMError as e:
            print(f"judge: query {first.query_id} attempt {attempt + 1}: {e}", flush=True)
        else:
            metas.append(meta)
            labels, missing = parse_labels(obj, ids)
            if not missing:
                break
        # A retry with the same prompt would read the same replay file; only real calls retry.
        if (provider or config.llm_provider()) == "replay":
            break
    return {
        "key": f"batch:{first.query_id}",
        "query_id": int(first.query_id),
        "mode": "batch",
        "labels": labels,
        "missing": missing,
        "meta": metas,
    }


def judge_single(
    rows: pd.DataFrame, provider=None, model=None, record=False, done=()
) -> list[dict]:
    """Batch-effect check: the same query judged one candidate per call (skips keys in done)."""
    out = []
    for _, row in rows.sort_values("product_id").iterrows():
        if f"single:{row.query_id}:{row.product_id}" in done:
            continue
        rec = judge_query(rows.loc[[row.name]], provider, model, record)
        rec.update(key=f"single:{row.query_id}:{row.product_id}", mode="single")
        out.append(rec)
    return out


def single_query_ids(queries: list[int]) -> set[int]:
    """Every k-th sampled query (sorted ids), N_SINGLE_QUERIES of them, for the batch check."""
    step = max(1, len(queries) // N_SINGLE_QUERIES)
    return set(sorted(queries)[::step][:N_SINGLE_QUERIES])


def cache_path(fixtures: bool):
    return config.llm_cache_dir(fixtures) / "judge.jsonl"


def labels_path(fixtures: bool):
    return config.eval_reports_dir(fixtures) / "judge_labels.parquet"


def read_cache(fixtures: bool) -> dict[str, dict]:
    path = cache_path(fixtures)
    if not path.exists():
        return {}
    recs = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    return {r["key"]: r for r in recs}


def run(fixtures: bool = False, provider=None, model=None, record=False) -> pd.DataFrame:
    """Judge the sample (resumable), then write the labels table next to the reports."""
    config.load_dotenv()
    split = config.processed_dir(fixtures) / "test.parquet"
    if not split.exists():
        raise SystemExit(f"{split} is missing: run `psr {'--fixtures ' * fixtures}prepare` first")
    picked = sample(pd.read_parquet(split, columns=COLUMNS))
    done = read_cache(fixtures)
    path = cache_path(fixtures)
    path.parent.mkdir(parents=True, exist_ok=True)
    queries = sorted(picked.query_id.unique())
    single_ids = single_query_ids(queries)
    with path.open("a") as f:
        for i, qid in enumerate(queries, 1):
            rows = picked[picked.query_id == qid]
            todo = []
            if f"batch:{qid}" not in done:
                todo.append(judge_query(rows, provider, model, record))
            if qid in single_ids:
                todo += judge_single(rows, provider, model, record, done)
            for rec in todo:
                f.write(json.dumps(rec) + "\n")
                f.flush()
                done[rec["key"]] = rec
            print(f"judge: {i}/{len(queries)} queries", flush=True)
    table = labels_table(picked, done)
    out = labels_path(fixtures)
    out.parent.mkdir(parents=True, exist_ok=True)
    table.to_parquet(out, index=False)
    print(f"wrote {out} ({len(table)} pairs, {int(table.label_judge.isna().sum())} missing)")
    return table


def labels_table(picked: pd.DataFrame, cache: dict[str, dict]) -> pd.DataFrame:
    """One row per judged pair: human label, batch judge label, single-call label (if any)."""
    batch, single = {}, {}
    for rec in cache.values():
        target = batch if rec["mode"] == "batch" else single
        for pid, lab in rec["labels"].items():
            target[(rec["query_id"], pid)] = lab
    keys = list(zip(picked.query_id.astype(int), picked.product_id, strict=True))
    return pd.DataFrame(
        {
            "query_id": picked.query_id.to_numpy(),
            "locale": picked.locale.to_numpy(),
            "bucket": picked.bucket.to_numpy(),
            "product_id": picked.product_id.to_numpy(),
            "label_human": picked.label.to_numpy(),
            "label_judge": [batch.get(k) for k in keys],
            "label_single": [single.get(k) for k in keys],
        }
    )
