"""Raw ESCI files -> data/processed/{train,val,test}.parquet and products_{us,es}.parquet."""

import hashlib
import json
from datetime import date

import duckdb

from product_search_ranker import config

ROW_COLUMNS = [
    "query_id",
    "query",
    "locale",
    "product_id",
    "label",
    "split",
    "product_title",
    "product_brand",
    "product_color",
    "product_bullet_point",
    "product_description",
    "title_raw_len",
    "bullet_raw_len",
    "description_raw_len",
]
PRODUCT_COLUMNS = [
    "product_id",
    "locale",
    "product_title",
    "product_brand",
    "product_color",
    "product_bullet_point",
    "product_description",
    "title_raw_len",
    "bullet_raw_len",
    "description_raw_len",
]

# Multiplicative hash of the integer query_id: the same split on every DuckDB version.
VAL_RULE = "((query_id * 2654435761) % 4294967296) // 65536 % 10 = 0"

MACROS = r"""
CREATE MACRO clean(x) AS trim(regexp_replace(regexp_replace(
    replace(replace(coalesce(x, ''), '&nbsp;', ' '), '&amp;', '&'),
    '<[^>]*>', ' ', 'g'), '\s+', ' ', 'g'));
"""


def data_sha() -> str:
    manifest = json.loads((config.raw_dir().parent / "MANIFEST.json").read_text())
    shas = sorted(v["sha256"] for v in manifest.values())
    return hashlib.sha256("".join(shas).encode()).hexdigest()


def _build_from_raw(con: duckdb.DuckDBPyConnection) -> None:
    raw = config.raw_dir()
    locales = ",".join(f"'{loc}'" for loc in config.LOCALES)
    con.execute(MACROS)
    con.execute(
        f"""
        CREATE TABLE kept AS
        SELECT query_id, query, product_locale AS locale, product_id, esci_label AS label,
               CASE WHEN split = 'train' AND {VAL_RULE} THEN 'val' ELSE split END AS split
        FROM '{raw}/shopping_queries_dataset_examples.parquet'
        WHERE small_version = 1 AND product_locale IN ({locales})
        """
    )
    con.execute(
        f"""
        CREATE TABLE products AS
        SELECT p.product_id, p.product_locale AS locale,
               clean(p.product_title) AS product_title,
               clean(p.product_brand) AS product_brand,
               clean(p.product_color) AS product_color,
               clean(p.product_bullet_point) AS product_bullet_point,
               clean(p.product_description) AS product_description,
               length(coalesce(p.product_title, '')) AS title_raw_len,
               length(coalesce(p.product_bullet_point, '')) AS bullet_raw_len,
               length(coalesce(p.product_description, '')) AS description_raw_len
        FROM '{raw}/shopping_queries_dataset_products.parquet' p
        WHERE p.product_locale IN ({locales})
          AND EXISTS (SELECT 1 FROM kept k
                      WHERE k.product_id = p.product_id AND k.locale = p.product_locale)
        """
    )
    cols = ", ".join(f"p.{c}" for c in PRODUCT_COLUMNS if c not in ("product_id", "locale"))
    con.execute(
        f"""
        CREATE TABLE rows AS
        SELECT k.query_id, k.query, k.locale, k.product_id, k.label, k.split, {cols}
        FROM kept k JOIN products p USING (product_id, locale)
        """
    )


def _build_from_fixture(con: duckdb.DuckDBPyConnection) -> None:
    con.execute(f"CREATE TABLE rows AS SELECT * FROM '{config.fixture_file()}'")
    cols = ", ".join(PRODUCT_COLUMNS)
    con.execute(f"CREATE TABLE products AS SELECT DISTINCT {cols} FROM rows")


def _stats(con: duckdb.DuckDBPyConnection) -> dict:
    def q(sql: str):
        return con.execute(sql).fetchall()

    counts = {
        f"{split}/{loc}": {"queries": nq, "judgements": n}
        for split, loc, nq, n in q(
            "SELECT split, locale, count(DISTINCT query_id), count(*) FROM rows GROUP BY ALL"
        )
    }
    totals = {
        loc: {"queries": nq, "judgements": n}
        for loc, nq, n in q(
            "SELECT locale, count(DISTINCT query_id), count(*) FROM rows GROUP BY 1"
        )
    }
    labels = {}
    for split, loc, lab, n in q("SELECT split, locale, label, count(*) FROM rows GROUP BY ALL"):
        labels.setdefault(f"{split}/{loc}", {})[lab] = n
    # A query text that appears under two splits would leak even with distinct query_ids.
    text_overlap = q(
        """SELECT count(*) FROM (SELECT locale, query FROM
           (SELECT DISTINCT locale, query, split FROM rows)
           GROUP BY ALL HAVING count(*) > 1)"""
    )[0][0]
    return {
        "counts": counts,
        "totals": totals,
        "labels": labels,
        "issues": {
            "empty_titles": q("SELECT count(*) FROM products WHERE product_title = ''")[0][0],
            "empty_brands": q("SELECT count(*) FROM products WHERE product_brand = ''")[0][0],
            "empty_descriptions": q("SELECT count(*) FROM products WHERE product_description = ''")[
                0
            ][0],
            "duplicate_query_product_pairs": q(
                """SELECT coalesce(sum(n - 1), 0) FROM
                   (SELECT count(*) n FROM rows GROUP BY query_id, product_id, locale
                    HAVING count(*) > 1)"""
            )[0][0],
            "bullets_over_2000_chars": q(
                "SELECT count(*) FROM products WHERE bullet_raw_len > 2000"
            )[0][0],
            "query_texts_in_two_splits": text_overlap,
        },
        "n_products": {loc: n for loc, n in q("SELECT locale, count(*) FROM products GROUP BY 1")},
    }


def check_counts(totals: dict) -> None:
    for loc, (exp_q, exp_n) in config.EXPECTED_COUNTS.items():
        got = totals[loc]
        for name, exp, val in (
            ("queries", exp_q, got["queries"]),
            ("judgements", exp_n, got["judgements"]),
        ):
            if abs(val - exp) / exp > config.COUNT_TOLERANCE:
                raise SystemExit(
                    f"STOP: {loc} {name} = {val}, SPEC 4.2 expects {exp} (+-0.5%). Ask the user."
                )


def prepare(fixtures: bool = False) -> None:
    out = config.processed_dir(fixtures)
    out.mkdir(parents=True, exist_ok=True)
    con = duckdb.connect()
    con.execute(f"SET temp_directory = '{config.root() / 'data' / 'duckdb_tmp'}'")
    (_build_from_fixture if fixtures else _build_from_raw)(con)

    rows = ", ".join(ROW_COLUMNS)
    for split in config.SPLITS:
        path = out / f"{split}.parquet"
        con.execute(
            f"COPY (SELECT {rows} FROM rows WHERE split = '{split}' ORDER BY query_id, product_id) "
            f"TO '{path}' (FORMAT parquet, COMPRESSION zstd)"
        )
        print(f"wrote {path}")
    prods = ", ".join(PRODUCT_COLUMNS)
    for loc in config.LOCALES:
        path = out / f"products_{loc}.parquet"
        con.execute(
            f"COPY (SELECT {prods} FROM products WHERE locale = '{loc}' ORDER BY product_id) "
            f"TO '{path}' (FORMAT parquet, COMPRESSION zstd)"
        )
        print(f"wrote {path}")

    stats = _stats(con)
    stats["date"] = date.today().isoformat()
    stats["duckdb"] = duckdb.__version__
    if fixtures:
        (out / "stats.json").write_text(json.dumps(stats, indent=2))
        return
    stats["data_sha"] = data_sha()
    config.reports_dir().mkdir(exist_ok=True)
    stats_path = config.reports_dir() / "data_stats.json"
    stats_path.write_text(json.dumps(stats, indent=2))
    print(f"wrote {stats_path}")
    write_data_card(stats)
    check_counts(stats["totals"])
    print("counts match SPEC 4.2 within 0.5%")


CARD = """# Data card: ESCI (Shopping Queries Dataset)

All numbers below come from `reports/data_stats.json` (written by `psr prepare`, {date}, data
sha `{sha}`).

## Source and licence
Amazon Shopping Queries Dataset (ESCI), https://github.com/amazon-science/esci-data, folder
`shopping_queries_dataset/`. Licence: Apache-2.0. The files are downloaded by `psr fetch` and are
not redistributed in this repository; only `tests/fixtures/esci_sample.parquet` (200 queries, text
cut to 300 characters) is committed.

Citation: Reddy et al., "Shopping Queries Dataset: A Large-Scale ESCI Benchmark for Improving
Product Search", arXiv:2206.06588, 2022.

## Filters
- `small_version == 1` (the reduced set that ESCI Task 1 uses).
- Locales `us` and `es`. Japanese is left out: BM25 would need a Japanese tokenizer, and two
  locales are enough to show cross-language behaviour.
- Examples are joined to products on `(product_id, product_locale)`. HTML tags are removed from
  text fields and whitespace is collapsed; raw lengths are kept (`*_raw_len`).

## Splits
The official `split` column gives train and test. Validation is the train queries with
`((query_id * 2654435761) % 4294967296) // 65536 % 10 == 0`, so a query is in exactly one split
(tested in `tests/test_data.py`).

| Split | Locale | Queries | Judgements |
|---|---|---|---|
{counts}

Totals match the esci-data README within 0.5% (SPEC 4.2): us {us_q} queries / {us_n}
judgements, es {es_q} queries / {es_n} judgements. Unique products kept: us {p_us}, es {p_es}.

## Label distribution (judgements)
| Split | Locale | E | S | C | I |
|---|---|---|---|---|---|
{labels}

## Known issues found during prepare
- Products with an empty title: {empty_titles}.
- Products with an empty brand: {empty_brands}; with an empty description: {empty_descriptions}.
- Duplicate (query, product) pairs: {dups}.
- Products whose bullet points are longer than 2,000 characters: {long_bullets}.
- Query texts that appear in two splits of the same locale: {overlap}.
"""


def write_data_card(stats: dict) -> None:
    order = [f"{s}/{loc}" for s in config.SPLITS for loc in config.LOCALES]
    counts = "\n".join(
        f"| {k.split('/')[0]} | {k.split('/')[1]} | {stats['counts'][k]['queries']:,} | "
        f"{stats['counts'][k]['judgements']:,} |"
        for k in order
    )
    labels = "\n".join(
        f"| {k.split('/')[0]} | {k.split('/')[1]} | "
        + " | ".join(f"{stats['labels'][k].get(lab, 0):,}" for lab in ("E", "S", "C", "I"))
        + " |"
        for k in order
    )
    t, iss = stats["totals"], stats["issues"]
    card = CARD.format(
        date=stats["date"],
        sha=stats["data_sha"],
        counts=counts,
        labels=labels,
        us_q=f"{t['us']['queries']:,}",
        us_n=f"{t['us']['judgements']:,}",
        es_q=f"{t['es']['queries']:,}",
        es_n=f"{t['es']['judgements']:,}",
        p_us=f"{stats['n_products']['us']:,}",
        p_es=f"{stats['n_products']['es']:,}",
        empty_titles=iss["empty_titles"],
        empty_brands=f"{iss['empty_brands']:,}",
        empty_descriptions=f"{iss['empty_descriptions']:,}",
        dups=iss["duplicate_query_product_pairs"],
        long_bullets=f"{iss['bullets_over_2000_chars']:,}",
        overlap=iss["query_texts_in_two_splits"],
    )
    path = config.root() / "docs" / "data_card.md"
    path.write_text(card)
    print(f"wrote {path}")
