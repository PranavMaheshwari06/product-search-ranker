"""Sample the small committed fixture (tests/fixtures/esci_sample.parquet) from data/processed."""

import duckdb
import numpy as np
import pandas as pd

from product_search_ranker import config
from product_search_ranker.data.prepare import ROW_COLUMNS

N_QUERIES = {"us": 150, "es": 50}
TEXT_LIMIT = 300
MAX_BYTES = 2_000_000
TEXT_COLUMNS = ["product_title", "product_brand", "product_color", "product_bullet_point",
                "product_description"]  # fmt: skip


def _allocate(sizes: pd.Series, total: int) -> pd.Series:
    """Split `total` over groups in proportion to `sizes` (largest remainder)."""
    exact = sizes / sizes.sum() * total
    base = exact.astype(int)
    rest = total - base.sum()
    base[(exact - base).sort_values(ascending=False).index[:rest]] += 1
    return base


def make_fixture() -> None:
    src = config.processed_dir()
    con = duckdb.connect()
    queries = con.execute(
        f"""SELECT DISTINCT query_id, locale, split FROM
            read_parquet(['{src}/train.parquet', '{src}/val.parquet', '{src}/test.parquet'])
            ORDER BY query_id"""
    ).df()
    chosen = []
    for loc, n in N_QUERIES.items():
        q = queries[queries.locale == loc]
        take = _allocate(q.groupby("split").size(), n)
        for split, k in take.items():
            pool = q[q.split == split].query_id.to_numpy()
            rng = np.random.default_rng(config.SEED)
            chosen.extend(rng.choice(pool, size=k, replace=False).tolist())
    ids = pd.DataFrame({"query_id": sorted(chosen)})
    con.register("ids", ids)
    cols = ", ".join(ROW_COLUMNS)
    df = con.execute(
        f"""SELECT {cols} FROM read_parquet(['{src}/train.parquet', '{src}/val.parquet',
            '{src}/test.parquet']) WHERE query_id IN (SELECT query_id FROM ids)
            ORDER BY query_id, product_id"""
    ).df()
    for c in TEXT_COLUMNS:
        df[c] = df[c].str.slice(0, TEXT_LIMIT)

    missing = set(config.LABELS) - set(df.label)
    if missing:
        raise SystemExit(f"fixture lacks labels {sorted(missing)}; change the seed in config")
    out = config.fixture_file()
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out, index=False, compression="zstd", compression_level=19)
    size = out.stat().st_size
    print(f"wrote {out}: {df.query_id.nunique()} queries, {len(df)} rows, {size / 1e6:.2f} MB")
    if size > MAX_BYTES:
        raise SystemExit(f"fixture is {size} bytes, limit {MAX_BYTES}")
