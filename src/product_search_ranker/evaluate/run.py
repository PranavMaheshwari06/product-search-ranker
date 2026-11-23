"""`psr eval`: score a split with each system and merge the results into metrics.json."""

import hashlib
import json
import subprocess
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

from product_search_ranker import config
from product_search_ranker.evaluate import metrics, stats
from product_search_ranker.text import SLICES, query_slices

COLUMNS = ["query_id", "query", "locale", "product_id", "label", "product_brand"]


def load_split(split: str, fixtures: bool = False) -> pd.DataFrame:
    path = config.processed_dir(fixtures) / f"{split}.parquet"
    if not path.exists():
        if not fixtures:
            raise SystemExit(f"{path} is missing: run `psr fetch && psr prepare` first")
        from product_search_ranker.data.prepare import prepare

        prepare(fixtures=True)
    return pd.read_parquet(path, columns=COLUMNS)


def score_system(name: str, df: pd.DataFrame, fixtures: bool, split: str = "test") -> np.ndarray:
    if name == "bm25":
        from product_search_ranker.retrieval import bm25

        return bm25.score(df, fixtures)
    if name == "dense":
        from product_search_ranker.retrieval import dense

        return dense.score(df, fixtures)
    if name == "hybrid":
        from product_search_ranker.retrieval import hybrid

        return hybrid.score(df, fixtures)
    if name in ("ltr", "ltr_qu"):
        from product_search_ranker.rank import ltr

        return ltr.score(df, split, fixtures, name)
    raise SystemExit(f"unknown system {name!r}; built so far: bm25, dense, hybrid, ltr, ltr_qu")


def runs_dir(split: str, fixtures: bool) -> Path:
    """Per-query results, kept so later systems can be compared with earlier ones."""
    return config.eval_reports_dir(fixtures) / "runs" / split


def report_path(split: str, fixtures: bool) -> Path:
    name = "metrics.json" if split == "test" else f"metrics_{split}.json"
    return config.eval_reports_dir(fixtures) / name


def slice_table(df: pd.DataFrame) -> pd.DataFrame:
    """One row per query with a boolean column per slice."""
    rows = []
    for (qid, query, locale), g in df.groupby(["query_id", "query", "locale"], sort=False):
        s = query_slices(query, locale, list(g.product_brand))
        rows.append({"query_id": qid, **{name: name in s for name in SLICES}})
    return pd.DataFrame(rows)


def summary(pq: pd.DataFrame) -> dict:
    kept = pq[~pq.excluded]
    if kept.empty:
        return {"ndcg@10": None, "ci95": None, "mrr": None, "n_queries": 0,
                "n_excluded": int(pq.excluded.sum())}  # fmt: skip
    lo, hi = stats.ci95(kept.ndcg)
    return {
        "ndcg@10": round(float(kept.ndcg.mean()), 4),
        "ci95": [round(lo, 4), round(hi, 4)],
        "mrr": round(float(kept.rr.mean()), 4),
        "n_queries": len(kept),
        "n_excluded": int(pq.excluded.sum()),
    }


def compare(a: pd.DataFrame, b: pd.DataFrame) -> dict:
    m = a.merge(b, on=["query_id", "locale"], suffixes=("_a", "_b"))
    m = m[~m.excluded_a & ~m.excluded_b]
    delta, (lo, hi) = stats.paired_diff(m.ndcg_a, m.ndcg_b)
    return {
        "delta": round(delta, 4),
        "ci95": [round(lo, 4), round(hi, 4)],
        "p_value": round(stats.sign_flip_p(m.ndcg_a, m.ndcg_b), 4),
        "n_queries": len(m),
    }


def git_sha() -> str:
    try:
        out = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True)
        return out.stdout.strip() or "unknown"
    except OSError:
        return "unknown"


def data_sha(fixtures: bool) -> str:
    if fixtures:
        return hashlib.sha256(config.fixture_file().read_bytes()).hexdigest()
    from product_search_ranker.data.prepare import data_sha as raw_sha

    return raw_sha()


def evaluate(systems: list[str], split: str = "test", fixtures: bool = False) -> Path:
    if "ltr_qu" in systems and "ltr" not in systems:
        raise SystemExit("ltr_qu is compared with ltr from the same call: --systems ltr,ltr_qu")
    df = load_split(split, fixtures)
    runs = runs_dir(split, fixtures)
    runs.mkdir(parents=True, exist_ok=True)
    for name in systems:
        pq = metrics.per_query(df, score_system(name, df, fixtures, split))
        pq.to_parquet(runs / f"{name}.parquet", index=False)
        print(f"wrote {runs / f'{name}.parquet'}")

    path = report_path(split, fixtures)
    report = json.loads(path.read_text()) if path.exists() else {}
    report.update(
        updated=date.today().isoformat(),
        git_sha=git_sha(),
        data_sha=data_sha(fixtures),
        split=split,
        gains=config.GAINS,
    )
    # Every system with a saved run is re-summarised; systems without one are left as they are.
    all_runs = {p.stem: pd.read_parquet(p) for p in sorted(runs.glob("*.parquet"))}
    report.setdefault("systems", {})
    for name, pq in all_runs.items():
        report["systems"][name] = {
            **{loc: summary(pq[pq.locale == loc]) for loc in config.LOCALES},
            "all": summary(pq),
        }

    comparisons = [
        c for c in report.get("comparisons", []) if (c["a"], c["b"]) not in config.COMPARISONS
    ]
    for a, b in config.COMPARISONS:
        if a in all_runs and b in all_runs:
            for loc in (*config.LOCALES, "all"):
                pa, pb = all_runs[a], all_runs[b]
                if loc != "all":
                    pa, pb = pa[pa.locale == loc], pb[pb.locale == loc]
                comparisons.append({"a": a, "b": b, "locale": loc, **compare(pa, pb)})
    report["comparisons"] = comparisons

    sl = slice_table(df)
    report.setdefault("slices", {})
    for name_slice in SLICES:
        ids = set(sl.query_id[sl[name_slice]])
        entry = report["slices"].setdefault(name_slice, {})
        for name, pq in all_runs.items():
            s = summary(pq[pq.query_id.isin(ids)])
            entry[name] = {k: s[k] for k in ("ndcg@10", "ci95", "n_queries")}

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2) + "\n")
    print(f"wrote {path}")
    if "ltr_qu" in systems and split == "test":
        from product_search_ranker.llm import query_understanding as qu

        print(f"wrote {qu.write_report(df, all_runs, fixtures)}")
    return path
