"""`psr eval`: score systems on a split, write per-query runs, merge results into metrics.json."""

import hashlib
import json
import subprocess
from datetime import date

import numpy as np
import pandas as pd

from product_search_ranker import config
from product_search_ranker.evaluate import metrics, stats
from product_search_ranker.text import SLICES, query_slices

ROW_COLUMNS = ["query_id", "query", "locale", "product_id", "label", "product_brand"]


def score(system: str, rows: pd.DataFrame, fixtures: bool) -> np.ndarray:
    if system == "bm25":
        from product_search_ranker.retrieval.bm25 import score_candidates

        return score_candidates(rows, "all", fixtures)
    raise SystemExit(f"unknown system {system!r}; known so far: bm25")


def load_rows(split: str, fixtures: bool) -> pd.DataFrame:
    path = config.processed_dir(fixtures) / f"{split}.parquet"
    if not path.exists():
        flag = "--fixtures " if fixtures else ""
        raise SystemExit(f"{path} is missing: run `psr {flag}prepare` first")
    return pd.read_parquet(path, columns=ROW_COLUMNS)


def query_slice_table(rows: pd.DataFrame) -> pd.DataFrame:
    """One row per query, one boolean column per slice."""
    recs = []
    for (qid, locale, query), g in rows.groupby(["query_id", "locale", "query"], sort=True):
        s = query_slices(query, locale, g.product_brand.dropna().unique().tolist())
        recs.append({"query_id": qid, **{name: name in s for name in SLICES}})
    return pd.DataFrame(recs).set_index("query_id")


def summary_with_ci(pq: pd.DataFrame) -> dict:
    out = metrics.summarize(pq)
    kept = pq.ndcg.dropna().to_numpy()
    out["ci95"] = list(stats.bootstrap_ci(kept)) if len(kept) else None
    return {k: out[k] for k in ("ndcg@10", "ci95", "mrr", "n_queries", "n_excluded")}


def by_locale(pq: pd.DataFrame) -> dict:
    res = {loc: summary_with_ci(pq[pq.locale == loc]) for loc in config.LOCALES}
    res["all"] = summary_with_ci(pq)
    return res


def compare(a: pd.DataFrame, b: pd.DataFrame) -> dict:
    """Paired difference a - b per locale, over queries included in both runs."""
    j = a.merge(b, on=["query_id", "locale"], suffixes=("_a", "_b")).dropna(
        subset=["ndcg_a", "ndcg_b"]
    )
    out = {}
    for loc in (*config.LOCALES, "all"):
        part = j if loc == "all" else j[j.locale == loc]
        if len(part):
            out[loc] = stats.paired_diff(part.ndcg_a, part.ndcg_b)
    return out


def git_sha() -> str:
    try:
        return subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=config.root(),
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def current_data_sha(fixtures: bool) -> str:
    if fixtures:
        return hashlib.sha256(config.fixture_file().read_bytes()).hexdigest()
    from product_search_ranker.data.prepare import data_sha

    return data_sha()


def evaluate(systems: list[str], split: str = "test", fixtures: bool = False) -> dict:
    rows = load_rows(split, fixtures)
    runs = config.runs_dir(fixtures) / split
    runs.mkdir(parents=True, exist_ok=True)
    slices = query_slice_table(rows)
    slices.to_parquet(runs / "slices.parquet")

    for system in systems:
        scored = rows.assign(score=score(system, rows, fixtures))
        pq = metrics.per_query(scored)
        pq.to_parquet(runs / f"{system}.parquet")
        print(f"wrote {runs / f'{system}.parquet'}")

    path = config.metrics_path(split, fixtures)
    report = json.loads(path.read_text()) if path.exists() else {}
    report.update(
        {
            "updated": date.today().isoformat(),
            "git_sha": git_sha(),
            "data_sha": current_data_sha(fixtures),
            "split": split,
            "gains": config.GAINS,
        }
    )
    report.setdefault("systems", {})
    report.setdefault("slices", {})
    have = {p.stem: pd.read_parquet(p) for p in sorted(runs.glob("*.parquet"))}
    have.pop("slices")

    for system in systems:
        report["systems"][system] = by_locale(have[system])
        for name in SLICES:
            pq = have[system]
            in_slice = pq[pq.query_id.map(slices[name]).fillna(False).astype(bool)]
            s = summary_with_ci(in_slice)
            report["slices"].setdefault(name, {})[system] = {
                k: s[k] for k in ("ndcg@10", "ci95", "n_queries")
            }

    # Comparisons are recomputed for every pair whose runs are on disk; older entries for pairs
    # without runs here (another VM) are kept.
    kept = [c for c in report.get("comparisons", []) if (c["a"], c["b"]) not in _pairs(have)]
    for a, b in _pairs(have):
        for loc, res in compare(have[a], have[b]).items():
            kept.append({"a": a, "b": b, "locale": loc, **res})
    report["comparisons"] = kept

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2) + "\n")
    print(f"wrote {path}")
    for system in systems:
        for loc, r in report["systems"][system].items():
            if r["n_queries"]:
                lo, hi = r["ci95"]
                print(
                    f"  {system:8s} {loc:3s} ndcg@10={r['ndcg@10']:.4f} [{lo:.4f}, {hi:.4f}] "
                    f"mrr={r['mrr']:.4f} n={r['n_queries']} excluded={r['n_excluded']}"
                )
    return report


def _pairs(have: dict) -> list[tuple[str, str]]:
    return [(a, b) for a, b in config.COMPARISONS if a in have and b in have]
