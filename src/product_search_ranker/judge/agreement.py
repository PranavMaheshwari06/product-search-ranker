"""Judge vs human labels: pair-level agreement and whether the judge ranks systems the same way.

Writes `reports/judge.json` and `reports/judge_confusion.png` (SPEC 6.3).
"""

import itertools
import json
from datetime import date

import numpy as np
import pandas as pd
from scipy.stats import kendalltau

from product_search_ranker import config
from product_search_ranker.evaluate import metrics, stats
from product_search_ranker.judge import judge
from product_search_ranker.llm.client import cost_usd

LABELS = list(config.LABELS)
N_EXAMPLES = 20


def confusion_counts(human, judged) -> np.ndarray:
    """4x4 counts, rows = human label, columns = judge label, in E, S, C, I order."""
    h = pd.Categorical(human, categories=LABELS).codes
    j = pd.Categorical(judged, categories=LABELS).codes
    m = np.zeros((4, 4), dtype=np.int64)
    np.add.at(m, (h, j), 1)
    return m


def kappa_from_counts(m: np.ndarray) -> np.ndarray:
    """Cohen's kappa from confusion matrices of shape (..., k, k)."""
    m = np.asarray(m, dtype=np.float64)
    n = m.sum(axis=(-2, -1))
    p_o = np.trace(m, axis1=-2, axis2=-1) / n
    p_e = (m.sum(axis=-1) * m.sum(axis=-2)).sum(axis=-1) / n**2
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(p_e < 1, (p_o - p_e) / (1 - p_e), np.nan)


def cohen_kappa(human, judged) -> float:
    return float(kappa_from_counts(confusion_counts(human, judged)))


def exact_vs_rest(m: np.ndarray) -> np.ndarray:
    """Collapse (..., 4, 4) counts to Exact vs the rest, (..., 2, 2)."""
    e_e, e_r = m[..., 0, 0], m[..., 0, 1:].sum(-1)
    r_e, r_r = m[..., 1:, 0].sum(-1), m[..., 1:, 1:].sum((-2, -1))
    return np.stack([np.stack([e_e, e_r], -1), np.stack([r_e, r_r], -1)], -2)


def pair_metrics(pairs: pd.DataFrame, judged_col: str = "label_judge") -> dict:
    """Accuracy, 4-class kappa and Exact-vs-rest kappa, CIs by resampling queries."""
    p = pairs[pairs[judged_col].notna()]
    if p.empty:
        return {"n_pairs": 0, "n_queries": 0}
    per_q = np.stack(
        [confusion_counts(g.label_human, g[judged_col]) for _, g in p.groupby("query_id")]
    )
    rng = np.random.default_rng(config.SEED)
    idx = rng.integers(0, len(per_q), size=(config.BOOTSTRAP_N, len(per_q)))
    boot = per_q[idx].sum(axis=1)
    total = per_q.sum(axis=0)

    def with_ci(fn):
        lo, hi = np.nanpercentile(fn(boot), [2.5, 97.5])
        return {"value": round(float(fn(total)), 4), "ci95": [round(lo, 4), round(hi, 4)]}

    return {
        "n_pairs": int(total.sum()),
        "n_queries": len(per_q),
        "accuracy": with_ci(lambda m: np.trace(m, axis1=-2, axis2=-1) / m.sum(axis=(-2, -1))),
        "kappa": with_ci(kappa_from_counts),
        "kappa_exact_vs_rest": with_ci(lambda m: kappa_from_counts(exact_vs_rest(m))),
        "confusion": total.tolist(),
    }


def system_level(pairs: pd.DataFrame, systems: list[str], fixtures: bool) -> dict:
    """NDCG@10 of each system on the judged queries, with human and with judge labels."""
    from product_search_ranker.evaluate.run import load_split, score_system

    complete = pairs.groupby("query_id").label_judge.apply(lambda s: s.notna().all())
    qids = complete.index[complete]
    split = load_split("test", fixtures)
    df = split[split.query_id.isin(qids)].reset_index(drop=True)
    judged = df.merge(
        pairs[["query_id", "product_id", "label_judge"]], on=["query_id", "product_id"], how="left"
    )
    per_sys = {}
    for name in systems:
        scores = score_system(name, df, fixtures)
        per_sys[name] = {
            "human": metrics.per_query(df, scores).set_index("query_id"),
            "judge": metrics.per_query(judged.assign(label=judged.label_judge), scores).set_index(
                "query_id"
            ),
        }
    table = {}
    for name, r in per_sys.items():
        table[name] = {src: round(float(r[src].ndcg.mean()), 4) for src in ("human", "judge")} | {
            "n_queries": int(r["human"].ndcg.notna().sum())
        }

    out = {"n_queries": len(qids), "n_dropped_incomplete": int((~complete).sum()), "systems": table}
    if len(systems) < 2:
        out["note"] = "fewer than two systems: no order to compare"
        return out
    human = [table[s]["human"] for s in systems]
    judge_scores = [table[s]["judge"] for s in systems]
    tau = kendalltau(human, judge_scores)
    out["kendall_tau"] = round(float(tau.statistic), 4)
    out["same_winner"] = bool(
        systems[int(np.argmax(human))] == systems[int(np.argmax(judge_scores))]
    )
    rows, sign_ok, sign_ok_sig, n_sig = [], 0, 0, 0
    for a, b in itertools.combinations(systems, 2):
        ha, hb = per_sys[a]["human"].ndcg, per_sys[b]["human"].ndcg
        ja, jb = per_sys[a]["judge"].ndcg, per_sys[b]["judge"].ndcg
        keep = ha.notna() & hb.notna() & ja.notna() & jb.notna()
        h_delta, (lo, hi) = stats.paired_diff(ha[keep], hb[keep])
        j_delta = float((ja[keep] - jb[keep]).mean())
        agree = bool(np.sign(h_delta) == np.sign(j_delta))
        significant = bool(lo > 0 or hi < 0)
        sign_ok += agree
        n_sig += significant
        sign_ok_sig += agree and significant
        rows.append(
            {
                "a": a,
                "b": b,
                "human_delta": round(h_delta, 4),
                "human_ci95": [round(lo, 4), round(hi, 4)],
                "judge_delta": round(j_delta, 4),
                "same_sign": agree,
                "human_ci_excludes_0": significant,
            }
        )
    out["pairs"] = rows
    out["same_sign"] = {
        "all": f"{sign_ok}/{len(rows)}",
        "human_ci_excludes_0": f"{sign_ok_sig}/{n_sig}",
    }
    out["wrong_winner_pairs"] = [f"{r['a']} vs {r['b']}" for r in rows if not r["same_sign"]]
    return out


def examples(pairs: pd.DataFrame, fixtures: bool) -> list[dict]:
    """The disagreements with the largest gain gap between human and judge labels."""
    d = pairs[pairs.label_judge.notna() & (pairs.label_judge != pairs.label_human)].copy()
    if d.empty:
        return []
    d["gap"] = (d.label_human.map(config.GAINS) - d.label_judge.map(config.GAINS)).abs()
    d = d.sort_values(["gap", "query_id", "product_id"], ascending=[False, True, True])
    text = pd.read_parquet(
        config.processed_dir(fixtures) / "test.parquet",
        columns=["query_id", "product_id", "query", "product_title"],
    )
    d = d.head(N_EXAMPLES).merge(text, on=["query_id", "product_id"], how="left")
    return [
        {
            "query": r.query,
            "locale": r.locale,
            "title": r.product_title,
            "human": r.label_human,
            "judge": r.label_judge,
        }
        for r in d.itertuples()
    ]


def usage(fixtures: bool) -> dict:
    metas = [m for rec in judge.read_cache(fixtures).values() for m in rec["meta"]]
    if not metas:
        return {"n_calls": 0}
    tin = sum(m["input_tokens"] or 0 for m in metas)
    tout = sum(m["output_tokens"] or 0 for m in metas)
    lat = np.array([m["latency_s"] for m in metas])
    models = sorted({m["model"] for m in metas})
    return {
        "model": models[0] if len(models) == 1 else models,
        "provider": sorted({m["provider"] for m in metas}),
        "n_calls": len(metas),
        "input_tokens": tin,
        "output_tokens": tout,
        "latency_s_p50": round(float(np.percentile(lat, 50)), 3),
        "latency_s_p95": round(float(np.percentile(lat, 95)), 3),
        "cost_usd": cost_usd(tin, tout),
    }


def plot_confusion(m: np.ndarray, path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rows = m / np.maximum(m.sum(axis=1, keepdims=True), 1)
    fig, ax = plt.subplots(figsize=(4.2, 3.6))
    ax.imshow(rows, cmap="Blues", vmin=0, vmax=1)
    for i, j in itertools.product(range(4), range(4)):
        ax.text(
            j,
            i,
            f"{rows[i, j]:.2f}",
            ha="center",
            va="center",
            color="white" if rows[i, j] > 0.5 else "black",
            fontsize=9,
        )
    ax.set_xticks(range(4), LABELS)
    ax.set_yticks(range(4), LABELS)
    ax.set_xlabel("judge label")
    ax.set_ylabel("human label")
    ax.set_title("Judge vs human (row-normalised)")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def run(systems: list[str] | None = None, fixtures: bool = False) -> dict:
    path = judge.labels_path(fixtures)
    if not path.exists():
        raise SystemExit(f"{path} is missing: run `psr {'--fixtures ' * fixtures}judge` first")
    pairs = pd.read_parquet(path)
    if systems is None:
        from product_search_ranker.evaluate.run import runs_dir

        systems = sorted(p.stem for p in runs_dir("test", fixtures).glob("*.parquet"))
    pair_level = {loc: pair_metrics(pairs[pairs.locale == loc]) for loc in config.LOCALES}
    pair_level["all"] = pair_metrics(pairs)
    single = pairs[pairs.label_single.notna()].assign(label_human=lambda d: d.label_judge)
    report = {
        "date": date.today().isoformat(),
        **usage(fixtures),
        "n_queries": int(pairs.query_id.nunique()),
        "n_pairs": len(pairs),
        "n_missing": int(pairs.label_judge.isna().sum()),
        "pair_level": pair_level,
        "system_level": system_level(pairs, systems, fixtures),
        # Batched labels take the place of the reference here: does one-call-per-pair agree?
        "batch_effect": pair_metrics(single.dropna(subset=["label_human"]), "label_single"),
        "examples": examples(pairs, fixtures),
    }
    out = config.eval_reports_dir(fixtures)
    out.mkdir(parents=True, exist_ok=True)
    (out / "judge.json").write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")
    plot_confusion(
        np.array(pair_level["all"].get("confusion", np.zeros((4, 4)))), out / "judge_confusion.png"
    )
    print(f"wrote {out / 'judge.json'} and {out / 'judge_confusion.png'}")
    return report
