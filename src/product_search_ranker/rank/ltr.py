"""LightGBM lambdarank over the SPEC 5.4 features; grid on val only, runs logged to MLflow."""

import itertools
import json
import os
from functools import lru_cache

import lightgbm as lgb
import numpy as np
import pandas as pd

from product_search_ranker import config
from product_search_ranker.evaluate import metrics
from product_search_ranker.rank.features import FEATURES, build

LABEL_ID = {"I": 0, "C": 1, "S": 2, "E": 3}
GRID = [
    {"num_leaves": n, "learning_rate": lr, "min_data_in_leaf": m}
    for n, lr, m in itertools.product((31, 63), (0.05, 0.1), (50, 200))
]
BASE = {
    "objective": "lambdarank",
    "metric": "ndcg",
    "eval_at": [config.K],
    "label_gain": [0, 0.01, 0.1, 1.0],  # the metric's gains, so training optimises what is scored
    "seed": config.SEED,
    "deterministic": True,
    "force_row_wise": True,
    "verbose": -1,
}
ROUNDS, EARLY_STOP = 1000, 100


def _dataset(feat: pd.DataFrame) -> lgb.Dataset:
    feat = feat.sort_values("query_id", kind="stable")  # lambdarank wants each query contiguous
    group = feat.groupby("query_id", sort=False).size().to_numpy()
    return lgb.Dataset(feat[FEATURES], feat.label.map(LABEL_ID), group=group, free_raw_data=False)


def fit(train: pd.DataFrame, val: pd.DataFrame, params: dict) -> lgb.Booster:
    dval = _dataset(val)
    return lgb.train(
        {**BASE, **params},
        _dataset(train),
        num_boost_round=ROUNDS,
        valid_sets=[dval],
        callbacks=[lgb.early_stopping(EARLY_STOP, verbose=False)],
    )


def predict(booster: lgb.Booster, feat: pd.DataFrame) -> np.ndarray:
    return booster.predict(feat[FEATURES], num_iteration=booster.best_iteration or None)


def val_ndcg(booster: lgb.Booster, val: pd.DataFrame) -> float:
    """Mean NDCG@10 with the project metric (not LightGBM's), IDCG = 0 queries excluded."""
    return float(metrics.per_query(val, predict(booster, val)).ndcg.mean())


def plot_importance(booster: lgb.Booster, path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    gain = pd.Series(booster.feature_importance("gain"), index=FEATURES).sort_values()
    fig, ax = plt.subplots(figsize=(7, 6))
    ax.barh(gain.index, gain.to_numpy())
    ax.set_xlabel("gain")
    ax.set_title("ltr feature importance (gain)")
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=120)
    plt.close(fig)


def train(fixtures: bool = False) -> dict:
    """Run the grid on val, keep the best config's model, write models/ and the importance plot."""
    os.environ.setdefault("MLFLOW_ALLOW_FILE_STORE", "true")  # SPEC 5.4: file store in ./mlruns
    import mlflow

    from product_search_ranker.evaluate.run import data_sha, git_sha

    train_f, val_f = build("train", fixtures), build("val", fixtures)
    mlflow.set_tracking_uri((config.root() / "mlruns").as_uri())
    mlflow.set_experiment("ltr")
    best = None
    for params in GRID:
        with mlflow.start_run(run_name="grid"):
            booster = fit(train_f, val_f, params)
            score = val_ndcg(booster, val_f)
            mlflow.log_params({**BASE, **params, "data_sha": data_sha(fixtures)})
            mlflow.log_metrics({"val_ndcg10": score, "best_iteration": booster.best_iteration})
            mlflow.log_dict({"features": FEATURES}, "features.json")
            mlflow.set_tag("git_sha", git_sha())
        print(f"{params}: val NDCG@10 {score:.4f} at {booster.best_iteration} rounds", flush=True)
        if best is None or score > best[0]:
            best = (score, params, booster)
    score, params, booster = best

    out = config.models_dir(fixtures)
    out.mkdir(parents=True, exist_ok=True)
    booster.save_model(str(out / "ltr.txt"), num_iteration=booster.best_iteration)
    meta = {
        "features": FEATURES,
        "params": {**BASE, **params},
        "val_ndcg10": round(score, 4),
        "best_iteration": booster.best_iteration,
        "data_sha": data_sha(fixtures),
        "git_sha": git_sha(),
    }
    (out / "ltr_meta.json").write_text(json.dumps(meta, indent=2) + "\n")
    plot = config.eval_reports_dir(fixtures) / "ltr_importance.png"
    plot_importance(booster, plot)
    with mlflow.start_run(run_name="best"):
        mlflow.log_params(params)
        mlflow.log_metric("val_ndcg10", score)
        mlflow.log_artifact(str(plot))
        mlflow.log_artifact(str(out / "ltr_meta.json"))
        mlflow.log_artifact(str(out / "ltr.txt"))
    _load.cache_clear()  # a later `score` in this process must read the model just written
    print(f"wrote {out / 'ltr.txt'}, {plot}; best {params}, val NDCG@10 {score:.4f}", flush=True)
    return meta


@lru_cache(maxsize=2)
def _load(fixtures: bool) -> lgb.Booster:
    path = config.models_dir(fixtures) / "ltr.txt"
    if not path.exists():
        raise SystemExit(f"{path} is missing: run `psr train` first")
    saved = json.loads((path.parent / "ltr_meta.json").read_text())["features"]
    if saved != FEATURES:
        raise SystemExit(f"{path} was trained on other features than features.FEATURES: retrain")
    return lgb.Booster(model_file=str(path))


def score(df: pd.DataFrame, split: str, fixtures: bool = False) -> np.ndarray:
    """ltr score for each (query_id, product_id) row of `df`, looked up in the split's features."""
    feat = build(split, fixtures)
    pairs = pd.MultiIndex.from_frame(feat[["query_id", "product_id"]])
    out = pd.Series(predict(_load(fixtures), feat), index=pairs).reindex(
        pd.MultiIndex.from_frame(df[["query_id", "product_id"]])
    )
    if out.isna().any():
        raise SystemExit(f"{int(out.isna().sum())} rows of `df` have no features in split {split}")
    return out.to_numpy()
