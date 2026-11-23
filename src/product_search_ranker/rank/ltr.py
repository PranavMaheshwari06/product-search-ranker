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
from product_search_ranker.llm import query_understanding as qu
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
QU_FEATURES = FEATURES + list(qu.FEATURES)
SYSTEM_FEATURES = {"ltr": FEATURES, "ltr_qu": QU_FEATURES}


def _dataset(feat: pd.DataFrame, features=FEATURES) -> lgb.Dataset:
    feat = feat.sort_values("query_id", kind="stable")  # lambdarank wants each query contiguous
    group = feat.groupby("query_id", sort=False).size().to_numpy()
    return lgb.Dataset(feat[features], feat.label.map(LABEL_ID), group=group, free_raw_data=False)


def fit(train: pd.DataFrame, val: pd.DataFrame, params: dict, features=FEATURES) -> lgb.Booster:
    dval = _dataset(val, features)
    return lgb.train(
        {**BASE, **params},
        _dataset(train, features),
        num_boost_round=ROUNDS,
        valid_sets=[dval],
        callbacks=[lgb.early_stopping(EARLY_STOP, verbose=False)],
    )


def predict(booster: lgb.Booster, feat: pd.DataFrame, features=FEATURES) -> np.ndarray:
    return booster.predict(feat[features], num_iteration=booster.best_iteration or None)


def val_ndcg(booster: lgb.Booster, val: pd.DataFrame, features=FEATURES) -> float:
    """Mean NDCG@10 with the project metric (not LightGBM's), IDCG = 0 queries excluded."""
    return float(metrics.per_query(val, predict(booster, val, features)).ndcg.mean())


def plot_importance(booster: lgb.Booster, path, features=FEATURES, name="ltr") -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    gain = pd.Series(booster.feature_importance("gain"), index=features).sort_values()
    fig, ax = plt.subplots(figsize=(7, 6))
    ax.barh(gain.index, gain.to_numpy())
    ax.set_xlabel("gain")
    ax.set_title(f"{name} feature importance (gain)")
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=120)
    plt.close(fig)


def _mlflow(experiment: str):
    os.environ.setdefault("MLFLOW_ALLOW_FILE_STORE", "true")  # SPEC 5.4: file store in ./mlruns
    import mlflow

    mlflow.set_tracking_uri((config.root() / "mlruns").as_uri())
    mlflow.set_experiment(experiment)
    return mlflow


def _save(name: str, booster: lgb.Booster, params: dict, score: float, fixtures: bool) -> dict:
    """models/<name>.txt + meta, reports/<name>_importance.png, and a `best` MLflow run."""
    from product_search_ranker.evaluate.run import data_sha, git_sha

    features = SYSTEM_FEATURES[name]
    out = config.models_dir(fixtures)
    out.mkdir(parents=True, exist_ok=True)
    booster.save_model(str(out / f"{name}.txt"), num_iteration=booster.best_iteration)
    meta = {
        "features": features,
        "params": {**BASE, **params},
        "val_ndcg10": round(score, 4),
        "best_iteration": booster.best_iteration,
        "data_sha": data_sha(fixtures),
        "git_sha": git_sha(),
    }
    if name == "ltr_qu":
        meta["qu_cache_sha"] = qu.cache_sha(fixtures)
    (out / f"{name}_meta.json").write_text(json.dumps(meta, indent=2) + "\n")
    plot = config.eval_reports_dir(fixtures) / f"{name}_importance.png"
    plot_importance(booster, plot, features, name)
    mlflow = _mlflow(name)
    with mlflow.start_run(run_name="best"):
        mlflow.log_params(params)
        mlflow.log_metric("val_ndcg10", score)
        mlflow.log_artifact(str(plot))
        mlflow.log_artifact(str(out / f"{name}_meta.json"))
        mlflow.log_artifact(str(out / f"{name}.txt"))
    _load.cache_clear()  # a later `score` in this process must read the model just written
    print(f"wrote {out / f'{name}.txt'}, {plot}; {params}, val NDCG@10 {score:.4f}", flush=True)
    return meta


def train(fixtures: bool = False) -> dict:
    """Run the grid on val, keep the best config's model, write models/ and the importance plot."""
    from product_search_ranker.evaluate.run import data_sha, git_sha

    train_f, val_f = build("train", fixtures), build("val", fixtures)
    mlflow = _mlflow("ltr")
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
    return _save("ltr", booster, params, score, fixtures)


def train_qu(fixtures: bool = False) -> dict:
    """ltr_qu: ltr's chosen config (no new grid) on the ltr features plus qu_* (SPEC 5.5)."""
    meta_path = config.models_dir(fixtures) / "ltr_meta.json"
    if not meta_path.exists():
        raise SystemExit(f"{meta_path} is missing: run `psr train` first")
    chosen = json.loads(meta_path.read_text())["params"]
    params = {k: chosen[k] for k in GRID[0]}
    train_f, val_f = (_table("ltr_qu", s, fixtures) for s in ("train", "val"))
    booster = fit(train_f, val_f, params, QU_FEATURES)
    return _save("ltr_qu", booster, params, val_ndcg(booster, val_f, QU_FEATURES), fixtures)


def _table(name: str, split: str, fixtures: bool) -> pd.DataFrame:
    feat = build(split, fixtures)
    if name != "ltr_qu":
        return feat
    if n := qu.missing(split, fixtures):
        raise SystemExit(f"{n} covered {split} queries are not in the qu cache: finish `psr qu`")
    return qu.add_features(feat, split, fixtures)


@lru_cache(maxsize=4)
def _load(name: str, fixtures: bool) -> lgb.Booster:
    path = config.models_dir(fixtures) / f"{name}.txt"
    if not path.exists():
        cmd = "psr train --qu" if name == "ltr_qu" else "psr train"
        raise SystemExit(f"{path} is missing: run `{cmd}` first")
    saved = json.loads((path.parent / f"{name}_meta.json").read_text())
    if saved["features"] != SYSTEM_FEATURES[name]:
        raise SystemExit(f"{path} was trained on other features than the current list: retrain")
    if name == "ltr_qu" and saved["qu_cache_sha"] != qu.cache_sha(fixtures):
        raise SystemExit(f"{path} was trained on another qu cache: rerun `psr train --qu`")
    return lgb.Booster(model_file=str(path))


def score(df: pd.DataFrame, split: str, fixtures: bool = False, name: str = "ltr") -> np.ndarray:
    """`name` score for each (query_id, product_id) row of `df`, from the split's features."""
    feat = _table(name, split, fixtures)
    pairs = pd.MultiIndex.from_frame(feat[["query_id", "product_id"]])
    pred = predict(_load(name, fixtures), feat, SYSTEM_FEATURES[name])
    out = pd.Series(pred, index=pairs).reindex(
        pd.MultiIndex.from_frame(df[["query_id", "product_id"]])
    )
    if out.isna().any():
        raise SystemExit(f"{int(out.isna().sum())} rows of `df` have no features in split {split}")
    return out.to_numpy()
