"""Paths, constants and environment overrides. Everything else imports from here."""

import os
from pathlib import Path

SEED = 0
LOCALES = ("us", "es")
LABELS = ("E", "S", "C", "I")
GAINS = {"E": 1.0, "S": 0.1, "C": 0.01, "I": 0.0}
SPLITS = ("train", "val", "test")

K = 10
N_BOOTSTRAP = 10_000
N_PERMUTATIONS = 10_000
# Paired differences reported in metrics.json, (a, b) means a minus b; a pair is skipped until
# both systems have been evaluated on the split.
COMPARISONS = (
    ("dense", "bm25"),
    ("hybrid", "bm25"),
    ("ltr", "bm25"),
    ("ltr", "hybrid"),
    ("ltr_qu", "ltr"),
)

ESCI_FILES = {
    "shopping_queries_dataset_examples.parquet": "media",
    "shopping_queries_dataset_products.parquet": "media",
    # Plain text in git, not LFS: the media host answers 404 for it.
    "shopping_queries_dataset_sources.csv": "raw",
}
ESCI_BASES = {
    "media": "https://media.githubusercontent.com/media/amazon-science/esci-data/main/shopping_queries_dataset",
    "raw": "https://raw.githubusercontent.com/amazon-science/esci-data/main/shopping_queries_dataset",
}

# Expected small_version counts (esci-data README): locale -> (queries, judgements).
EXPECTED_COUNTS = {"us": (29_844, 601_354), "es": (8_049, 218_774)}
COUNT_TOLERANCE = 0.005


def root() -> Path:
    return Path(os.environ.get("PSR_HOME", ".")).resolve()


def fixtures_mode(flag: bool = False) -> bool:
    return flag or os.environ.get("PSR_DATA_MODE") == "fixtures"


def raw_dir() -> Path:
    return root() / "data" / "raw"


def processed_dir(fixtures: bool = False) -> Path:
    sub = "fixtures_processed" if fixtures else "processed"
    return root() / "data" / sub


def fixture_file() -> Path:
    return root() / "tests" / "fixtures" / "esci_sample.parquet"


def reports_dir() -> Path:
    return root() / "reports"


def indexes_dir(fixtures: bool = False) -> Path:
    base = root() / "indexes"
    return base / "fixtures" if fixtures else base


def runs_dir(fixtures: bool = False) -> Path:
    """Per-query results of each eval, kept so later evals can make paired comparisons."""
    return root() / "data" / ("fixtures_runs" if fixtures else "runs")


def metrics_path(split: str = "test", fixtures: bool = False) -> Path:
    # Fixture runs never touch reports/: those numbers would mean nothing.
    base = root() / "data" / "fixtures_reports" if fixtures else reports_dir()
    return base / ("metrics.json" if split == "test" else f"metrics_{split}.json")
