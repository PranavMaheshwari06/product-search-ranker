"""Paths, constants and environment overrides. Everything else imports from here."""

import os
from pathlib import Path

SEED = 0
LOCALES = ("us", "es")
LABELS = ("E", "S", "C", "I")
GAINS = {"E": 1.0, "S": 0.1, "C": 0.01, "I": 0.0}
SPLITS = ("train", "val", "test")

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


def eval_reports_dir(fixtures: bool = False) -> Path:
    """Fixture runs (tests, CI) never touch the committed reports/."""
    return root() / "data" / "fixtures_reports" if fixtures else reports_dir()


def indexes_dir(fixtures: bool = False) -> Path:
    return root() / "indexes" / ("fixtures" if fixtures else "")


K = 10
BOOTSTRAP_N = 10_000
PERMUTATION_N = 10_000
# Paired differences written to metrics.json whenever both systems have a run.
COMPARISONS = (("dense", "bm25"), ("hybrid", "bm25"), ("ltr", "bm25"), ("ltr", "hybrid"))


# LLM layer (SPEC 6). Read at call time so tests and `.env` can set them.
GEMINI_BASE = "https://generativelanguage.googleapis.com/v1beta"
LLM_BACKOFF_S = (5, 15, 45)
LLM_TIMEOUT_S = 120


def load_dotenv() -> None:
    """Copy KEY=VALUE lines from the git-ignored `.env` into the environment (never printed)."""
    path = root() / ".env"
    if not path.exists():
        return
    for line in path.read_text().splitlines():
        key, sep, value = line.strip().partition("=")
        if sep and not key.startswith("#"):
            os.environ.setdefault(key.strip(), value.strip().strip("\"'"))


def llm_provider(fixtures: bool = False) -> str:
    return "replay" if fixtures else os.environ.get("PSR_LLM_PROVIDER", "gemini")


def gemini_model() -> str | None:
    """None until the user sets PSR_GEMINI_MODEL; `psr llm-ping` then picks the current Flash."""
    return os.environ.get("PSR_GEMINI_MODEL") or None


def ollama_model() -> str | None:
    return os.environ.get("PSR_OLLAMA_MODEL") or None


def llm_rpm() -> int:
    return int(os.environ.get("PSR_LLM_RPM", "10"))


def llm_prices() -> tuple[float, float] | None:
    """USD per million input and output tokens, only when the user set both. Never guessed."""
    p_in, p_out = os.environ.get("PSR_PRICE_IN_PER_M"), os.environ.get("PSR_PRICE_OUT_PER_M")
    return (float(p_in), float(p_out)) if p_in and p_out else None


def replay_dir() -> Path:
    return root() / "tests" / "fixtures" / "replay"


def llm_cache_dir(fixtures: bool = False) -> Path:
    return root() / "data" / ("fixtures_llm_cache" if fixtures else "llm_cache")
