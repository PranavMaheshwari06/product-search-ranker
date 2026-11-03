"""Paired bootstrap CIs and a paired sign-flip test over per-query scores (SPEC 3).

Every call resamples with the same seed, so for the same number of queries the resampled query
sets are identical across systems: the system CIs and the difference CIs are paired.
"""

import numpy as np

from product_search_ranker import config

CHUNK = 500


def bootstrap_means(values, n: int = config.BOOTSTRAP_N, seed: int = config.SEED) -> np.ndarray:
    x = np.asarray(values, dtype=np.float64)
    rng = np.random.default_rng(seed)
    out = np.empty(n)
    for start in range(0, n, CHUNK):
        m = min(CHUNK, n - start)
        out[start : start + m] = x[rng.integers(0, len(x), size=(m, len(x)))].mean(axis=1)
    return out


def ci95(values, n: int = config.BOOTSTRAP_N, seed: int = config.SEED) -> tuple[float, float]:
    """Percentile 95% CI of the mean."""
    lo, hi = np.percentile(bootstrap_means(values, n, seed), [2.5, 97.5])
    return float(lo), float(hi)


def paired_diff(a, b, n: int = config.BOOTSTRAP_N, seed: int = config.SEED):
    """Mean of a - b over the same queries, with its percentile 95% CI."""
    d = np.asarray(a, dtype=np.float64) - np.asarray(b, dtype=np.float64)
    return float(d.mean()), ci95(d, n, seed)


def sign_flip_p(a, b, n: int = config.PERMUTATION_N, seed: int = config.SEED) -> float:
    """Two-sided p-value: randomly swap the two systems' scores within each query."""
    d = np.asarray(a, dtype=np.float64) - np.asarray(b, dtype=np.float64)
    observed = abs(d.mean())
    rng = np.random.default_rng(seed)
    hits = 0
    for start in range(0, n, CHUNK):
        m = min(CHUNK, n - start)
        signs = rng.choice((-1.0, 1.0), size=(m, len(d)))
        hits += int((np.abs((signs * d).mean(axis=1)) >= observed - 1e-12).sum())
    return (hits + 1) / (n + 1)
