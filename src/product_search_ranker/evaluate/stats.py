"""Paired bootstrap CIs and a sign-flip randomization test over per-query scores."""

import numpy as np

from product_search_ranker import config

# Resamples are drawn in blocks so 10,000 x n_queries indexes never sit in memory at once.
BLOCK = 500


def _resampled_means(values: np.ndarray, n: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    means = np.empty(n)
    for start in range(0, n, BLOCK):
        m = min(BLOCK, n - start)
        idx = rng.integers(0, len(values), size=(m, len(values)))
        means[start : start + m] = values[idx].mean(axis=1)
    return means


def bootstrap_ci(
    values, n: int = config.N_BOOTSTRAP, seed: int = config.SEED
) -> tuple[float, float]:
    """Percentile 95% CI of the mean, resampling queries with replacement."""
    values = np.asarray(values, dtype=float)
    lo, hi = np.percentile(_resampled_means(values, n, seed), [2.5, 97.5])
    return float(lo), float(hi)


def paired_diff(a, b, n: int = config.N_BOOTSTRAP, seed: int = config.SEED) -> dict:
    """Mean of a - b over the same queries, its bootstrap CI and the sign-flip p-value."""
    d = np.asarray(a, dtype=float) - np.asarray(b, dtype=float)
    lo, hi = bootstrap_ci(d, n, seed)
    return {
        "delta": float(d.mean()),
        "ci95": [lo, hi],
        "p_value": sign_flip_p(d, n, seed),
        "n_queries": int(len(d)),
    }


def sign_flip_p(d, n: int = config.N_PERMUTATIONS, seed: int = config.SEED) -> float:
    """Two-sided p-value: under H0 each query's difference is equally likely to have either sign."""
    d = np.asarray(d, dtype=float)
    observed = abs(d.mean())
    rng = np.random.default_rng(seed)
    hits = 0
    for start in range(0, n, BLOCK):
        m = min(BLOCK, n - start)
        signs = rng.integers(0, 2, size=(m, len(d)), dtype=np.int8) * 2 - 1
        hits += int((np.abs((signs * d).mean(axis=1)) >= observed - 1e-12).sum())
    # The +1s count the observed assignment, so p is never 0.
    return (hits + 1) / (n + 1)
