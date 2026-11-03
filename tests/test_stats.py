import numpy as np

from product_search_ranker.evaluate import stats


def test_bootstrap_ci_contains_the_true_mean_on_a_synthetic_case():
    rng = np.random.default_rng(1)
    x = rng.uniform(0, 1, size=1_000)  # true mean 0.5
    lo, hi = stats.ci95(x)
    assert lo < 0.5 < hi
    assert lo < x.mean() < hi
    # Width close to the normal approximation 2 * 1.96 * sd / sqrt(n).
    assert abs((hi - lo) - 3.92 * x.std() / np.sqrt(len(x))) < 0.005


def test_bootstrap_is_deterministic_and_paired():
    x = np.arange(50.0)
    assert stats.ci95(x) == stats.ci95(x)
    # Same seed and length: the same query resamples are used for every system.
    a, b = np.linspace(0, 1, 50), np.linspace(0, 1, 50) ** 2
    diffs = stats.bootstrap_means(a) - stats.bootstrap_means(b)
    np.testing.assert_allclose(diffs, stats.bootstrap_means(a - b))


def test_paired_difference_and_sign_flip_test():
    rng = np.random.default_rng(2)
    base = rng.uniform(0, 1, size=400)
    better = base + 0.05 + rng.normal(0, 0.1, size=400)
    delta, (lo, hi) = stats.paired_diff(better, base)
    assert lo < 0.05 < hi and lo > 0
    assert stats.sign_flip_p(better, base) < 0.001
    assert stats.sign_flip_p(base, base) == 1.0
    noise = base + rng.normal(0, 0.1, size=400)
    assert stats.sign_flip_p(noise - noise.mean() + base.mean(), base) > 0.5
