"""
multiple_comparisons.py
------------------------
If a team measures 5 metrics per experiment and calls it a win when ANY ONE crosses
p<0.05, the real false positive rate isn't 5% - it's roughly 1-(0.95)^5 ~ 23% per
experiment (for independent metrics; correlated metrics do a bit better). This module
simulates that and shows the Benjamini-Hochberg (FDR) correction fixing it.
"""
import numpy as np


def simulate_multi_metric_nulls(n_experiments=1000, n_metrics=5, alpha=0.05, seed=3):
    """For each of n_experiments (all true nulls), simulate n_metrics independent
    p-values ~ Uniform(0,1) under H0, and compare:
      - naive: declare a 'win' if ANY metric's raw p < alpha
      - BH: declare a 'win' if any metric survives Benjamini-Hochberg at alpha
    Returns (naive_fpr, bh_fpr).
    """
    rng = np.random.default_rng(seed)
    naive_wins = 0
    bh_wins = 0
    for _ in range(n_experiments):
        pvals = rng.uniform(0, 1, n_metrics)
        if (pvals < alpha).any():
            naive_wins += 1
        if benjamini_hochberg(pvals, alpha).any():
            bh_wins += 1
    return naive_wins / n_experiments, bh_wins / n_experiments


def benjamini_hochberg(pvals, alpha=0.05):
    """Returns a boolean array: which p-values survive BH correction at level alpha."""
    pvals = np.asarray(pvals)
    n = len(pvals)
    order = np.argsort(pvals)
    ranked = pvals[order]
    thresholds = (np.arange(1, n + 1) / n) * alpha
    passed = ranked <= thresholds
    if not passed.any():
        return np.zeros(n, dtype=bool)
    max_i = np.max(np.where(passed))
    reject = np.zeros(n, dtype=bool)
    reject[order[:max_i + 1]] = True
    return reject
