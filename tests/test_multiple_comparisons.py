import numpy as np
from experiment_autopsy.multiple_comparisons import benjamini_hochberg, simulate_multi_metric_nulls


def test_bh_matches_textbook_example():
    # classic example: 5 p-values, alpha=0.05
    pvals = np.array([0.01, 0.02, 0.03, 0.04, 0.20])
    reject = benjamini_hochberg(pvals, alpha=0.05)
    # thresholds: 0.01, 0.02, 0.03, 0.04, 0.05 -> all but the last pass
    assert reject.tolist() == [True, True, True, True, False]


def test_bh_controls_fdr_below_naive():
    naive_fpr, bh_fpr = simulate_multi_metric_nulls(n_experiments=3000, n_metrics=5, alpha=0.05, seed=0)
    assert bh_fpr < naive_fpr
    assert abs(bh_fpr - 0.05) < 0.02
