import numpy as np
from experiment_autopsy.srm import srm_chi_square


def test_srm_detects_no_bug_on_perfect_split():
    chi2, p, flagged = srm_chi_square(10000, 10000)
    assert not flagged
    assert p > 0.5


def test_srm_flags_obvious_skew():
    # 44/56 split on 20k users is a massive, unmistakable SRM
    chi2, p, flagged = srm_chi_square(8800, 11200)
    assert flagged
    assert p < 0.01


def test_srm_false_positive_rate_near_alpha():
    # under the true null (perfect 50/50 generative process with sampling noise),
    # the chi-square test should flag close to alpha fraction of the time
    rng = np.random.default_rng(0)
    alpha = 0.01
    flags = 0
    n_sims = 2000
    for _ in range(n_sims):
        n_c = rng.binomial(20000, 0.5)
        n_t = 20000 - n_c
        _, _, flagged = srm_chi_square(n_c, n_t, alpha=alpha)
        flags += flagged
    fpr = flags / n_sims
    assert abs(fpr - alpha) < 0.01  # within 1pp of nominal
