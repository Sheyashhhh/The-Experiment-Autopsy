"""
srm.py
------
Sample Ratio Mismatch: a chi-square goodness-of-fit test on the actual split vs the
intended split (usually 50/50). SRM means the randomization or logging pipeline is
broken, which invalidates the experiment's causal claim REGARDLESS of what the
outcome metric says. Convention in the industry (Microsoft's ExP team, among others)
is to flag SRM at a strict threshold like p < 0.01, because false alarms here are
cheap but missed SRM silently poisons every downstream number.
"""
from scipy import stats


def srm_chi_square(n_control, n_treatment, expected_ratio=0.5, alpha=0.01):
    """Returns (chi2_stat, p_value, is_srm_flagged)."""
    total = n_control + n_treatment
    expected_c = total * expected_ratio
    expected_t = total * (1 - expected_ratio)
    chi2, p = stats.chisquare([n_control, n_treatment], f_exp=[expected_c, expected_t])
    return chi2, p, (p < alpha)
