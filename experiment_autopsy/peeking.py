"""
peeking.py
----------
Quantifies the false positive inflation from "optional stopping": checking the
p-value every day and declaring a winner the moment it crosses 0.05, instead of
waiting for a pre-registered sample size / horizon.

This is the single most common way experimentation programs lie to themselves.
"""
import numpy as np
from scipy import stats


def two_prop_ztest(x1, n1, x2, n2):
    """Two-proportion z-test. Returns (z, p_two_sided). NaN-safe for zero-traffic days."""
    if n1 == 0 or n2 == 0:
        return 0.0, 1.0
    p1, p2 = x1 / n1, x2 / n2
    p_pool = (x1 + x2) / (n1 + n2)
    se = np.sqrt(p_pool * (1 - p_pool) * (1 / n1 + 1 / n2))
    if se == 0:
        return 0.0, 1.0
    z = (p2 - p1) / se
    p = 2 * (1 - stats.norm.cdf(abs(z)))
    return z, p


def evaluate_stop_when_significant(exp, alpha=0.05, min_day=1):
    """
    Simulates a team that checks the cumulative p-value every day starting on
    min_day, and stops (declares a winner) the first day it dips below alpha.
    Returns dict: declared (bool), day_stopped, p_at_stop.
    """
    n_days = len(exp["daily_n_t"])
    cum_nt = np.cumsum(exp["daily_n_t"])
    cum_nc = np.cumsum(exp["daily_n_c"])
    cum_ct = np.cumsum(exp["daily_conv_t"])
    cum_cc = np.cumsum(exp["daily_conv_c"])

    for day in range(min_day - 1, n_days):
        _, p = two_prop_ztest(cum_cc[day], cum_nc[day], cum_ct[day], cum_nt[day])
        if p < alpha:
            return {"declared": True, "day_stopped": day + 1, "p_at_stop": p}
    # never crossed threshold - evaluate at final day for completeness
    _, p_final = two_prop_ztest(cum_cc[-1], cum_nc[-1], cum_ct[-1], cum_nt[-1])
    return {"declared": False, "day_stopped": n_days, "p_at_stop": p_final}


def evaluate_fixed_horizon(exp, alpha=0.05):
    """Only looks at the final day's cumulative data - the statistically valid version."""
    n_c = exp["daily_n_c"].sum()
    n_t = exp["daily_n_t"].sum()
    c_c = exp["daily_conv_c"].sum()
    c_t = exp["daily_conv_t"].sum()
    _, p = two_prop_ztest(c_c, n_c, c_t, n_t)
    return {"declared": p < alpha, "day_stopped": len(exp["daily_n_t"]), "p_at_stop": p}


def false_positive_rate(null_batch, method="peeking", alpha=0.05):
    """Runs a stopping rule against a batch of KNOWN-NULL experiments and returns
    the empirical false positive rate."""
    declared = 0
    for exp in null_batch:
        result = (evaluate_stop_when_significant(exp, alpha=alpha) if method == "peeking"
                   else evaluate_fixed_horizon(exp, alpha=alpha))
        declared += result["declared"]
    return declared / len(null_batch)
