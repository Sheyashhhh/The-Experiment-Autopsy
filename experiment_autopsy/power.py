"""
power.py
--------
Standard power analysis for a two-proportion test, plus an MDE (minimum detectable
effect) calculator given fixed traffic. Most experimentation programs never do this
step and run tests that could never have detected the effect they were hoping for.
"""
import numpy as np
from scipy import stats


def required_n_per_arm(baseline_rate, relative_lift, alpha=0.05, power=0.8):
    """Sample size per arm needed to detect a given RELATIVE lift on a baseline
    conversion rate, using the standard two-proportion normal approximation."""
    p1 = baseline_rate
    p2 = baseline_rate * (1 + relative_lift)
    z_alpha = stats.norm.ppf(1 - alpha / 2)
    z_beta = stats.norm.ppf(power)
    pbar = (p1 + p2) / 2
    numerator = (z_alpha * np.sqrt(2 * pbar * (1 - pbar)) + z_beta * np.sqrt(p1 * (1 - p1) + p2 * (1 - p2))) ** 2
    denominator = (p2 - p1) ** 2
    return int(np.ceil(numerator / denominator))


def mde_given_traffic(baseline_rate, n_per_arm, alpha=0.05, power=0.8, lo=0.001, hi=1.0, tol=1e-4):
    """Solves for the minimum detectable RELATIVE lift given a fixed sample size per
    arm, via binary search on required_n_per_arm (monotonic in lift)."""
    while hi - lo > tol:
        mid = (lo + hi) / 2
        n_needed = required_n_per_arm(baseline_rate, mid, alpha, power)
        if n_needed > n_per_arm:
            lo = mid
        else:
            hi = mid
    return hi


def days_to_reach_n(n_per_arm, daily_traffic_per_arm):
    return int(np.ceil(n_per_arm / daily_traffic_per_arm))
