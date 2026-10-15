"""
Experimental sequential mixture-test prototype inspired by mSPRT.

IMPORTANT: This implementation approximates the two-sample Bernoulli problem using an
effective sample size and a pooled, data-estimated variance. It has NOT been shown to
provide formally always-valid p-values under all supported data-generating processes.
The observed false-positive rate in a finite simulation is not proof of a stopping-time
type-I-error guarantee. Do not use this prototype for live experiment decisions
without a more rigorous derivation and broad calibration against a trusted reference.

The module is retained as an exploratory comparison against naive peeking and
fixed-horizon testing.
"""
import numpy as np


def mixture_lambda(n, sigma2, tau2, mean_diff):
    """
    Likelihood ratio (mixture over N(0, tau2) prior on the true mean difference)
    for n paired-equivalent observations with per-observation variance sigma2 and
    observed standardized cumulative mean difference `mean_diff` (treatment - control).

    Formula (Johari et al., normal mixture SPRT):
        Lambda_n = sqrt(sigma2 / (sigma2 + n*tau2)) *
                   exp( n^2 * tau2 * mean_diff^2 / (2 * sigma2 * (sigma2 + n*tau2)) )
    where mean_diff is the raw (unstandardized) difference of sample means and
    sigma2 is the variance of a single observation's contribution to that difference.
    """
    denom = sigma2 + n * tau2
    term1 = np.sqrt(sigma2 / denom)
    term2 = np.exp((n ** 2) * tau2 * (mean_diff ** 2) / (2 * sigma2 * denom))
    return term1 * term2


def evaluate_msprt_stopping(exp, alpha=0.05, tau2=None, min_day=1):
    """
    Runs the experimental mixture-test statistic day by day on an experiment dict
    (same format as peeking.py). The reciprocal likelihood ratio is treated as an
    exploratory diagnostic; formal always-valid p-value guarantees are not asserted.
    tau2 defaults to a plausible prior variance on the per-user effect if not supplied.
    """
    n_days = len(exp["daily_n_t"])
    cum_nt = np.cumsum(exp["daily_n_t"])
    cum_nc = np.cumsum(exp["daily_n_c"])
    cum_ct = np.cumsum(exp["daily_conv_t"])
    cum_cc = np.cumsum(exp["daily_conv_c"])

    if tau2 is None:
        p_hat = exp["baseline_rate"] if "baseline_rate" in exp else 0.08
        tau2 = (0.15 * p_hat) ** 2  # prior: effect sizes up to ~15% relative lift are plausible

    for day in range(min_day - 1, n_days):
        n_t, n_c = cum_nt[day], cum_nc[day]
        if n_t == 0 or n_c == 0:
            continue
        p_t = cum_ct[day] / n_t
        p_c = cum_cc[day] / n_c
        mean_diff = p_t - p_c
        # Reduce the two-sample problem to an equivalent one-sample mSPRT: define an
        # effective sample size n_eff such that Var(mean_diff) = sigma2 / n_eff, with
        # sigma2 the per-observation Bernoulli variance under the pooled rate. This
        # is exact when n_eff = harmonic-mean-style n_t*n_c/(n_t+n_c).
        p_pool = (cum_ct[day] + cum_cc[day]) / (n_t + n_c)
        sigma2 = p_pool * (1 - p_pool)
        n_eff = (n_t * n_c) / (n_t + n_c)
        lam = mixture_lambda(n_eff, sigma2, tau2, mean_diff)
        p_always_valid = min(1.0, 1.0 / lam) if lam > 0 else 1.0
        if p_always_valid < alpha:
            return {"declared": True, "day_stopped": day + 1, "p_at_stop": p_always_valid}

    return {"declared": False, "day_stopped": n_days, "p_at_stop": 1.0}
