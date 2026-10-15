"""
cuped.py
--------
CUPED (Controlled-experiment Using Pre-Experiment Data), Microsoft's 2013 technique:
use a covariate measured BEFORE the experiment started (so it can't be affected by
treatment) to strip out variance that has nothing to do with the treatment effect.

theta = Cov(Y, X) / Var(X)
Y_adjusted = Y - theta * (X - mean(X))

This doesn't change the expected treatment effect (X is uncorrelated with treatment
by design, since it's pre-period), it just shrinks the noise around it. Less noise
means you need fewer users to hit the same statistical power -> the "time-to-money"
part: variance reduction translates directly into a runtime reduction.
"""
import numpy as np


def cuped_adjust(y, x):
    """y, x are 1D arrays (same length). Returns (y_adjusted, theta, variance_reduction_pct)."""
    y = np.asarray(y, dtype=float)
    x = np.asarray(x, dtype=float)
    cov_yx = np.cov(y, x, ddof=1)[0, 1]
    var_x = np.var(x, ddof=1)
    theta = cov_yx / var_x if var_x > 0 else 0.0
    y_adj = y - theta * (x - x.mean())
    var_before = np.var(y, ddof=1)
    var_after = np.var(y_adj, ddof=1)
    variance_reduction_pct = 100 * (1 - var_after / var_before) if var_before > 0 else 0.0
    return y_adj, theta, variance_reduction_pct


def runtime_reduction_pct(variance_reduction_pct):
    """Required sample size (and therefore runtime, holding daily traffic fixed) is
    proportional to variance. If variance drops by v%, required N drops by the same
    v% - this is the direct algebraic link CUPED papers use to report 'days saved'."""
    return variance_reduction_pct
