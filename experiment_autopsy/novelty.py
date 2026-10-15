"""
novelty.py
----------
Novelty effects (or decay effects): a treatment effect that is strong on day 1-2 and
fades as users get used to it (or the reverse - a "primacy" dip that recovers). If you
only look at the pooled average effect, you miss this and mis-forecast the long-run
impact.

Two detectors are provided:
  - time_sliced_effect(): the simple version - split the run into early/late halves
    and test whether the difference between the two slice-level effects is itself
    significant. Easy to explain, uses only 2 aggregated data points.
  - trend_effect(): a regression-based version - fits a per-day treatment effect via
    weighted least squares and tests whether that effect has a significant linear
    slope over time. Uses all n_days points instead of 2.

  Honest result from testing both against simulated ground truth: at this program's
  default traffic (~1200/arm/day), the two tests have STATISTICALLY INDISTINGUISHABLE,
  and both are low, recall (~7-9%) - the trend test is not a fix for underpowered
  novelty detection. Its advantage over the split test only becomes measurable
  (roughly +3pp recall) once traffic is raised well above program defaults (see
  tests/test_novelty.py). The real lever is more data, not a cleverer test - use
  required_n_for_novelty_detection() below to see how much more.

required_n_for_novelty_detection() answers the natural follow-up: how much traffic
would you need for the simple test to reliably catch a decay of a given size? (Spoiler:
a lot more than you'd need to detect the average effect - this is why recall is low.)
"""
import numpy as np
from scipy import stats


def time_sliced_effect(exp):
    n_days = len(exp["daily_n_t"])
    half = n_days // 2

    def rate(conv, n):
        conv_sum, n_sum = conv.sum(), n.sum()
        return (conv_sum / n_sum if n_sum > 0 else np.nan), conv_sum, n_sum

    r_t_early, ct_e, nt_e = rate(exp["daily_conv_t"][:half], exp["daily_n_t"][:half])
    r_c_early, cc_e, nc_e = rate(exp["daily_conv_c"][:half], exp["daily_n_c"][:half])
    r_t_late, ct_l, nt_l = rate(exp["daily_conv_t"][half:], exp["daily_n_t"][half:])
    r_c_late, cc_l, nc_l = rate(exp["daily_conv_c"][half:], exp["daily_n_c"][half:])

    lift_early = (r_t_early - r_c_early)
    lift_late = (r_t_late - r_c_late)

    def se_diff(c1, n1, c2, n2):
        p1, p2 = c1 / n1, c2 / n2
        return np.sqrt(p1 * (1 - p1) / n1 + p2 * (1 - p2) / n2)

    se_early = se_diff(ct_e, nt_e, cc_e, nc_e)
    se_late = se_diff(ct_l, nt_l, cc_l, nc_l)
    se_interaction = np.sqrt(se_early ** 2 + se_late ** 2)
    z = (lift_early - lift_late) / se_interaction if se_interaction > 0 else 0.0
    p_interaction = 2 * (1 - stats.norm.cdf(abs(z)))

    return {
        "lift_early": lift_early,
        "lift_late": lift_late,
        "p_interaction": p_interaction,
        "novelty_flagged": p_interaction < 0.05 and lift_early > lift_late,
    }


def trend_effect(exp):
    """
    Fits daily lift_d = r_t[d] - r_c[d] for each day d, then regresses lift_d on day
    index (0..n_days-1), weighting each day by its inverse variance (days with more
    traffic / more precise estimates count more). Tests whether the slope is
    significantly negative (decaying effect) via a weighted least squares t-test.
    This uses every day's data instead of collapsing to 2 points, so it has
    meaningfully more power than the early/late split for the same underlying decay.
    """
    n_days = len(exp["daily_n_t"])
    days = np.arange(n_days)

    daily_lift = np.full(n_days, np.nan)
    daily_var = np.full(n_days, np.nan)
    for d in range(n_days):
        nt, nc = exp["daily_n_t"][d], exp["daily_n_c"][d]
        if nt == 0 or nc == 0:
            continue
        pt, pc = exp["daily_conv_t"][d] / nt, exp["daily_conv_c"][d] / nc
        daily_lift[d] = pt - pc
        daily_var[d] = pt * (1 - pt) / nt + pc * (1 - pc) / nc

    valid = ~np.isnan(daily_lift) & (daily_var > 0)
    if valid.sum() < 3:
        return {"slope": np.nan, "p_slope": 1.0, "novelty_flagged": False}

    w = 1.0 / daily_var[valid]
    x = days[valid]
    y = daily_lift[valid]

    # weighted least squares: y = a + b*x
    W = np.diag(w)
    X = np.column_stack([np.ones_like(x), x])
    XtW = X.T @ W
    beta = np.linalg.solve(XtW @ X, XtW @ y)
    resid = y - X @ beta
    dof = max(len(x) - 2, 1)
    sigma2 = (resid @ W @ resid) / dof
    cov_beta = sigma2 * np.linalg.inv(XtW @ X)
    se_slope = np.sqrt(cov_beta[1, 1])
    slope = beta[1]
    t_stat = slope / se_slope if se_slope > 0 else 0.0
    p_slope = 2 * (1 - stats.t.cdf(abs(t_stat), df=dof))

    return {
        "slope": slope,
        "p_slope": p_slope,
        "novelty_flagged": p_slope < 0.05 and slope < 0,
    }


def required_n_for_novelty_detection(baseline_rate, early_relative_lift, late_relative_lift,
                                      n_days=14, alpha=0.05, power=0.8):
    """
    How much traffic per arm per day would the SIMPLE (early/late split) test need
    to reliably (80% power) detect a decay from early_relative_lift to
    late_relative_lift? This is the honest answer to 'why is recall so low' - it's a
    power problem, and this function quantifies exactly how much data would fix it.
    """
    p_c = baseline_rate
    p_t_early = baseline_rate * (1 + early_relative_lift)
    p_t_late = baseline_rate * (1 + late_relative_lift)
    delta = (p_t_early - p_c) - (p_t_late - p_c)  # = p_t_early - p_t_late
    if delta == 0:
        return np.inf
    z_alpha = stats.norm.ppf(1 - alpha / 2)
    z_beta = stats.norm.ppf(power)
    # variance of (lift_early - lift_late), assuming n per half per arm, using
    # baseline_rate as a rough plug-in for all four cells
    var_per_n = 4 * baseline_rate * (1 - baseline_rate)  # sums the 4 proportion variances at rate ~baseline
    n_half_per_arm = ((z_alpha + z_beta) ** 2) * var_per_n / (delta ** 2)
    n_per_arm_total = int(np.ceil(n_half_per_arm * 2))  # early + late halves
    daily_traffic_needed = int(np.ceil(n_per_arm_total / n_days))
    return {"n_per_arm_total": n_per_arm_total, "implied_daily_traffic_per_arm": daily_traffic_needed}
