"""
simulate.py
-----------
Builds a synthetic but realistic experimentation program where we, the simulator,
control the ground truth (which experiments have a real effect and which don't).
This is what makes the rest of the audit checkable: we can compute the TRUE false
positive rate, because we know which "wins" are fake.

Design choices (stated up front, not hidden):
- Binary outcome (conversion) is the primary metric, generated day by day so we can
  simulate a team "peeking" at results daily.
- ~40% of experiments carry a real lift (drawn from a plausible, not generous, range).
  60% are true nulls. This ratio is a modeling assumption, not a fact about all
  companies - I'm using it because internal industry write-ups from Bing/Netflix/
  Airbnb-style experimentation teams commonly cite that a large majority of tested
  ideas don't move the needle.
- ~7% of experiments get a randomization/logging bug that skews the split away from
  50/50 (SRM), independent of whether they have a real effect.
- Each experiment also gets a per-user pre-period covariate (correlated with the
  outcome) so CUPED has something real to adjust on, and a "novelty" flag that decays
  the treatment effect over time for a subset of true winners.
"""
import numpy as np
import pandas as pd


def simulate_experiment_program(
    n_experiments=200,
    frac_real_effect=0.40,
    baseline_rate=0.08,
    daily_traffic_per_arm=1200,
    n_days=14,
    srm_rate=0.07,
    novelty_rate=0.25,
    seed=42,
):
    """
    Returns a list of experiment dicts. Each dict contains day-by-day cumulative
    counts for control/treatment, plus ground truth fields used ONLY for grading
    the audit afterwards (a real analyst would never get these).
    """
    rng = np.random.default_rng(seed)
    experiments = []

    for i in range(n_experiments):
        has_real_effect = rng.random() < frac_real_effect
        has_srm_bug = rng.random() < srm_rate
        has_novelty = has_real_effect and (rng.random() < novelty_rate)

        # true relative lift for winners: modest, realistic range (2%-15% relative)
        true_relative_lift = rng.uniform(0.02, 0.15) if has_real_effect else 0.0

        # allocation: bugged experiments get a skewed split (e.g. 44-48% to treatment)
        if has_srm_bug:
            treat_share = rng.uniform(0.42, 0.48)
        else:
            treat_share = 0.50

        # per-day daily traffic split according to treat_share
        daily_n_t = rng.poisson(daily_traffic_per_arm * treat_share / 0.5, n_days)
        daily_n_c = rng.poisson(daily_traffic_per_arm * (1 - treat_share) / 0.5, n_days)

        # day-varying treatment effect (novelty decays the lift toward 0 by day 14)
        if has_novelty:
            decay = np.linspace(1.0, 0.15, n_days)  # effect fades over the run
        else:
            decay = np.ones(n_days)

        daily_rate_t = baseline_rate * (1 + true_relative_lift * decay)
        daily_rate_c = np.full(n_days, baseline_rate)

        daily_conv_t = rng.binomial(daily_n_t, np.clip(daily_rate_t, 0, 1))
        daily_conv_c = rng.binomial(daily_n_c, np.clip(daily_rate_c, 0, 1))

        # pre-period covariate: user-level propensity correlated with conversion,
        # used for CUPED. We simulate this separately at user level for a cleaner demo
        # (see cuped_userlevel_sample below) rather than polluting the daily counts.

        experiments.append({
            "exp_id": f"exp_{i:03d}",
            "daily_n_t": daily_n_t,
            "daily_n_c": daily_n_c,
            "daily_conv_t": daily_conv_t,
            "daily_conv_c": daily_conv_c,
            "baseline_rate": baseline_rate,
            # ground truth - for grading only
            "true_has_effect": has_real_effect,
            "true_relative_lift": true_relative_lift,
            "true_has_srm_bug": has_srm_bug,
            "true_has_novelty": has_novelty,
        })

    return experiments


def simulate_null_batch(n=1000, baseline_rate=0.08, daily_traffic_per_arm=1200, n_days=14, seed=1):
    """A batch of PURE NULL experiments (no effect, no SRM) - used to measure false
    positive rate under different stopping rules, cleanly, with no confound from
    real winners mixed in."""
    rng = np.random.default_rng(seed)
    batch = []
    for i in range(n):
        daily_n_t = rng.poisson(daily_traffic_per_arm, n_days)
        daily_n_c = rng.poisson(daily_traffic_per_arm, n_days)
        daily_conv_t = rng.binomial(daily_n_t, baseline_rate)
        daily_conv_c = rng.binomial(daily_n_c, baseline_rate)
        batch.append({
            "exp_id": f"null_{i:04d}",
            "daily_n_t": daily_n_t,
            "daily_n_c": daily_n_c,
            "daily_conv_t": daily_conv_t,
            "daily_conv_c": daily_conv_c,
        })
    return batch


def simulate_cuped_userlevel(n_per_arm=8000, true_absolute_lift=5.0, covariate_corr=0.6,
                              revenue_mean=50.0, revenue_sd=10.0, seed=7):
    """
    User-level simulation for the CUPED demo, using a CONTINUOUS metric (e.g. weekly
    revenue per user) rather than a binary conversion flag. This is deliberate: CUPED's
    variance reduction is capped by the squared correlation between the pre-period
    covariate and the outcome (rho^2), and that correlation is naturally much higher
    for a continuous, autocorrelated metric like revenue (week-to-week revenue
    correlation of ~0.5-0.7 is typical) than for a single binary conversion event.
    Using a binary outcome here would understate what CUPED is actually good for.

    covariate_corr is exactly rho between the pre-period covariate and the (untreated)
    outcome, so the theoretical variance reduction is ~covariate_corr^2 - stated here
    so the result below isn't a mystery number.
    """
    rng = np.random.default_rng(seed)
    n = n_per_arm * 2
    treat = np.array([0] * n_per_arm + [1] * n_per_arm)

    cov = [[1, covariate_corr], [covariate_corr, 1]]
    z = rng.multivariate_normal([0, 0], cov, size=n)
    pre_covariate = revenue_mean + revenue_sd * z[:, 0]
    y_base = revenue_mean + revenue_sd * z[:, 1]  # potential outcome under control
    y = y_base + treat * true_absolute_lift  # treatment adds a flat revenue bump

    return pd.DataFrame({"treat": treat, "y": y, "pre_covariate": pre_covariate})
