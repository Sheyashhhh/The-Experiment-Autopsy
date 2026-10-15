from experiment_autopsy.power import required_n_per_arm, mde_given_traffic


def test_required_n_decreases_with_larger_lift():
    n_small_lift = required_n_per_arm(0.08, 0.02)
    n_large_lift = required_n_per_arm(0.08, 0.10)
    assert n_large_lift < n_small_lift


def test_mde_roundtrip_matches_required_n():
    # if N is exactly what's required for a given lift, mde_given_traffic should
    # recover approximately that same lift
    baseline = 0.08
    true_lift = 0.06
    n = required_n_per_arm(baseline, true_lift)
    recovered_mde = mde_given_traffic(baseline, n)
    assert abs(recovered_mde - true_lift) / true_lift < 0.02  # within 2%


def test_mde_shrinks_with_more_traffic():
    mde_small_n = mde_given_traffic(0.08, 5000)
    mde_large_n = mde_given_traffic(0.08, 500000)
    assert mde_large_n < mde_small_n
