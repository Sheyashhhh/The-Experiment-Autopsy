from experiment_autopsy.simulate import simulate_null_batch
from experiment_autopsy.peeking import false_positive_rate
from experiment_autopsy.sequential import evaluate_msprt_stopping


def test_peeking_inflates_fpr_above_fixed_horizon():
    null_batch = simulate_null_batch(n=1500, seed=5)
    fpr_peek = false_positive_rate(null_batch, method="peeking", alpha=0.05)
    fpr_fixed = false_positive_rate(null_batch, method="fixed", alpha=0.05)
    assert fpr_peek > fpr_fixed
    assert fpr_peek > 0.10          # inflated well above nominal
    assert abs(fpr_fixed - 0.05) < 0.03  # fixed horizon stays near nominal


def test_msprt_prototype_has_expected_behavior_on_reference_seed():
    null_batch = simulate_null_batch(n=1500, seed=6)
    declares = sum(evaluate_msprt_stopping(exp, alpha=0.05)["declared"] for exp in null_batch)
    fpr_msprt = declares / len(null_batch)
    # Regression check on this fixed simulation seed only; not a proof of sequential validity.
    assert fpr_msprt <= 0.05 + 0.01  # small slack for simulation noise
