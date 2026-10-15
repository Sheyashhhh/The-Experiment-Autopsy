import numpy as np
import pandas as pd
import json

from experiment_autopsy.simulate import (
    simulate_experiment_program, simulate_null_batch, simulate_cuped_userlevel
)
from experiment_autopsy.peeking import (
    evaluate_stop_when_significant, evaluate_fixed_horizon, false_positive_rate
)
from experiment_autopsy.power import required_n_per_arm, mde_given_traffic, days_to_reach_n
from experiment_autopsy.srm import srm_chi_square
from experiment_autopsy.cuped import cuped_adjust
from experiment_autopsy.sequential import evaluate_msprt_stopping
from experiment_autopsy.multiple_comparisons import simulate_multi_metric_nulls
from experiment_autopsy.novelty import time_sliced_effect

results = {}

# ---------------------------------------------------------------------------
# 1. PEEKING / OPTIONAL STOPPING - FPR inflation on a clean null batch
# ---------------------------------------------------------------------------
null_batch = simulate_null_batch(n=2000, seed=1)
fpr_peeking = false_positive_rate(null_batch, method="peeking", alpha=0.05)
fpr_fixed = false_positive_rate(null_batch, method="fixed", alpha=0.05)
results["fpr_peeking_naive"] = fpr_peeking
results["fpr_fixed_horizon"] = fpr_fixed

# Exploratory mixture-test prototype on the same null batch; this is not a validity proof.
msprt_declares = sum(evaluate_msprt_stopping(exp, alpha=0.05)["declared"] for exp in null_batch)
fpr_msprt = msprt_declares / len(null_batch)
results["fpr_msprt_prototype"] = fpr_msprt

# ---------------------------------------------------------------------------
# 2. POWER / MDE - given the program's typical traffic
# ---------------------------------------------------------------------------
baseline_rate = 0.08
daily_traffic_per_arm = 1200
n_days = 14
n_per_arm_available = daily_traffic_per_arm * n_days
mde = mde_given_traffic(baseline_rate, n_per_arm_available)
n_needed_for_5pct = required_n_per_arm(baseline_rate, 0.05)
days_needed_for_5pct = days_to_reach_n(n_needed_for_5pct, daily_traffic_per_arm)
results["mde_relative_at_current_traffic"] = mde
results["n_per_arm_available_14days"] = n_per_arm_available
results["n_per_arm_needed_for_5pct_lift"] = n_needed_for_5pct
results["days_needed_for_5pct_lift"] = days_needed_for_5pct

# ---------------------------------------------------------------------------
# 3. FULL PROGRAM: 200 experiments, run under stop-when-significant policy
# ---------------------------------------------------------------------------
program = simulate_experiment_program(n_experiments=200, seed=42)

rows = []
for exp in program:
    peek_result = evaluate_stop_when_significant(exp, alpha=0.05)
    fixed_result = evaluate_fixed_horizon(exp, alpha=0.05)
    srm_c = exp["daily_n_c"].sum()
    srm_t = exp["daily_n_t"].sum()
    chi2, p_srm, srm_flag = srm_chi_square(srm_c, srm_t)
    novelty = time_sliced_effect(exp)

    rows.append({
        "exp_id": exp["exp_id"],
        "true_has_effect": exp["true_has_effect"],
        "true_relative_lift": exp["true_relative_lift"],
        "true_has_srm_bug": exp["true_has_srm_bug"],
        "true_has_novelty": exp["true_has_novelty"],
        "peek_declared_winner": peek_result["declared"],
        "peek_day_stopped": peek_result["day_stopped"],
        "fixed_declared_winner": fixed_result["declared"],
        "srm_chi2": chi2,
        "srm_p": p_srm,
        "srm_flagged": srm_flag,
        "novelty_p_interaction": novelty["p_interaction"],
        "novelty_flagged": novelty["novelty_flagged"],
        "lift_early": novelty["lift_early"],
        "lift_late": novelty["lift_late"],
    })

df = pd.DataFrame(rows)
df.to_csv("program_results.csv", index=False)

declared_winners = df[df["peek_declared_winner"]]
n_declared = len(declared_winners)
n_false_positive_winners = len(declared_winners[~declared_winners["true_has_effect"]])
pct_winners_false = 100 * n_false_positive_winners / n_declared if n_declared else 0

results["n_declared_winners_peeking"] = int(n_declared)
results["n_false_positive_winners"] = int(n_false_positive_winners)
results["pct_declared_winners_that_are_false"] = pct_winners_false

# what would fixed-horizon testing on the SAME 200 experiments have declared?
declared_fixed = df[df["fixed_declared_winner"]]
n_declared_fixed = len(declared_fixed)
n_fp_fixed = len(declared_fixed[~declared_fixed["true_has_effect"]])
pct_fp_fixed = 100 * n_fp_fixed / n_declared_fixed if n_declared_fixed else 0
results["n_declared_winners_fixed_horizon"] = int(n_declared_fixed)
results["pct_declared_winners_that_are_false_fixed_horizon"] = pct_fp_fixed

# SRM
results["pct_experiments_srm_flagged"] = 100 * df["srm_flagged"].mean()
results["pct_experiments_true_srm_bug"] = 100 * df["true_has_srm_bug"].mean()
srm_detection_recall = df[df["true_has_srm_bug"]]["srm_flagged"].mean() * 100
results["srm_detection_recall_pct"] = srm_detection_recall
# how many declared winners were sitting on top of an undetected SRM bug?
winners_with_srm = declared_winners[declared_winners["true_has_srm_bug"]]
results["n_declared_winners_with_srm_bug"] = int(len(winners_with_srm))

# novelty
results["pct_true_winners_with_novelty"] = 100 * df[df["true_has_effect"]]["true_has_novelty"].mean()
results["pct_novelty_correctly_flagged"] = 100 * df[df["true_has_novelty"]]["novelty_flagged"].mean()
results["pct_false_novelty_alarms_on_stable_winners"] = 100 * df[(df["true_has_effect"]) & (~df["true_has_novelty"])]["novelty_flagged"].mean()

# ---------------------------------------------------------------------------
# 4. CUPED - variance reduction and runtime reduction
# ---------------------------------------------------------------------------
cuped_data = simulate_cuped_userlevel(n_per_arm=8000, seed=7)
y = cuped_data["y"].values
x = cuped_data["pre_covariate"].values
y_adj, theta, var_reduction_pct = cuped_adjust(y, x)
results["cuped_theta"] = float(theta)
results["cuped_variance_reduction_pct"] = float(var_reduction_pct)
results["cuped_runtime_reduction_pct"] = float(var_reduction_pct)  # same number, see cuped.py docstring

# translate into a concrete "days saved" example at the program's traffic level
days_before = days_needed_for_5pct
days_after = int(np.ceil(days_before * (1 - var_reduction_pct / 100)))
results["days_to_detect_5pct_lift_before_cuped"] = days_before
results["days_to_detect_5pct_lift_after_cuped"] = days_after

# ---------------------------------------------------------------------------
# 5. MULTIPLE COMPARISONS
# ---------------------------------------------------------------------------
naive_fpr_mc, bh_fpr_mc = simulate_multi_metric_nulls(n_experiments=2000, n_metrics=5, alpha=0.05, seed=3)
results["multi_metric_naive_fpr_5metrics"] = naive_fpr_mc
results["multi_metric_bh_fpr_5metrics"] = bh_fpr_mc

with open("results.json", "w") as f:
    json.dump(results, f, indent=2, default=float)

for k, v in results.items():
    print(f"{k}: {v}")
