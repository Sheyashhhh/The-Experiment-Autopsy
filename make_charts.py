import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams.update({"font.size": 11, "figure.dpi": 150})

with open("results.json") as f:
    r = json.load(f)
df = pd.read_csv("program_results.csv")

# 1. FPR comparison bar chart
fig, ax = plt.subplots(figsize=(6, 4))
methods = ["Naive\npeeking", "Fixed\nhorizon", "Mixture-test\n(prototype)"]
values = [r["fpr_peeking_naive"] * 100, r["fpr_fixed_horizon"] * 100, r["fpr_msprt_prototype"] * 100]
colors = ["#d62728", "#2ca02c", "#1f77b4"]
bars = ax.bar(methods, values, color=colors)
ax.axhline(5, color="gray", linestyle="--", linewidth=1, label="Nominal 5% target")
for b, v in zip(bars, values):
    ax.text(b.get_x() + b.get_width() / 2, v + 0.5, f"{v:.1f}%", ha="center", fontweight="bold")
ax.set_ylabel("False positive rate (%)")
ax.set_title("False Positive Rate by Stopping Rule\n(on known-null experiments; prototype is exploratory)")
ax.legend()
plt.tight_layout()
plt.savefig("charts/01_fpr_by_method.png")
plt.close()

# 2. Declared winners: true vs false, peeking vs fixed horizon
fig, ax = plt.subplots(figsize=(6.5, 4))
labels = ["Peeking\n(stop-when-sig)", "Fixed horizon"]
n_declared = [r["n_declared_winners_peeking"], r["n_declared_winners_fixed_horizon"]]
n_false = [r["n_false_positive_winners"], int(r["n_declared_winners_fixed_horizon"] * r["pct_declared_winners_that_are_false_fixed_horizon"] / 100)]
n_true = [d - f for d, f in zip(n_declared, n_false)]
x = np.arange(len(labels))
ax.bar(x, n_true, label="Real winners", color="#2ca02c")
ax.bar(x, n_false, bottom=n_true, label="False positives", color="#d62728")
for i, (t, f) in enumerate(zip(n_true, n_false)):
    ax.text(i, t + f + 2.5, f"{f}/{t+f} false ({100*f/(t+f):.0f}%)", ha="center", fontsize=9)
ax.set_xticks(x)
ax.set_xticklabels(labels)
ax.set_ylabel("Declared winners (of 200 experiments)")
ax.set_title("Declared Winners: Real vs False, by Policy")
ax.set_ylim(0, max(n_declared) * 1.22)
ax.legend(loc="upper right", bbox_to_anchor=(1.0, 0.92))
plt.tight_layout()
plt.savefig("charts/02_declared_winners_composition.png")
plt.close()

# 3. MDE curve: relative lift detectable vs sample size per arm
from experiment_autopsy.power import mde_given_traffic
ns = np.logspace(3, 5.5, 25).astype(int)
mdes = [mde_given_traffic(0.08, n) * 100 for n in ns]
fig, ax = plt.subplots(figsize=(6.5, 4))
ax.plot(ns, mdes, marker="o", markersize=3, color="#1f77b4")
ax.axvline(r["n_per_arm_available_14days"], color="gray", linestyle="--",
           label=f"Current 14-day traffic\n({r['n_per_arm_available_14days']:,}/arm)")
ax.axhline(r["mde_relative_at_current_traffic"] * 100, color="gray", linestyle=":")
ax.set_xscale("log")
ax.set_xlabel("Sample size per arm (log scale)")
ax.set_ylabel("Minimum detectable relative lift (%)")
ax.set_title("Power Curve: What Can You Even Detect?\n(baseline rate = 8%)")
ax.legend()
plt.tight_layout()
plt.savefig("charts/03_power_curve.png")
plt.close()

# 4. CUPED variance reduction: before/after distribution
from experiment_autopsy.simulate import simulate_cuped_userlevel
from experiment_autopsy.cuped import cuped_adjust
cuped_data = simulate_cuped_userlevel(n_per_arm=8000, seed=7)
y = cuped_data["y"].values
x_cov = cuped_data["pre_covariate"].values
y_adj, theta, var_reduction = cuped_adjust(y, x_cov)
fig, ax = plt.subplots(figsize=(6.5, 4))
ax.hist(y - y.mean(), bins=60, alpha=0.5, label=f"Before CUPED (var={np.var(y):.1f})", color="#d62728", density=True)
ax.hist(y_adj - y_adj.mean(), bins=60, alpha=0.5, label=f"After CUPED (var={np.var(y_adj):.1f})", color="#2ca02c", density=True)
ax.set_xlabel("Centered outcome (revenue units)")
ax.set_ylabel("Density")
ax.set_title(f"CUPED Variance Reduction: {var_reduction:.1f}%")
ax.legend()
plt.tight_layout()
plt.savefig("charts/04_cuped_variance_reduction.png")
plt.close()

# 5. SRM: p-value distribution across the program, flagged vs not
fig, ax = plt.subplots(figsize=(6.5, 4))
flagged = df[df["srm_flagged"]]
not_flagged = df[~df["srm_flagged"]]
ax.scatter(not_flagged.index, -np.log10(not_flagged["srm_p"] + 1e-300), color="#1f77b4", s=15, label="Not flagged")
ax.scatter(flagged.index, -np.log10(flagged["srm_p"] + 1e-300), color="#d62728", s=25, label="SRM flagged (p<0.01)")
ax.axhline(-np.log10(0.01), color="gray", linestyle="--", label="Flag threshold")
ax.set_xlabel("Experiment index")
ax.set_ylabel("-log10(SRM p-value)")
ax.set_title(f"SRM Screen Across the Program\n({r['pct_experiments_srm_flagged']:.1f}% flagged)")
ax.legend()
plt.tight_layout()
plt.savefig("charts/05_srm_screen.png")
plt.close()

# 6. Novelty: example decaying experiment, daily lift over time
from experiment_autopsy.simulate import simulate_experiment_program
program = simulate_experiment_program(n_experiments=200, seed=42)
novelty_exps = [e for e in program if e["true_has_novelty"]]
example = max(novelty_exps, key=lambda e: e["true_relative_lift"])  # clearest signal for illustration
daily_lift = (example["daily_conv_t"] / example["daily_n_t"]) - (example["daily_conv_c"] / example["daily_n_c"])
n_days_ex = len(daily_lift)
true_decay = np.linspace(1.0, 0.15, n_days_ex)
true_daily_lift = example["baseline_rate"] * example["true_relative_lift"] * true_decay
fig, ax = plt.subplots(figsize=(6.5, 4))
ax.plot(range(1, n_days_ex + 1), daily_lift * 100, marker="o", color="#9467bd",
        alpha=0.55, label="Observed daily lift (noisy)")
ax.plot(range(1, n_days_ex + 1), true_daily_lift * 100, linestyle="--", color="#333333",
        linewidth=2, label="True underlying effect (ground truth)")
ax.axhline(0, color="gray", linewidth=0.8)
ax.set_xlabel("Day of experiment")
ax.set_ylabel("Daily lift (percentage points)")
ax.set_title("Novelty/Decay Effect: Real Signal vs. What You Actually See\n(this is why detection recall is low)")
ax.legend(fontsize=9)
plt.tight_layout()
plt.savefig("charts/06_novelty_example.png")
plt.close()

print("charts written:", __import__("os").listdir("charts"))
