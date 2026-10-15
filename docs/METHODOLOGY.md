# Methodology, validation, and limitations

## Purpose

Experiment Autopsy is a simulation-based portfolio project for exploring common A/B-testing reliability failures. Synthetic data are useful here because the simulator knows the true treatment effect, whether allocation was deliberately corrupted, and whether a treatment effect was programmed to decay. Those labels make it possible to estimate error rates that cannot normally be observed directly in a live experimentation program.

## Current simulation design

- A simulated program contains 200 experiments with a mixture of treatment effects, null effects, SRM scenarios, and time-varying effects.
- A separate null batch is used to compare fixed-horizon and repeated daily significance checks.
- Power and minimum detectable effect calculations use a baseline conversion rate of 8% and the configured traffic assumptions.
- CUPED is demonstrated on a continuous revenue-style metric with an engineered pre/post correlation near 0.6.
- Multiple-testing behavior is assessed across five null metrics.
- The sequential mixture-test prototype is evaluated on the simulated null batch, but has not been calibrated broadly enough to claim formal always-valid inference.

## How to interpret the numbers

All percentages in `results.json`, `program_results.csv`, the dashboard, and the scorecard depend on the chosen data-generating process, sample sizes, effect distribution, seeds, and test implementation. They should be read as **scenario results**, not general performance guarantees.

The empirical false-positive rate is a proportion from a finite number of simulations. A more complete report should attach binomial confidence intervals and repeat the experiment over multiple independent seeds. A result near 5% is not proof of calibration by itself.

## Statistical and engineering limitations

### Sequential testing

The current implementation approximates a two-sample Bernoulli problem using an effective sample size and a pooled variance estimate. The current code is a prototype for exploring sequential monitoring. Its observed null rejection rate does not establish an always-valid p-value or a formal type-I-error guarantee for every stopping time. Do not use it to make live decisions without replacing it with, or validating it against, a method with clearly stated assumptions and established guarantees.

### CUPED

The demonstration reports variance reduction for the pooled adjusted continuous outcome. In a real analysis, the quantity of interest is typically the variance of the treatment-effect estimator under random assignment. The conversion from outcome variance reduction to runtime reduction assumes that the variance reduction transfers to the estimator, that daily traffic is stable, and that other constraints do not dominate runtime. The reported ~35% reduction is specific to this synthetic correlation setting.

### Multiple comparisons

The simulation includes an all-null setup with five independent metrics. In an all-null family, FDR and FWER coincide for the event of at least one rejection. They are not interchangeable in general. Real metrics can be correlated, and the appropriate correction depends on the decision objective and the dependence assumptions.

### SRM

The simulated allocation bugs are deliberately detectable. The reported recall is not a guarantee of sensitivity to subtle imbalances, low-traffic experiments, or logging failures that do not change the overall allocation ratio. A production diagnostic also needs the intended allocation ratio and experiment-specific eligibility rules.

### Power and MDE

Normal approximations are convenient but can be inaccurate for sparse outcomes, extreme baseline rates, or small samples. Validate with simulation or an appropriate exact method when those conditions matter. A non-significant result is not proof of no effect.

### Novelty detection

The current time-slice diagnostic has low recall in the selected simulation. It only targets certain forms of early-versus-late effect change and should not be described as a general novelty detector.

## Current automated tests

The pytest suite checks selected numerical properties and expected behavior for the implemented methods. Passing tests confirms those assertions, not the full validity of every statistical guarantee. In particular, broad sequential calibration, realistic data-generating scenarios, and treatment-effect-estimator variance benchmarking remain future work.

## Recommended next validation milestone

1. Run at least tens of thousands of null simulations across several independent seeds for each decision rule.
2. Report empirical type-I-error estimates with binomial confidence intervals.
3. Evaluate power across a grid of baseline rates, effect sizes, sample sizes, and traffic imbalance.
4. Compare the sequential method with a trusted reference implementation and validate optional-stopping behavior under its stated assumptions.
5. Benchmark CUPED on repeated randomized treatment-effect estimates.
6. Add edge cases and continuous integration, and make the results reproducible from a clean clone.
