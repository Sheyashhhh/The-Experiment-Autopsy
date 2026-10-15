"""
experiment_autopsy
-------------------
A reusable toolkit for auditing the trustworthiness of an A/B testing program.

Modules:
    simulate               - generate a synthetic experimentation program with known ground truth
    peeking                - quantify false-positive inflation from optional stopping
    power                  - power analysis / minimum detectable effect (MDE)
    srm                    - Sample Ratio Mismatch detection (chi-square)
    cuped                  - variance reduction using pre-experiment covariates
    sequential             - experimental mixture-test prototype (not validated as always-valid)
    multiple_comparisons   - FDR control across multiple metrics
    novelty                - detect novelty / decay effects via time-sliced treatment effects
    scorecard              - roll everything into an Experimentation Health Scorecard
"""
