# Experiment Autopsy: Executive Readout

**Audience:** Product and Growth leadership  
**Purpose:** Demonstrate how experiment decisions can become unreliable under common testing practices.

## Executive summary

This synthetic audit illustrates three practical risks: repeated unadjusted significance checks can inflate false positives, experiments may not have enough traffic to detect business-relevant effects, and randomization diagnostics should be part of the readout process.

The simulation includes 200 experiments with known, simulator-defined ground truth. These findings are useful for illustrating failure modes, not for estimating the health of a real company's experimentation program.

## Findings from the configured scenarios

### 1. Repeated peeking can inflate false positives

In a separate simulated batch of known-null experiments, checking an ordinary p-value daily and stopping at the first p-value below 0.05 produced a 23.0% empirical false-positive rate. The fixed-horizon comparison produced 5.3%.

**Recommendation:** pre-register the analysis horizon and primary metric, or use a sequential-testing method with established guarantees and assumptions that match the experiment. Do not treat the experimental mixture-test prototype in this repository as a validated production method.

### 2. Traffic constrains what can be learned

At the configured baseline conversion rate of 8% and 16,800 users per arm over 14 days, the normal-approximation power calculation gives an MDE of about 10.6% relative lift. Detecting a 5% relative lift requires an estimated 62 days at the configured daily traffic.

**Recommendation:** calculate power and MDE before launch. Align the sample size with the smallest effect that would change the business decision. A non-significant result should not be described as proof of no effect.

### 3. Check randomization before interpreting outcomes

The simulated SRM diagnostic flagged 8.5% of the 200-experiment program. The simulated allocation bugs are deliberately detectable, so this result should not be interpreted as a real-world sensitivity benchmark.

**Recommendation:** include an SRM check using the intended allocation ratio and experiment eligibility rules as part of every experiment readout. Investigate a mismatch before trusting causal conclusions.

### 4. Multiple metrics change the error profile

Across five independent all-null metrics, the simulation observed a 25.1% probability of at least one unadjusted rejection and 5.2% after Benjamini–Hochberg correction.

**Recommendation:** define metric families and the decision objective in advance. Use a correction appropriate to the desired error control and dependence assumptions. FDR and FWER are not generally interchangeable.

### 5. CUPED can help when pre-period data are predictive

For a simulated continuous metric with pre/post correlation around 0.6, the adjusted outcome showed approximately 34.5% pooled variance reduction.

**Recommendation:** evaluate CUPED where pre-experiment covariates are predictive and measured before treatment. In real analyses, benchmark variance reduction on the treatment-effect estimator and validate that the covariate adjustment preserves the target estimand.

### 6. Novelty detection remains a weak point

The current time-slice diagnostic detected only 8.3% of the simulated novelty cases. This is evidence that the current diagnostic is underpowered in this scenario, not evidence that novelty is generally rare or impossible to detect.

**Recommendation:** treat time-varying effects as a diagnostic question requiring adequate sample size, pre-specified analysis, and suitable modeling.

## Suggested operating controls

1. Pre-register the primary metric, analysis horizon, minimum relevant effect, and stopping policy.
2. Require power/MDE calculations before launch.
3. Check SRM and instrumentation before interpreting experiment outcomes.
4. Define how multiple metrics and segments will be handled.
5. Use CUPED only when its covariates and assumptions are appropriate.
6. Validate sequential methods independently before production use.

## Limitations

All results are synthetic and depend on simulator assumptions, sample sizes, and random seeds. Empirical rates are finite-simulation estimates; this report does not include confidence intervals. The sequential mixture-test implementation is exploratory and does not currently establish formal always-valid inference. See [`docs/METHODOLOGY.md`](docs/METHODOLOGY.md) for detailed caveats.

**Bottom line:** the strongest lesson is process discipline. A dashboard can surface risk, but statistical validity depends on design, assumptions, calibration, and careful interpretation—not on a green badge alone.
