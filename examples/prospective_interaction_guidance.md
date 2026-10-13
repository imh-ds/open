# When to trust an interaction finding

The tested spline procedure can find interactions and show useful relative patterns. It often misses weak interactions and interactions between strongly overlapping predictors. **A missing interaction in the graph is not evidence that the interaction is absent.**

This guidance describes the [frozen prospective study](prospective_interaction_protocol.md) and its [condition-by-condition results](results/2026-10-13-prospective-interaction-signal/REPORT.md). It is research evidence for one procedure, not a general guarantee for an individual study or every ML engine.

## Three questions to keep separate

1. **Did this dataset produce a finding?** The selected pair improved held-out prediction beyond the additive comparison and passed the frozen selection rule. This is a predictive finding; it does not establish causation.
2. **Does the method find this kind of interaction reliably?** In simulations, the known interaction must be detected often enough while false findings remain sufficiently uncommon. A finding in one dataset does not answer this question.
3. **Can we interpret its plot?** Displayed curves need common data support and consistent relative patterns across folds. Simulation evidence additionally checks whether those relative patterns resemble the known truth. This is conditional on the plot being displayed; it does not validate the numerical magnitude or every point on a surface.

## Labels and their meaning

| Description | What it means | What it does not establish |
|---|---|---|
| Exploratory interaction finding | A dataset passes the predictive selection rule; broader reliability is not established for its conditions. | A validated probability that this individual finding is true. |
| Benchmark-supported detection | A specific simulated condition passed both null tests, the unrelated-pair control test and the detection test. | That any real dataset with the same sample size will behave similarly. |
| Benchmark-supported relative pattern | That simulated condition also passed the conditional pattern test. | Accurate effect-size magnitude, a causal interaction or a fully accurate surface. |
| Unresolved or insufficient evidence | Support is missing, incomplete or fails the prespecified requirement. | Proof that no interaction exists. |

These describe evidence levels for interpretation. They do not add production API fields, automatically relabel existing results, or convert descriptive fold consistency into statistical confidence.

## What the study found

All 10,800 simulated datasets completed. Each of the 24 combinations used **450 separate datasets**, with 300 or 600 people **in each dataset**. Four combinations passed all requirements; six passed detection alone. Successful detection could still fail the unrelated-pair control requirement.

For mixed interactions with the same distinct contribution of 4% of outcome variance:

| People per dataset | Predictors | Correlation between the interaction predictors | Interactions detected | Combined requirements passed? |
|---:|---:|---:|---:|---|
| 300 | 5 | 0 | 338/450 (75.1%) | No |
| 300 | 5 | 0.7 | 227/450 (50.4%) | No |
| 600 | 5 | 0 | 436/450 (96.9%) | Yes |
| 600 | 5 | 0.7 | 348/450 (77.3%) | No |
| 300 | 15 | 0.7 | 246/450 (54.7%) | No |

More data helped in this comparison, but 600 people was not sufficient in every tested setting. The 15-predictor result does not demonstrate that adding predictors caused lower detection: its rate was similar to the five-predictor counterpart, and the complete benchmark grid did not isolate a universal predictor-count rule.

The four combinations that passed detection, false-finding and relative-pattern requirements were:

| People | Predictors | Correlation | Simulated interaction | Detected / 450 | Displayed / 450 | Relative patterns matching truth / displayed |
|---:|---:|---:|---|---:|---:|---:|
| 300 | 5 | 0 | Mixed, coefficient 0.8 | 447 | 425 | 423/425 (99.5%) |
| 600 | 5 | 0 | Mixed, distinct contribution 4% | 436 | 399 | 388/399 (97.2%) |
| 600 | 5 | 0 | Mixed, coefficient 0.8 | 450 | 447 | 447/447 (100%) |
| 600 | 5 | 0.7 | Pure, coefficient 0.8 | 438 | 433 | 432/433 (99.8%) |

“Mixed” includes additive effects plus an interaction. “Pure” contains the interaction without separate additive effects. Fixed-coefficient benchmarks and fixed-distinct-information benchmarks answer different questions: the same coefficient does not imply the same standardized distinct strength when predictors overlap. Do not rank their results as if they were equal-strength comparisons.

At correlation 0.9 the study used smaller feasible distinct-information targets (0.2% and 0.5%, rather than 1% and 4%) and found only 1–6 interactions per 450 datasets across the four tested benchmarks. These results show poor detection for those specific combinations, but they cannot separate correlation from signal strength. They do not establish a universal correlation cutoff.

## Why the training samples are smaller than the dataset

Three outer folds assess predictions on people who were not used to fit that fold's model. Three inner folds tune the model inside its outer training sample.

| People in the dataset | Outer assessment sample | Outer training sample | Smallest/largest inner tuning training sample | Full-data fit used for the separate diagnostic |
|---:|---:|---:|---:|---:|
| 300 | 100 | 200 | 133–134 | 300, with inner training samples of 200 |
| 600 | 200 | 400 | 266–267 | 600, with inner training samples of 400 |

These are verified sizes for this study's three-by-three CV design. They are not missing participants. The separate 1,000-row independent diagnostic is synthetic assessment data, not extra training people or part of selection. Different fold counts and sample sizes have not been validated by this benchmark.

## Practical rules for researchers

- Treat interaction findings as exploratory when your engine, data type, screening strategy or study conditions have not been validated.
- Report sample size, predictor count, predictor overlap, folds, the smallest tuning training sample, and the selection settings. Nominal sample size alone does not describe the fitted models.
- Expect weak interactions or highly similar predictors to be harder for this tested procedure to detect. Do not interpret a sparse graph as a test demonstrating no interactions.
- Report both the number of interactions detected and the number receiving a displayed curve. Report pattern agreement among displayed curves separately; high agreement among a small subset does not mean most interactions were recovered.
- Explain plots as relative predictive patterns. Keep numerical effect-size accuracy, individual-result confidence and causality separate.
- Do not infer a universal minimum sample size from these results, interpolate untested cells into validated conditions, or use the known-truth simulation signal strength as an automatic classifier for real data.

## What remains outside this evidence

The study used Gaussian data, one frozen weighted spline comparison, two prespecified pairs and fixed thresholds. It does not validate RF, boosting, LightGBM, arbitrary pair screening, all pairs in a large network, ordinal measures, causal moderation or the complete recursive workflow. A general production confidence feature would need a separate validation decision.

The selection rule, thresholds and 36/24/24 multiplicity allocations were frozen before these results. A failed requirement is not rescued by pooling conditions, changing denominators or changing a cutoff afterward. Original results, reproducible independent checks and raw preservation receipts accompany the report.
