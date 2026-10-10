# Interaction-selection investigation

Fixed before execution, 2026-10-10. This is a development diagnosis, not a
calibration study or an attempt to improve the previous held-out score.

Keep GBM, its shipped tuning grid, base seed 613 (613 + original outer-fold
index for exact replay), 80% subsampling, two tuning folds,
all relevance thresholds, and the 70% stability rule unchanged. Investigate
only the root comparison `y ~ a + b`.

1. Read only the original outer-training rows. Replay all nine original root
   assessments with two local performance folds; verify original risks and
   promotions. Extend to ten assessments per training partition.
2. On the same training rows and identical sampled positions/seeds, use five
   local performance folds instead of two. Record model-training sizes,
   full-model gain, broad/conditional contributions, and classifications.
3. For both settings, summarize the first three and all ten assessments at
   the unchanged 70% cutoff. Prefix comparisons share data and are not
   independent replications.
4. Run fresh independent Gaussian interaction datasets, `y = 2*a*b + error`,
   error SD .3, at n=54 and n=216, seeds 10401 and 10402. Use five local folds
   and ten assessments. Include one independent-noise check n=216, seed10403.
   The smaller dataset is a prefix in predictor values of the larger dataset
   for its seed; outcome noise is generated after predictors, so outcomes
   are not a nested sample. No paired sample-size inference is claimed.

Planned total: 110 local assessments, including nine exact replays. Any failure
stops execution and must be documented; never omit it from a denominator.
No new outer-test score is computed. Save all local evidence, sample/split
identities, configurations, input data, and a source fingerprint. Do not pick
a threshold, engine, or tuning grid from these results. If changing production
behavior is justified, specify a separate validation protocol with null,
redundancy, additive and interaction cases, multiple sample sizes and seeds.

Questions: Is there an arithmetic/aggregation defect? Does broad screening
block interaction inputs? Is low full-model gain the main loss mechanism?
Does increasing repetitions alone help? Does increasing per-fit training data
help? Are any isolated one-variable retentions left after consensus?
