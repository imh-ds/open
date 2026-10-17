# N=200 interpretation: weak recovery under the shipped LightGBM settings

[Completed run](https://github.com/imh-ds/open/actions/runs/39000850264).
[Durable original reports and raw evidence](https://github.com/imh-ds/open/releases/tag/lightgbm-moderate-effects-2026-10-16-n200-39000850264).

All30 cases completed and passed the workflow's independent no-fit replay.
There are no missing/unknown cases. All34 jobs are accounted for:33 successful,
registration skipped. The source is8108299ffc777e8ec584bc08fef475ed833f7247,
first manual attempt, and the live manifest equals the approved frozen N200 manifest.

| Population correlation on each true link | Direct X2 -> Y found /10 | Indirect X1 -> X2 found /10 | Indirect target tested /10 | Exact network /10 | Extra-finding networks /10 |
|---|---:|---:|---:|---:|---:|
| .20 |0|0|0|0|0|
| .30 |1|0|1|0|0|
| .40 |0|0|0|0|0|

LightGBM rarely recovered even the first link, so recursion almost never reached
the next target. The indirect zeros mostly describe an unvisited stage, rather
than30 independent unsuccessful tests of the indirect relationship. The result
is predominantly false negatives/empty networks, not excessive false positives.
No false findings in nearly empty networks does not establish useful error control.

## Where detection was lost

The review reader verifies all30 raw archive hashes, their sealed contents,
frozen input hashes and producer identities, then reads all3,000 root assessments
without fitting a model. The existing full auditor already replayed the complete
checkpoints in Actions; this supplement independently verifies publication,
counts and root prediction arithmetic, rather than claiming another complete
checkpoint reconstruction.

| True r | Root assessments where full model was worse than mean baseline | Root assessments passing the 2% full-gain gate | Median direct-link selections /100 |
|---|---:|---:|---:|
| .20 |1000/1000|0/1000|0|
| .30 |837/1000|127/1000|4|
| .40 |751/1000|167/1000|11|

These are overlapping subsamples of ten independent datasets per row, not1,000
independent datasets. The baseline uses training-fold means. Passing the gain
gate only permits further selection; it does not by itself select a link.

The median across datasets of each dataset's median baseline-relative full gain
was -23.2%,-9.7%,-4.8% at r=.20,.30,.40. Negative means higher held-out prediction
error than the baseline, not negative population explained variance. The same
descriptive calculation for the true predictor X2 used alone was -10.2%,-0.8%,
+0.2%. Thus the difficulty is not confined to competition for importance among
predictors. These saved-model results cannot diagnose a unique cause, demonstrate
that no alternative tuning works, or show the relationship is undetectable by
another learner. They also do not directly measure training/test overfitting.

Raising retention from70% to90% cannot repair poor prediction before selection.
The r=.30 condition's one selected link versus none at .40 is not evidence that
stronger effects are harder to detect: ten datasets yield coarse counts, while
the descriptive prediction/selection metrics improve with effect size overall.

## Implication and next step

The shipped LightGBM configuration did not provide useful recursive recovery for
this small, clean, linear N200 example at r=.20–.40. This is a meaningful
limitation of this configuration and intended use. It is not a verdict on every
LightGBM configuration, nonlinear relationships, larger samples, Elastic Net/OLS,
or the logical feasibility of recursive predictive networks.

Leave the approved N300 wave running unchanged. Then propose a matched Elastic
Net comparison and an OLS reference on these same frozen datasets. This isolates
learner limitations from the selection/recursion policy without changing many
things at once. OLS is a proposed reference; no OLS engine was added here. No new
fits, retries, tuning, threshold changes or Actions dispatches occurred during
this review. Findings remain exploratory with ten datasets per condition.

`review-verification.json` contains the independent checks, per-case root
descriptions and warning-message counts. Reproduce from the repository root with:

```powershell
python examples/results/2026-10-16-lightgbm-effect-pilot/n200-39000850264/review.py --bundle <downloaded-lgbm-n200-part01.zip>
```

All ten original published assets were downloaded and checked against GitHub
release SHA256 metadata,
including the266,653,279-byte raw bundle with30 cases. Original publication
verification is retained; no original results were rewritten. The raw
evidence stays in the durable prerelease and an ignored local review directory;
ordinary Git retains the small reports, controls and review evidence.
