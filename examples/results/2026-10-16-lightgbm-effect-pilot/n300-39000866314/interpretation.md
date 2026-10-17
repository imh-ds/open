# N300 LightGBM: prediction improved, network recovery remained poor

[Completed run](https://github.com/imh-ds/open/actions/runs/39000866314).
[Durable reports and raw evidence](https://github.com/imh-ds/open/releases/tag/lightgbm-moderate-effects-2026-10-16-n300-39000866314).

All30 analyses completed and passed the workflow's independent no-fit case replay;
no unknowns. All34 jobs accounted for:33 successful,registration skipped. First
manual attempt at the approved source8108299ffc777e8ec584bc08fef475ed833f7247.
The published manifest matches the frozen N300 controls. Ten datasets per effect;
N300 adds100 participants to each matching N200 dataset.

| True correlation on both links | N200 direct link found /10 | N300 direct link found /10 | N300 complete chain /10 | N300 extra-finding networks /10 |
|---|---:|---:|---:|---:|
| .20 |0|0|0|0|
| .30 |1|1|0|0|
| .40 |0|1|0|1|

N300 found one additional true direct link across30 datasets. The same .30
repetition6 retained X2 -> Y at both sample sizes. At .40,repetition3 retained
X2 -> Y but also added X1 -> Y and the unrelated Z1 -> Y. X1 is related to Y
indirectly in the generator, but adds no population information about Y after
conditioning on X2; its direct promotion is an extra finding under this benchmark.
Both sample sizes recovered zero complete chains. The indirect target X2 was
reached only2/30 times at N300; in the .40 case,X1 was already assigned directly
to Y and therefore absent from the later candidate pool. The zero indirect
recovery count should not be described as30 independent direct tests of that link.

## Why the final graphs stayed sparse

| True r | Root assessments with full model worse than training-mean baseline | Root assessments passing the2% full-gain requirement | Median true direct-link selections /100 |
|---|---:|---:|---:|
| .20 |997/1000|1/1000|0|
| .30 |779/1000|149/1000|0.5|
| .40 |504/1000|355/1000|32.5|

These are dependent subsamples of ten datasets,not1,000 independent experiments.
The corresponding N200 full-gain pass counts were0,127,167,and median direct
selection counts0,4,11. Stronger signals/N300 generally improved held-out prediction,
especially at .40,but the changes did not yield consistent enough useful models
and links to pass the complete selection procedure. Median-of-dataset-median
full prediction gains relative to baseline changed from -23.2% to -14.3% at .20,
-9.7% to -7.9% at .30,and -4.8% to approximately -0.1% at .40. Negative means
higher held-out error than the training-mean baseline,not negative population R².

At .40 the true predictor used alone had a descriptive median gain of4.3%,while
the full-model median was approximately zero. This suggests that learning with
the additional covariates remains difficult under these settings; it does not
isolate a single cause or establish a benefit for a particular retuning change.
The final70% cutoff is not the only bottleneck. Raising it to90% cannot fix models
that fail the earlier prediction-gain requirement. Ten datasets are too few to
make a precise sample-size requirement or claim statistical significance for the
observed N200-to-N300 differences.

## Applied meaning and next step

The shipped LightGBM configuration did not reliably reconstruct these clean
linear .20–.40 chains at either N200 or N300. Adding100 participants helped some
prediction metrics but did not make the final networks useful in this pilot.
This is a limitation of this tested configuration/use case,not proof that all
LightGBM settings,larger samples,nonlinear problems or the recursive framework
must fail. The main error was missed relationships; almost-empty networks do not
provide useful reassurance from their low observed false-positive counts.

The already-approved matched Elastic Net/OLS N200 comparison is the appropriate
next diagnostic. Wait for its results before changing retention,adding more
features,or launching additional LightGBM experiments. No new model fits,tuning,
threshold changes,retries or Actions dispatches were performed in this review.

## Verification and reproduction

All ten release assets were downloaded and verified against GitHub SHA256 metadata,
including the438,873,043-byte raw bundle. The review checks all30 nested archive
hashes,sealed contents,input hashes and run/source/attempt identities,independently
reproduces summary counts and root prediction arithmetic,and verifies that each
N300 dataset's first200 rows exactly match the frozen paired N200 input hash.
The original complete checkpoint replay occurred in Actions; this review is not
claimed as an additional full checkpoint replay. Original reports remain unchanged.

`review-verification.json` records checks and per-case descriptions. From the
repository root,using the downloaded original release bundle:

```powershell
python examples/results/2026-10-16-lightgbm-effect-pilot/n300-39000866314/review.py --bundle <lgbm-n300-part01.zip>
```

The raw bundle remains in the durable prerelease and an ignored local review
directory. Small reports,API snapshots,verification and interpretation are kept
in Git with exact-byte preservation.
