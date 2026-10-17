# Matched linear references: recursive recovery improves substantially

[Completed run](https://github.com/imh-ds/open/actions/runs/39004942944).
[Durable reports and raw evidence](https://github.com/imh-ds/open/releases/tag/linear-moderate-effects-2026-10-16-n200-39004942944).

All60 analyses completed and passed the existing independent no-fit checkpoint
audit in Actions. No missing/unknown cases. All64 jobs are accounted for:
63 successful,registration skipped. First manual attempt at approved public
source2e0da27ec17745af37085b02f2627df7756507e4. The released manifest equals the
frozen launch manifest. There are30 independent N200 datasets paired across
engines,not60 independent datasets. All inputs and nonlearner settings match
the original LightGBM N200 pilot;retention remains70%.

| Engine | True population r on both links | Direct X2 -> Y /10 | Indirect X1 -> X2 /10 | Indirect target tested /10 | Exact network and depths /10 | Extra-finding networks /10 |
|---|---|---:|---:|---:|---:|---:|
| Elastic Net |.20|1|0|1|0|0|
| OLS |.20|1|0|1|0|0|
| Elastic Net |.30|9|6|9|5|1|
| OLS |.30|9|6|9|5|1|
| Elastic Net |.40|10|9|10|8|1|
| OLS |.40|10|10|10|9|1|

For comparison,LightGBM recovered zero complete networks at all three effect
strengths at both N200 and N300. Its N200 direct recovery was0/1/0 per10.
The same recursive framework with linear learners can therefore recover useful
structure in these linear examples,including genuine indirect paths. The earlier
LightGBM failure is not evidence that recursive execution is fundamentally broken.
This comparison supports learner/settings suitability as a major explanation,
not a universal superiority claim or evidence about nonlinear data.

## Simple interpretation

- At r=.20,both references usually missed the first link and consequently did
  not reach the indirect stage. Neither reconstructed a full chain. This remains
  a major limitation at N200 under these selection rules.
- At r=.30,both found the direct link9/10 times,but the indirect link6/10.
  Five networks were exactly correct. The depth expansion works but loses
  sensitivity before producing a consistently correct complete network.
- At r=.40,both found every direct link;OLS found every true indirect link and
  Elastic Net found9/10. OLS's9 exact networks versus Elastic Net's8 is only a
  one-dataset difference in this small paired pilot,not a demonstrated general
  performance advantage.
- Both engines added Z1 -> X2 in effect.30 repetition1 and Z2 -> Y in effect.40
  repetition10. Those same false links occurred on the same paired datasets.
  Both true links were also present in those cases. Recovering all true links
  therefore does not guarantee an exactly correct network or controlled errors.

OLS retained sparse networks without coefficient shrinkage or p-value selection.
The external held-out gain,unique/shared relevance and layer-retention rules
perform the network pruning. Elastic Net supplied additional within-fit
regularization but gave little extra network recovery in this particular pilot.
That does not rule out benefits with other data dimensions,correlation patterns,
sample sizes or effect functions.

Recorded median analysis time across30 cases was20.3 minutes for Elastic Net and
2.5 minutes for OLS,excluding some job setup/packing costs. OLS used489,000 outer
fits and zero tuning fits;Elastic Net used484,000 outer fits and20,775,000 tuning
fits. Different recovered depths can change outer-fit counts. The computation
difference is expected because OLS has no search while Elastic Net tunes25
settings;this is not an equal-compute contest or a timing guarantee.

## Prediction before retention

| Engine | True r | Root assessments passing2% full-gain gate /1000 | Median true direct selections /100 |
|---|---:|---:|---:|
| Elastic Net |.20|217|10|
| OLS |.20|240|12|
| Elastic Net |.30|888|98|
| OLS |.30|912|98.5|
| Elastic Net |.40|994|100|
| OLS |.40|995|100|

These are overlapping subsamples of ten datasets per cell,not1,000 independent
datasets. At .20 the median across datasets of within-dataset median full gain
was approximately zero for Elastic Net and -0.7% for OLS. At .30 those descriptive
values were+8.8%/+8.0%,and at .40+11.5%/+11.2%. Negative means worse held-out
prediction than a training-mean baseline,not negative population R². These numbers
describe the fitted procedures and do not establish an exact failure mechanism.

## Applied implication and recommendation

The linear recursive framework shows promising recovery,particularly at r=.40,
in this N200,small Gaussian linear network. Exact-network recovery at .30 remains
inconsistent,despite good direct-link recovery.
It is not yet a general reliability claim or release qualification. Ten paired
datasets per effect give coarse estimates;false findings remain,and r=.20
recovery is poor. The simulations do not establish causal direction,performance
on nonlinear functions/interactions,larger variable sets,measurement error or
universal sample-size requirements.

Recommended next development priority:broaden OLS validation with more independent
datasets and sample sizes,explicitly checking r=.20 sensitivity and false-link
control,with Elastic Net as a reference. OLS is attractive here because recovery
is similar and computation is far lower. Keep current settings and original
results unchanged until a prospective protocol and compute allocation are
approved. Do not simply lower retention to seek more discoveries,or conclude the
ML engines have no use before testing their intended nonlinear settings.

## Evidence checks and reproduction

All nine published assets were downloaded and checked against GitHub SHA256
metadata,including the965,163,585-byte raw bundle. The no-fit reader verifies all
60 nested archives,sealed contents,input hashes,configurations and run/source/
attempt identities. It independently recomputes exact/extra structure and the
reported cell counts,and checks6,000 root-assessment gain calculations. A saved
full-model checkpoint per case verifies the actual engine label,hyperparameter
schema and correct25-setting/zero-setting tuning accounting. The original full
checkpoint replay occurred in Actions;this supplement is not another complete
loss-by-loss checkpoint replay. No new scientific model fitting or dispatch
occurred in this review. Original published reports are unchanged.

`review-verification.json` contains per-case descriptions,root warning counts and
verification scope. With the original release bundle,from the repository root:

```powershell
python examples/results/2026-10-16-linear-effect-pilot/n200-39004942944/review.py --bundle <linear-n200-part01.zip>
```

Small reports,provenance and interpretation are committed with exact-byte
preservation. Raw evidence remains in the durable prerelease and an ignored
local review directory.
