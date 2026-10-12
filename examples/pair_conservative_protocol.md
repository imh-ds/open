# Fresh conservative joint-selection development protocol

The D-037 saved small-N diagnosis motivates one bounded research change.
All production defaults and earlier frozen studies remain unchanged. No
automatic interaction rule, confirmation claim or minimum-N requirement.

## Selection policy

Compare production v2, minimum-CV joint weighting, and a one-SE-style joint
penalty preference. Keep the original 24 settings: knots3/5, alpha.1/1/10/100,
joint block weight1/10/100. Marginal scaling, natural tensor columns, training
projection and LSQR tolerance1e-8 are unchanged. Share the additive baseline.

For each pair, compute the mean of its three inner-fold MSEs for each setting.
Let b be the minimum-mean candidate. The band is mean(b) plus
sampleSD(its three losses)/sqrt3. Among candidates inside that band, choose
largest alpha/weight squared, then fewer knots, then larger alpha, then smaller
weight. This prespecified penalty preference does not guarantee lowest total
effective degrees of freedom across spline bases. Save every candidate's
three fold losses, means, band, eligible identities and selected indices.
Only training observations enter this calculation.

The [Stanford course notes](https://web.stanford.edu/class/stats202/notes/Resampling/Kfold-CV.html)
describe one-SE parsimony. Our extension to joint-block penalties is a
development hypothesis. The fold-SE is a heuristic, not a valid confidence
interval, significance test or interaction certificate. CV folds share data;
see [Bates, Hastie and Tibshirani](https://arxiv.org/abs/2104.00673).

The exact minimum-CV competitor comes from the same search. Its 73 fits include
72 inner fits and one final refit; the conservative setting adds one final
refit even if settings coincide. Shared production costs25baseline+2*25pairs.
Total223 fitting/tuning fits per comparison, cap224. Four comparisons per
dataset cost892fits,214080for240datasets. Saved-setting verification adds
28refits per dataset,6720total. It checks predictions/settings, not re-tuning.
No clipping, omission of difficult rows, new search settings or basis changes.

## Frozen fresh design

240 new datasets: rho0/.7/.9; noise/additive/pure/mixed signals; N,p=(80,5),
(300,5),(600,5),(300,15); five replications per48cells. Fresh discovery/test/
split seeds178/188/198million, IDs prefixed one-se--. Three outer folds,
three inner folds and1000independent assessment observations. Fixed x1:x2 and
x1:x3 pairs avoid selection-availability denominators. All data generation
and Gaussian population references reuse D-036's unchanged generator.
The previous240datasets and diagnosed case are development inputs, never
fresh validation cases for this policy. Native preflight exercises two of
these new task identities, not additional independent replications.

Gain is (assessment additive MSE-joint MSE)/assessment training-mean MSE.
Pool CV risks with assessment counts, rather than averaging fold ratios.
Report raw predictions/settings, independent gains, Gaussian target/pair
references, maximum prediction in training outcome SD, worst error share,
all candidate selection traces, cost/warnings and failures. Descriptive
cutoffs0/.01/.02/.05 remain fixed; no cutoff is accepted. All missing/error
identities remain in planned denominators. The fixed control pair does not
calibrate all-pair FPR or validate RF/GBM/network recovery.

Prespecified interpretation: assess whether restrained selection reduces
negative-gain tails at N80 and high correlation, and whether it sacrifices
pure/mixed independent gains where joint information is substantial. An
improvement in the diagnosed case alone cannot qualify the selector. Five
replications per cell support development comparison, not precision sample
size guidance. Do not change the ordering or band after seeing fresh results;
failed candidates remain evidence against adoption.

## Execution and evidence

Dedicated source/runtime/task manifest, runner, setup and workflow. Hash all
production source plus imported frozen study helpers and this protocol/setup/
workflow before dispatch. Workers re-fit saved settings and digest-bind raw
results, predictions, receipts and timing. Offline audit regenerates data,
checks risk arithmetic, complete CV grids and selection-band arithmetic.
Conflicting artifact bytes reject; checkpoints are validated before reuse.
Checkpoint digests are bound to complete-task verification. Top-level costs
cover verified complete tasks; overlapping checkpoint totals separately retain
completed work from partial tasks. Unfinished fits have unknown cost, so saved
costs are lower bounds. Prior errors remain historical after successful reuse.

120two-task shards, two single-thread workers each, maxparallel256 subject
to account limits. Ordinary windows-2022 runners; no paid runners or automatic
retry. Job cap30minutes, setup4/study22, prepare15/aggregate60:3675maximum
runner minutes. Native preflight and independent review precede dispatch.
Retain downloaded raw evidence alongside findings after completion.
Production adoption, a calibrated interaction rule and disjoint confirmation
require a later decision; this dispatch is fresh development only.
