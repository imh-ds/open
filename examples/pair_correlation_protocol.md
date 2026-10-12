# Joint shrinkage and correlated inputs: fresh development protocol

This research study follows the v2 numerical repair. It does not alter the
recursive engine, adopt a comparator, or calibrate an interaction decision.
Earlier frozen studies and their denominators remain unchanged.

## Frozen design

240 independent datasets: input correlation 0, .7, .9; noise, additive, pure
product and mixed signals; (N,p)=(80,5),(300,5),(600,5),(300,15); five replications
per cell. Fresh discovery/test/split seeds start at 148/158/168 million. Inputs
are standard Gaussian, with only x1/x2 correlated. Signal R-squared is .60;
noise-only outcomes have unit noise SD. Exactly two fixed pairs (x1,x2) and
(x1,x3) are assessed. This isolates the comparator, without tree shortlisting.
It does not establish recovery of a whole network or all-pair false-positive rates.

Each dataset has three outer folds, three training-only inner folds, and one
separate 1,000-observation assessment after full-discovery fitting. All spline
knots, scaling, projection, penalty and weight selection use training data only.
Baseline marginal curves are shared by production v2 and the research comparison.
Production v2 keeps joint tensor columns on their natural scale. The research
version multiplies the entire joint block by one factor selected from 1,10,100.
It never divides individual joint columns by their observed SD. With ridge this
changes the relative penalty on joint directions to alpha/weight squared.
Both use knots 3/5 and alpha .1/1/10/100, LSQR tolerance 1e-8. Research inner CV
chooses among 24 settings, costing 73 fits per pair; production costs 25 baseline
and 25 per pair. Total 221 fits per comparison, cap 224; 884 per dataset,
212,160 fitting/tuning fits for the planned study, plus 20 saved-setting refits
per dataset. These refits verify saved predictions, not winner selection anew.

## Known Gaussian reference (derived for these simulations)

Write rho for correlation and Z=x1*x2-rho. The best population additive
approximation of Z is a*(x1 squared+x2 squared-2), a=rho/(1+rho squared).
Indeed E[Z|x1]=rho*(x1 squared-1) and
E[x2 squared-1|x1]=rho squared*(x1 squared-1), so the residual has zero
conditional expectation given either input, hence is orthogonal to all
square-integrable univariate functions. Its variance is
(1-rho squared) squared/(1+rho squared).

Signals are: noise 0; additive x1+.6*(x2 squared-1); pure Z;
mixed x1+.6*x2+.8*Z. Their variances are respectively 0, 1.72,
1+rho squared, and 1.36+1.2*rho+.64*(1+rho squared).
Noise variance is signal variance*2/3 (unit variance for noise-only).
The expected population incremental skill gap is product coefficient squared
times residual variance divided by total outcome variance. It is zero for
noise/additive signals. For mixed rho=.7 it is about .02126; for rho=.9 about
.00214. Closely related inputs allow separate curves to capture much of the
product signal. Dependent-input decompositions generally require care; see
[Chastaing, Gamboa and Prieur](https://arxiv.org/abs/1112.1788).

This reference is specific to this Gaussian construction. It is neither a
universal ceiling for researchers' data nor a strict bound on finite fitted
gains: baseline underfit and assessment noise can produce larger gains. We
retain raw oracle predictions and empirical risks to expose that distinction.

## Evidence, interpretation and bounded execution

Gain = (assessment baseline MSE - joint MSE)/assessment training-mean MSE.
Pool fold numerators and denominators, never average fold ratios. Report
independent-test gain, both oracle MSEs, expected population gap, maximum
absolute prediction in training outcome SD, and worst-observation error share.
Gain cutoffs 0,.01,.02,.05 are descriptive sensitivities, not accepted decisions.
Every cell retains its planned five datasets and all missing/error identities;
incomplete evidence is unknown, never a negative interaction finding.

Source hashes, pinned runtime and task identities are frozen before dispatch.
Workers save raw predictions, settings, checkpoints, receipts, warnings and
timing; re-fit saved settings and digest-bind the verification to the evidence.
Aggregation independently checks arithmetic and regenerates the simulated data.
Conflicting artifact bytes are rejected. A two-case local preflight checks
construction; it is not another independent study or a calibration claim.

120 two-task shards with two single-thread workers, max-parallel 256 subject
to account limits, ordinary windows-2022 runners. Each worker job has a
30-minute cap (setup 4, study step 22), preparation 15 and aggregation 60:
at most 3,675 runner minutes. No paid runners or automatic retry. Dedicated
workflow and manifest leave all older frozen protocols/workflows untouched.
Preserve full downloaded evidence in the repo after completion. Final method
adoption requires completed evidence and a separate decision.
