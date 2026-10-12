# Pair validation and adaptive detail — development panel

User approved the recommendation after the completed pair-screen investigation.
Freeze before dispatch. No production interaction rule or cutoff is adopted.

240 independent RF tasks: six prior generating scenarios, four N,p points and
ten fresh replications per cell. Seeds start at 68/78/88 million (discovery/test/
splits), disjoint from previous studies. Reuse the prior analytic generator;
continuous IID data, correlated inputs rho=.7, oracle signal R2=.60. This is a
smaller development panel, not confirmation or a minimum-sample study.

For each of three outer splits: fit/tune RF on outer training data with its
unchanged shipped grid and three inner folds. Screen only training covariates,
using training outcome SD; shortlist the two highest supported coarse pair
scores, without marginal filtering. Assessment rows/outcomes do not select pairs.
Fit an additive cubic-spline/ridge comparator (all predictors) and a comparator
adding each selected pair's tensor spline basis, residualized against that pair's
training marginal spline columns. Tune each independently on training-only
three-fold CV: knots3/5, ridge alpha .1/1/10/100; imputation, knots, projection and
scaling are fitted inside each partition. One additive fit is shared per outer
fold. Cap100validation fits per call; default cost25baseline plus25per pair.

Evaluate on untouched outer assessments. Gain is (additive MSE-joint MSE) divided
by training-mean prediction MSE on assessment data. It is a predictive skill
improvement, not a coefficient, p-value or certified interaction. Matched risk
arithmetic, selected settings and raw predictions are retained. Missing/unselected
folds are unknown; a pair gets a complete pooled gain only when assessed in all
three planned folds. Conditional gains and fold counts remain separately visible.
This deliberately conservative complete-pair endpoint must not hide lost coverage.

After train-only shortlist locking, explain each pair using the fitted RF and
assessment covariates with opt-in adaptive details. Prefer a complete4/3/2bin
grid, otherwise mask unsupported two-bin cells. Heldout outcomes never choose
the grid. Fixed/coarse versus adaptive availability, realized bins, covered rows,
support counts and prediction-query budgets are saved. Availability is not
statistical detection. Correlated-cell corners can still leave observed support.
Fixed and adaptive assessment details each have a separate 100,000 prediction-row
cap shared across the two locked pairs; comparison predictions are counted separately.

Final full-discovery RF predicts1000independent test observations. Select its
shortlist using discovery data only, fit/tune the spline comparators using only
discovery data and evaluate the locked pair comparisons on the independent test.
This provides an additional diagnostic reference, not calibration for CV rules.

Primary summaries: true selected/assessed fold coverage, complete versus
conditional gains, incidental gains, separate/joint heldout predictive skill,
fixed versus adaptive plot support and actual additional fit/runtime cost.
Prespecified gain cutoffs0/.01/.02/.05 are sensitivity summaries only. Report
complete and conditional pair endpoints, missing/unknown counts and all ten
planned independent datasets per cell. No favorable cutoff selection, automatic
retention, p-value or broad scientific acceptance. A searched rule requires a
separately frozen disjoint confirmation. Splines can underfit other relationships;
validation is model-family dependent and may fail on higher-order interactions.
Incidental summaries cover shortlisted pairs only; complete/conditional assessed
pair counts and unselected unknown counts accompany them. They are not false-positive
rates for all possible pairs. Fitting/tuning and saved-setting verification costs
are reported separately, with fitting and total task elapsed time retained.

120 two-task shards/two single-threaded workers, maxparallel256 subject to shared
account scheduling;30minute job with4minute setup/22minute fit step,15minute
prepare/60aggregate. Maximum allocation3675runner minutes, no paid runners or
automatic retries. Dedicated setup verifies actual frozen versions. Receipts
precede fits, atomic fold checkpoints and failure artifacts retained. Workers
refit saved RF/comparator settings (not retune winners), checking raw predictions
and screen/detail reproduction. Verification stamps bind evidence digests.
Aggregate separately downloaded artifacts with duplicate conflict checks and
retain incomplete/error identities in planned denominators. Preserve raw evidence,
audit, hashes, source/run identity, reports and graphs in the repository after
completion; artifacts retain30days. Old frozen protocols/manifests stay unchanged.
