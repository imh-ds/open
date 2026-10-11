# Sample-size guidance feasibility pilot

Version 1, 2026-10-10. This is a computational/scoring pilot, not a minimum-N
validation study. The preceding 48-case study remains unchanged. No production
selection threshold, engine grid or recursive rule changes here.

## Frozen proposed scope

Twelve independently generated datasets: four scenarios crossed with three
design points (N80/p5, N300/p5, N300/p15). p counts all candidate predictors,
excluding the root y. One dataset per cell cannot estimate reliability.
Scenario order: noise, additive, interaction, chain; design-point order as above.
Case index i=0..11; training seed=410000+i*1000, independent test seed=7410000+i*1000,
discovery seed=610000+i*1000. Test N=2000. No seed selection or exclusions.

All candidates initially independent standard normal x1..xp, except the chain:

- Noise: y is independent standard normal. Oracle conditional mean zero.
- Additive: y=x1+e; oracle conditional mean x1.
- Interaction: y=x1*x2+e; oracle conditional mean x1*x2.
- Chain: x3 is standard normal; x2=.6*x3+.8*z2;
  x1=.6*x2+.8*z1; y=x1+e, with independent normal z1,z2.
  All other candidates are independent noise. Oracle root mean x1.

In the three signal scenarios, e is independent normal with variance 2/3.
Signal variance is one, so population oracle root R-squared is .60.
Chain local x2->x1 and x3->x2 R-squared is .36, not .60.
This pilot covers clean continuous IID data only, moderate chain correlation,
one signal strength, one engine and one fold combination.

GBM with shipped fixed grid; local performance CV5 folds/2 repeats, inner tuning
3 folds, subsampling80%, B100, retention70%, other thresholds unchanged.
Root-only scenarios maximum depth1; chain maximum depth3. The actual production
layer-consensus function runs, including freezing each layer before recursion.
An orchestration adapter checkpoints each production local assessment; singleton
calls restore original sample IDs before production aggregation. No alternate
selection implementation. Independent test data never enter discovery or tuning.

## Scoring definitions

Expected root inputs: none for noise; x1 for additive/chain; x1 and x2 for
interaction. Expected chain predictive edges x1->y, x2->x1, x3->x2 and depths
y0,x1=1,x2=2,x3=3. This is the declared predictive traversal under these
candidate restrictions, not a causal discovery claim or a universal DAG metric.
Track retained relation kinds/shared groups separately from member-level edges.

For a complete network record required-root recovery, false root inputs,
required predictive-edge recovery, extra structural edges, complete expected
path recovery and depth matches. Full-pipeline misses remain misses even when
an upstream nonselection prevents downstream evaluation. Report attempted
target comparisons and their actual candidate counts separately.
For incomplete networks withhold structural scores and test refits; preserve
status, failure reason and checkpoints. Complete but empty networks are valid
results, not execution failures. Budget stops are incomplete, not negative links.

Train-only three-fold tuning/refitting compares all-input and retained-root-input
models on identical independent test rows; no retained inputs uses training mean.
Save observed y, training-mean baseline, oracle conditional mean, both predictions,
signed baseline-relative skills, oracle skill and full-minus-selected skill loss.
Prediction does not certify recursive recovery. No pilot pass/fail N threshold.

Record every assessment's sampled rows/splits, warnings, stop reason, promotions,
gain, fit/tuning counts and actual fit/tuning/validation row sizes by depth/target.
Retain full serialized production network and per-assessment objects for audit.
Record Python/dependency/platform versions, commit, source/protocol/config hashes,
elapsed time, orchestration workers and GitHub run identity. Record OS worker
process-lifetime peak working-set memory, including imports; this is not the
simultaneous total job-memory peak. Source hashes normalize
CRLF to LF for text reproducibility; exact checkpoint/data fingerprints also apply.

## Proposed bounded Actions dispatch

Windows2022, Python3.11.9 and dependencies matching the earlier Actions study.
Twelve case jobs, maximum three concurrent, four single-threaded assessment
processes per case. A case may start work for at most90minutes; check the deadline
before each new assessment batch and test refits. A running batch can overrun that
soft budget. Hard case-job timeout110minutes includes installation/artifact work;
if killed, available checkpoints survive the always-run artifact step where possible.
Preparation15minutes; combined audit/report30minutes. Maximum allocated budget
12*110+15+30=1365 runner minutes (22.75 runner hours), excluding optional reruns.
No automatic reruns, case expansion or fallback settings. Dispatch approval applies
only to this manifest and cap; further study needs a separately frozen budget.

Completed cases and partial checkpoints are always collected. The combined audit
must report all12 planned identities, including absent/timed-out cases; it cannot
silently exclude them. Audit structural aggregation and prediction arithmetic,
and independently refit the final full/retained models with grid-valid saved
parameters to verify test predictions. Do not retune on test data or claim an
independent refit of every local model. Validate saved incomplete networks too.

## Pilot disposition

Use observed runtime/memory/fit counts to propose affordable development and
confirmation panels. Report one-case results descriptively, including misses.
If p15/recursive jobs cannot finish within the cap, preserve incomplete evidence
and redesign the compute plan; do not lower B100 silently. No minimum sample size,
fold superiority, false-positive calibration or scientific P6 acceptance follows.

Planning/reporting follows
[Morris, White & Crowther (2019)](https://pmc.ncbi.nlm.nih.gov/articles/PMC6492164/).
Nested evaluation follows the
[scikit-learn explanation](https://scikit-learn.org/stable/auto_examples/model_selection/plot_nested_cross_validation_iris.html).
