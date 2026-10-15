# Recursive tree-engine timing preflight launched

[GitHub Actions run 38775613523](https://github.com/imh-ds/open/actions/runs/38775613523)
was dispatched once from ffe85ca57112530871a8ab9e9db7ec5ca8fdf9b2.
Preparation passed the frozen-source and installed-runtime checks. Its saved
authorization matches exactly six cases: Random Forest, GBM and LightGBM, each
at N=80 and N=200. All six case jobs were created on the first attempt.

This measures runtime and checks recursive execution before a larger reliability
study. It cannot establish reliability from one fixture per engine/sample size.
Each case uses shipped defaults, including 100 resamples and repeated 5-fold CV.
The jobs can run in parallel. Reserved compute is 586 runner-minutes combined,
within the owner's new 600-minute ceiling; each case job has a 90-minute limit.
No retries or larger study are authorized automatically.

Software verification: 38 focused tests and 29 inherited checks passed.
The versioned auditor exactly reproduced a preserved ElasticNet case's full audit
row: 300 assessments, 307,814 observation checks, no refits. Independent review
caught and resolved interrupted-run handling defects; regression tests cover them.

After completion, preserve and independently audit all six raw case archives,
including incomplete attempts. Report runtime, correct/missed/extra links and
variable depths separately. Review timing and storage needs before proposing a
funded reliability study. The 990-dataset proposal remains unapproved.

## Two Random Forest cases stopped at the time limit

Both Random Forest cases stopped at the prespecified 4,500-second soft deadline,
not an estimator exception. Saved attempt ledgers contain no failed assessments.
N=80 completed 216 assessments (100 at Y, 100 at X2, 16 at X1) in 75.65 minutes.
N=200 completed 192 (100 at Y, 92 at X2) in 75.17 minutes. They remain incomplete;
partial results cannot establish final network accuracy.

GBM completed in 29.06 minutes (N=80) and 45.88 minutes (N=200). LightGBM completed
in 16.12 and 20.95 minutes, respectively. All four saved 300 completed assessments.
These are computation statuses from receipts, not independent accuracy conclusions.
The cloud arithmetic audit was still running when this diagnosis was recorded.

All six artifacts were downloaded locally and their raw ZIP CRCs checked.
`case-status-diagnosis.json` records raw archive SHA256 values and checkpoint
counts. No software change, scientific refit, rerun or additional Actions dispatch
was made. The next step is to review the completed audit, then prepare a separately
budgeted checkpoint continuation for RF rather than repeating completed work.

## What made Random Forest slow?

The completed independent cloud audit verifies the four GBM/LightGBM cases each
recovered X1 -> X2 -> Y with correct depths. Both RF prefixes passed their saved
observation-loss/split/parameter/count checks, with no failed or unmatched
attempts. The parent aggregate correctly remains failed with two incomplete cases.
LightGBM recorded156 warnings per case; these remain visible for later inspection.

Recorded median worker time for one root assessment (four predictors):

|Engine|N80 seconds|N200 seconds|
|---|---:|---:|
|Random Forest|94.265|102.038|
|GBM|27.750|43.484|
|LightGBM|14.328|17.797|

Each target has100 resamples. Each assessment uses10 outer partitions (5folds,
repeatedtwice). Every tuned model searches4 candidates with3 inner folds:12tuning
fits per outer partition. At four predictors, the full model and four separate
single-predictor models are tuned:5x10x12=600tuningfits per assessment. Ordinary
full/broad/leave-one-out prediction adds90 outer fits, with occasional additional
group-ablation fits. Every RF fit builds100 trees. Thus one root assessment
requires about69,000 trees, and the root alone about6.9million across100resamples.
Later targets cost less as candidate sets shrink: tuning480then360 per assessment.

The two unfinished RF prefixes already required113,760 and104,160 tuning fits,
plus16,911 and15,441 outer fits. About87% of the recorded fits were tuning fits.
This is a fit-count share, not a measured share of elapsed time. The total is
408completed assessments and250,272model fits, roughly25million RF trees already
built. Those results can be reused; they should not be recomputed.

Four worker processes run assessments concurrently. Each estimator uses one
thread, avoiding nested parallelism. The runner waits for batches of four; batch
barriers may add overhead, but the saved records do not isolate that overhead.
Equal numbers of100-tree fits are not equal work: RF trees and shallow GBM stages
have different training costs, and LightGBM uses another implementation. The
observed comparisons establish runtime differences, not a causal performance
ablation of each setting. No per-risk timers or controlled setting comparisons
were collected. Candidate filters, repeat count, tuning size, tree count and
worker scheduling are future optimization candidates, each needing matched
accuracy/stability evidence before changing defaults.

Keep these settings unchanged in the RF-only continuation. Reuse216assessments
atN80 and192atN200. From completed per-depth medians, roughly20-30additional
minutes of computation may suffice for the unfinished depths; this is a planning
estimate, not a guarantee. The150-minute job cap gives room for download, prefix
verification, runner variability and upload. Its132-minute soft compute allowance
is additional time, not a change to scientific settings. No larger study follows
automatically.

A concrete avoidable cost also appears in the pinned implementation: RF searches
max_features=.7 and1.0 even for a single-predictor model. sklearn1.3.0 resolves
both to one candidate feature (max(1,int(fraction*n_features))). The two search
entries therefore have the same effective feature-sampling setting for each
min_samples_leaf choice. With unchanged seeds/data, this is a candidate for
numerical-equivalence testing and deduplication. Across three complete targets
with4/3/2predictors, this could remove54,000of144,000tuning fits, aboutone-third
of all ordinary model fits. This is a source-derived fit-count opportunity,
not an experimentally verified runtime saving. It is not applied to this frozen
continuation: grid counts, parameter/tie conventions and exact prediction parity
would need explicit regression evidence in a later optimization change.

## RF-only continuation launched

[Run38788630751](https://github.com/imh-ds/open/actions/runs/38788630751) continues
only the two unfinished RF fixtures. Each job allows150minutes, including132
additional compute minutes; the total new reservation is346runner-minutes.
No scientific defaults, seeds or datasets change. Preparation passed frozen
source/runtime and producer transport checks; downloaded authorization exactly
matches the two RF cases on attempt1/sourcefa709359a461a561de7ccc3aa2fc3f9de483e59c.
Both case jobs started. The code reuses408 audited completed assessments and
refuses failed or unmatched attempts. Four original GBM/LightGBM cases are reused
in aggregation, with no rerun. This is still timing evidence, not qualification.

45focused software checks pass. A Windows report line-ending fingerprint mismatch
was found and fixed before dispatch; raw ZIP/checkpoint SHA256 checks remain
strict. Implementation, normalization fix, cost evidence and actual launch
records were committed separately. See rf-continuation-38788630751/launch.json
and saved authorization for provenance. Review completed mixed results before
considering any larger reliability study or default optimization.

## Completed continuation

Run 38788630751 passed: all six cases complete and exactly recover the chain,
with correct layers and no extra links. RF needed 21.5/31.1 additional compute
minutes (97.1/106.3 total). All six raw archives, audit, status/timings and simple
interpretation are preserved in [the completed results](completed-38788630751/README.md).
This completes the timing check, not broad scientific qualification.
