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
