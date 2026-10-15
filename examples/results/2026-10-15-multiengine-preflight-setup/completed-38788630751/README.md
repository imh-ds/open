# Completed recursive timing check

[Continuation run](https://github.com/imh-ds/open/actions/runs/38788630751)
completed successfully on 2026-10-15, attempt 1, source
`fa709359a461a561de7ccc3aa2fc3f9de483e59c`.

All six cases completed and independently recovered exactly `X1 -> X2 -> Y`,
with X2 at depth 1 and X1 at depth 2. There were no missing or extra links,
no nuisance variables promoted, no unknown cases, and no audit integrity errors.
Each case completed 300 target/resample assessments (100 at each of three targets).
These are six dataset/engine cases, not 1,800 independent datasets.

| Engine | N=80 compute minutes | N=200 compute minutes | Exact pathway |
| --- | ---: | ---: | --- |
| Random forest | 97.1 | 106.3 | Both |
| Gradient boosting | 29.1 | 45.9 | Both |
| LightGBM | 16.1 | 20.9 | Both |

RF totals combine the original run and this continuation. New computation took
21.5 and 31.1 minutes, respectively; 216 and 192 completed assessments were reused,
and 84 and 108 new assessments finished. Scientific settings were unchanged.
The continuation workflow took about 37.5 minutes of wall-clock time, including
preparation, concurrent jobs, artifact transfer and the final independent audit.
Compute times in the table exclude those surrounding workflow activities.

The earlier RF failures were time-budget failures. With additional time, both
cases completed and produced the expected structure. This supports proceeding
to design a broader reliability study, rather than diagnosing these failures
as evidence that RF cannot perform recursion. One known strong continuous chain
at each sample size does not establish recovery rates, false-positive rates,
performance with correlated predictors, or a general minimum sample size.
The audit explicitly records `scientific_qualification=false` and `stage=timing_only`.

LightGBM recorded 156 assessment warnings per case. Inspection of every saved
assessment warning identified only `PyparsingDeprecationWarning` messages from
matplotlib's font parser (`oneOf`, `parseString`, `resetCache` deprecated names).
These are dependency deprecation notices, not estimator convergence warnings.
The counts and messages remain preserved in `archive-verification.json`.

The high fit count remains the main identified workload: 144,000 tuning fits
and roughly 21,000 other fits per complete RF case, with 100 trees per fit.
The earlier source inspection identified duplicate effective RF grid settings
for single-predictor models. They were not changed here. Recommended next step:
test prediction equivalence and tuning-accounting correctness of deduplicating
those settings before committing to a large RF reliability allocation. This
could reduce fit counts substantially; runtime savings have not been measured.
GBM/LightGBM can use their measured timing to budget a separately approved study.
No further scientific fits or workflow dispatches are included in this review.

`audit.json` is the unmodified cloud audit. `run.json` and `jobs.json` preserve
GitHub status and timing evidence. `raw/` contains all six completed case ZIPs,
their control files and runtime records. Four cases come from original run
38775613523; two RF cases come from continuation run 38788630751 and retain their
original producer evidence internally. All six ZIP byte counts, SHA256 digests,
and CRC checks match the audited archives. No local scientific fits were run.
This preserves these six cases independently of Actions artifact expiration;
it does not resolve the separate historical archive-transfer backlog.
