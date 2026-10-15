# Optimized RF recursive validation: both cases passed

[GitHub Actions run 38816428068](https://github.com/imh-ds/open/actions/runs/38816428068)
completed successfully. Preparation, both fresh cases, uploads and the independent
audit passed on the first attempt. All two planned datasets remain in the results.

## What this means simply

The optimization removed duplicate fitting work. On these two datasets, it did
not change the answers: both the predictions and the recursive network matched
the original implementation, with no numerical differences in the comparison.
Both cases found the intended chain:

```text
X1 -> X2 -> Y
```

The engine first identified X2 as a direct predictor of Y, then identified X1
as a predictor of X2. It assigned Y to depth 0, X2 to depth 1, and X1 to depth 2.
It did not add either unrelated variable, Z1 or Z2, to the network. This confirms
that the engine progressed beyond the initial outcome model in both fixtures.

|Sample size|Original compute minutes|Optimized compute minutes|Observed reduction|Result|
|---|---:|---:|---:|---|
|80|97.14|44.67|54.0%|Same predictions and correct network|
|200|106.26|71.92|32.3%|Same predictions and correct network|

Times are recorded computation durations, not whole workflow or billing times.
The original RF durations combine their preserved initial and continuation
computation. These are observed comparisons across runs, not controlled runtime
benchmarks or guaranteed speedups on other datasets or runners.

Each case used the shipped 100 resamples, five-fold performance CV repeated twice,
three inner tuning folds, and 70% retention threshold. Each completed 300 local
assessments across the recursive targets. Each used 90,000 tuning fits rather
than the original 144,000, a 37.5% reduction. Outer model fits were unchanged:
21,111 at N=80 and 21,001 at N=200. The search-policy change saved 54,000 tuning
fits per case without reducing the resample or fold counts.

## Verification and preserved evidence

The independent cloud audit checked 1,076,653 observation-loss records in total,
fit accounting, parameters, splits, inputs, attempt coverage and network/depth
decisions. All 600 completed checkpoint comparisons passed; changed float fields
and maximum absolute difference were both zero. There were no missing cases,
integrity errors, warnings, or unknown fit costs. No old checkpoint was reused.

`verification.json` records the local download verification: ZIP CRCs, original
audit fingerprints, every receipt-listed file hash, run/source/attempt identity,
and a second comparison against the original reference archives. This verification
performs no scientific fits. `file-sha256.json` fingerprints the preserved packet
except itself. Raw ZIPs retain all inputs, controls, predictions, losses, fitted
parameters, checkpoint ledgers and network output.

- `audit/`: original cloud audit, case comparisons, interpretation, runtime and
  artifact download inventory, with original bytes preserved.
- `raw/`: both full case ZIPs, controls and pinned runtime records.
- `authorization/`: exact approved case matrix and authorization.
- `run.json`, `jobs.json`, `artifacts.json`, `logs.txt`: completed Actions evidence.
- Dispatch source: `c9b512459be657d0e11c33fb808e4616708a6de9`.
- Manifest: `840948cb5d67b0f5b28f784209b474835c6851b7426c23c074134e54048cebc7`.

The approved ceiling was 300 additional runner-minutes, with 286 reserved.
Both cases completed within their 120-minute job caps. The original allocation
and historical failed preflight outcomes are unchanged; this run does not refund
or rewrite earlier reservations.

## What remains to do

This completes the two-case timing/equivalence check. It does not establish how
often RF finds the correct network across new datasets, whether it invents links
under the null, or general minimum sample sizes. N=80 succeeding here is not a
general N=80 recommendation. This check does not qualify GBM or LightGBM.

The recommended next step is a frozen, budgeted broader reliability study for
RF, GBM and LightGBM, starting with the null conditions to check invented links,
then the matched signal conditions if each engine passes its null gate. Use the
observed RF runtime to revise its allocation, and preserve all planned cases,
including failures. Correlated predictors and nonlinear recursion remain separate
conditions requiring their own evidence. No new scientific run was dispatched
while reviewing these results, and no additional compute is authorized here.
