# Tree-engine recursive null validation protocol

**Status:** Protocol and no-fit implementation checks are frozen. No 330-case
scientific run has been dispatched. This protocol describes the frozen null
stage; it does not report model findings.

## What this study checks

The study asks whether Random Forest, GBM, or LightGBM invents recursive links
when every candidate predictor is independent of the outcome. Each dataset has
four independent standard-normal predictors (`X1`, `X2`, `Z1`, `Z2`) and an
independent standard-normal outcome (`Y`). The true recursive network is empty.

There are 55 fresh datasets for each engine at N=80 and N=200: six cells and 330
planned cases total. Repetitions 1–55 exclude the prior timing fixture at
repetition 0. The same generated rows and analysis seeds are paired across the
three engines. A case that is missing, incomplete, unaudited, or has unknown fit
cost remains unknown; no case is replaced.

## Analysis settings

Keep the shipped thresholds and analysis settings fixed: 100 subsamples, 80%
subsampling, 5-fold performance CV repeated twice, three inner tuning folds,
100 trees, four assessment workers per case, and one numerical thread per
worker. RF uses its verified two-candidate singleton predictor grid and its
four-candidate wider grid. GBM and LightGBM use four candidates for every
non-empty predictor set. Different tuning grids mean this study compares
recursive findings under each engine's shipped setup; it is not an equal-cost
predictive-performance contest.

## How the null gate works

For each engine and sample size, count a false network if the finalized result
contains any promoted node, edge, or group. Calculate a two-sided exact 95%
Clopper–Pearson interval with the planned denominator of 55. The upper bound
must be at most .10: zero or one false network passes (upper bounds about .0649
and .0972); two fail (about .1253). Any unknown case blocks that cell from
passing. The engine advances only when both N=80 and N=200 cells pass and all
case audits and artifact provenance checks are verified.

A pass means the engine is eligible for a separately reviewed matched-signal
study. It does not show that the engine reliably finds real links. The later
signal phase needs a new budget and explicit approval.

## Runtime and evidence controls

Run two concurrent Actions matrix jobs, one for each N. Each matrix has 165
jobs, with at most ten parallel jobs; at most 20 case jobs are requested at
once. Each case has a hard cap of 60/100 minutes for RF, 45/70 for GBM, and
25/35 for LightGBM at N=80/200. The runner stops new assessment work early to
leave time to pack and upload evidence. A timeout is an unknown case, never an
automatic retry or a reason to change folds, resamples, or sample size.

The null stage has a 20,000 runner-minute ceiling. The case-job caps reserve
18,425 minutes; setup, preparation, auditing, report, and publication reserve
256, for 18,681 total. The remaining 1,319 minutes are not automatically
allocated. No signal-stage compute is included.

Retain per-case audit rows, raw case archives, the combined report, and a SHA256
inventory. Bundle raw archives by engine and N into prerelease assets no larger
than 1.5 GiB each; verify each remote asset after upload. Actions artifacts are
temporary transport and are not the durable record. Release-write permission is
limited to the publication job.

## Interpretation limits

This design tests false recursive links for independent Gaussian predictors at
N=80 and N=200. It does not test recovery of real links, correlated predictors,
nonlinear populations, interactions, causal claims, or a universal minimum
sample size. The confidence bounds are marginal for each engine/sample-size
cell and do not control a family-wise error rate across all six cells.
