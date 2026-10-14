# Continuation stopped at the per-dataset time allowance

Run [38621370295](https://github.com/imh-ds/open/actions/runs/38621370295), attempt1, source84c933c6bfb6fb82680bb262d0d82c8d71809b42, failed because the first new dataset exceeded its15-minute computation allowance. The case stopped after903.781seconds at a batch boundary. This was the study's own soft deadline, not GitHub's45-minute job timeout. The job took16minutes53seconds including installation, no-fit fixture verification and preservation. Both artifact uploads succeeded; later stages were skipped.

## Simple interpretation

The previous comparison problem was resolved. All200old fixture assessments were verified and reused, with zero new fixture fits. Their original archive fingerprint matches exactly.

The recursive engine advanced normally on new chain case160(N80): it completed all100outcome assessments, finalized X2as the direct predictor, then began predicting X2at depth1. It completed44of100assessments there before the deadline. This supports correct layer advancement on the observed prefix; it does not establish that the whole network, remaining depths or scientific accuracy criteria pass. Case240(N200)never started. There are zero completed new study networks, so no error-rate or recovery qualification is available.

Independent no-fit replay validated all144completed checkpoints:174477observation-loss checks, saved identities/settings/inventories,12191recorded outer fits and507000recorded tuning fits, with zero assessment warnings and no unfinished started assessments. The case-level receipt conservatively remains incomplete/fit-cost-unknown; the checkpoint audit documents known completed work without rewriting that receipt. These fit counts describe saved original computation, not new audit fits.

## Why the allowance was insufficient

This dataset contains four candidate predictors. Each outcome assessment tunes one full model and four individual-predictor models across10outer folds,25parameter combinations and3inner folds:3750tuning fits per assessment. The100outcome assessments alone therefore required375000tuning fits. At depth1, three candidates required3000tuning fits per assessment. Small Ndoes not remove this repeated fitting cost.

Median worker times were25.3515seconds per outcome assessment and20.234seconds per depth1assessment. At the observed four-worker throughput,100assessments in each of these two layers project to about19minutes in total, excluding startup/other overhead and any deeper layers. This is a partial-case projection, not a measured full-network runtime or a timing guarantee for other scenarios/N200. The earlier smaller fixed fixture was insufficient evidence for the assumed15-minute new-case allocation; this preflight was specifically intended to test that assumption and it failed.

## Recommended next step

Revise the execution plan before another scientific dispatch. Keep shipped statistical settings and the original failure/unknown denominators unchanged. Use the saved144checkpoints to avoid wasting completed work in any separately approved continuation. Establish realistic complete-network timing and then choose a feasible shard layout and replication budget; increasing only GitHub's job timeout cannot fix the per-case900-second limit or the total study allocation. The current480-dataset/9917-minute plan is not demonstrated feasible. No automatic retry, cutoff reduction, smaller CV/resample setting, scientific qualification or release acceptance follows from this diagnosis.

## Saved evidence and reproduction

`raw/` preserves both downloaded artifacts, the producer manifest, complete run log and run/jobs API snapshots. `audit/raw-sha256.json` records every raw file's SHA256. `audit/diagnosis.json` and `audit/assessment-timing.csv` preserve the independent partial-case checks and timing evidence. No production code, scientific settings or historical manifest was changed.

From the repository root, use the pinned installed environment:

```powershell
python -I examples/results/2026-10-14-recursive-preview-validation/preflight-failure-38621370295/audit/reproduce.py
```

This replays saved observations and arithmetic with zero model fits. It does not resume the scientific case or make its failed timing gate pass.
