# Shipped-default recursive pilot

GitHub Actions run [38525190948](https://github.com/imh-ds/open/actions/runs/38525190948), source revision `284b474e0c26b8fb348e6c5ae3306ff53184490b`. All three fixtures ran with the shipped settings: 100 resamples, 5-fold performance CV repeated twice, 3-fold tuning CV, and default thresholds. All 500 planned target-by-resample assessments completed; none failed and none recorded warnings. The three saved bundles passed a strict reload and the expected-structure audit without new model fits. Original sample/config hashes match.

## What it found

- **Chain fixture:** the network was `Stress → Depression → Anxiety`. Depression was assigned depth 1 and Stress depth 2. Both links were selected in **100 of 100** resamples.
- **Same chain with one missing Depression value:** it recovered exactly the same two links and depths, again **100 of 100** times per link. In 85 resamples the missing row was included in the subsample, so the Depression assessment used 63 observed outcomes; in the other 15 it used 64. The missing value was excluded, not imputed.
- **Near-duplicate predictors:** Scale A and Scale B formed **one joint group** predicting Outcome in **100 of 100** resamples. The results did not claim either scale had a unique arrow by itself.

In plain language, the recursive engine followed the chain all the way back, and it handled the missing outcome and the redundant-predictor case as intended. On these fixtures, each displayed relation was very stable across resampled portions of the same dataset.

## Time and limits

The analysis steps took 144.7 minutes for the chain, 149.5 minutes for the missing-value chain, and 89.0 minutes for the shared-predictor fixture: about **383 minutes of model work**. Including setup, uploads and aggregation, the Actions jobs used about **386.7 runner-minutes** of the approved 560-minute cap. They ran in parallel, so wall-clock time from job start through aggregation was about 151 minutes. This is an actual cost measurement for these fixtures and this runner setup.

These are three fixed, synthetic 80-row fixtures, not hundreds of newly generated independent studies or real participants. The 100 resamples are repeated subsamples from each one dataset. Therefore, “100/100” means the same relation survived all 100 subsamples here; it does not mean a 100% chance that a real-world relationship is true, and these results do not estimate a false-positive rate or prove general scientific accuracy.

The run validates full-default execution, recursive layer construction, missing-outcome handling, joint grouping, artifact saving and reloading for these cases. The broader scientific confirmation study remains open. Full output ledgers, configurations, environment freezes, run receipts and SHA-256 file hashes are stored alongside this interpretation.
