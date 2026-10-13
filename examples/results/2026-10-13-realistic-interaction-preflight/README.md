# D-043 native calibration preflight

The representative N=600, latent correlation 0.5, reliability 0.8 observed-additive shard completed all 80 planned datasets in **160.406 seconds** (12-minute limit). Each dataset includes 692 fitting/tuning calls and 20 saved-setting verification refits. This is extra native computation on existing planned identities, not 80 additional independent replicates.

Fresh worker processes resumed the same 80 datasets in 6.062 seconds with **zero fits and zero refits**. Independent vector-score recomputation differed by at most 3.36e-17; all 80 raw tasks passed saved-evidence checks, with zero warnings or audit errors.

Raw evidence is80,554,615bytes across721 files, losslessly compressed to22,019,620bytes. The streamed one-shard audit took7.75seconds; linear extrapolation is5.8minutes for calibration and17.4minutes for development, below the60-minute aggregation limit. Estimated development disk headroom is6.1GB using duplicate compressed archives and one extracted shard. These are preflight estimates; noisy-saturation/strong-reference timing and size are checked again after calibration.

The full3600-task calibration audit is intentionally incomplete here: this file is native readiness evidence, not completed calibration or learned thresholds. The3520 absent planned identities retain unknown execution costs. `preflight.json`, `audit.json` and `dataset_metrics.csv` preserve scope, identities, costs and numerical checks.

`raw-044/` contains six bounded lossless archive parts, every part/archive SHA256, the original raw-file inventory and a reassembly verifier. Run `python raw-044/reassemble.py`, then audit the restored archive with the frozen realistic runner. Original native files and the complete original ZIP are also retained locally. Publication proof is separate; local preservation alone does not establish remote archival completion.

`source-publication.json` verifies the frozen source and manifest; `sparse-check.json` verifies the runtime and legacy guards in a sparse checkout fixture. This preflight ran alongside a separate test-suite process; the measured timing is not a dedicated-runner performance guarantee. Historical and new test computation is separate from the approved3750 Actions runner-minute timeout ceiling.
