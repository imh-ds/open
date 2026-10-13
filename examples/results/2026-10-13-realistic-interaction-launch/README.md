# D-043 calibration launch

[Calibration run 38477737386](https://github.com/imh-ds/open/actions/runs/38477737386) was dispatched once from public source `9ece0250e04a6213b397dd871961759db6f0fdd3`, after approval and verified implementation gates.

This first stage checks **3,600 datasets with no interaction in the measured predictors**. It establishes fresh cutoffs for sample sizes 150/300/600, latent predictor correlations 0.1/0.3/0.5 and predictor reliability 1.0/0.8. Its purpose is to learn how large a score can arise by chance before interpreting the real interaction cases.

The planned matrix has 45 shards, 80 datasets per shard and two single-thread workers. `max-parallel=256` allows available account capacity; it does not promise 256 simultaneous runners. Worker timeouts and the total two-stage 3,750 runner-minute ceiling follow the approved protocol.

The implementation passed **712 tests**, independent review and the native 80-dataset preflight in **160.406 seconds**, with a verified zero-fit resume. All frozen source/runtime/manifest and native raw-evidence blobs were checked on GitHub before dispatch. Native raw evidence and test logs are preserved in the adjacent preflight folder; this launch folder retains public Git proofs, dispatch receipt and API snapshots.

Next gate: complete and independently audit all 3,600 calibration identities, preserve their raw evidence, and commit/verify the fresh thresholds. Then check representative noisy-saturation and strong-reference cases before launching the separately seeded 10,800-dataset development stage. That stage examines detection and plot accuracy across the approved conditions. Neither stage grants production confidence or establishes a universal minimum sample size.
