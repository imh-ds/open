# Study ablation: what the noise bar and recursion contribute

Run: GitHub Actions 40742913135 (240 of 240 shards, 5,040 of 5,040 tasks).
Protocol: `examples/study_ablation_protocol.md`. Datasets are study v2's core
cells (3 engines × 8 cells × 100 datasets), discovery half only, max_depth 3.

**Integrity.** The depth-1 slices equal the `max_depth = 1` reruns exactly
(0 mismatches in 9,180 `full` and 11,727 `nobar` links).

## Arms, per network (mean over the eight cells)

| Engine | Arm | False solid links | True links kept |
|---|---|---|---|
| OLS | full | 0.04 | 2.00 |
| | no bar | 10.9 | 3.49 |
| | no recursion | 0.01 | 0.92 |
| | neither | 2.2 | 1.81 |
| Elastic-Net | full | 0.30 | 3.73 |
| | no bar | 3.5 | 4.14 |
| | no recursion | 0.08 | 1.62 |
| | neither | 1.3 | 1.86 |
| LightGBM | full | 0.02 | 1.25 |
| | no bar | 19.4 | 4.59 |
| | no recursion | 0.01 | 0.60 |
| | neither | 2.6 | 2.32 |

("False solid links" are false kept links in `per-network.csv`, averaged over
cells; per-cell values and paired intervals are in `per-network.csv` and
`contrasts.csv`.)

## Predictions

**P1: removing the bar at least doubles false solid links. Met in 24 of 24 cells.**
- The increase is large: 3× to 25× for Elastic-Net, 75× to 630× for OLS and
  hundreds of times for LightGBM, whose full-rule rate is near zero.
- It is larger at depth 2–3 than at depth 1 for every engine: false links per
  network at depth 1 / depth 2–3 are 2.2 / 8.7 (OLS), 1.3 / 2.2 (Elastic-Net)
  and 2.6 / 16.7 (LightGBM), against 0.01–0.08 / 0.01–0.22 with the bar.
  Without the bar, recursion compounds the mistakes.
- The no-bar arm keeps more true links as well (mean gain 1.5 OLS, 0.4
  Elastic-Net, 3.3 LightGBM per network), but they come with 10–30 false ones.

**P2: recursion finds true links that depth 1 cannot. Met in 17 of 24 cells.**
- It is met in every linear cell and in every LightGBM cell.
- It fails in the seven OLS and Elastic-Net U-shaped cells (one Elastic-Net
  cell has 0.01), because those engines keep almost nothing in them at any
  depth. This is the known blindness to U-shapes, not a recursion effect.
- Gains are modest where it works: 0.9 to 3.4 extra true links per network in
  linear cells for OLS, 3 to 5 for Elastic-Net, 1 to 1.3 for LightGBM at
  N = 2,000 and about 0.1 to 0.2 at N = 1,000.

**P3: with the bar on, recursion does not raise false solid links. Met in 19 of 24 cells.**
- LightGBM: 8 of 8; OLS: 7 of 8 (N = 2,000, pool 25, linear misses by .02:
  0.07 against 0.00 + .05).
- **Elastic-Net: 1 of 8.** Recursion adds about 0.2 false links per network
  (up to 0.5; the paired interval excludes zero in 7 of 8 cells). Elastic-Net's
  full-rule rate is already the highest of the three (0.1 to 0.6 per network),
  and recursion adds to it.

## Conclusions

- **The noise bar is what makes recursion safe.** Without it, every engine
  fills the network with false links, and recursion makes it worse at depth
  2–3. With it, false solid links stay below 1 per network in all cells.
- **Recursion adds real discoveries at low cost.** It finds extra true links
  wherever the engine can see the effect, with no added false links for OLS
  and LightGBM. For Elastic-Net it adds true links (2.1 per network) and some
  false ones (0.2).
- **Elastic-Net does not behave like OLS.** I expected it to show little
  difference between `full` and `nobar`, because its noise columns are often
  shrunk to zero. It does not: removing the bar adds 1 to 10 false links per
  network, so the bar works for it as well.
- The "just repeated feature selection" objection has an answer: the screening
  arm (neither component) keeps 1 to 4 false links per network and 0.8 to 2.3
  true ones, and the full rule removes nearly all the false links while
  keeping as many true links overall (2.0 against 1.8 for OLS, 3.7 against 1.9
  for Elastic-Net; LightGBM keeps fewer, 1.3 against 2.3) because the depth-2–3
  links are added on top.

## Limits

- "Gain above zero in 80% of folds" is one no-bar rule; no other variants.
- Discovery only. 100 datasets per cell, so cell-level rates carry Monte Carlo
  error; paired intervals are in `contrasts.csv`.
- Raw shard outputs (27 MB) are in the Actions artifacts of the run above, not
  committed here.
