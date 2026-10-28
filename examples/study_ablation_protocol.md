# Study ablation protocol: noise bar and recursion

Date: 2026-10-28. Status: **DRAFT** (to be frozen after the timing probe; the
predictions below are fixed before any full run).

Source: `examples/study_ablation.py`. Workflows: `study-ablation-probe.yml`,
`study-ablation.yml`. Reuses study v2's generator, truth scoring and dataset
seeds (`examples/study_v2.py`).

## 1. Question

Is OPEN "just feature selection performed repeatedly"? Two components are
claimed to matter:

- the **noise bar** (a link passes a fold only if its held-out gain beats the
  95th percentile of the permuted-noise columns);
- **recursion** (kept links become targets and are explored at the next depth).

Does each contribute, and does recursion, which compounds mistakes, stay safe
only because of the noise bar?

## 2. Arms

| Arm | Recursion | Noise bar | Source |
|---|---|---|---|
| `full` | yes (max_depth 3) | yes | `explore`, defaults |
| `nobar` | yes (max_depth 3) | no | `explore` with the bar replaced by 0 |
| `norec` | no | yes | depth-1 slice of `full` |
| `screen` | no | no | depth-1 slice of `nobar` |

- **No bar** means a link passes a fold when its held-out gain is above zero.
  The 80% keep and 40% flag fold rules, the hide floor (0.001) and every
  other setting are unchanged. The bar is replaced by monkeypatching
  `open.explore._apply_bars` inside the study script; the library is untouched.
  It applies to interactions as well as single links.
- **Slices are exact.** Depth 1 does not depend on later depths. For datasets
  0–4, `full_d1` and `nobar_d1` rerun with `max_depth = 1`, and the aggregate
  reports any tier mismatch against the slices (expected 0).

## 3. Design

- Engines: OLS, Elastic-Net, LightGBM.
- Cells: study v2's eight core cells (N 1,000 / 2,000 × pool 13 / 25 ×
  linear / U-shaped), correlated normal predictors.
- Datasets 0–99 per cell, the same datasets as study v2, each split 50/50;
  every arm is run on the exploration half. Confirmation is out of scope.
- 3 × 8 × 100 × 2 arms = 4,800 explorations, plus 240 integrity reruns.
- Truth and scoring are study v2's (d-separation given the pool).

## 4. Outcomes, per network

False kept links (direct + interaction), by depth (1 vs 2–3); false flagged
links; true designed links kept, by depth. Paired differences between arms are
taken per dataset and summarised per cell with a normal 95% interval
(`contrasts.csv`):

- `nobar − full` (the bar's effect with recursion);
- `full − norec` (recursion's effect with the bar);
- `nobar − screen`, `norec − screen` (the same effects without the other).

## 5. Predictions (set before the run), per cell

1. **P1.** Removing the bar at least doubles false kept links per network
   (`nobar ≥ 2 × full`), and the increase is larger at depth 2–3 than at depth 1.
2. **P2.** Recursion finds true links that depth 1 cannot: `full` keeps more
   than 0 true links at depth 2–3 per network (`norec` keeps none by construction).
3. **P3.** With the bar on, recursion does not raise false kept links:
   `full ≤ norec + 0.05` per network.

Failing P3 or P1 is reported as a finding, not tuned away.

## 6. Known limits

- "Gain above zero in 80% of folds" is one reasonable no-bar rule. No other
  variants are run.
- The Elastic-Net noise columns are often shrunk to zero, so its noise bar is
  often near zero already; Elastic-Net may therefore show little difference
  between `full` and `nobar`. That is a result, not an error.
- The hide floor stays on in `nobar`; at 0.001 it removes only negligible links.
- Discovery stage only; the studies of confirmation are Studies 5 and 6.
- Real-data replication is a separate, later study.

## 7. Outputs

`integrity.json`, `per-network.csv`, `contrasts.csv`, `designed-keep.csv`,
`criteria.csv` (P1–P3 per cell), then `interpretation.md` and a section in
`docs/validation-evidence.md` once results are in.
