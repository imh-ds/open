# Matched-chain multi-engine signal validation launch

Status: [dispatched once as run38891293628](https://github.com/imh-ds/open/actions/runs/38891293628) from `e1dc2fa5cf25611ff2ab54499594f6121298bfd7`; preparation passed and both case matrices started. The new30,000 runner-minute ceiling applies. No scientific model has been fitted locally while preparing this study.

## What will be tested

330 analyses: RF/GBM/LightGBM x N80/200 x55 repetitions. These are110 fresh independent datasets, paired across engines. Known network X1 -> X2 -> Y at depths2 ->1 ->0; two unrelated nuisance variables must be excluded. X2=.6X1+.8e2 and Y=.8X2+.6ey, with independent standard-normal X1/e2/ey/Z1/Z2.

Keep shipped100 subsamples,80% sampling, performance CV5x2, inner CV3,100 trees, effective engine grids and thresholds including70% retention. At each engine/N, require all55 cases audited, at least50 exact networks+depths, and no more than1 extra-finding network. Unknowns prevent a pass. The independent audit checks saved losses, splits, selected parameters, fit counts, inputs and full recursive structure.

## Budget and evidence

Case caps (N80/N200 minutes): RF90/150,GBM60/100,LightGBM40/60. Case sum27,500. Two165-case matrices, max10 parallel each. Six120-minute audit caps, prepare30, combine15, publication270, registration1: reserved28,536 of30,000, leaving1,464 unallocated. Six separate audit jobs and sequential cell publication bound disk usage. All case jobs remain below6hours. No automatic retries/replacements or changed settings after seeing results.

Raw archives and case-level audit evidence are retained in prerelease bundles below1.5GiB with download/hash verification. Actions artifacts are temporary transport. Only the publication job has write permission.

## Verified before launch

- 75 focused tests passed under the pinned Python3.11.9 runtime; inherited null/audit/RF guards included.
- All six source-checkout variants import in isolation AND reproduce the frozen manifest/source checks.
- Final no-fit preparation generated330 controls, each bound to the manifest below; zero scientific fits.
- No collisions against146,854 historical seed values in retained study manifests.110 independent analysis blocks reserve500 seeds each, spaced1,000 apart, and use different blocks from the historical study.
- Publication tests reject missing audited archives and verify six distinct engine/N inventories. Complete and expired no-fit case paths are tested.
- Parent corrected null report qualifies all engines at both N values and is frozen as a source of this study.

Frozen manifest canonical SHA256: `afbff171ed5bae801c366530028c4fd24bac3f87adb44c73bd904ed18be32c53`.
Protocol: [matched-chain validation protocol](../../tree_recursive_signal_validation_protocol.md).
Manifest: `examples/study_manifests/tree-recursive-signal.json`.

This study tests recovery of one strong linear Gaussian chain. It does not establish causal direction, weak/nonlinear effects, correlated direct predictors, interactions, high-dimensional performance or a universal minimum sample size. Interpret each engine and N separately. Scientific results are pending.
