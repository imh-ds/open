# Validation results index

Tracked summary results of each validation study, one dated folder per study.
Folder names are cited by workflows, protocols, manifests and docs, so they do
not move. Raw per-cell outputs live in GitHub releases (see
[docs/reproducing.md](../../docs/reproducing.md)); the interpretation of each
study is in [docs/validation-evidence.md](../../docs/validation-evidence.md).

| Folder | Role | Produced by | Notes |
|---|---|---|---|
| `2026-10-17-phase-a-linear-reference` | Phase A: linear reference | `examples/phase_a_linear_reference.py` | Reference for the later phases |
| `2026-10-17-phase-a-linear-reference-check` | Pre-run check record for Phase A | `examples/phase_a_linear_reference.py` | Protocol and verification files only |
| `2026-10-17-phase-a-prime-small-sample` | Phase A′: small-sample OLS calibration | workflow `phase-a-prime-small-sample.yml`, `examples/phase_a_prime_actions.py` | |
| `2026-10-17-phase-a-prime-small-sample-check` | Pre-run check record for Phase A′ | `examples/phase_a_prime_actions.py` | |
| `2026-10-18-phase-b-local-detector` | Phase B: engine-based p-values | workflows `phase-b-check.yml`, `phase-b-local-detector.yml`, `examples/phase_b_actions.py` | Evidence for dropping random forest and gradient boosting ([dropped engines](../../docs/dropped-engines.md)) |
| `2026-10-18-phase-b-local-detector-check` | Pre-run check record for Phase B | `examples/phase_b_actions.py` | `check.json`, `protocol-draft.json`, `seed-audit.json` |
| `2026-10-19-exploratory-validation` | Discovery validation (study 1) | workflow `exploratory-validation.yml`, `examples/exploratory_validation.py` | |
| `2026-10-20-threshold-sensitivity` | Re-analysis of study 1 at other thresholds | `examples/threshold_sensitivity.py` | |
| `2026-10-20-study-v2` | Study v2: split-and-confirm, correlated pools, comparators | workflow `study-v2.yml`, `examples/study_v2.py` | |
| `2026-10-20-study-v2r` | Study v2-R: repeated splits | workflow `study-v2r.yml`, `examples/study_v2r.py` | |
| `2026-10-22-pilot-pls-svr` | Pilot of PLS and SVR engines | workflow `pilot-pls-svr.yml`, `examples/pilot_pls_svr.py` | Engines dropped ([dropped engines](../../docs/dropped-engines.md)) |
| `2026-10-22-svr-validation` | SVR validation rates | workflow `svr-validation.yml` | Engine dropped |
| `2026-10-28-study-ablation` | Noise-bar and recursion ablation | workflow `study-ablation.yml`, `examples/study_ablation.py` | |

Each study folder holds its own `interpretation.md` where one exists.
