# Reproducing the evidence

## Environment

The studies ran on Python 3.11.9 with these pinned versions:

```bash
python -m pip install numpy==1.26.4 scipy==1.11.2 pandas==2.2.3 scikit-learn==1.3.0 lightgbm==4.6.0 statsmodels==0.14.4 patsy==1.0.1 packaging==23.2
```

The reference-study scripts check their own source hashes before running.
The files in `examples/` are byte-identical to the versions that produced the
results. Do not edit them.

## Studies

| Study | Run it with | Summary results | Raw outputs (GitHub release) |
|---|---|---|---|
| Exploratory validation | Workflow `exploratory-validation.yml` (probe: `exploratory-validation-probe.yml`); one cell locally: `python examples/exploratory_validation.py run --engine ols --n 300 --shape linear --reps 0:200 --out out` | `examples/results/2026-10-19-exploratory-validation/` | `exploratory-validation-2026-10-19-39184696821` |
| Phase A (linear reference) | `python examples/phase_a_linear_reference.py --help` | `examples/results/2026-10-17-phase-a-linear-reference/` | `phase-a-linear-reference-2026-10-17-local` |
| Phase A′ (OLS small-sample calibration) | Workflow `phase-a-prime-small-sample.yml` (harness `examples/phase_a_prime_actions.py`) | `examples/results/2026-10-17-phase-a-prime-small-sample/` | `phase-a-prime-small-sample-2026-10-17-39105797558` |
| Phase B (engine p-values) | Workflows `phase-b-check.yml`, then `phase-b-local-detector.yml` (harness `examples/phase_b_actions.py`) | `examples/results/2026-10-18-phase-b-local-detector/` | `phase-b-local-detector-2026-10-18-39144406136` |

Each results folder contains its protocol reference, frozen manifest hash, a
verification record and an `interpretation.md`.

## Release checksums (SHA-256)

| Release asset | SHA-256 |
|---|---|
| `exploratory-validation-2026-10-19-39184696821-raw.zip` | `fc036001b3afe656aa3e857c6cbc4da99c6862519969fcd17f3093413b26d55b` |
| `phase-b-local-detector-2026-10-18-39144406136-raw.zip` | `7eabf07b8fbdc16931cea85ef91228594cbccb46dcaabf207b924a8e275e5793` |
| `phase-a-prime-pvalues.zip` | `40129d65aa9bd4ed04fedcb135efeaf40049530c2eaee7395b59289fdbb526e3` |
| `phase-a-pvalues.zip` | `a87a177282bc0493879907e35f874a60d5a53cbe1199043a0de5bdf448454744` |

## Notes

- The workflows were written for the development repository. Their `push`
  triggers name development branches, so in this repository run them with
  **workflow_dispatch**.
- Commit SHAs and run IDs in the results files refer to the development
  repository [imh-ds/open-archive](https://github.com/imh-ds/open-archive).
  See [provenance.md](provenance.md).
