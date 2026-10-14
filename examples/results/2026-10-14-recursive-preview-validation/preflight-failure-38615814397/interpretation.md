# Preflight failure: numerical-comparison checker

Original run [38615814397](https://github.com/imh-ds/open/actions/runs/38615814397), source `ddc6df755305fef39898ae752262276784042cde`, failed its original strict comparison. That original result remains **failed**; this diagnostic does not retrospectively qualify it.

## Simple interpretation

The engine completed the chain and found exactly the same network. The checking script then rejected extremely small floating-point differences as though they were meaningful calculation changes. This was a checker defect, not a timeout or evidence that recursion stopped working.

- All200 planned fixture assessments completed, with0warnings and known fit costs.
- Both runs have `Stress -> Depression -> Anxiety`, depths2/1/0, identical samples/splits, classes, decisions, counts and all parameter fields actually retained in the old comparison.
- All573 differing fields were floating-point derived statistics. The largest absolute difference was `2.220446049250313e-16`; for example `.40676605290832350` versus `.40676605290832346`. This is consistent with ordinary floating-point roundoff. The physical cause (such as processor or reduction-order differences) is not established by this audit.
- Independent no-fit replay passed140,761observation-loss checks and all saved risk/contrast/identity checks. It recorded7000outer fits and375000tuning fits from the original computation; the diagnostic performed0new fits.
- The fixture fit took363.469seconds (about6.1minutes), compared with144.7minutes in the historical sequential pilot. This is encouraging execution evidence on one fixed fixture, not a measured runtime for the still-unrun proposed datasets.
- The whole preflight job ran448seconds (about7.5minutes), far below its120-minute cap. Both artifact uploads succeeded.
- The planned independent N80/N200 cases160/240 were never started; `planned_cases` is empty. No scientific dataset identity has been consumed by a model fit, and no noise/signal stage launched.

## Correction and remaining gate

The checker now compares derived floats at relative tolerance1e-12 and absolute tolerance1e-14, without rounding or changing the saved values. Configuration, sample/split identity, graph construction, decisions, counts, data observations, frequencies and available fitted-parameter fields must still match exactly. Material numeric changes fail. Focused regression checks cover tiny roundoff, genuine numeric changes, changed discrete decisions, parameter/data/frequency changes, missing keys, type changes and nonfinite values.

The saved fixture passes this corrected numerical diagnostic with573changed floats and0rejected differences. The complete raw observation audit passes independently. The original fixture did not save full/broad prediction objects; the comparison covers every field actually saved, while the new run's captured predictions are audited independently. It does not claim recovery of unavailable original predictions or model settings.

The original producer manifest is retained byte-for-byte in `raw/producer-manifest.json`. It has not been rewritten to bless the patched checker. A revised execution freeze and specifically approved continuation are required before more fitting. The [continuation proposal](../../../recursive_preview_continuation_proposal.md) reuses this completed fixture with zero new fixture fits, preserves all scientific settings/IDs, and requests an additional45-minute computing-job cap plus1registration minute. Including all original allocated caps, the combined ceiling would be9917runner-minutes, within10000. No continuation has been dispatched.

## Durable evidence and reproduction

`raw/` retains the exact original604-file checkpoint/network archive, control receipt, runtime lock, complete logs, run/jobs API responses and producer manifest. `audit/diagnosis.json` contains the numerical comparison and independent audit; `audit/raw-sha256.json` hashes each original retained file. Original raw bytes are protected from Git newline conversion.

Run the no-fit diagnostic with the installed package and the repository examples directory available:

```powershell
python examples/audit_recursive_preflight_failure.py --evidence examples/results/2026-10-14-recursive-preview-validation/preflight-failure-38615814397/raw --out NEW_DIRECTORY
```

This reproduces the diagnosis, not the failed model run. No scientific operating-characteristic or release gate is accepted by this diagnostic.
