# Preserved investigation evidence

`actions-evidence.zip` contains the complete downloaded final artifact from
[Actions run 38116199436](https://github.com/imh-ds/open/actions/runs/38116199436):
74 diagnostic results, 64 model prediction sets and fold records, ten LOCO risk
replays, the completed chain's 300 checkpoints and graph, and the original pilot
evidence used for comparison. Synthetic data only. This is a local repack of the
downloaded files, not GitHub's original artifact ZIP byte stream.

All 96 jobs succeeded. The local offline audit returned 74/74 diagnostics,
completed continuation, zero audit errors and zero warnings. Its results and
report exactly matched the original Actions audit and report.

Extract the archive into a new directory and run:

```sh
python examples/selection_investigation.py audit --output <extracted-directory>
```

Use frozen public source `7358226a804efd359ed7043836cc3eb51046785f`, Python 3.11.9,
NumPy 1.26.4, pandas 2.2.3, SciPy 1.11.2 and scikit-learn 1.3.0. The manifest,
source hashes and protocol reject incompatible changes. The auditor refits saved
model settings but does not independently repeat their tuning searches.

`python render_closeout.py <extracted-directory>` regenerates the interpreted
report and figure in this folder without fitting models. The figure uses
Matplotlib 3.10.7. `REPORT-from-actions.md` preserves the complete original table;
`execution.json` retains run/job metadata, the archive checksum and audit scope.

The run used a peak of 20 concurrent jobs and about 48 minutes of wall time.
Summed job duration includes setup, fitting, transfer and auditing, but excludes
queue time. These are operational observations, not a controlled runtime benchmark.
Prediction rows repeat participants across model comparisons; they are not
independent population replications.

The historical pilot remains 11/12 complete in its own report. This follow-up
successfully completed the previously interrupted case without changing that
historical denominator. No production defaults, reliability recommendation or
minimum sample size follows from these development results.
