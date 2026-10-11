# Preserved pilot evidence

The twelve case records, 1,616 assessment checkpoints, model evidence and test
predictions are preserved in `actions-evidence.zip` (synthetic data only).
The archive includes the original Actions report and audit.

Extract the archive into a new directory, then run
`python examples/sample_size_pilot.py audit --output <extracted-directory>`
from the repository with the frozen source and pinned study dependencies.
The manifest rejects changed production/study sources or settings.
The original run used public source commit `dd8a176`.

`python render_closeout.py <extracted-directory>` regenerates the interpreted
report and figure in that extracted directory. It fits no models or changes
no study settings. Use Matplotlib 3.10.7 for the original plotting environment.

See `execution.json` for the archive checksum and verification provenance.
The study has one dataset per condition and cannot establish a minimum N.
