# Running the frozen selection study on GitHub Actions

The manually runnable **Selection reliability study** workflow runs one frozen
dataset per job, up to twelve jobs concurrently. Each dataset uses four workers
for its independent local assessments. Models and numeric libraries remain
single-threaded. Statistical settings, 48 dataset identities, 100 assessments
per dataset, and the 70% retention rule remain unchanged.

The synthetic transfer archive preserves the interrupted local run, including
completed datasets and atomic assessment checkpoints. It also preserves exact
source bytes: Windows checkout line-ending conversions must not silently
invalidate source fingerprints. Python and numerical dependencies are pinned
to the local study versions. Every reused checkpoint is validated against its
data, settings, production sources, and planned sample/splits.

Each dataset is independently audited before its job succeeds. All jobs preserve
their evidence and checkpoints as artifacts even on failure. The final job
combines the case artifacts, requires all planned cases, runs the existing
independent audit, and renders the report. A missing case fails aggregation;
it is never removed from the study. `fail-fast: false` preserves other jobs.

Each case records its execution platform, package versions, GitHub run/commit,
checkpoint reuse/computation counts, and elapsed time. These results combine
local and hosted execution and are not a pristine runtime benchmark. Identical
settings and seeds do not promise bit-for-bit equality across platforms.

To prepare the current local evidence for transfer:

```powershell
python examples/selection_actions.py pack
```

Commit the resulting synthetic archive together with the tested workflow.
Its initial push to `codex/actions-selection-reliability` triggers the workflow; later runs can
use manual dispatch after the workflow is registered. No merge into `main`
is required to execute the branch's push workflow. The final artifact is
`selection-study-audited-results`; case artifacts include restart checkpoints.
If a job is interrupted, retain its artifact and incorporate its checkpoints
into a new validated transfer before rerunning. The committed transfer remains
an initial checkpoint snapshot, not a claim that the study has completed.

The Actions branch exports the public implementation onto the published base.
Internal decision-log revisions stay in the local development branch rather
than being added to the public export history.
