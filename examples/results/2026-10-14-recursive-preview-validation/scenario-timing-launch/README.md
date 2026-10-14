# Four-condition recursive timing launch

The owner approved implementation and launch of the [revised execution plan](../../../validation_execution_revision.md). This is a timing study, not scientific qualification or a release announcement.

Four original conditions: noise N80/N200 and correlated direct predictors N80/N200, indices0/80/320/400. One original dataset/job; four jobs may run concurrently. Each60-minute job includes a50-minute computation/audit step and a48-minute case soft scheduling deadline, leaving preservation time. Registration1+fourjobs240+no-fitaggregation15=256 reserved runner-minutes, reallocated inside the existing10000 ceiling. No further study stage follows automatically.

The independent review found and the regression-first fix pass addressed three problems: regression-test fitting boundaries, interrupted ZIP preservation, and continuing other audits after one damaged case. The new test module now forbids real fits by default. Pack validates full byte inventory/CRC and atomically publishes an archive; aggregate retains every planned row and fails overall on any integrity error while continuing other available audits. Both previously completed chains are audited/reused without refitting.

During development, one initially unsafe regression test briefly entered real local fitting before interruption. Its invalid test-source/test-run attempt2 output is quarantined in ignored development storage, excluded from all study results and never reused. The saved incident summary records8completed assessments,12starts/8finishes,720known completed outer fits/30000tuning fits and4unknown started costs. No complete network was produced, and no Actions runner-minutes were consumed. No observed local selection result was used to change a case, seed, setting or criterion. The approved Actions identities remain unchanged; the invalid diagnostic is not an additional independent replication or a producer-valid scientific result. This history is retained rather than asserting zero local fits.

Source/runtime, software checks, original raw evidence hashes and remote Git blob verification must pass before dispatch. Run/source/event/attempt and startup receipts will be added after the once-only dispatch. Timing outcomes remain pending until independently audited and saved.

## Launched once

[Actions run38645875109](https://github.com/imh-ds/open/actions/runs/38645875109), workflow_dispatch attempt1, sourcef1c1e3851bc19504a3ff6131c2f6bc3524ce5450. Registration38645814299succeeded. All63remote source/manifest Git blobs match;50focused tests passed114.77sec and pinned installed-source check passed. The four timing jobs started concurrently. Startup/source/API receipts are retained here; completion and upload have not yet been accepted. No full-study stage launched.

Startup verified: all4/4jobs passed the pinned runtime/installed-source checks and entered computation. See startup-verification.json and startup-jobs.json. These are startup receipts; final networks, timings, integrity audit and uploads remain pending.
