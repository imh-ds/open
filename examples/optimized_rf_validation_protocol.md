# Fresh optimized RF recursive validation

Owner approved up to **300 additional runner-minutes** for two fresh RF cases.
Reserve 286: registration 1 + preparation 15 + two case jobs at 120 each +
independent audit 30. Leave 14 unallocated; prior approvals remain separate.
Run at most two case jobs concurrently, with four assessment workers per job
and one numerical thread per worker. Compute step cap 108 minutes; soft
assessment deadline 105 minutes. No automatic fit retry or replacement seeds.

Use original chain fixtures at indices 160 and 240 (N=80 and N=200, repetition 0),
with new `optimized-` case IDs. Preserve original generator/analysis seeds,
rows, shipped thresholds, 100 resamples, 80% subsampling, performance CV of
five folds repeated twice, three inner folds, 100 RF trees and ablation policy.
The known chain is X1 -> X2 -> Y, with X2=.6 X1+.8 e2 and Y=.8 X2+.6 ey;
X1/e2/ey/Z1/Z2 are independent standard-normal draws, as in the frozen parent.
The production change removes only equivalent singleton RF grid candidates.
Both cases start from empty directories; no old checkpoint is reused.

Freeze current source, runtime, case/config/data identities, workflow, budget,
explicit `rf-singleton-v1` audit policy, historical audit, and reference archive
SHA256/size. Authorize exactly these two cases on the first dispatch/attempt.
The new auditor independently expects two candidates for a tuned singleton,
four for a tuned multivariable RF, and zero tuning fits for baselines or
parameter-reuse ablations. Require effective tuned parameters; parameter-reuse
ablations may retain a full-model feature fraction of 1.0. Reject historical
four-candidate singleton counts under this policy. Keep the historical auditor
and manifests unchanged and restore temporary audit-policy hooks on every exit.

Independently replay observed losses, split/input identities, every fitted
parameter, selected graph/depth decisions, attempt coverage, and fit counts.
Hash-check the two original completed RF archives against the preserved audited
reference before fitting. After auditing fresh output, compare its network and
all checkpoint risk records with the original reference. Omit only the independently
checked `tuning_model_fit_count` field from that comparison. Keep parameters,
inputs, frequencies and discrete decisions exact; use the existing frozen
derived-float tolerance (rtol 1e-12, atol 1e-14). Runtime/Git/memory metadata are
reported separately. No fitting occurs in the audit or comparison.

Retain all two planned identities, including incomplete/error outcomes. Save
per-case audit progress and preserve checkpoints after timeouts. Successful
execution, exact known-chain recovery and original-result equivalence are
separate endpoints. A failed endpoint stops for interpretation, without rerun.
Compare actual fit counts, compute times and archive sizes with the original
RF cases. Two timing/equivalence fixtures do not establish recovery rates,
false-positive rates or general sample-size requirements. No larger study is
authorized or dispatched automatically.

Upload complete raw archives, controls, receipts, pinned runtime, loss records,
assessment ledgers, audit, comparison and interpretation. Preserve them in the
repository after completion, with hashes, beyond Actions artifact expiration.
Local preparation/tests may inspect or replay saved records but must not run
new scientific fits. Larger qualification and historical backup backlog remain
separate pending tasks.
