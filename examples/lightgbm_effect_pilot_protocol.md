# LightGBM moderate-effect recursive pilot

Owner approved the N=200 design and requested the matching N=300 pilot on
2026-10-16. Apply the same 2,000 runner-minute ceiling per sample size (4,000
total), 30 analyses per wave, first attempts only, no automatic fit retries.
This is exploratory development, not scientific qualification or release approval.

## Frozen design

For each r in {.20,.30,.40}, generate ten independent datasets. Independent
standard-normal X1,e2,ey,Z1,Z2 produce X2=r*X1+sqrt(1-r²)*e2 and
Y=r*X2+sqrt(1-r²)*ey. The two true population links each have correlation r;
their simple oracle explained variances are 4%,9%,16%. X1's marginal correlation
with Y is r²; it must be discovered through recursion rather than required as a
direct Y predictor. Expected network/depths: X1(2) -> X2(1) -> Y(0), with Z1/Z2
unrelated. These are predictive direction/reconstruction endpoints, not causal identification.

Pair N across conditions: generate a five-by-300 array once per effect/repetition;
use the first 200 observations for N=200 and all 300 for N=300. Distinct
effect/repetition conditions have distinct generator seeds. Generator seed:
3,400,000,000+1,000*effect_percent+repetition. Analysis seed:
3,500,000,000+100,000*N+10,000*effect_percent+1,000*repetition.
Check new identities against saved prior study manifests before dispatch. Analysis
seed blocks are spaced by at least 1,000; only generator seeds are paired across N.

Use the installed production package, Python3.11.9, pinned historical study
dependencies, fixed LightGBM/effective shipped grid, 100 trees, 100 80%-subsamples,
performance CV5x2, tuning CV3, retention .70 and other shipped thresholds,
maximum depth3, four assessment workers and one numerical thread per worker.
Do not change production behavior to improve this pilot.

## Outputs and limits

Report each strength/sample-size condition separately: ten planned identities,
complete/unknown cases, direct-link and indirect-link recovery, exact-network and
depth recovery, extra-finding networks, missing and extra edges, shared groups,
warning counts, observed runtime and fit counts. Keep incomplete/missing/audit-failed
cases in planned denominators. Do not count an untested deeper edge as a tested
negative. Raw candidate evidence enables descriptive .70–.90 snapshots in .02
increments at already-tested targets only; these are not alternative full-recursive
runs or evidence of a validated replacement cutoff. Do not pool them with the
historical strong-chain qualification. With ten datasets, estimates are rough;
zero observed errors is not proof of error control.

## Budget and preservation

Per wave: 30 case jobs capped at60 minutes each, prepare30, aggregate90,
publish60, registration1 =1,981 reserved runner-minutes, below2,000. Maximum20
parallel case jobs. Serialize the two waves with a shared concurrency group;
runner-minutes are summed across jobs, not wall time. Reserve10 minutes within
each case cap for packing/upload; stop starting assessments with five additional
minutes of batch grace before the bounded fit-step timeout. A separate always-run
preservation step packs interrupted checkpoints before upload; record deadline
failures without retry. An externally cancelled runner can still prevent upload.

Preparation freezes normalized source hashes, grids, exact config/input hashes,
seeds and runtime. Each case preserves checkpointed production assessments and
sealed receipts; aggregation uses the existing independent no-fit tree auditor
to replay losses, splits, parameters, fit counts and network decisions. Keep
run/source/attempt provenance and explicit audit failures. Publish reports,
manifest, protocol and all available raw archives to a new durable prerelease
per N/run; verify remote assets by download and SHA256. Keep artifacts even if
some analyses fail. A run is launch-complete only after its source/input identity
and matrix are checked; study findings await completed execution and audit.

## Implementation and verification steps

1. Add frozen generation/controls and tests for pairing, config, tampering,
   budgets, no-retry behavior and unknown denominators; observe tests fail first.
2. Add checkpoint-runner orchestration, no-fit audit, aggregation, durable evidence
   publication and a manually dispatched workflow; verify targeted controls and
   historical helper regressions without scientific model fitting locally.
3. Commit tools/protocol, then workflow, in meaningful steps. Transfer only public
   files to the Actions branch; preserve private outline edits.
4. Prepare and preserve both manifests/seed audit, dispatch each wave once,
   verify run SHA/inputs/attempt/matrix, and save launch provenance.
