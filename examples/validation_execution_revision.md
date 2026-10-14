# Revised recursive-validation execution plan

2026-10-14. **Planning revision requested by the owner after the completed timing probe.** This document proposes the next concrete allocation and implementation steps. It does not authorize a dispatch, refreeze the scientific study, or amend a historical run's outcome. The old source manifests and protocols remain unchanged.

## What changes, and why

The [completed chain timing probe](results/2026-10-14-recursive-preview-validation/timing-completed-38629900589/interpretation.md) needed 27m22s for N80 cumulatively and 27m24s for N200 uninterrupted. Two such cases require about 55 computation minutes, exceeding the original 40-minute two-case job. This is an execution-budget failure, not evidence that recursion failed: both completed chains were correct.

Replace two-case jobs with **one original dataset per job**, with scenario-specific time limits selected after measuring the other scenarios. Keep four single-threaded assessment processes per dataset and at most 20 concurrent jobs. Splitting jobs reduces the chance that a slow case prevents another from starting; it does not reduce total runner-minutes. The existing two-case workflow is historical and must not be dispatched as this revised plan.

Keep the scientific target at **480 datasets, 80 per scenario/N cell**, pending the feasibility decision below. Preserve the generator, original identities/seeds, B100, performance CV5x2, tuning CV3, retention .70, all other defaults, and the original statistical criteria. Do not edit the old manifest in place or silently narrow its denominators.

## Next measured step: four missing timing conditions

The two chain conditions are already measured. Propose running only the first original identity in each remaining condition, once:

|Original index|Original ID|Condition|Job cap|
|---:|---|---|---:|
|0|noise-n80-r00|Noise, N80|60 min|
|80|noise-n200-r00|Noise, N200|60 min|
|320|correlated-n80-r00|Correlated direct predictors, N80|60 min|
|400|correlated-n200-r00|Correlated direct predictors, N200|60 min|

Use a separate manual-only timing workflow, four independent jobs, the same pinned Windows/Python runtime, and the existing production traversal/checkpoint adapter. Each 60-minute job reserves up to 3 minutes for install/source checks, 50 for computation plus case audit, 2 for checkpoint packing and 4 for upload, leaving 1 minute of job headroom. No per-dataset 15-minute cutoff applies. Stop scheduling before the computation step expires, retain attempts/checkpoints on interruption, and classify hard-killed work conservatively as unknown. Evidence preservation is attempted on failure but cannot be guaranteed after a hard kill.

Reserve a separate 15-minute no-fit aggregation job and a 1-minute no-fit workflow registration. Total new reservation: **4 x 60 + 15 + 1 = 256 runner-minutes**. Four parallel jobs could shorten waiting; actual elapsed time depends on runner availability. These caps are stop limits, not a promise that these previously unmeasured conditions will finish.

Freeze this new runner/workflow/protocol in a separate manifest, with exact original source/config/480-identity references and producer-bound proof for the two saved chain cases. Do not replay those chains or the historical fixed fixture. No timing outcome can change relevance/retention settings or exclude a scenario. Audit and save every timing result, including incorrect structures, warnings and failures.

The four cases remain their original study identities, not extra replications. Reuse any completed, independently audited output with zero new fits if the 480-case study proceeds. Noise-stage inference must wait for all 80 planned identities in each noise cell; the two early correlated cases do not bypass the null-first signal-stage gate. Timing is the sole purpose of their early execution, as with the original chain preflight cases. No automatic retry or broader stage follows this probe.

## Reallocation within the same 10,000-minute ceiling

The earlier ledger reserved all future jobs, so it has zero unallocated minutes. Adding 256 minutes on top would exceed the ceiling. Instead, the proposed revision **withdraws the unlaunched stage reservations and replaces them**. Retain the full historical caps, without refunds for jobs finishing early:

|Reservation|Runner-minutes|
|---|---:|
|Historical first registration and preflight|1 + 120 = 121|
|Historical continuation registration and job|1 + 45 = 46|
|Historical chain timing registration and job|1 + 82 = 83|
|Historical allocations retained|**250**|
|Proposed four-case timing, registration and aggregation|**256**|
|Future noise/signal preparation and aggregation, retained provisionally|**150**|
|Remainder available for a redesigned study execution allocation|**9,344**|
|Total ceiling|**10,000**|

The replaced original future reservations were 9,600 shard minutes plus 150 preparation/aggregation minutes, or 9,750. Their replacement is 256 + 150 + 9,344 = 9,750. The remainder is **not permission to dispatch unspecified jobs**. Any further workflow registration, verification or preservation job must fit inside it. Actual usage must also be recorded separately from reserved caps.

## Feasibility decision after the four cases

Before fitting any other dataset, retain a table by scenario/N of elapsed computation, audit/pack/upload time, recursive targets, assessment/fit counts, warnings and archive bytes. The chain measurement is not a cost estimate for every scenario. One observation per other condition is likewise not a worst-case bound.

For a candidate single-case job cap C in each cell, calculate the full reservation as:

`250 + 256 + 150 + additional_control_minutes + sum(remaining_cases_in_cell * C_cell)`

For the original 80-per-cell design, each cell has 79 remaining fits after its first case has been audited: four newly timed cases and the two saved chains cover six identities. Thus 474 identities remain. Every C must include setup, audit and preservation, with explicit headroom above the observed runtime. Freeze the exact caps and their rationale before launching; do not assume 474 average-duration jobs constitute a hard ceiling.

The mean observed chain case time is 27.3854 minutes. The remaining 158 chain cases alone would project about 4,327 computation minutes, before their setup and preservation. This is a two-case projection, not a runtime guarantee. For comparison, uniform 40-minute single-case caps would reserve 18,960 minutes for the remaining 474 jobs alone and therefore cannot fit the ceiling.

If all 480 identities can fit a defensible cap allocation, retain the original scientific design. Otherwise, present a narrower prospective design with exact new independent-dataset counts, confidence-bound calculations and claim limits before further fitting. Decide scope from execution feasibility, not observed recovery or incorrect-link rates. Do not drop inconvenient cases or rescue a failed criterion. If the design changes, explicitly preserve the original incomplete 480-case study and record which pilot outputs are reused or descriptive; do not rewrite its history. A new smaller study must have its own frozen identity list and denominators.

Fewer datasets mean less precise reliability estimates. They are not a reason to lower CV/resamples or silently loosen acceptance criteria. If neither defensible full nor narrower validation fits, stop and report the limit rather than treating two correct chains as scientific qualification.

## Implementation and acceptance checklist

- [x] Preserve the completed chain timing/audit and original failed preflights.
- [x] Document the one-case layout, missing timing conditions and replacement budget.
- [ ] Review this concrete next-run allocation before new scientific fits or dispatch.
- [ ] Implement a separate timing adapter/workflow without editing frozen historical sources; pin source/runtime/defaults and four original identities.
- [ ] Verify fail-closed source/identity checks, bounded attempt handling, exact saved-chain reuse, no duplicate fitting, negative/incomplete result preservation and no-fit aggregation with focused regression tests. Verify all source/input blobs before dispatch.
- [ ] Run the four approved timing cases once on Actions; audit and durably save raw results, hashes and simple interpretation.
- [ ] Freeze a costed final study plan or narrower prospective design before any other datasets run. Retain null-first qualification and independent integrity audit.

This revision changes execution planning only. Scientific qualification and research-preview release readiness remain open. No release publication, new engine, threshold adjustment or automatic follow-on dispatch is included.
