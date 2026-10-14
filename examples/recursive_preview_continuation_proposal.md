# Proposed no-refit continuation after the preflight checker failure

Status: approved by the owner's “Proceed” on2026-10-14; implementation and launch verification in progress. The approved scientific design,480identities,100resamples/CV5x2/inner3/defaultelasticnet/.70retention and .10/.80reference criteria remain unchanged. The original failed run is preserved as failed.

## Why an execution amendment is needed

The original preflight finished200/200fixed-fixture assessments but failed a bit-identical layer-hash comparison on573derived float fields; maximum difference2.220446049250313e-16. Exact sample/split/graph/decision/parameter-field comparisons and independent replay of140761observation losses pass. See the [saved diagnosis](results/2026-10-14-recursive-preview-validation/preflight-failure-38615814397/interpretation.md).

The existing protocol says that failed equivalence must stop execution and receive a separately reviewed change, and that the129unallocated minutes do not automatically authorize a rerun. This proposal uses that explicit review gate. It does not silently loosen scientific cutoffs or relabel old evidence.

## Exact proposed continuation

1. Preserve the old source/config/manifest, failed-control receipt and exact raw archive/fixture receipt fingerprints. Publish those bytes and the independently reproduced diagnosis before dispatch.
2. Freeze the corrected numerical checker and new continuation source separately. Only derived floating-point comparison permits rtol1e-12/atol1e-14; observed data, parameters, sample/splits, config, counts and graph/selection decisions stay exact. Raw values stay unchanged. This is an execution-equivalence correction, not a selection-threshold change.
3. Download the original successful raw artifact from run38615814397 and verify its original fingerprints. Re-audit and compare the already-computed fixture under its original producer manifest. Do not recompute any of its200assessments. Verify that production-package sources match between producer and continuation; any difference stops reuse.
4. Execute the originally planned cases160(N80) and240(N200) once each using unchanged approved settings. Neither has been fitted yet. Each must finish within its original15-minute computation allocation; all original risk, storage, audit and transport gates apply.
5. Bound the new continuation computing job to45minutes, including setup, no-fit original-fixture verification, at most30minutes for the two new cases, audit and preservation. Allow one no-fit workflow-registration job capped at1minute. No automatic retries, expanded sample sizes, replacement IDs, lower folds/resamples or new engine selection.
6. Preserve old and new attempts/source/runtime/costs separately. The continuation receives a new run ID/attempt1 and an explicit link to the failed producer; the original run is never called successful. Accept preflight continuation only after all checks and both uploads succeed.
7. StageA remains80two-case40-minute shards, StageB160two-case40-minute shards. StageB reuses the exact completed160/240outputs with zero new fits and requires the original independently audited/preserved/committed null selection.

## Budget and stop condition

Retain all original9871allocated runner-minutes, even though the first preflight used only about7.5actual job minutes. Add45minutes for continuation plus1registration minute: **9917total allocated runner-minutes**, leaving83within the owner-approved10000ceiling. The owner's approval explicitly allocates46of the previously unallocated129minutes to this continuation.

If the corrected no-fit equivalence check, case15-minute timing, source fingerprints, raw replay or preservation fail, stop again without new fits/retries or scientific qualification. Passing this continuation would permit the already approved noise stage; it would not itself establish scientific accuracy or authorize a release publication.
