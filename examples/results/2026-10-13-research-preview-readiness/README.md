# Research-preview release-readiness audit

Status: installed software checks passed; scientific confirmation and release publication remain open. The models, thresholds and recursive rules are unchanged. No release tag, version bump, main-branch merge or PyPI upload has occurred.

## What passed

The development wheel (`0.1.0.dev0`, initial source f736222) built, includes the repository GPL license, and installed non-editably into a new Python 3.11.9 environment with exactly pinned dependencies. `pip check` found no broken requirements. Installed imports were verified under `site-packages` with Python isolated mode, rather than from this checkout's source.

81 fresh focused source tests passed (four known missing-font warnings); these are distinct from the installed-wheel checks. The previous full-suite evidence remains 725 passed; no fresh full-suite claim is made because production code was not changed.

|Installed check|Result|Meaning|
|---|---|---|
|Historical chain|40/40 assessments complete; Anxiety depth0, Depression depth1, Stress depth2|Recursive expansion and original fixture structure reproduced|
|Same chain, missing Depression row|40/40 complete; same depths; missing targets excluded|The previously troublesome missing-target recursion works|
|Near-copy scales|20/20 complete; both depth1 in one joint relation; no unique member arrows|Redundant predictors retain joint representation|
|Direct-predictor researcher report|24/24 people held out once; two evaluation folds; effects and three plots exported|Installed report/export workflow works; descriptive and held-out evidence remain distinct|

There were zero recorded assessment warnings in the three fixtures using 20 assessments per target (100 total). Fixed fixture success is regression evidence, not independent scientific validation or an estimated error rate. The report's held-out risk arithmetic and partition membership are retained in its saved evaluation.

The first audit recorder failed to serialize a read-only mapping after all workflow/export assertions completed. The package did not fail. Initial logs are retained; the recorder now uses the existing portable serializer. The [no-fit completion audit](audit_installed_outputs.py) rechecks the complete exports and can be rerun without repeating model fitting. Do not interpret its zero audit fits as zero fits in the original regression executions. Assessment warnings are preserved in the exported ledgers; original wrapper-only plotting warnings were not separately retained by the failed recorder.

The first wheel-build attempt also hit a sandbox-only temporary-file rename denial; the workspace-temp build succeeded without code changes. Both logs are saved.

## Runtime and remaining choices

The owner subsequently selected shipped defaults with a strict compute budget. The earlier demonstration profile is not the validation target. A [manual-only full-default pilot](../../default_preview_pilot.md) is prepared, with a proposed 560 runner-minute ceiling and no dispatch yet.

A [diagnostic replay](assessment_timing_check.py) of the same two B=1 local assessments without allocation tracing took [17.795 seconds](default-timing/no-tracing-diagnostic.json). Both complete local payloads matched the saved traced run exactly. This identifies substantial memory-telemetry overhead; production behavior is unchanged and the proposed pilot retains it. This is not a full 100-resample timing or a scientific confirmation result.

The historical 20-assessment checks took 316.3 seconds (chain), 316.9 seconds (missing chain), and 168.8 seconds (near-copy scales), measured while the independent checks ran concurrently. These timings are not a worst-case bound.

A separate default-settings timing probe used the same 80-row chain, default performance CV5 repeated twice/inner CV3, with one effective assessment per target. It completed in 85.46 seconds. Multiplying by the configured 100 assessments gives a rough142.4 minutes, assuming unchanged expansion and per-assessment cost. This is not a measured full-budget run, a reliable timing bound, or a scientific result. See [timing/projection and limits](default-timing/timing.json).

This runtime difference is why the confirmation configuration and compute budget must be chosen explicitly. Testing a smaller reference profile cannot certify the shipped default profile. A fresh operating-characteristic study is proposed separately; none has been generated or dispatched here.

The public default branch is still at the earlier e1f55f8 revision, whereas the audited implementation/results are on codex/actions-selection-reliability. Release integration must use the reviewed implementation; this audit has not merged branches.

## Reproduction and evidence

[Scope and gates](../../../docs/research-preview-readiness.md), [installed verification](installed-verification.json), [wheel contents/hash](distribution/wheel-verification.json), [dependency lock](distribution/requirements.lock), [source tests](distribution/focused-tests.log), and [public branch status](public-branch-status.json).

Use a clean virtual environment, install the saved wheel plus the exact dependency lock from an available wheelhouse, and run `installed_workflow_check.py CASE --root REPO --out NEW_DIRECTORY` with that environment's `python -I`. Run from outside `src`; never install editable for this check. Cases are chain, chain_missing, shared_predictors and direct_report. To audit existing outputs with zero new fits, run `audit_installed_outputs.py CASE --root REPO --out SAVED_CASE_DIRECTORY` under the same installed environment. Original data/config bytes are saved per case.

[Chain graph](installed/chain/network.png), [joint-scale graph](installed/shared_predictors/network.png), and [researcher report and plots](installed/direct_report/report/REPORT.md). The chain graph was visually inspected: all three nodes, both 20/20 links and depth labels are legible. Numerical magnitudes and layer frequencies retain their stated predictive interpretation.

The documentation-updated candidate was rebuilt at b6734e3 and reinstalled non-editably. All40 `open/` members are byte-identical to the originally tested wheel; only dist-info METADATA/RECORD changed. [Final candidate proof](distribution/final-candidate-verification.json) records both hashes. The candidate remains a development build and has not been released.
