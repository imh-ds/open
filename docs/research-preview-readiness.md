# OPEN research-preview release readiness

The target is a v0.1 **research preview** of the current method. This document freezes the intended claim scope; it does not announce a release. The current package version remains `0.1.0.dev0`.

## Intended scope

- Independent cross-sectional participants, one fully observed continuous root outcome and MSE loss.
- Numeric scale-level predictors. Numeric ordinal scores are treated as continuous numbers; ordinal item models and their methodological validity are not established.
- Layer-consensus recursion, with a configurable retention threshold (default 70%). Frequencies describe selection conditional on earlier finalized layers, not confidence or causal paths.
- Elastic net is the default reference learner. RF, GBM and optional LightGBM retain their implemented APIs, with engine-specific evidence limits; simulation findings from one comparator do not certify every engine.
- Predictive networks, original-unit/standardized predictive effects and complete-procedure held-out reports. Descriptive reference graphs and held-out prediction scores answer different questions.
- Optional interaction screens/comparators remain experimental. Later calibrated spline studies cover prespecified pairs and specific simulation conditions; they are separate from automatic recursive network construction and do not establish general per-finding confidence.

Grouped/repeated-measures or temporal data, binary outcomes, causal mediation/moderation, universal sample-size rules, and automatic scientific confidence guarantees are outside this preview's claims. No LM/GAM companion or all-product AutoML screen is added.

## Release gates

|Gate|Current status|Required evidence|
|---|---|---|
|Scope and claims|Defined for review|API/documentation consistency and explicit experimental boundaries|
|Software regression|81 fresh focused tests passed; previous full suite 725 passed|Focused fresh core/report checks; production fixes require fresh full suite|
|Distribution|Initial and documentation-updated wheels passed on pinned Windows/Python 3.11.9 environment|Wheel contents, license/metadata, clean installation, dependency check and outside-source import proof|
|Installed user workflow|Four fixed regression cases passed|Recursive fixtures, exports/reloaded graphs, researcher report and no-overwrite behaviour|
|Scientific operating characteristics|Incomplete|Fresh end-to-end validation of network selection and layer placement; existing fixed demonstrations cannot supply rates|
|Interaction claims|Scoped research evidence only|Keep prespecified-pair benchmark scope distinct from arbitrary pair screening|
|Release publication|Not authorized by this audit|Finish gates, choose release artifact/revision and review actual publication action|

The immediate work uses existing fixed fixtures and unchanged settings. Independent scientific confirmation requires a separately frozen design and approved compute budget. Passing packaging and fixture checks does not accept the scientific gate.

## Existing evidence

[Current interaction findings and limits](../examples/results/2026-10-13-realistic-interaction-recovery/interpretation.md), [prior prospective interaction guidance](../examples/prospective_interaction_guidance.md), [layer-regression demonstrations](../examples/results/2026-10-08-layer-fixtures/REPORT.md), and [complete-procedure report contract](procedure-evaluation.md).

The first three implementation/export paths are substantial software evidence. They do not establish false-selection rates, recursive layer accuracy, or robustness for all practical populations. Release-audit measurements and remaining work will be retained under `examples/results/2026-10-13-research-preview-readiness/`.

The [shipped-default recursive fixture pilot](../examples/results/2026-10-14-default-preview-pilot/interpretation.md) completed all three fixed cases and matched their expected chain, missing-outcome and redundant-predictor structures. It confirms full-default execution and artifact reload on those fixtures; it does not estimate scientific error rates or close the broader validation gate.

[Installed release audit and saved evidence](../examples/results/2026-10-13-research-preview-readiness/README.md) distinguish source tests, clean installed checks, and the remaining scientific/runtime gates. The default-settings runtime projection is not a completed 100-assessment benchmark.
