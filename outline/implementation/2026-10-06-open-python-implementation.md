# OPEN Python Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development or superpowers:executing-plans to implement this plan task by task. Checkboxes track deliverables; the [progress register](OPEN-progress.md) records evidence and gate decisions.

**Goal:** Build a reproducible Python implementation of the Outcome-centered Predictive Expansion Network (OPEN), then evaluate its intended predictive objects before claiming methodological validity.

**Architecture:** A local-target engine estimates held-out risk differences and relationship classes. A layer-synchronous network builder promotes qualifying nodes. Whole-pipeline resampling aggregates runs into a consensus network; export and visualization consume those results without changing them.

**Tech Stack:** Python package under `src/open/`; initial numeric continuous reference path using NumPy, pandas, SciPy, and scikit-learn; pytest for implementation checks. Introduce other libraries only with a recorded decision.

**Spec:** [OPEN technical implementation specification v3](../OPEN_v3_technical_implementation_spec.md). Its numbered sections and ten final algorithm rules are authoritative. This plan fixes implementation order and provisional software choices; record any methodological ambiguity before resolving it in code.

## Global constraints

- An OPEN arrow means predictive contribution under the stated learner, candidate set, loss, splits, and thresholds. It does not establish causal direction (spec §§1, 27, 49).
- Fit imputation, encoding, scaling, dependence estimation, and tuning on training folds only. Use identical held-out folds for baseline, full, single-variable, and group-ablation comparisons (spec §§6–7, 24).
- Preserve raw negative loss differences; clamp them only for visual strength. Keep the specified loss fixed within each local target (spec §§4, 10, 25).
- Test candidate pairs before pruning weak individuals. A group-important pair with one strong member is screening, not symmetric redundancy. Promote every validated shared-group member (spec §§11–17).
- Run all targets at a depth against one snapshot of unassigned variables. Assign the next depth only after the whole current layer completes (spec §§18–20).
- Rebuild preprocessing, dependence groups, classifications, and the full recursive network in every stability resample. Divide structural probabilities by all planned completed runs; separately report how often a target was evaluated (spec §§21–24).
- Treat prototype thresholds as configurable engineering defaults, never calibrated scientific constants (spec §§31, 43–45).
- An accepted software phase means its defined checks passed. It does not imply simulation calibration, applied validity, or support for deferred outcome types.

## Review focus

Each condition has an owning task and a required check.

| Risk | Owning task and check |
|---|---|
| A target total score leaks its own items or a subject ID into predictors | P0.2 excludes the identifier and target items and rejects unsupported item-level mode. |
| Preprocessing or pair selection sees held-out data | P0.3 and P1.3 use a held-out sentinel; changing test-fold values must not change fitted preprocessing or candidate pairs. |
| A target has zero/negative gain and normalized relevance becomes misleading | P1.2 records the low-gain stop reason before classifying or promoting nodes. |
| A pair is group-important but only one member is uniquely strong | P2.1 checks `SCREENED_BY_PARTNER`, never `PAIR_SHARED_REDUNDANT`. |
| A current-layer peer enters another target's candidate set | P3.1 tests both target orders and requires the same candidate snapshot and depths. |

## Scope and phase gates

The first usable path is cross-sectional, numeric scale-level data with a continuous target, one fixed deterministic learner, fold-local single imputation, and `retune_ablation=false`. It follows the spec's staged optimization advice (§47). Binary outcomes, mixed-type dependence plugins, grouped/longitudinal data, multiple imputation, per-target selection, ensembles, and research-scale calibration are separate gates in Phase 6. The initial prototype must reject unsupported modes clearly.

Default configuration comes from spec §§7, 20, 31, 38, 40: `performance_cv.folds=5`, `repeats=2`; `tuning_cv.folds=3`; `max_depth=3`; `tau_gain=.02`, `tau_b=.05`, `tau_u=.05`, `tau_group=.10`, `tau_dep=.70`, `pi_min=.70`; stability subsampling fraction `.80`, development `B=100`, final `B=500`. Small deterministic fixtures may reduce folds and runs explicitly. Calibration may change defaults only through a new logged decision and fresh evidence.

Do not mark a phase accepted until its check list, focused checks, output artifact, and review have been recorded in [OPEN-progress.md](OPEN-progress.md). Work in small commits with task IDs in the message or progress entry. A failed check stays visible with its exact command and result. The implementer can choose normal Python internals where interfaces below leave room; changes to statistical definitions require a decision record.

## File and interface map

These are intended responsibilities, not a requirement to create empty files early. Keep public results as typed dataclasses or equivalent records with stable serialized names.

| Area | Planned files | Required interface / responsibility |
|---|---|---|
| Configuration and input | `src/open/config.py`, `metadata.py` | `ThresholdConfig`; validate root, variable roles/types, exclusions, thresholds, seed, loss, and supported mode. |
| Split and prediction | `src/open/splits.py`, `preprocessing.py`, `learners/base.py`, `learners/elastic_net.py`, `risk.py` | `make_split_plan(data, config) -> SplitPlan`; predictor `tune/fit/predict/clone`; `evaluate_risk(..., predictors, split_plan) -> RiskResult` with held-out losses. |
| Local method | `src/open/relevance.py`, `dependence/spearman.py`, `grouping.py`, `classify.py`, `local_target.py` | `analyze_target(data, target, candidates, config, split_plan) -> LocalResult`; raw and normalized broad/unique/group values, classes, stop reason, promoted set. |
| Network | `src/open/recurse.py`, `graph.py` | `build_network_once(data, root, all_variables, config) -> NetworkRun`; depth, parents, local results, core edges, virtual shared groups. |
| Stability | `src/open/stability.py`, `consensus.py` | `fit_stable_open(data, root, variables, config) -> OPENResult`; run-level records, unconditional structural probabilities, consensus, target evaluation frequencies. |
| Reporting | `src/open/io.py`, `visualization.py`, `diagnostics.py` | Save exact spec §39 outputs, metadata, fit counts; draw the spec §25 graph from saved result objects. |
| Evidence | `tests/`, `simulations/`, `examples/`, `configs/` | Unit checks, synthetic scenarios, benchmark/sensitivity outputs, reproducible applied example. |

An interface change propagates to dependent phase sheets and receives a decision entry. Do not create a second definition of a statistical quantity in visualization or export code.

## Phase 0 — Reproducible project and data contract

**Outcome:** A small continuous-data analysis can be configured, validated, split, and preprocessed without leakage. No network claim yet. Source: spec §§3–7, 30, 37–40.

### P0.1 Package and configuration

- [ ] Create `pyproject.toml`, `src/open/__init__.py`, `src/open/config.py`, `configs/default.yaml`, and a concise usage section in root `README.md`.
- [ ] Parse the spec §40 shape. Validate numeric thresholds in `[0,1]`, positive depth/folds/resample counts, one root, fixed loss, seed, and supported first-path options; reject unknown mode values.
- [ ] Add `tests/test_config.py` for default values, invalid threshold/mode, and configuration round-trip. Record Python/dependency floors as D-001 in the decision log before installing dependencies.

### P0.2 Dataset and metadata contract

- [ ] Create `src/open/metadata.py` and `tests/test_metadata.py`. Return a validated root and ordered candidate names; preserve display labels and the spec §3 metadata fields.
- [ ] Exclude IDs, fold labels, explicit excludes, duplicated columns, direct target derivatives, and target scale items by default. Flag overlapping psychometric totals in output metadata. Prospective mode must reject post-outcome candidates when later added.
- [ ] Check fixtures containing duplicate columns, missing root, and target item overlap. Document the only supported first-path types: numeric continuous or ordinal scale-level predictors and continuous target.

### P0.3 Split plan and fold-local transformations

- [ ] Create `src/open/splits.py`, `preprocessing.py`, `learners/base.py`, `learners/elastic_net.py`, and `tests/test_splits_preprocessing.py`.
- [ ] Fix split indices once per local target/run; seed and serialize them. Fit imputation and scaling inside each training fold. Tune inside training data when tuning is enabled; initial ablations reuse full-model parameters.
- [ ] Prove held-out values cannot alter train-fold transformation parameters. Check that the same seed reproduces splits; unsupported grouped/repeated-measures requests fail clearly until Phase 6.

**Gate P0:** Run `python -m pytest tests/test_config.py tests/test_metadata.py tests/test_splits_preprocessing.py`; record the result. A documented small fixture produces validated metadata and reproducible fold assignments. Log the command, result, commit, and remaining type limits.

## Phase 1 — Held-out local relevance

**Outcome:** A target gets baseline/full gain and paired broad, individual LOCO, and candidate pair-ablation estimates on the same splits. Source: spec §§8–12, 24, 32–34.

### P1.1 Risk engine

- [ ] Create `src/open/risk.py`, `tests/test_risk.py`. For a continuous target, compute held-out MSE and the training-mean baseline; return per-observation losses and fold/aggregate risk.
- [ ] Test the training-mean baseline and common split indices across full and ablated fits. Store learner, loss, split seed, candidate set, and model-fit count.

### P1.2 Relevance quantities and low-gain stop

- [ ] Create `src/open/relevance.py`, `tests/test_relevance.py`. Compute `G=R0-R_full`, `g=G/(R0+eps)`, `B_j=R0-R_j`, `b_j=B_j/(G+eps)`, `U_j=R_-j-R_full`, and `u_j=U_j/(G+eps)` as in §§8–10.
- [ ] If `g < tau_gain`, return `LOW_TOTAL_GAIN` before normalized classes or promotion. Retain signed raw and normalized U where evaluated; clamp only in Phase 5 display.
- [ ] Check exact arithmetic on fixed risk fixtures, including a negative `U_j`; check that a zero/negative gain stops cleanly.

### P1.3 Dependence candidates and pair ablation

- [ ] Create `src/open/dependence/spearman.py`, `src/open/grouping.py`, and `tests/test_dependence.py`. Use absolute Spearman score for supported numeric scale-level predictors and `tau_dep=.70` for candidate generation only.
- [ ] Build candidate pairs separately from each training fold's dependence estimates, never held-out values, before any weak-individual pruning. Evaluate each selected pair against the full model on that fold's same held-out rows. Pool the pair's raw contrast and pair-specific total-gain denominator over only folds where selected, then normalize; retain selected/evaluated fold counts and frequencies and the evaluated row positions. If pair-specific total gain is not positive, retain raw contrast and leave the normalized score undefined.
- [ ] D-002 is resolved. Check that a held-out sentinel cannot alter that fold's pair selection, paired risk rows match exactly, folds without pair selection do not enter its score, and weak individual members still receive a group ablation when selected.

**Gate P1:** The risk and relevance components produce a complete signed ledger for one continuous target; arithmetic and leakage checks pass. Classification remains pending Phase 2. Run `python -m pytest tests/test_risk.py tests/test_relevance.py tests/test_dependence.py`; record the result.

## Phase 2 — Pair, group, and individual classification

**Outcome:** Local targets have the exact relationship classes and promotion set required by the spec. Source: spec §§13–17, 34, 41.2.

### P2.1 Pair decisions

- [ ] Create `src/open/classify.py`, `tests/test_classification.py`. `classify_pairs(broad, unique, pair_group, thresholds)` uses `tau_u`, `tau_b`, and `tau_group` without rounding intermediate scores.
- [ ] Check spec §41.2 fixtures: `(.40,.01,.02)` yields `PAIR_SHARED_REDUNDANT`; `(.40,.25,.01,b_b=.20)` yields proximal `a` and `SCREENED_BY_PARTNER(a)` for `b`; `(.40,.20,.18)` keeps both proximal. A group-weak pair is never called redundant or partner-screened.

### P2.2 Validated higher-order groups

- [ ] Extend `src/open/grouping.py` and `tests/test_grouping.py`. Form connected components only from shared pair edges, group-ablate each component, and validate only if `u_G >= tau_group` and every member has `u_j < tau_u`.
- [ ] If a merged component fails, retain supported shared pairs and promote every member of each retained validated pair; if it has strong members, apply `SCREENED_BY_SET(S)` to qualifying weak broad predictors. Resolve D-003 for overlapping supported pairs and membership representation.
- [ ] Test a valid three-member group, a merged group that fails validation, and a mixed strong/weak component.

### P2.3 Final local result

- [ ] Create `src/open/local_target.py` and `tests/test_local_target.py`; compose Phase 1 results and Phase 2 classifiers into `LocalResult` with the spec §16 precedence and `promoted` set.
- [ ] Promote proximal unique variables and every validated shared-group member. Keep screened, unstable, and negligible candidates unassigned. Compute optional `C_G=U_G-sum(U_j)` as a labeled diagnostic only.
- [ ] Check direct unique, redundant substitutes, and chain local-target fixtures. In the chain, the root must classify the distal candidate as screened rather than redundant.

**Gate P2:** Local-target checks demonstrate all pair branches and promotion behavior. Run `python -m pytest tests/test_classification.py tests/test_grouping.py tests/test_local_target.py`; record the result. Do not infer stability from a single run.

## Phase 3 — Layer-synchronous network expansion

**Outcome:** One complete network run preserves shallowest depth and all qualifying parents without using already assigned variables as core candidates. Source: spec §§18–20, 35, 41.3.

### P3.1 Expansion engine

- [ ] Create `src/open/recurse.py`, `graph.py`, `tests/test_recursion.py`. Initialize `assigned_depth[root]=0`; for each depth, snapshot all unassigned variables once, analyze every target against that same set, then assign new depths together.
- [ ] Retain all qualifying parent edges for nodes found by multiple same-layer targets. Never model a virtual group node. Store shared visual relations separately from unique edges.
- [ ] Check root exclusion, peer exclusion, order independence, shallowest depth, multiple parents, screened candidate eligibility at the next layer, and promotion of every shared member.

### P3.2 Stopping and diagnostics

- [ ] Stop on low gain, no promotion, empty unassigned set, maximum depth, or insufficient effective sample size; keep a reason per evaluated target.
- [ ] Limit early development examples to depth 2 and record when the default depth 3 is enabled. Keep optional within-layer/backward cross-links out of the core graph.
- [ ] Check the synthetic chain `S -> D -> A` yields screened `S` at root and eligible `S` at the `D` target, with predictive depth 2 when thresholds allow.

**Gate P3:** One-run graph, depth table, parent sets, local ledgers, and stop reasons can be inspected and reproduce under a fixed seed. Run `python -m pytest tests/test_recursion.py`; record the result. Any failure of the chain or shared-group checks blocks stability work.

## Phase 4 — Whole-pipeline stability and consensus

**Outcome:** Independent complete runs yield unconditional structural probabilities and a consensus graph. Source: spec §§21–24, 36, 41.4.

### P4.1 Whole-run resampling

- [ ] Create `src/open/stability.py`, `tests/test_stability.py`. Subsample independent units without replacement at fraction `.80`; seed each run reproducibly. Rebuild preprocessing, pair candidates, groups, local classes, and depths inside each run.
- [ ] Track requested, completed, and failed runs. Do not silently drop failed runs or claim the requested denominator if fewer completed. Define retry/failure policy in D-004 before the first large run.
- [ ] Use 50–100 development runs first; use the configured final `B=500` only after the checks and a documented computational estimate.

### P4.2 Structural aggregation

- [ ] Compute node-depth, overall node, edge-class, pairwise shared co-membership, screened-by, and whole-path frequencies over the same completed-run denominator. Never match groups by run-local IDs or multiply edge probabilities to obtain path probability.
- [ ] Preserve target-reached/evaluated frequency separately. Compute retained edge magnitude from runs in which its target was evaluated, with the spec §23 mean/median raw and normalized values.
- [ ] Check exact counts on hand-built run records, including a target evaluated in only some runs and two differently numbered but equivalent shared groups.

### P4.3 Consensus

- [ ] Create `src/open/consensus.py`, `tests/test_consensus.py`. Include nodes, classes, edges, and shared relations at probability `>= pi_min=.70`; choose modal displayed depth, breaking ties toward shallower depth.
- [ ] Keep descriptive loss-difference intervals separate from the inclusion decision. Record threshold, denominator, evaluation frequency, and omissions in the result.
- [ ] Check threshold equality, depth ties, and target reachability. A displayed edge must not imply a causal or formal path metric.

**Gate P4:** A fixed-seed small run has repeatable run IDs, complete denominators, exact synthetic aggregation checks, and a saved consensus object. Run `python -m pytest tests/test_stability.py tests/test_consensus.py`; record the result. Full-scale evidence remains a separate gate.

## Phase 5 — Outputs, visualization, and diagnostics

**Outcome:** Users can inspect machine-readable results and a faithful graph without altering analytical decisions. Source: spec §§25–26, 38–40, 46, 49–50.

### P5.1 Export and provenance

- [ ] Create `src/open/io.py`, `diagnostics.py`, `tests/test_io.py`. Emit spec §39 files: `nodes.csv`, `edges.csv`, `shared_groups.csv`, `shared_comembership.csv`, `local_targets.csv`, `analysis_metadata.json`.
- [ ] Preserve all named columns from §39; add explicit evaluated counts/denominators where needed. Record package/git version, config, seed, learner and library versions, preprocessing, loss, thresholds, resampling design, metadata hash, model-fit counts, target/pair/group evaluations, elapsed time, and peak memory when measurable.
- [ ] Round-trip fixtures, empty/low-gain results, and shared members without individual unique edges. Distinguish unmeasured values from measured zero.

### P5.2 Visualization

- [ ] Create `src/open/visualization.py`, `tests/test_visualization.py`. Place root at depth 0; radial layers follow consensus depth. Unique edge width uses `max(u,0)`, shared virtual edge uses `u_G`, opacity uses stability; screened edges are dashed and shown only when broad and stability display rules pass.
- [ ] Draw a hull or bracket for shared members, with separate member nodes and one virtual group-to-target edge. Label visual distance as display-only; never use it for a statistical decision.
- [ ] Check visual-data preparation on a fixed consensus fixture, including negative `u`, multiple parents, a shared group, and a screened relation. Inspect one rendered figure before accepting layout.

**Gate P5:** A saved analysis reloads with matching counts and a reviewed plot. Run `python -m pytest tests/test_io.py tests/test_visualization.py`; record the result. A report template uses predictive language and includes the spec §50 minimum fields.

## Phase 6 — Scientific evaluation and scoped extensions

**Outcome:** Methodological claims are backed by synthetic and applied evidence; additional modes receive their own evidence rather than inheriting the continuous prototype's status. Source: spec §§28–33, 42–48, 54.

### P6.1 Simulation suite and operating characteristics

- [ ] Build `simulations/generators.py`, scenario modules, `simulations/evaluate.py`, and `tests/test_synthetic_scenarios.py`. Cover direct unique predictor, redundant substitutes, chain, two independent direct predictors, interaction/complementarity, consequence of root, collider, latent common cause, and measurement error (§42).
- [ ] For stochastic scenarios, predeclare seeds, sample sizes, predictor counts, correlation/reliability/noise grids, learners, success metrics, and failure rules in a protocol file before running the final batch. Use pilot runs to debug; keep pilot and final evidence separate.
- [ ] Report the spec §43 metrics, especially incorrect redundancy, chain reconstruction, depth error, false discovery, predictive risk, runtime, and number of fits. Retain all attempted replicates and errors in denominators; do not relabel a passing smoke check as validation.

### P6.2 Sensitivity and comparison

- [ ] Vary `tau_dep=[.60,.70,.80]`, `tau_u=[.025,.05,.10]`, `tau_b=[.025,.05,.10]`, `tau_group=[.05,.10,.20]`, `pi_min=[.60,.70,.80]`, learner family, depth, and missing-data strategy as applicable (§45).
- [ ] Add only comparators whose assumptions, outcome type, and outputs permit a fair named contrast (§44). Prestate what each metric measures; no blanket superiority claim.
- [ ] Recommend or revise defaults only from recorded operating characteristics, with a new decision ID and separate held-out confirmation when hypotheses or thresholds were chosen on earlier results.

### P6.3 Learners and data extensions, each separately gated

- [ ] Add Random Forest and histogram gradient boosting behind the same predictor contract; then `per_target` tuning/selection and ensemble mode. Verify equal splits, fold-local tuning, reproducible selection, and correct metadata.
- [ ] Add binary log-loss and training-prevalence baseline; then any further outcome type from spec §4 with its own loss, baseline, and scenario checks.
- [ ] Add grouped/person-level splitting and resampling before claiming repeated-measures support; then temporal candidate restrictions and prospective examples. Keep between-person, within-person, and prospective modes distinct.
- [ ] Add mixed-data dependence, multiple imputation, retuned ablations, conditional importance, and diagnostic cross-links only as independently documented extensions. Fast mode remains labeled approximate.

### P6.4 Applied reproducible example

- [ ] Produce `examples/anxiety_example.ipynb` or a script, frozen config, input contract, outputs, plot, sensitivity report, and environment/seed instructions. Use only data with permission to distribute or a generator plus documented recipe.
- [ ] Audit every narrative statement for predictive language and trace it to an output field. Report which claims are supported by unit checks, simulations, and applied analysis separately.

**Gate P6:** Publish a scoped evidence summary with exact protocol, seeds, counts, failures, metrics, code revision, and unsupported modes. The phase is not accepted merely because the software ran or generated a graph.

## Coverage and handoff rule

| Source spec | Plan phase |
|---|---|
| §§1–7, 30–31, 37–40 | P0, P5, P6 extensions |
| §§8–12, 24, 32–34 | P1 |
| §§13–17 | P2 |
| §§18–20, 35 | P3 |
| §§21–23, 36 | P4 |
| §§25–29, 46, 49–50 | P5 and P6 extensions |
| §§41–45, 47–48, 54–55 | Checks throughout; P6 evaluation and extensions |

At each handoff, update [OPEN-progress.md](OPEN-progress.md) and [OPEN-decisions.md](OPEN-decisions.md). The next agent should read the current phase, its prerequisites, the source sections, unresolved decisions, and the latest progress entry before coding. Implementation should stop at the first failed phase gate, fix it, and record the evidence; later phases do not waive earlier gates.
