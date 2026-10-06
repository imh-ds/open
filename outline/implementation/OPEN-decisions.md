# OPEN implementation decision log

Use this append-only record for choices the [method specification](../OPEN_v3_technical_implementation_spec.md) leaves open or that alter a public interface, numerical definition, computational policy, or validation claim. The [implementation plan](2026-10-06-open-python-implementation.md) names decisions that must be resolved before dependent code. A superseding decision cites the old ID; keep the old entry for auditability.

## Entry template

```markdown
### D-### — Short title

- **Date / status:** YYYY-MM-DD; proposed | accepted | superseded
- **Applies to:** phase/task, files, source-spec sections
- **Question:** precise ambiguity or tradeoff
- **Options considered:** A / B, with meaningful consequences
- **Decision:** exact behavior or `pending`
- **Reason:** evidence and tradeoff that support the choice
- **Effect on compatibility / inference:** changed outputs, cost, or claims
- **Evidence:** experiment, review, issue, or commit; `none yet` if pending
- **Supersedes:** D-### or `none`
```

## Initial decisions and required resolutions

### D-001 — Python and dependency baseline

- **Date / status:** 2026-10-06; accepted as a package-support policy; runtime installation and project compatibility checks remain pending
- **Applies to:** P0.1; packaging and continuous reference learner; spec §§5, 37, 47
- **Question:** Which Python and core dependency versions form the reproducible baseline?
- **Options considered:** A) Python 3.11+ with major-version-bounded dependency ranges; B) broader older Python support, which expands compatibility and CI work.
- **Decision:** Require Python `>=3.11`. Use runtime ranges NumPy `>=1.24,<3`, pandas `>=2.0,<4`, SciPy `>=1.10,<2`, scikit-learn `>=1.3,<2`, and PyYAML `>=6,<7`; use pytest `>=7.4,<9` for development. These are resolver ranges, not a lockfile; the installed environment must resolve and test a concrete set before P0 is accepted.
- **Reason:** Python 3.11 matches the proposed baseline and has CPython 3.11 distributions for the selected reference-library floors. Major-version ceilings make compatibility changes explicit while allowing patch/minor updates.
- **Effect on compatibility / inference:** Sets runtime support only; makes no scientific claim.
- **Evidence:** Verified PyPI metadata and CPython 3.11 artifacts for the selected NumPy 1.24.0, pandas 2.0.0, SciPy 1.10.0, scikit-learn 1.3.0, and PyYAML 6.0.2 floors, and Python 3.11 support for pytest 7.4.0. No project install or tests were possible because `py -0p` reports no installed runtimes.
- **Supersedes:** none

### D-002 — Fold-specific dependence candidates

- **Date / status:** 2026-10-06; accepted as the initial leakage-safe evaluation policy; implementation and operating characteristics pending
- **Applies to:** P1.3; `src/open/dependence/`, `grouping.py`; spec §§6, 11–12
- **Question:** How do training-fold pair scores form a candidate set when full and pair-ablated risks use repeated held-out splits?
- **Options considered:** A) Fold-local pair selection and risk contrasts, with explicit selection/evaluation frequencies and a prespecified aggregation rule; B) a separate discovery sample that selects pairs before performance folds. A union across all training folds is unsuitable without further separation because observations held out in one fold occur in another fold's training data.
- **Decision:** Use fold-local selection. In each outer training fold, estimate absolute Spearman dependence using only candidate predictors in that fold's training rows and select every pair meeting `tau_dep`. Evaluate each selected pair's group-ablation predictions only on that same fold's held-out rows, using the same fitted full-model reference and split. For a pair, pool its held-out observation losses over exactly the folds that selected it; compute its raw group loss contrast and pair-specific total-gain denominator on that identical set of held-out rows, then form one pooled normalized pair score. Apply `tau_group` to that score for pair classification. Record selected-fold count, evaluated-fold count, their frequencies over all planned outer folds, and the exact evaluation rows. A pair not selected in a fold has no pair score for that fold. If pooled pair-specific total gain is not positive, retain raw group contrast and mark its normalized score undefined; it cannot pass the group threshold. Do not union pairs across folds before evaluation or force an unselected pair into an evaluation fold.
- **Reason:** Each observation's held-out outcome and predictor values remain unavailable to its own pair selection; matching full and ablated losses row-for-row makes the group contrast auditable. Pooling only the folds where training data selected the pair gives a defined statistic without a post-hoc global pair union. Selection frequency exposes how often that conditional statistic was available.
- **Effect on compatibility / inference:** Pair scores use pair-specific held-out rows and denominators and may therefore differ from all-fold individual relevance scores. Candidate selection and evaluation frequencies are required context. This is a leakage-safe prototype policy, not an unbiased estimate of population group importance; calibration and sensitivity to selection frequency remain pending.
- **Evidence:** Design review against spec §§6, 11–12, 24; P1.3 sentinel and paired-fold checks pending.
- **Supersedes:** none

### D-003 — Overlapping supported shared pairs

- **Date / status:** 2026-10-06; proposed
- **Applies to:** P2.2; `grouping.py`, result schema; spec §14
- **Question:** What is represented when a connected shared-pair component fails whole-component validation but some constituent pairs pass?
- **Options considered:** A) Keep supported pairwise shared relations as separate virtual groups; B) discard all pairs with the failed component.
- **Decision:** Propose A, matching the explicit spec instruction. Preserve each pair's member tuple and group importance; promote each member of a retained validated pair, never a synthetic aggregate.
- **Reason:** A failed merge must not erase locally supported pair relations.
- **Effect on compatibility / inference:** A variable may appear in more than one virtual pair; export must preserve this without inventing a partition.
- **Evidence:** Required P2.2 overlap fixture pending.
- **Supersedes:** none

### D-004 — Failed stability runs and denominator

- **Date / status:** 2026-10-06; proposed
- **Applies to:** P4.1–P4.3; `stability.py`, `consensus.py`, metadata; spec §§21–24
- **Question:** How should a failed full-pipeline resample affect the requested `B` runs and reported probabilities?
- **Options considered:** A) Prespecified bounded retry with failure log and probability denominator equal to completed runs; B) silently drop errors; C) stop at first failure.
- **Decision:** Propose A with zero silent retries until a bounded retry count is explicitly configured. Always report requested, attempted, failed, and completed counts; do not accept a final run if completed count is below the prespecified gate.
- **Reason:** Structural probabilities require an auditable denominator and failures may be data-dependent.
- **Effect on compatibility / inference:** An incomplete final batch is evidence of a failed gate, not a passing consensus estimate.
- **Evidence:** Required P4.1 failure-injection check pending.
- **Supersedes:** none

### D-005 — YAML configuration parser

- **Date / status:** 2026-10-06; accepted
- **Applies to:** P0.1; `src/open/config.py`, `configs/default.yaml`; spec §40
- **Question:** Which parser should load and round-trip the YAML configuration example?
- **Options considered:** A) Add a narrowly scoped YAML parser dependency; B) provide YAML only as a non-loadable example or implement a custom parser.
- **Decision:** Use PyYAML's safe loader/dumper for configuration files, with the supported range defined in D-001.
- **Reason:** The package needs to parse the spec's YAML shape, and maintaining a custom YAML parser is out of scope.
- **Effect on compatibility / inference:** Adds one runtime dependency for configuration I/O; it does not change any statistical definition or inference.
- **Evidence:** PyYAML 6.0.2 publishes a CPython 3.11 wheel; runtime behavior remains unverified until Python is available.
- **Supersedes:** none

### D-006 — First-path performance split plan

- **Date / status:** 2026-10-06; accepted as the initial cross-sectional implementation policy; runtime behavior remains unverified
- **Applies to:** P0.3 and subsequent local-target evaluation; `src/open/splits.py`; spec §§6–7
- **Question:** Which row-level split scheme should define repeatable performance folds, and what must be serialized?
- **Options considered:** A) seeded repeated shuffled K-fold CV with explicit positional indices; B) repeated random holdout splits; C) generate folds lazily from a seed without persisting indices.
- **Decision:** Use scikit-learn `RepeatedKFold` for the independent-row continuous first path, with `folds`, `repeats`, and the configured seed. Persist the realized train/test positional indices and split parameters in a versioned JSON `SplitPlan`; validate index bounds, nonempty folds, and train/test disjointness when loading. Reject grouped/repeated-measures requests until their phase is implemented. Inner tuning receives folds built only from the outer training rows.
- **Reason:** Explicit folds make paired model comparisons auditable and preserve the exact evaluation rows when a run is resumed. Repeated K-fold matches the plan's 5-fold, 2-repeat default for independent cross-sectional observations.
- **Effect on compatibility / inference:** Results depend on the realized split plan; this does not support clustered or longitudinal rows and does not establish that the CV estimate is calibrated.
- **Evidence:** Implemented in commit `d4391ae`, merged to `main` at `8698681`; the combined metadata/fold fixture is in `8698681`; independent static review passed and `git diff --check b971c92..HEAD` passed. Runtime and P0 gate tests remain unrun because no Python interpreter is installed.
- **Supersedes:** none

### D-007 — Initial elastic-net tuning grid

- **Date / status:** 2026-10-06; accepted as an engineering default for the prototype; not calibrated
- **Applies to:** P0.3 and P1 learner tuning; `src/open/learners/elastic_net.py`; spec §§5–7, 33
- **Question:** What deterministic search defines the initial elastic-net learner when the specification does not provide an alpha grid?
- **Options considered:** A) one fixed estimator with no tuning; B) a finite grid selected by inner cross-validation; C) an automatic path search whose preprocessing would need separate nested-fold safeguards.
- **Decision:** Use `GridSearchCV` with negative mean squared error on `alpha` values `(0.0001, 0.001, 0.01, 0.1, 1.0)` and `l1_ratio` values `(0.1, 0.5, 0.9, 0.95, 1.0)`. Run the full median-imputation and standardization pipeline inside every inner fit, use `max_iter=10000`, and retain the configured no-retuning policy for ablations by reusing the full-model parameters.
- **Reason:** A finite ordered grid gives deterministic, auditable tuning and allows preprocessing to be refit separately inside each inner training fold. The chosen values are an initial engineering range, not a claim of optimality.
- **Effect on compatibility / inference:** The first learner is limited to continuous outcomes scored by MSE. Predictions may depend on the fixed grid; calibration and sensitivity to the grid remain Phase 6 work.
- **Evidence:** Implemented in commit `d4391ae`, merged to `main` at `8698681`; independent static review passed. Runtime and P0 gate tests remain unrun because no Python interpreter is installed; model/grid calibration remains pending Phase 6.
- **Supersedes:** none

## Verification addendum

### 2026-10-06 — Phase 0 runtime evidence for D-001 and D-005–D-007

The original decision entries above are retained as written; this addendum records later verification.

- **D-001:** Python 3.11.9 executed the P0 suite with NumPy 1.26.4, pandas 2.2.3, SciPy 1.11.2, scikit-learn 1.3.0, PyYAML 6.0.2, and pytest 8.3.3, all within their declared ranges. The suite passed 37/37. Python 3.14.3 also passed 37/37 with core packages within range; its pytest 9.0.3 is outside the declared development range and is supplementary evidence only.
- **D-005:** Safe YAML load/dump and configuration round-trip behavior are covered by `tests/test_config.py` and passed in both recorded runs.
- **D-006:** Split reproducibility, JSON round-trip, partition validation, and the combined metadata/fold fixture are covered by the P0 tests and passed in both runs.
- **D-007:** The configured GridSearchCV path, fold-local preprocessing, parameter validation, fit/predict, and cloning checks are covered by the P0 tests and passed in both runs. This does not calibrate the grid or validate OPEN's scientific performance.
- **Exact gate:** `& 'C:\Users\imhoh\AppData\Local\Programs\Python\Python311\python.exe' -m pytest tests/test_config.py tests/test_metadata.py tests/test_splits_preprocessing.py` -> 37 passed. The same command using `C:\Users\imhoh\AppData\Local\Programs\Python\Python314\python.exe` also passed 37 tests with pytest 9.0.3.
- **Commit / review:** pytest src-path configuration committed as `462ad23`; full-branch static review passed; `git diff --check b971c92..HEAD` passed.

Before implementing a proposed decision, the agent must either mark it `accepted` with the check or review that supports it, or append a replacement entry and cite the replacement in the progress register. Proposed entries are not validated method rules.
