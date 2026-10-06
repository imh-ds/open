# OPEN implementation progress

This is the live status and evidence register for the [implementation plan](2026-10-06-open-python-implementation.md). The [method specification](../OPEN_v3_technical_implementation_spec.md) defines behavior. A checked box in the plan needs an evidence entry here; a commit or document alone does not establish a passed gate.

## Status vocabulary

`not_started` means no implementation evidence; `in_progress` means work is under way; `blocked` means a named dependency prevents work; `gate_failed` means a required check failed; `accepted` means the phase gate has recorded passing evidence and review. Use these exact labels. Keep failed attempts in the history below.

## Phase register

| Phase | Status | Current gate / next action | Evidence entry |
|---|---|---|---|
| P0 — Project and data contract | accepted | Required gate passed on Python 3.11.9; whole-branch review and combined fixture are recorded. This is software acceptance only. | `462ad23` |
| P1 — Local relevance | not_started | P0 accepted and P1 policies/contracts resolved; implement held-out risk and relevance ledger | — |
| P2 — Classification | not_started | P1 accepted; implement pair/group and promotion rules | — |
| P3 — Network expansion | not_started | P2 accepted; pass chain and layer-snapshot checks | — |
| P4 — Stability and consensus | not_started | P3 accepted; rebuild whole pipeline per run and verify denominators | — |
| P5 — Outputs and visualization | not_started | P4 accepted; export and inspect saved graph | — |
| P6 — Scientific evaluation and extensions | not_started | P5 accepted; freeze protocols and evaluate each supported claim | — |

## Active work

- **Current phase/task:** P0 accepted as the software setup gate at `462ad23`; P1 has not started.
- **Next concrete action:** Begin P1.1 `risk.py` and tests using the D-009 `RiskResult` contract; apply D-008 denominator handling and D-002 fold-local pair policy in the later P1 tasks.
- **Known limits:** No full OPEN analysis, simulation result, or validated methodological result exists. Python 3.11.9 passed the P0 gate; Python 3.14.3 also passed a supplementary run using pytest 9.0.3, outside the declared development range. `py -0p` still fails to enumerate either installation, but direct interpreter paths work. D-002, D-008, and D-009 are accepted implementation policies whose code checks remain pending.
- **Unresolved decisions:** See the [decision log](OPEN-decisions.md); D-003 and D-004 remain proposed, alongside later-phase questions.

## Update protocol for the coding agent

1. Before a task, read its source-spec sections and unresolved decisions. Add a dated `in_progress` entry with task ID, branch, and intended deliverable.
2. Make one independently reviewable change. Record any methodological or public-interface choice in the decision log **before** relying on it. Do not revise past decision entries; supersede them.
3. Run the focused checks named in the task, then the phase gate when applicable. Capture exact command, outcome, counts, failures, artifact paths, and commit SHA. State when a wider suite was not run.
4. Update the phase table only when the evidence supports the new status. If a gate fails, set `gate_failed`, retain the failure in history, and identify the next fix. If externally blocked, state the blocker and what would unblock it.
5. For simulation or applied results, keep protocol/seed/denominator and code revision in the evidence entry. Separate implementation acceptance from methodological validation.

## Evidence history

### 2026-10-06 — Planning baseline

- **Tasks:** Plan and handoff files only.
- **Status:** All implementation phases `not_started`.
- **Evidence:** Source spec reviewed for phase mapping; no Python implementation or tests run.
- **Decision provenance:** Initial software choices and unresolved questions are listed in `OPEN-decisions.md`.
- **Next action:** P0.1.

### 2026-10-06 — P0.1 kickoff

- **Status:** in_progress
- **Branch / commit:** `codex/open-p0-1`; base `b971c92`; task commit pending
- **Scope and decision IDs:** package metadata, configuration defaults/validation, root README usage, `tests/test_config.py`; D-001
- **Checks:** pending; Python runtime discovery command `py -0p` -> no installed Python runtimes
- **Artifact / data provenance:** source specification §§5, 37–40; implementation plan P0.1
- **What works:** implementation is isolated from the existing main checkout
- **Limits / failures:** Python tests cannot run until a runtime is available
- **Next action:** finish, review, and integrate P0.1

### 2026-10-06 — P0.2 kickoff

- **Status:** in_progress
- **Branch / commit:** `codex/open-p0-2`; base `b971c92`; task commit pending
- **Scope and decision IDs:** `src/open/metadata.py`, `tests/test_metadata.py`; data contract §§3 and 38
- **Checks:** pending; Python runtime discovery command `py -0p` -> no installed Python runtimes
- **Artifact / data provenance:** source specification §3; implementation plan P0.2
- **What works:** implementation is isolated from the existing main checkout
- **Limits / failures:** Python tests cannot run until a runtime is available
- **Next action:** finish, review, and integrate P0.2

### 2026-10-06 — P0.1 package/configuration integrated

- **Status:** in_progress pending executable tests
- **Branch / commit:** `codex/open-phase0`; `d933b7b` (P0.1 implementation commit)
- **Scope and decision IDs:** package metadata, configuration defaults/validation, root README usage, and `tests/test_config.py`; D-001 and D-005
- **Checks:** independent static review passed; `git diff --check` passed for tracked changes; pytest not run because `py -0p` reports no installed Python runtimes
- **Artifact / data provenance:** source spec §§5, 37–40; implementation plan P0.1
- **What works:** package/configuration files are integrated; missing `analysis` now raises the documented `ValueError`
- **Limits / failures:** runtime import, YAML round-trip, and config tests remain unverified
- **Next action:** finish P0.3 and run the full P0 gate in a Python 3.11+ environment

### 2026-10-06 — P0.2 metadata contract integrated

- **Status:** in_progress pending executable tests
- **Branch / commit:** `codex/open-phase0`; `c2573ba` (P0.2 integration commit; task commit `5c4baa3`)
- **Scope and decision IDs:** ordered candidate validation, scale-level exclusions, display-label preservation, prospective-time rejection, and symmetric overlap flags; no new decision ID
- **Checks:** independent static review passed, including the predictor-only overlap regression; `git diff --check` passed; all nine pytest cases remain unrun because Python is unavailable
- **Artifact / data provenance:** source spec §3; implementation plan P0.2
- **What works:** metadata records are preserved, supported candidates are ordered, and shared-item totals are flagged in output metadata
- **Limits / failures:** runtime behavior and all nine test cases remain unverified
- **Next action:** complete P0.3, then execute the specified P0 gate

### 2026-10-06 — P0.3 kickoff

- **Status:** in_progress
- **Branch / commit:** `codex/open-p0-3`; based on integrated P0.1/P0.2 commit `c2573ba`
- **Scope and decision IDs:** `splits.py`, `preprocessing.py`, `learners/base.py`, `learners/elastic_net.py`, and `tests/test_splits_preprocessing.py`; D-006 and D-007
- **Checks:** pending; Python runtime discovery `py -0p` reports no installed runtimes
- **Artifact / data provenance:** source spec §§5–7; implementation plan P0.3
- **What works:** split serialization and learner behavior are scoped before implementation
- **Limits / failures:** executable leakage and reproducibility checks cannot run until Python is available
- **Next action:** implement and review the P0.3 interfaces, then execute the P0 gate when an interpreter is available

### 2026-10-06 — P0.3 split and learner implementation integrated

- **Status:** blocked pending executable P0 gate
- **Branch / commit:** `codex/open-phase0`; `d4391ae` (P0.3 implementation), `37fc306` (terminal blank-line cleanup), `8698681` (combined metadata/split fixture)
- **Scope and decision IDs:** versioned repeated-K-fold plans, JSON persistence, numeric fold-local imputation/scaling, predictor interface, elastic-net MSE tuning, and `tests/test_splits_preprocessing.py`; D-006 and D-007
- **Checks:** independent whole-branch static review and focused fixture review passed; `git diff --check b971c92..HEAD` passed at `8698681`; focused P0.3 tests are authored but unrun because `py -0p` reports no Python runtime
- **Artifact / data provenance:** source spec §§5–7, 37; implementation plan P0.3
- **What works:** code stores realized split indices, validates loaded partitions, keeps preprocessing inside inner CV fits, rejects grouped splits and invalid CV indices, and a combined fixture checks metadata candidates plus same-seed fold reproducibility
- **Limits / failures:** Python syntax/import/runtime behavior, the P0 pytest gate, and numerical behavior remain unverified; no methodological validity claim is made
- **Next action:** make Python 3.11+ and D-001 dependencies available and run the full P0 gate; only then consider P0 accepted

### 2026-10-06 — Phase 0 implementation merged to main

- **Status:** blocked pending executable P0 gate; Phase 0 is not accepted
- **Branch / commit:** `main`; `8698681` (fast-forward from `codex/open-phase0`)
- **Scope and decision IDs:** integrated P0.1 package/configuration, P0.2 metadata contract, P0.3 split/preprocessing/learner, plus the combined metadata/fold fixture; D-001, D-005, D-006, and D-007
- **Checks:** final whole-branch static review passed; `git diff --check b971c92..HEAD` passed; `py -0p` -> `No installed Pythons found!`; the P0 pytest command was not run
- **Artifact / data provenance:** no analysis data or scientific results; code only
- **What works:** all planned Phase 0 code and focused tests are committed; the combined fixture checks metadata candidates and reproducible fold plans
- **Limits / failures:** runtime imports, dependency resolution, preprocessing/model behavior, and all pytest cases remain unverified; no method-validity claim is made. Local status still reports `.gitignore` modified, but its worktree blob equals `HEAD` and it was not changed in this work. The outline and progress/decision documents remain ignored local files.
- **Next action:** make Python 3.11+ and D-001 dependencies available, run `python -m pytest tests/test_config.py tests/test_metadata.py tests/test_splits_preprocessing.py`, record the exact result, and review before P1

### 2026-10-06 — P0 executable gate passed

- **Status:** accepted as a software phase; no methodological validation claim
- **Branch / commit:** `main`; `462ad23` (`Configure pytest to import src package`), following P0 integration head `8698681`
- **Scope and decision IDs:** P0.1 configuration, P0.2 metadata, P0.3 splits/preprocessing/learner, and combined metadata/fold fixture; D-001 and D-005–D-007
- **Checks:** `& 'C:\Users\imhoh\AppData\Local\Programs\Python\Python311\python.exe' -m pytest tests/test_config.py tests/test_metadata.py tests/test_splits_preprocessing.py` -> 37 passed in 1.52s using Python 3.11.9, NumPy 1.26.4, pandas 2.2.3, SciPy 1.11.2, scikit-learn 1.3.0, PyYAML 6.0.2, pytest 8.3.3. Supplementary `& 'C:\Users\imhoh\AppData\Local\Programs\Python\Python314\python.exe' -m pytest tests/test_config.py tests/test_metadata.py tests/test_splits_preprocessing.py` -> 37 passed in 8.76s using Python 3.14.3 and pytest 9.0.3 (outside the declared pytest `<9` development range). Final whole-branch static review passed; `git diff --check b971c92..HEAD` passed.
- **Artifact / data provenance:** deterministic six-row metadata/fold fixture in `tests/test_splits_preprocessing.py`; synthetic unit data only
- **What works:** configured source imports, YAML config round-trip, metadata validation, ordered candidates, split serialization/reproducibility, fold-local preprocessing, and elastic-net tune/fit/predict checks pass in the P0 suite
- **Limits / failures:** the initial Python 3.11 gate attempt without `PYTHONPATH` failed collection with three `ModuleNotFoundError: open` errors. Setting process-local `PYTHONPATH=src` passed 37/37 and confirmed the src-layout cause; `pyproject.toml` now declares `pythonpath = ["src"]` (commit `462ad23`), after which the direct-path Python 3.11 gate passed. `py -0p` still reports no runtimes; direct interpreter paths were used. P0 does not validate OPEN's statistical method, simulations, or applied performance. `.gitignore` remains locally marked modified but its content hash equals `HEAD`; it was not changed. This evidence entry is local because `outline/` is ignored by Git.
- **Next action:** start P1.1 risk engine; preserve P0 as software-only acceptance

### 2026-10-06 — P1 method contracts resolved

- **Status:** P1 not_started; implementation policies recorded, no P1 software gate evidence
- **Branch / commit:** `main`; `47db527` (D-002 fold-local pair evaluation), `6adfa14` (D-008 exact relevance denominators), `2dbc66b` (D-009 RiskResult contract), `0304708` (formal D-002 pooled-score equation), `5b57679` (specification whitespace cleanup)
- **Scope and decision IDs:** fold-local dependence selection and pair evaluation, including pair-specific held-out risk pooling; no-epsilon denominator and low-gain stop; immutable row-keyed `RiskResult`, `FoldRisk`, and `ObservationLoss` contract. Updated the Phase 1 plan and method specification.
- **Checks:** `git diff --check` passed for each documentation increment and across the committed decision range after `5b57679`. No Python tests run; implementation has not started. The ignored outline documents are now explicitly tracked and committed.
- **Artifact / data provenance:** design decisions reviewed against method specification §§6–12, 24, 32, 34, 38 and Phase 1 task dependencies; no analysis data used.
- **What works:** D-002, D-008, and D-009 now specify pair selection/evaluation denominators, relevance denominator guards, result fields, paired-row keys, pooled aggregation, and fit-count meanings. The decision log records rationale, consequences, and pending evidence checks.
- **Limits / failures:** Leakage sentinel, arithmetic, pairing, baseline, aggregation, and fit-count tests remain pending their P1 tasks. These decisions do not establish population-level validity or calibration.
- **Next action:** start P1.1 risk engine under the D-009 interface.

Copy this form for each subsequent entry:

```markdown
### YYYY-MM-DD — Pn.m short result

- **Status:** in_progress | blocked | gate_failed | accepted
- **Branch / commit:** branch name; SHA or `uncommitted`
- **Scope and decision IDs:** exact files or behavior; D-### references
- **Checks:** `exact command` -> result, pass/fail counts, relevant output path
- **Artifact / data provenance:** path, config, seed, input version/hash when relevant
- **What works:** observable behavior supported by the checks
- **Limits / failures:** exact failing case or untested boundary
- **Next action:** one concrete task or remedy
```
