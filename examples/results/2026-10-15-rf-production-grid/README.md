# Production RF singleton-grid optimization

RF now skips duplicate feature-fraction settings when tuning one numeric
predictor. The original `.7` setting is retained, together with both leaf-size
settings. With three-fold inner CV, singleton tuning uses six fits instead of
twelve. RF searches with multiple predictors and other engines keep their grids.

`TreePredictor.tuning_grid_size` describes the most recent search on that instance
(the nominal size before its first tune). Risk evaluation reads this value from
the tuned clone, after tuning. Reusing an instance for another predictor width
refreshes its size. Effect supplements report the effective full-model grid
and size, rather than the nominal size of an unfitted prototype.

The comparison tool now has an explicit `OriginalGridPredictor` reference that
performs the historical duplicate fits. Its production branch calls the normal
optimized engine against the unchanged nominal grid, so it tests the automatic
reduction itself. The historical four-candidate reference exists only in the
example tool, not as a new production configuration option.

The fresh comparison in `comparison.json` passed all 12 conditions: N=80 and
N=200 crossed with linear, U-shaped, 20%-missing, constant-predictor,
all-missing-predictor, and constant-outcome inputs. Every selected parameter,
inner-fold/mean CV score, and held-out prediction matched exactly. Each condition
used one paired repetition: actual tuning fit counts were 12 versus six, with
one final fit each. These are software fixtures, not independent scientific
reliability-study repetitions. Exact arrays and their hashes accompany the report.

This run verifies production parity and accounting. Its timing fields were
collected while the broader test suite was active and are not used as a speed
estimate. The earlier [isolated timing comparison](../2026-10-15-rf-grid-equivalence/README.md)
found about 50% lower singleton tuning time. Full recursive runtime savings remain
unmeasured; full models, ablations, groups and other work also consume time.

Regression coverage checks actual RF fit calls, original tie order, retuning
and cloning across widths, real risk counts and identical fold losses, effect
metadata, and a predictor missing in just one inner training partition. A
read-only independent review found no correctness issues; its missing-partition
coverage suggestion was incorporated.

The broader suite exposed historical research fixtures that tried to validate
current source against calibration bound to older source. The guard correctly
rejected that mismatch. These unit tests now use temporary manifests and
synthetic calibration rows, retaining the real provenance and selection checks.
Their caches and paths are restored afterward. Separate regression tests verify
that changed source remains rejected and a missing null audit still blocks the
signal stage.

Two other tests imported an archived replay CLI that intentionally installs a
process-wide no-fit guard. That guard leaked into later tests and blocked their
scripted recursion. The imports now restore the original function afterward;
the archived CLI's guard is untouched. Positive historical reuse tests supply
their original declared manifest; source-tampering tests still reject changes.

The complete initial suite ran 856 tests: 834 passed and 22 failed. All 22
failures were repaired and retested successfully: five are covered in a
34-test replay/guard run, and the remaining 17 passed pytest's last-failed rerun.
Additional focused runs passed 14 production/provenance checks and 24 research
fixture checks. These groups overlap and must not be summed. The current
collection has 859 tests, including three added edge-case/isolation checks.
A second complete suite run after all repairs was not performed. The original
run also recorded four missing-font-glyph warnings and 25 constant-outcome
elastic-net convergence warnings. Detailed results, failure identities and
retained logs are in `verification.json` and `software-checks/`.

Historical manifests, checkpoints, calibration evidence and reported findings
retain their original context. This change does not establish additional
scientific reliability or a sample-size rule. No scientific workflow was
dispatched. A future optimized recursive study requires a fresh frozen manifest
and an auditor that explicitly checks the new per-risk effective grid sizes.
The historical multi-engine auditor retains its four-candidate accounting for
the original runs; it must not silently accept either count for new evidence.
