# Prospective interaction design verification — 2026-10-13

This verifies a proposal and its data-independent calculations, not empirical interaction reliability. No new model fits or Actions dispatch occurred.

- Re-ran `prospective_interaction_assurance.py` with the existing Python 3.11/SciPy 1.11.2 study runtime. Parsed output exactly equals `prospective_interaction_assurance.json`.
- The script independently cross-checks each Clopper-Pearson acceptance boundary using binomial CDF/SF probabilities. Verified null gate <=36/1200, alternative-control gate <=8/450, detection >=387/450, and supported-pattern gate >=194/200 displayed curves.
- Verified all 12 matched-q coefficients recover their specified population information targets; every denominator is positive.
- Verified totals: 14,400 null + up to10,800 alternatives =25,200 datasets; 17,438,400 fit/tune calls, 504,000 saved-setting refits, 180+135 worker shards; timeout cap6,450 runner minutes.
- Independently verified the historical threshold normalized-LF SHA256 equals the checksum in the protocol. Production and prior frozen source/workflow diffs are empty against development base `fa929b4`.
- Independent statistical review accepted simultaneous directional coverage, selection-conditional pattern inference and prospective null-first skipping. Review caught an underpowered alternative-control gate at the initial 300-replication proposal. Increased alternatives to450, disclosed hypothetical assurance assumptions, and narrowed shape claims to supported relative pattern; re-review found no remaining statistical design blocker.
- Whitespace verification: `git diff --check` and staged equivalent passed. No full package test-suite run: this change adds a proposal and a standalone no-fit calculation, not a candidate or production change.

Remaining gates: written design approval; written implementation plan; tested stage rules/manifests/audit; representative shard timing and storage estimate; pinned runtime/source freeze and native preflight; independent implementation review; exact public export verification before any launch. Historical confirmation remains failed. New reliability labels and exact-effect-size accuracy remain unestablished.
