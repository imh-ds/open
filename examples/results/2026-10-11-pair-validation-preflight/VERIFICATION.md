# Verification and launch

Implemented diagnostic validation and opt-in adaptive details; default recursive
selection and automatic interaction acceptance are unchanged.

- Full native suite: 578 passed, 29 existing font/elastic-net warnings, 558.52s.
  [Complete output](full-suite.log). The final manifest roundtrip regression was
  added during that run and also verified separately in the final seven-test
  runner suite; final runner/helper/adaptive suite: 16 passed in 13.93s.
- Focused integration suite: 55 passed in 34.65s before the manifest roundtrip
  regression was added. Public export suite: 35 passed in 18.25s.
- Independent review: implementation and fresh-runner findings repaired;
  final source freeze independently checked, no remaining blocker.
- Thirteen changed/new public source, test, protocol, workflow and documentation
  files match the development worktree after line-ending normalization. Frozen
  runtime/source check passed in both worktrees. Raw ZIP checksum and CRC passed.
- Public incremental commits: c134e4f validation; 8f7fd7e adaptive details;
  2b9cfe6 duplicate-pair fix; c73ae02 audited study; c47a5b1 frozen workflow;
  ef0742f documentation/native evidence. Internal outline updates remain internal.
- [Actions run 38196995390](https://github.com/imh-ds/open/actions/runs/38196995390)
  was dispatched once at ef0742fe0b72b8463ed80a02cad84931c2254922. Pinned
  preparation passed; first worker snapshot: 20 running, 100 queued, zero failures.
  [Launch metadata](launch.json); [startup snapshot](startup.json).

The 240-task study remains pending. The two preflight tasks are construction
checks and will be recomputed on Actions. Their negative validation gains are
preserved; no cutoff or production interaction rule has been accepted. Recorded
successful-task cost excludes interrupted attempts and later repeated audits.
Once Actions completes, download, verify and retain full raw evidence alongside
the results before artifacts expire; report examination coverage and unfavorable
results as well as successful detections.
