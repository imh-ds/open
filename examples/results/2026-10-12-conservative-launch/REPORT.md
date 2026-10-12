# Conservative-selection study launch

[GitHub Actions run](https://github.com/imh-ds/open/actions/runs/38279504044) dispatched once from `87420b857ef9cbc65f6713a9c8ed9a652727843d`. Source SHA and workflow_dispatch event verified through the API.

Frozen 240 fresh datasets, 120 two-task shards, two single-thread workers per shard. Three competitors: production v2, minimum-CV weighting, conservative one-SE-style penalty preference. Five datasets per cell; development only.

Startup snapshot: 121 jobs registered, 20 running, 100 queued, 1 completed. Actual concurrency is account-limited despite max-parallel256.

19 focused development tests and two native preflight identities passed before publication. Public checkout tests separately verified before dispatch. Preflight raw evidence is retained in ../2026-10-12-conservative-preflight. Earlier frozen studies and production defaults are unchanged. Full scientific results are pending.

Full API provenance: [run](run.json), [jobs](jobs.json), [receipt](receipt.json). Download and retain raw completed-study evidence in the repository before artifact expiry. No automatic retries, rule adoption, minimum-N guidance or confirmation claim.
