# Proposed bounded recursive-network confirmation

Status: proposal only. No data generation, new scientific fitting, or Actions dispatch is authorized by this artifact. The owner approved the release-readiness audit; the new simulation profile and budget are awaiting review. Passing installed fixed fixtures does not close this scientific gate.

## Purpose

Check incorrect links, redundant-group classification and layer placement for the unchanged recursive method, rather than adding interaction screens or more engines. This is a narrow reference benchmark, not general validation of all configurations, sample sizes or real populations.

## Proposed common design

- Four independent continuous Gaussian scenarios, three columns (one outcome Y and two eligible predictors X1/X2), 80 participants per dataset and 80 independent datasets per scenario (320 total).
- Direct: X1/X2 independent standard normal; Y=X1+normal noise SD 0.1; expected X1 at depth 1, X2 unpromoted.
- Redundant: X1 standard normal; X2=X1+normal noise SD 0.01; Y=X1+normal noise SD 0.15; expected one joint root group containing X1/X2, no unique member arrows.
- Chain: X1 standard normal; X2=X1+normal noise SD 0.7; Y=X2+normal noise SD 0.3; expected X2 depth 1, X1 depth 2, with no additional root X1 link. This checks the algorithm's stipulated predictive construction, not causal direction in empirical data.
- Noise: Y/X1/X2 independent standard normals; expected no promotion.
- All innovations mutually independent except the specified construction. Do not generate until the selected profile and budget are approved.
- Proposed immutable task ordering: noise IDs 0â€“79, direct 80â€“159, redundant 160â€“239, chain 240â€“319. Generator seed 940000000+ID; analysis/split/subsampling seed 950000000+ID. Audit these proposed namespaces against all historical manifests before freezing; a collision is a design error requiring an explicit revision, never silent replacement.
- Fixed elastic net, max depth3, independent rows, no missingness, MSE, fold-local median imputation/linear standardization,80% sampling without replacement,70% retention. No engine/threshold tuning from these results. One scientific execution per identity; preserve attempts/errors/costs and outputs.

## Configuration choice still requiring an owner decision

|Profile|Resamples per target|Performance CV|Tuning CV|Unique relevance threshold|Claim boundary|
|---|---:|---|---:|---:|---|
|Shipped defaults|100|5-fold repeated twice|3-fold|.05|Checks the shipped default profile at these narrow scenarios/N|
|Existing demonstration profile|20|2-fold once|2-fold|.15|Checks only this explicitly non-default profile; defaults remain unconfirmed|

All other thresholds remain their existing configuration values: total relative gain.02, broad.05, joint.10, dependence.70, retention.70. The demonstration profile is copied from historical fixture configs, not selected to improve the new results. The owner's preference question is pending; no profile is silently chosen and no default is modified.

## Metrics and proposed acceptance

Retain planned denominators of 80 per scenario. Report exact structure recovery, incorrect selected relations, missing expected relations, exact depth placement, joint-vs-unique classification, incomplete analyses, warnings, root local predictive risks, fits, elapsed time and peak traced memory. A wrong unique edge in the redundant scenario counts as incorrect; a wrong direct root edge or depth in the chain counts as structural failure. These are known-generator checks, not claims of causal identification.

All 320 identities must be complete and auditable before a positive qualification. Prespecified per-scenario two-sided 95% exact binomial intervals: noise-any-finding and signal-any-incorrect-relation upper bounds <= 0.05; exact structure-recovery lower bound >= 0.80 for each of the three signal scenarios. Missing/failed cases retain denominators and cannot qualify; report inclusive best/worst bounds if incomplete. Intervals are per-scenario, not simultaneous coverage or probabilities that individual real-study links are true. No claim of universal minimumN, effect magnitude certification, or general false-discovery control.

A failed gate remains a failed result. It does not authorize retuning thresholds, dropping scenarios, changing denominators, or relabeling a development run as confirmation. Fixed protocol evidence is prerequisite to any scoped release claim based on this benchmark.

## Cost and staged execution proposal

The proposed maximum total Actions timeout ceiling is 3300 runner-minutes across both stages, ordinary runners, no paid capacity or automatic model retries. Stage A: 80 noise datasets, 40 two-case shards. Stage B: 240 signal datasets, 120 two-case shards, only after complete audited StageA passes its bound. Each worker 20 min (3setup/15computation/2preservation); each stage 10 min preparation and 30 min aggregation;160*20+2*(10+30)=3280min, with 20 min unallocated headroom. Transport retries preserve original bytes; they do not permit new fits.

A native preflight of the first two planned chain identities must show the two-case shard fits its 15-minute computation allocation before implementation/dispatch is accepted. Preserve those costs separately; Actions recomputation of the same IDs is not extra scientific replication. Measure raw/disk/aggregation cost and freeze exact package/runtime/source/manifest identities before dispatch. Matrix concurrency may use available account capacity without increasing the total ceiling. Every missing job stays in the planned denominator.

The **shipped-default option has a serious timing feasibility concern**: the one-assessment installed probe took 85.46 sec, and its rough 100-assessment projection is 142.4 min for a single chain case. That projection greatly exceeds the proposed two-case 15-minute computation allocation. It is not a measured full-budget run, but it is sufficient to withhold launch under this proposal. Do not silently reduce resamples/folds or increase timeouts. If the owner chooses defaults, first reconcile performance/budget and approve a feasible revised protocol. The lighter profile's measured existing fixture time (approximately5.3min per chain case under concurrent native checks) is closer to the allocation but still requires the exact native preflight.

No expensive run is launched by choosing a preference alone: freeze the concrete profile, feasible measured budget and source guards, then obtain approval of that reviewable dispatch. A research-preview software release must keep its exploratory boundaries if scientific acceptance remains open.
