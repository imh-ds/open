# Recursive tree-engine matched-chain validation protocol

Approved by the owner on 2026-10-16: 330 analyses and up to 30,000 additional runner-minutes. This is a new signal study; the completed null runs and corrected no-fit audit remain unchanged.

## Population and planned cases

Independent standard-normal X1, e2, ey, Z1 and Z2. X2=.6*X1+.8*e2; Y=.8*X2+.6*ey. Expected network X1 -> X2 -> Y, with Y depth0, X2 depth1, X1 depth2; nuisance variables Z1/Z2 are absent. There are 55 independent datasets per sample size N80/N200, paired across RF/GBM/LightGBM (330 engine cases). Fresh generator seeds 2,800,000,000+100*N+r and analysis seeds 2,900,000,000+100,000*N+1,000*r, r1..55; exclude prior timing fixtures. The 1,000-seed spacing keeps within-case resampling seed blocks disjoint across independent repetitions. Check for collisions against retained study manifests before launch.

## Frozen analysis and gates

Use shipped settings: 100 80%-subsamples, performance CV5x2, tuning CV3, 100 trees, fixed engine, production effective grids, thresholds including retention .70. Four assessment workers and one numerical thread per worker. Root Y, max depth3. No new tuning to favor these populations. Parent corrected null report must qualify all three engines at both N values before launch.

Every engine/N cell keeps 55 planned identities. A pass requires all cases complete and independently audited, no unknown fit costs, exact full network AND depth recovery in at least50/55 (two-sided exact95% lower bound>=.80), and extra findings in at most1/55 (two-sided exact95% upper bound<=.10). Extra findings are any incorrect edge, group or promoted nuisance node. Missing links/wrong depths fail exact recovery, even if they add no false edge. These are marginal per-cell intervals, without family-wise control. Reports must show missing/extra links and depth matches, not only pass/fail. No missing/failed case is replaced or retried automatically.

## Actions, compute and evidence

Two 165-case matrices, each max-parallel10: at most20 case jobs. Case hard caps (N80/N200 minutes): RF90/150, GBM60/100, LightGBM40/60. Reserve10 minutes inside each cap for setup/evidence; no new assessment starts after its soft budget. Sum of case caps27,500. Six independent engine/N audit jobs capped120 minutes each; prepare30, combined gate15, sequential durable publication270, registration1. Total reserved28,536; unallocated1,464 under30,000. Per-cell audit/download/publication avoids retaining all expanded raw evidence on one runner. No paid-runner escalation or automatic retry.

Pin Python3.11.9 and the previously verified package versions; install the frozen package. Freeze normalized source hashes, data/config hashes, exact seeds and thresholds. Push only registers the workflow; manual first-attempt dispatch performs fits. Verify run/source/attempt and every expected case artifact upload. Independently replay observation losses, splits, parameters, tuning/outer-fit counts, checkpoint/network decisions and input identity. Retain raw case archives including incomplete evidence. Publish engine/N bundles below1.5GiB, split when needed, and verify each remote asset by download and SHA256. Only publication has repository write permission. Preserve reports and raw evidence in a prerelease, rather than relying on expiring Actions artifacts.

## Interpretation limits

A pass supports recovery of this strong linear Gaussian chain with four candidate predictors at these sample sizes and shipped settings. It does not establish causal direction, weak effects, nonlinear chains, interactions, high dimensionality, correlated direct predictors, or a universal minimum N. RF/GBM/LightGBM have different tuning grids: this is not an equal-cost predictive-performance competition. Read N80 and N200 outcomes separately; do not pool a failing cell with a passing one. No future validation allocation is included.
