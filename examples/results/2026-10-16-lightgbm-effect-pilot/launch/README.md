# Approved LightGBM effect pilot: launch controls

Two paired sample-size waves: N=200 and N=300. Each uses ten datasets at each
population link correlation .20/.30/.40, giving 30 analyses per wave and 60 total.
Both true links X1 -> X2 -> Y have the specified correlation; Z1/Z2 are unrelated.
N=200 takes the first200 observations of its paired N=300 dataset. These are
exploratory ballpark estimates, not qualification or minimum-N evidence.

The owner approved the N=200 protocol and then requested N=300 under the same
plan. The operational interpretation, stated before implementation, is a2,000
runner-minute ceiling per wave,4,000 combined. Reserved maxima are1,981 each,
3,962 combined. The waves serialize; up to20 case jobs run in parallel within a
wave. Each case has a60-minute hard job cap, a bounded fit step and separate
interrupted-checkpoint packaging/upload time. No automatic fit retries.

The shipped LightGBM engine/grid,100 resamples,80% subsampling,CV5x2,tuningCV3,
retention70%,other thresholds and recursion are unchanged. Report true direct and
indirect links, exact full-network/depth recovery, extra findings, unknown cases
and whether the indirect target was reached. Saved .70–.90 snapshots at2-point
increments describe already-tested comparisons only, not alternate recursive runs.

Frozen manifests and prelaunch verification are beside this file. The complete
[protocol](../../../lightgbm_effect_pilot_protocol.md) and
[documented historical seed check](../../../study_manifests/lightgbm-effect-seed-audit.json)
define the controls. All14 new controls and39 historical helper checks passed;
an independent review's timeout-preservation and partial-provenance findings were
fixed and rechecked. No scientific model fitting occurred locally during setup.

Only public pilot files transfer to the Actions branch; private outline history
is retained locally. Each wave will preserve its reports and available raw case
archives in a dedicated prerelease, verify remote asset hashes by download, and
retain incomplete evidence and planned denominators. Dispatch provenance is saved
separately once run IDs and the actual public source are verified.

## Dispatched waves

- [N=200 run39000850264](https://github.com/imh-ds/open/actions/runs/39000850264):
  first manual attempt, source8108299ffc777e8ec584bc08fef475ed833f7247;
  preparation succeeded, live authorization equals the frozen N200 manifest,
  all30 expected case jobs verified. Scientific outcomes pending.
- [N=300 run39000866314](https://github.com/imh-ds/open/actions/runs/39000866314):
  first manual attempt at the same source, queued behind N200. Its live
  preparation/matrix check awaits execution; the local frozen controls passed.

`dispatch-verification.json` records the checked scope. Run API snapshots,
N200 startup job records and its live producer authorization are retained beside
the frozen manifests. These are startup records, not completion evidence.
