# Saved-record diagnosis of N=200 nuisance links

The engines found real paths, but occasionally kept unrelated variables too. The saved records reproduce those selections without an arithmetic discrepancy. They do not establish a single software defect that explains them.

This supplement examines all 165 N=200 cases from [the frozen signal study](../2026-10-16-multiengine-signal-validation/README.md), including all 13 false links. No estimators were fitted, no data generated, no Actions jobs dispatched, and no defaults or historical gates changed.

## What the false links looked like

All 13 were nuisance predictors of X2 at the indirect stage. They survived in 71–88 of 100 subsamples. Their median apparent unique improvements were 1.95–4.46% of baseline cross-validated prediction error. This is a prediction-error contrast, not population variance explained or a standardized regression coefficient.

Every selected nuisance predictor performed worse than the baseline when used alone. Its apparent benefit arose when added alongside the other predictors. This is useful diagnostic evidence, but not a reason to reject every predictor lacking a standalone association: genuine conditional relationships, suppressors and interactions can behave that way too.

Their absolute correlations with the known generator error were at most .137. That does not rule out nonlinear sample patterns or influential observations; this analysis did not perform an outlier/influence investigation.

## Did Random Forest feature sampling cause it?

The suspected mechanism exists: with `max_features=.7`, removing a predictor changes the nominal feature budget from two of three to one of two. That changes the chance of initially offering the true predictor at a split. Trees may inspect additional features when no valid split is found; these nominal budgets are not observed split counts.

However, all six false RF links also had a positive average drop-one loss difference in saved contexts using `max_features=1.0`, where all features were offered. The rounding mechanism is therefore insufficient as an explanation. The contexts were selected by tuning, not randomized; folds and subsamples overlap. This is not a controlled comparison proving that feature sampling has no effect.

All 110,000 saved RF paired fold contexts shared their full/reduced split identities and fitted hyperparameters. The raw drop-one risk arithmetic reproduced the saved relevance scores. No RF patch is justified by this evidence alone.

## A candidate to test next

For the **already-scored X2 comparisons only**, changing the retention cutoff gives:

| Engine | True links at 70% | False links at 70% | True links at 90% | False links at 90% |
|---|---:|---:|---:|---:|
| RF |55/55|6/110|55/55|0/110|
| GBM |55/55|2/110|55/55|0/110|
| LightGBM |53/55|5/110|50/55|0/110|

There are two nuisance opportunities per dataset, hence 110 false-link opportunities per engine. These counts are not full-network error rates. The 90% choice removes the observed false links while also losing three additional true LightGBM links. It is an exploratory candidate selected after seeing these data, not a validated correction.

Changing a cutoff can change assignments earlier in recursion and therefore the candidate pool at later targets. The table cannot predict the full rerun: nuisance variables excluded from later models under the original assignments could reenter. Weak effects and other data structures are also untested here.

Recommended next step: propose a small, fresh, matched 70% versus 90% **full-recursive** development study at N=200 before choosing a replacement rule. Include both recovery and false-network endpoints; freeze its datasets, budget and acceptance criteria before dispatch. Keep 70% as the shipped default for now. A successful development result would still require independent confirmation.

## Why missing the qualification gate needs careful wording

The original 55-dataset gate permits at most one false network, because the two-sided exact 95% interval must have its upper endpoint at or below 10%. GBM's two false networks therefore failed that gate; the original result remains unchanged. Failure means insufficient evidence to qualify under that rule, rather than proof that the population error rate exceeds 10%.

For illustration, **hypothetically** suppose the true error rate were 5%. The chance of passing that gate with 55 independent validation datasets would be only 23.2%. This follows analytically from a binomial distribution, not a fitted model or new simulation. At 200 validation datasets that probability would be 70.0%, and at 300 it would be 88.1%. These counts refer to repeated datasets, not participants within a dataset. `gate-assurance.csv` preserves the assumptions and exact calculations. Greater replication narrows uncertainty; it does not fix an algorithm's false links.

## Reproduction and evidence

`diagnosis.json` records the original run/source/manifest, three published raw-bundle hashes, diagnostic script hash and all summaries. `target-comparisons.csv` contains 495 candidate comparisons (165 cases times three predictors). `rf-feature-sampling.csv` describes the saved paired RF contexts; these rows are dependent descriptions, not independent tests.

The reader verified the outer bundles, every inner archive against the original audited case rows, receipt identities, frozen data hashes, sealed network/checkpoint digests, all 100 completed X2 assessments per case, promotion numerators, baseline-normalized medians and RF drop-one arithmetic. The X2 warning messages examined were LightGBM dependency deprecations, not convergence warnings; this does not describe warnings at unexamined targets.

With Python 3.11.9 and the frozen study dependencies, from the repository root:

```powershell
python examples/diagnose_signal_false_links.py --bundles <downloaded-N200-release-bundles> --out <new-output-directory>
python examples/signal_gate_assurance.py --out <assurance.csv>
```

Download the three N=200 bundles from the [original durable release](https://github.com/imh-ds/open/releases/tag/tree-recursive-signal-2026-10-16-38891293628). The diagnostic output directory must not already exist. Reproduction involves reading approximately 3.48 GB of compressed evidence; it performs zero model fits. `verification.json` preserves independent output checks and file hashes.
