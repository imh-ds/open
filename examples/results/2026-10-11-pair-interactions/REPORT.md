# Pair interaction smoke demonstration

600 synthetic observations, five predictors and Well-being as the outcome. Sleep Quality, Physical Activity, Social Support and Perceived Stress have additive contributions; Support and Stress also have a product interaction. Work Demands is independent noise. These are illustrative standard-normal scores, not actual survey measurements.

Analytic checks: nonlinear separate effects produce no candidate junction; pure and mixed functions identify the known Support–Stress pair without marginal importance filtering. This checks calculation, not statistical detection performance.

Actual fitted RF: CV R² 0.657; known interacting pair ranks 1 of ten pairs. Screening used 12,480 evaluated prediction rows across three folds, with zero additional model fits or tuning searches for the pair supplement. The larger effect comparison still performs its usual full, ablation, tuning and explanatory fits.

Whole comparison/export took 14.7 seconds. This is a single machine/synthetic-case smoke timing, not a general runtime guarantee.

[RF candidate graph and supported local-contrast plots](rf/INTERACTIONS.md), [pure known interaction](pure_interaction/INTERACTIONS.md), [known additive function](additive/INTERACTIONS.md), [mixed effects](mixed/INTERACTIONS.md).

RF can learn incidental nonadditivity involving other pairs. A shortlist entry does not establish a true population interaction; this demo does not calibrate false positives, stable recovery or minimum sample sizes. The recursive graph is not modified. Pure/mixed interactions are visually distinguished using a separate exploratory junction view, not automatic relabelling of ordinary arrows.

In this RF fit, 3 additional pairs received supported detailed examinations across folds despite not being interaction terms in the generating function. This is evidence of incidental fitted-model patterns and a reason to keep the candidate label; the screen is not a validated population interaction detector.

Raw model settings, fold scores/support, budgets and identities are in rf/comparison.json and CSVs. receipt.json records runtime, warnings and source hashes; sample.csv preserves the synthetic data.
