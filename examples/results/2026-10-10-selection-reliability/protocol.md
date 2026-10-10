# Root-selection reliability development study

Frozen before dispatch, 2026-10-10. Authorized follow-up to the interaction
diagnosis. This study investigates current behavior; it changes no production
selection rule and is not final scientific calibration.

## Aim and methods

Determine whether ordinary research settings preserve useful root inputs, and
whether loss coincides with weak local prediction. Use the shipped GBM tuning
grid; layer-consensus root comparison `y ~ a + b`, maximum depth1. Performance
CV5folds/2repeats, tuning3folds, subsampling80%, B100, discoveryseed613, and all
shipped relevance/group/stability thresholds (retention70%). GBM is fixed because
it exhibited the original failure; do not select an engine from test performance.
Use the actual production sample/assessment/aggregation functions. This isolates
root selection and does not evaluate deeper layers or same-depth relationships.

48 independently generated discovery datasets: four scenarios, n80/180/400,
four repetitions per cell. Repetitions are independent datasets, distinct from
the100 dependent stability assessments within one dataset. Larger n values are
not prefixes of smaller datasets. Noise SD.3 in all signal scenarios:

- Noise: a,b,y independent standard normal. Neither input useful.
- Ordinary: independent normal a,b; y=2a+error. a useful, b noise.
- Redundant: a normal, b=a+normal noise SD.03; y=2a+error. Either input can predict;
  omitting one is not itself a predictive failure. Track preservation of the
  shared-group relation as well as availability of at least one useful input.
- Interaction: independent normal a,b; y=2ab+error. Both inputs needed to represent
  the true conditional mean; neither has a marginal mean association.

Train seed=171001+scenario_index*10000+size_index*100+repetition_index;
scenario order noise/ordinary/redundant/interaction, size order80/180/400,
repetition_index0..3. Independent test seed=train_seed+1000000; n_test1000.
No missingness, measurement error, additional nuisance inputs or non-Gaussian
predictors are included. Do not generalize results beyond these cases.

## Evaluation and evidence

Finalize root consensus before predicting test participants. Independently
tune/refit retained-input and all-input GBM on the full discovery dataset using
three-fold CV. No retained inputs means training-mean prediction. No test score
changes the retained set or settings. Root inputs are not recursively predicted.

Record every planned local assessment, warning/failure, full-model gain, stop,
broad/conditional contributions, classes/promotions, selected joint identities,
actual sample/fit/tuning row sizes, and realized sample/split fingerprints.
Record conditional relation counts and consensus retained inputs. Preserve all
training/test rows, test predictions, refit parameters, source/config/protocol
fingerprints and elapsed cost. Failures retain planned denominators and withhold
prediction for incomplete cases. Catch worker failure into an explicit case file.

Report root recovery (a for ordinary, both for interaction, at least one for
redundant), full shared-group preservation for redundant, any false input in
noise and b retained in ordinary, and incomplete-case counts. Report all-input
and retained-input baseline-relative prediction skill on identical test rows.
Skill loss=all-input skill minus retained-input skill, signed. Also record losses
>.05 as a predefined descriptive flag; this is not a calibrated acceptable-gap
cutoff or a significance claim. All scores remain signed.

Within each scenario/n cell report counts out of four planned datasets, range,
mean skill loss and Monte Carlo SE of that mean (sample SD/sqrt completed count).
Show Wilson95% intervals for dataset-level recovery/false-input counts when all
four cases complete; failures withhold these performance proportions. Four
datasets per cell give low precision, even if all succeed or none fail. Never
treat B100 as100 independent replications or as confidence in an edge.

## Decision rule and stop condition

Do not patch simply because a graph is sparse. Determine whether misses coincide
with low local gain and/or appear despite strong full-input prediction. A loss
of useful inputs at n400 in multiple independently generated datasets warrants
designing a separate selection safeguard, with new development and confirmation
cases. Loss concentrated in the smaller samples supports training-size/weak-fit
diagnostics first; it cannot establish a safe minimum sample size. Ordinary/noise
false retention or redundant-class fragmentation must be reported alongside any
interaction recovery. No production policy is accepted from this small study.

Do not retune settings, extend sample sizes, drop failed datasets or select seeds
after viewing results. Any new proposal requires a separately recorded study and
fresh confirmation identities. Final P6 remains unaccepted. Follow simulation
reporting guidance: [Morris, White & Crowther (2019)](https://pmc.ncbi.nlm.nih.gov/articles/PMC6492164/).
