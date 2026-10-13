# Proposed realistic-interaction development study (D-043)

**Status: proposed parameters, awaiting owner approval. No datasets have been generated, models fitted or Actions workflows dispatched under this protocol.**

## Purpose and decision

Find where the current interaction procedure works with more typical predictor correlations, noisy continuous questionnaire-like scores and weaker or saturating interactions. Produce understandable detection and plot summaries to select a later, separate confirmation study. This is a development map, not a production-confidence certification.

The owner requested a new GitHub study covering realistic correlations, measurement error and weaker or differently shaped interactions. The exact levels, counts and budget below are proposed design choices, not previously specified owner parameters.

Options considered: a focused development map; full confirmation of every new condition; or a large crossed grid including ordinal items, missingness, many predictor counts and searched pairs. Recommend the development map because the new conditions need fresh calibration and known-truth calculations first. A full confirmation grid would multiply simulation costs before identifying useful conditions. This proposal limits the study to the priority factors.

## Design

| Factor | Proposed levels |
|---|---|
| People per dataset | 150, 300, 600 |
| Predictors | 5; only the first two affect the signal outcome |
| Correlation between the true underlying interaction predictors | 0.1, 0.3, 0.5 |
| Predictor reliability | 1.0 and 0.8 |
| Main signal family | Mixed: separate effects plus an interaction |
| Interaction shapes | Product and a smoothly saturating moderator |
| Lower/higher coefficients within each shape | 0.2 and 0.4 |
| Positive reference | Product interaction with coefficient 0.8 |
| Outer / inner CV | 3 / 3 |
| Independent diagnostic assessment | 1,000 separately generated rows per task, never used for selection |

There are 18 design contexts: three sample sizes × three underlying correlations × two reliability levels. The remaining three predictors are independent Gaussian covariates. The study does not add the pure-interaction family; that family already has separate D-042 evidence and would enlarge this run.

“Reliability 0.8” means 80% of a measured predictor's variance is true-score variance under a classical independent-error model. It is not a claim that Cronbach's alpha, omega or an observed reliability estimate can automatically be substituted into this model. No extra measurement error is added to the outcome; outcome residual noise is specified below.

Keeping underlying correlations fixed isolates the measurement change in the population generator. Measurement error reduces expected **observed** correlations:

| Underlying correlation | Reliability 1.0: observed correlation | Reliability 0.8: observed correlation |
|---:|---:|---:|
| 0.1 | 0.1 | 0.08 |
| 0.3 | 0.3 | 0.24 |
| 0.5 | 0.5 | 0.40 |

The noisy panel therefore covers observed correlations 0.08–0.40, rather than pretending they remain exactly 0.1/0.3/0.5. Both underlying and observed values must appear in reports. If the owner instead wants exactly matched observed correlations across reliability levels, that requires a different underlying correlation design and must be decided before implementation.

## Generators and comparable interpretation

Generate latent predictors `L` with standard-normal marginals, correlation `gamma` between L1 and L2 and independent remaining predictors. Generate independent standard-normal measurement errors `E` for all five predictors, and use:

`Z = sqrt(R) * L + sqrt(1 - R) * E`

Models receive measured predictors Z, not L or E. Their marginals retain unit variance; the first pair's observed correlation is `R * gamma`. Always consume the measurement-error draws, including R=1, so generation order is explicit. Use a separate residual draw after predictor and measurement-error generation. Training and independent-assessment seeds are disjoint.

For product signals, h(u)=u. For saturating signals, `h(u)=tanh(u)/s`, where `s=sqrt(E[tanh(G)^2])` for standard-normal G. This fixes the moderator function's marginal variance at one; it does not make the two shapes equal in unique interaction information.

For each mixed signal:

`f(L) = L1 + 0.6 * L2 + b * (L1 * h(L2) - E[L1 * h(L2)])`

Set `Y = f(L) + epsilon`, with independent Gaussian residual variance `Var(f(L)) * 2/3`. Overall latent-data predictability is therefore 0.6. Adding predictor measurement error naturally reduces what can be predicted from Z; do not compensate by increasing b or decreasing the residual noise.

For product interactions, `Var(f)=1.36+1.2*gamma+b^2*(1+gamma^2)`. For saturation, compute the interaction variance from Gaussian moments:

`Var(L1*h(L2)) = (1-gamma^2)*E[h(G)^2] + gamma^2*E[G^2*h(G)^2] - gamma^2*E[G*h(G)]^2`.

The interaction is even under simultaneous sign reversal, while the additive terms are odd, so their population covariance is zero. Use this variance in the residual-noise calculation. The 0.2/0.4 levels are fixed coefficients, not universal “small/medium” effect-size definitions. The 0.8 product condition is an explicit positive reference. Do not claim equal distinct standardized strength across correlations or shapes, or describe this as a matched-q experiment.

## Correct truth for measured predictors

Evaluate plots against the conditional predictive relationship **E[Y | Z]**, not the unobserved latent formula evaluated as if measurement error were absent.

For the first two predictors, let `S_L=[[1,gamma],[gamma,1]]`, `S_Z=[[1,R*gamma],[R*gamma,1]]`, `K=sqrt(R)*S_L*inverse(S_Z)`, and `V=S_L-R*S_L*inverse(S_Z)*S_L`. Then `L | Z=z` has mean `mu=K*z` and covariance V.

For product interactions, the conditional product expectation is `mu1*mu2+V12`. For saturation it is `mu1*E[h(U)]+V12*E[h'(U)]`, where `U ~ Normal(mu2,V22)`. Include the centered population product expectation and the conditional linear main effects. Handle zero conditional variance directly at R=1.

Compute the Gaussian moments and conditional expectations with 256-node normalized Gauss-Hermite quadrature, independently checked against 128 nodes. Require absolute differences below 1e-8 for the global moments and every required truth surface and assessment prediction. A disagreement is a numerical-truth failure: stop rather than fit or label that case. The generator is approved only after tests compare these formulas with independent mathematical checks. Static design calculations already show the global saturation-normalization moments agree; this is not generated-data evidence.

Use the existing five-by-five predictor grid. Anchor the conditional truth surface by subtracting both zero-coordinate slices and adding the zero/zero prediction. Compare its relative supported pattern with each estimated outer-fold curve. All known-truth quantities are diagnostic only: they cannot affect tuning, score, threshold or display eligibility.

A product-specific ideal-additive risk or distinct-gap diagnostic may be reported only when independently derived for the measured-input conditional mean. Do not apply the old latent-product formula to noisy or saturating data. Unsupported ideal-additive diagnostics remain explicitly unassessed; fitted additive comparison risks are still reported.

## Null classes and a measurement diagnostic

For each observed-input context, define the two genuine predictive null classes:

- Noise: Y is independent standard-normal noise.
- Observed additive curvature: `Y = Z1 + 0.6*(Z2^2-1) + epsilon`, with residual variance `1.72*2/3`. The conditional mean is additive in the actual model inputs Z.

These are predictive nulls in the measured variables. They must not be replaced by a latent curved-additive generator that can become nonadditive after conditioning on noisy correlated measurements.

Include a separate diagnostic in the nine R=0.8 contexts: `Y = L1 + 0.6*(L2^2-1) + epsilon`, with residual variance `1.72*2/3`. Its latent formula contains no interaction, but its observed conditional mean is `mu1+0.6*(mu2^2+V22-1)` and can contain an observed predictive interaction. Report it as **measurement-induced predictive structure**, not a false latent-moderation discovery or a genuine predictive null. R=1 observed-additive null results provide the error-free reference; do not pool them with the noisy diagnostic.

This diagnostic prevents a scientifically misleading interpretation of measurement-error results. It adds no causal claim, and is not used to learn thresholds.

## Fresh calibration, then development assessment

Keep the minimum-CV weighted spline candidate, its existing grids, separate tuned additive baseline and matched-settings additive counterpart. Assess only x1:x2 and x1:x3. The latter is an unrelated-predictor control in the main alternatives. An eligible candidate has positive pooled overall CV gain and positive matched gain in at least two outer folds; its score is the nonnegative pooled matched gain when eligible, otherwise zero.

**Calibration:** 100 noise and 100 observed-additive datasets per context: 18×200=3,600 datasets. For each context, freeze the maximum nonnegative candidate score across both pairs and both null classes, following the existing null-maximum recipe. Do not reuse D-040/D-042 cutoffs for these new correlations. Preserve all calibration scores. Every calibration identity must complete and be independently audited before committing the threshold artifact; otherwise development dispatch remains blocked.

**Development assessment:** independently seeded datasets, with 80 per context and benchmark:

- Two observed predictive null classes: 18×2×80=2,880.
- Product and saturation, each at b=0.2 and b=0.4: 18×4×80=5,760.
- Product b=0.8 reference: 18×80=1,440.
- Latent-additive measurement diagnostic, R=0.8 only: 9×80=720.

Total development assessment: **10,800 datasets**. Total with calibration: **14,400 datasets**. All development identities are planned before calibration; no post-result condition selection or changing coefficients/counts/cutoffs. There is no null-pass filtering of development conditions: this map intentionally records failures as well as successes. Do not use this design to change historical confirmation labels.

## Outputs and interpretation

Report point estimates and exact **two-sided 95% descriptive binomial intervals**, unadjusted for the many conditions, for null false findings, unrelated-pair selection, known-pair detection and conditional relative-pattern agreement. These intervals are development summaries, not simultaneous assurance or proof that an individual finding is true. Do not assign D-042-style qualification or production-confidence labels from 80 datasets per group.

Display eligibility remains truth-blind: selection, common supported locations, minimum four noncentral locations and minimum fold-to-fold cosine 0.8. Count a relative-pattern match at minimum fold-to-conditional-truth cosine 0.8. Preserve displayed/80 counts separately from correct/displayed counts. Zero displays is unresolved, not zero accuracy. For the measurement-induced diagnostic, show estimated and true observed patterns separately from latent formula interpretation.

Collect supported-grid standardized contrast RMSE and relative curve-magnitude discrepancies against the conditional teacher as descriptive diagnostics, without extra fits or an accuracy-certification cutoff. Do not infer magnitude validity from cosine agreement. For all rates, preserve missing/error cases as unknown, keep the planned denominator visible and prohibit support claims from incomplete groups.

Show actual sample sizes: N150 outer training100/inner66–67; N300 outer200/inner133–134; N600 outer400/inner266–267. Full-data independent diagnostics use N people for fitting and 1,000 independent rows for assessment; they do not add training participants or influence decisions.

Produce one simple plot for detection versus sample size within each correlation/reliability/shape panel, a separate display-coverage/pattern summary and a practical report explaining which factors harmed performance. Do not rank all scenarios by raw coefficients or pool away failures. Recommendations for a later confirmation study require a separate frozen protocol, new disjoint identities and appropriately powered counts.

## GitHub Actions and cost boundaries

Propose two manual stages in a new workflow, with independent namespaces and receipts: calibration, then development after audited committed threshold publication. Reuse ordinary windows-2022 runners, the pinned D-042 runtime, two single-thread workers, 80 tasks per shard and sparse guarded checkout. The matrices are 45 calibration and 135 development workers. Set max-parallel=256; actual capacity remains account-limited. Both matrices fit GitHub's [256-job matrix limit](https://docs.github.com/en/actions/reference/limits). Do not promise that every shard runs simultaneously.

Worker limit20min, setup4min, computation12min, finalization4min; prepare15min and aggregate60min per stage. The maximum timeout-based cap is **180×20+2×(15+60)=3,750 runner minutes**, not an expected duration or billing estimate. Planned computation is **9,964,800 fitting/tuning calls plus 288,000 saved-setting verification refits**. Native preflight/test computation is recorded separately.

Before any Actions dispatch, pass mathematical-truth/generator tests, legacy-freeze checks, fresh independent review and a representative native 80-task calibration preflight within12min, with verified zero-fit resume. After calibration, check representative noisy saturation and strong-reference signal cases for timing before development dispatch. Count those repeats separately, never as new independent replicates. If timing, truth accuracy, storage, source or budget checks fail, stop for a revision; do not silently reduce cases, change the method, increase timeouts or add paid runners. No automatic workflow/model retries.

Use new fixed seed namespaces (propose708000000 training,718000000 independent assessment,728000000 splits, plus global task indices); freeze exact ordering and audit collisions against all historical manifests before generating datasets. New manifest names should use the `prospective-interaction-realistic` prefix so the existing frozen historical-seed scan remains unchanged. Freeze all source/runtime/manifest/threshold hashes before their respective stages, verify exact public blobs and dispatch each stage once.

Retain every raw prediction, curve, checkpoint, attempt-cost record, inventory, digest, API receipt and standalone verifier. Stream one extracted shard at a time during aggregation. Measure actual raw/compressed/temporary-disk sizes in preflight; require aggregate completion within60min. Commit raw preservation separately from interpretation, use bounded public transfers and verify remote hashes before claiming public archival completion. Retain partial work and all known/unknown failed costs.

## Scope and launch gate

This proposal covers continuous Gaussian latent variables with classical predictor measurement error, five predictors, two fixed pairs and one spline candidate. It does not validate ordinal item scores, missingness, outliers, clusters, searched-pair multiplicity, other engines, causal moderation, a universal minimum N or a production confidence feature. Those remain separate investigations.

Owner approval of these parameters and the development-only purpose is required before implementation and generation. This document is the concrete proposed design, not an implemented or launched study. Following approval, prepare the implementation plan, add tests/independent verifiers and pass the frozen launch gates. Preserve the existing studies and their decisions.
