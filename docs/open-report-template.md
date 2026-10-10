# OPEN Analysis Report

For a unified current-method report with training-isolated root evaluation,
reference network, root effects and explicit unassessed diagnostics, use
[`analyze_researcher` and `write_researcher_outputs`](procedure-evaluation.md).
That workflow generates `REPORT.md`; its fold networks and full-data reference
have distinct evidence directories. The historical template below remains for
existing saved bundles and does not itself run complete-procedure evaluation.

> Complete the fields appropriate to the saved analysis method. Keep its artifact bundle with this report so reported values can be traced to their source. Historical placeholders below belong to whole-pipeline bundles.

## Layer-consensus reporting (current shipped configuration)

Identify `method=layer_consensus` and status before interpreting the network.
Use `analysis_metadata.json`, `layer_evidence.json`, `assessment_ledger.json`,
`nodes.csv`, `edges.csv`, `shared_groups.csv`, and `local_targets.csv`.
The historical sections below apply only to `whole_pipeline` bundles.

Report assigned depths and parents, not modal depth distributions or node
selection probabilities. For each finalized target, state its frozen candidate
set, selected/planned/completed counts, retained unique or exact joint relation,
signed strength medians and separate measurement counts. Report any unfinalized
layer separately; it contributes no new reference nodes. Low-gain fits are
completed nonselections; failures are not nonselections.

Use one main reference graph. Frequencies are conditional on earlier finalized
layers; they are neither confidence nor empirical whole-path probabilities.
Reconstruct the complete network for threshold sensitivity. Local CV performance
is conditional on discovered candidates and is not held-out evaluation of discovery.

## Analysis overview

- **Root:** `{{root}}`
- **Max depth:** `{{max_depth}}`
- **Thresholds:** `{{thresholds}}`
- **Learner configuration:** `{{learner_configuration}}`
- **Resampling procedure:** `{{resampling_procedure}}`
- **Sensitivity analysis:** `{{sensitivity_analysis}}`

## Network results

### Node selection probabilities and depth distributions

| Node | Selection probability | Depth distribution |
| --- | --- | --- |
| `{{node_name}}` | `{{selection_probability}}` | `{{depth_distribution}}` |

### Edge-class probabilities

| Target | Predictor or group | Edge class | Probability |
| --- | --- | --- | --- |
| `{{target}}` | `{{predictor_or_group}}` | `{{edge_class}}` | `{{edge_class_probability}}` |

### Shared co-membership probabilities

| Target | Member pair | Co-membership probability |
| --- | --- | --- |
| `{{target}}` | `{{member_pair}}` | `{{shared_co_membership_probability}}` |

### Screening-partner probabilities

| Target | Screened candidate | Partner | Screening-partner probability |
| --- | --- | --- | --- |
| `{{target}}` | `{{screened_candidate}}` | `{{screening_partner}}` | `{{screening_partner_probability}}` |

## Local target results

Repeat this section for every target evaluated by OPEN.

### Target name: `{{target_name}}`

- **Depth:** `{{depth}}`
- **Candidate set:** `{{candidate_set}}`
- **Baseline risk:** `{{baseline_risk}}`
- **Full-model risk:** `{{full_model_risk}}`
- **Total relative gain:** `{{total_relative_gain_g_T}}`
- **Learner:** `{{learner}}`
- **Target-evaluation count:** `{{target_evaluation_count}}` resamples

| Candidate or tested group | Broad relevance | Unique relevance | Tested group relevance | Classification | Stability probability |
| --- | --- | --- | --- | --- | --- |
| `{{candidate_or_group}}` | `{{broad_relevance_b_j}}` | `{{unique_relevance_u_j}}` | `{{tested_group_relevance_u_G}}` | `{{classification}}` | `{{stability_probability}}` |

## Interpretation

Describe each relation as a **predictive contribution under the configured analysis**. Predictive relations do not establish causality. Avoid causal language unless a separate research design supports it.

Any network visualization is supplementary: **visual distances are display-only** and do not encode additional evidence or alter the reported analysis.

### Exploratory display and consensus

- **View and display filter:** `{{exploratory_or_consensus}}`, `{{minimum_display_frequency}}`
- **Consensus cutoff:** `{{consensus_cutoff}}` over all completed networks
- **Whole-path evidence:** `{{measured_path_counts_or_unavailable}}`; do not assemble paths from pooled arrows

For new bundles, read the versioned `exploration` field in `analysis_metadata.json`.
Historical bundles without that field support only the consensus display.
Report whole-network repeatability separately from eligibility and strength:

| Relation | Selected / complete | Selected / eligible | Unknown eligibility | Signed median ΔMSE | Median per-run ΔMSE / baseline | Raw / baseline score counts | Measurement scope |
| --- | --- | --- | --- | --- | --- | --- | --- |
| `{{relation}}` | `{{selected_complete}}` | `{{selected_eligible_or_unknown}}` | `{{unknown_count}}` | `{{signed_raw_median}}` | `{{signed_baseline_relative_median}}` | `{{raw_and_relative_counts}}` | `{{strength_scope}}` |

The exploratory default shows every observed unique, screened-by, and joint
shared relation, highlighting the consensus subset. Opacity is
`0.25 + 0.75 × whole-network frequency`; width is
`0.8 + 3.2 × sqrt(clip(median per-run ΔMSE / baseline, 0, 1))`.
The fixed width scale avoids making a weak sole predictor look maximally strong.
Keep negative scores in the evidence table. Baseline-normalized predictive loss
reductions are not additive explained-variance shares or calibrated confidence.
Unique strength includes available scored nonselections; shared strength currently
uses selected joint ablations only. The optional gain-relative median has its own
measurement count, which can be smaller when low-gain scores lack normalization.
Missing measurements remain unavailable.

Eligibility requires an evaluated target and the source or all group members in
its actual candidate set. Low-gain stops count as opportunities, even if no
ablation score was computed. When legacy eligibility is incomplete, report the
conditional frequency as unknown. A high eligible rate can be based on very few
opportunities; it does not replace the whole-network consensus denominator.

Pooled arrows with modal node depths may form cycles or combine mutually
incompatible runs. They are not a single fitted network. Dashed arrows show
screening; dotted connectors link shared members to one joint predictive arrow.
Display filtering changes only visibility. An omitted relation is not evidence
of independence.

## Saved P5.1 artifacts

Set `P5.1_OUTPUT_DIR` to the directory containing the exported analysis files. These links use that placeholder directory:

- [`nodes.csv`](<P5.1_OUTPUT_DIR/nodes.csv>)
- [`edges.csv`](<P5.1_OUTPUT_DIR/edges.csv>)
- [`shared_groups.csv`](<P5.1_OUTPUT_DIR/shared_groups.csv>)
- [`shared_comembership.csv`](<P5.1_OUTPUT_DIR/shared_comembership.csv>)
- [`local_targets.csv`](<P5.1_OUTPUT_DIR/local_targets.csv>)
- [`analysis_metadata.json`](<P5.1_OUTPUT_DIR/analysis_metadata.json>)
