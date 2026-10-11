# Exploratory fitted-model interaction patterns

A junction is a candidate joint pattern in fitted predictions, not a validated population interaction. It does not establish causal moderation. Ordinary predictive arrows do not certify separate main effects.

Shortlisting is computational triage. Scores describe local mixed prediction contrasts, not variance explained, coefficients, p-values or full second-order ALE. They can include higher-order interactions. Absence of a junction is not evidence of no interaction.

## random_forest

Planned folds: 3; evaluated prediction rows: 12480; additional model fits: 0.

|Pair|Supported screens / planned folds|Shortlisted folds|Supported details|Screen RMS (outcome SD)|Status|
|---|---:|---:|---:|---:|---|
|perceived_stress + physical_activity|3/3|0|0|0.2368|complete_descriptive|
|perceived_stress + sleep_quality|3/3|1|1|0.3086|complete_descriptive|
|perceived_stress + social_support|3/3|3|3|0.4383|complete_descriptive|
|perceived_stress + work_demands|3/3|0|0|0.07063|complete_descriptive|
|physical_activity + sleep_quality|3/3|1|1|0.2841|complete_descriptive|
|physical_activity + social_support|3/3|0|0|0.1609|complete_descriptive|
|physical_activity + work_demands|3/3|0|0|0.0707|complete_descriptive|
|sleep_quality + social_support|3/3|1|1|0.2473|complete_descriptive|
|sleep_quality + work_demands|3/3|0|0|0.07935|complete_descriptive|
|social_support + work_demands|3/3|0|0|0.05359|complete_descriptive|

Displayed 4 of 4 supported detailed candidates; rank is not statistical acceptance.

```mermaid
flowchart LR
  target["wellbeing"]
  n0["perceived_stress"]
  n1["physical_activity"]
  n2["sleep_quality"]
  n3["social_support"]
  joint0{"Candidate joint pattern"}
  n0 --> joint0
  n3 --> joint0
  joint0 --> target
  joint1{"Candidate joint pattern"}
  n0 --> joint1
  n2 --> joint1
  joint1 --> target
  joint2{"Candidate joint pattern"}
  n1 --> joint2
  n2 --> joint2
  joint2 --> target
  joint3{"Candidate joint pattern"}
  n2 --> joint3
  n3 --> joint3
  joint3 --> target
```

![Local joint contrasts](interaction-random_forest-0.png)

This shows the first supported detailed fold in saved order, not an aggregate population surface. Cell counts refer to sampled reference observations. Signed means may cancel when effects vary with other variables; RMS retains their magnitude. Bin corners may leave the observed manifold. Other folds and boundaries remain in comparison.json.

![Local joint contrasts](interaction-random_forest-1.png)

This shows the first supported detailed fold in saved order, not an aggregate population surface. Cell counts refer to sampled reference observations. Signed means may cancel when effects vary with other variables; RMS retains their magnitude. Bin corners may leave the observed manifold. Other folds and boundaries remain in comparison.json.

![Local joint contrasts](interaction-random_forest-2.png)

This shows the first supported detailed fold in saved order, not an aggregate population surface. Cell counts refer to sampled reference observations. Signed means may cancel when effects vary with other variables; RMS retains their magnitude. Bin corners may leave the observed manifold. Other folds and boundaries remain in comparison.json.

![Local joint contrasts](interaction-random_forest-3.png)

This shows the first supported detailed fold in saved order, not an aggregate population surface. Cell counts refer to sampled reference observations. Signed means may cancel when effects vary with other variables; RMS retains their magnitude. Bin corners may leave the observed manifold. Other folds and boundaries remain in comparison.json.
