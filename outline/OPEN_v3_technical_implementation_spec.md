# Outcome-centered Predictive Expansion Network (OPEN)
## Technical Specification and Repository Implementation Guide — Version 3

> **Status:** Working methodological specification for implementation and simulation testing.  
> **Purpose:** Define the exact statistical objects, processing order, classification rules, recursion logic, resampling strategy, output schema, visualization rules, and software architecture required to build the method reproducibly.

---

# 0. Executive specification

OPEN is a **rooted, outcome-centered predictive network procedure** for psychological and behavioral data.

Given a prespecified focal outcome \(Y\), the framework answers:

1. Which variables broadly predict \(Y\)?
2. Which variables contribute **unique** out-of-sample predictive information about \(Y\)?
3. Which variables matter **jointly** because they substitute for one another?
4. Which variables are broadly relevant but become **screened** once a stronger partner or the full predictor set is available?
5. What predicts the variables in the proximal predictive neighborhood of \(Y\)?
6. How does this structure expand recursively into Layer 2, Layer 3, and later layers?
7. Which nodes, edges, classifications, and paths are stable when the **entire procedure** is rerun under resampling?

The output is not a causal DAG and not a global conditional-association graph. It is a **recursive map of predictive relevance centered on a researcher-defined outcome**.

The primary objects are:

- **Broad relevance** \(B\): does a variable predict the target at all?
- **Unique conditional relevance** \(U\): how much predictive performance is lost if a variable is removed while competitors remain available?
- **Group relevance** \(U_G\): how much predictive performance is lost if a related predictor group is removed together?
- **Relationship class**: proximal, shared/redundant, screened, distal, unstable, or negligible.
- **Recursive depth**: the shallowest layer at which a node enters the target-centered network.
- **Visual distance**: a plotting transformation of predictive relevance only; it is not a formal path metric.

The core algorithm expands **outward only into previously unassigned nodes**. This gives every node a shallowest predictive depth and prevents the focal outcome or already assigned nodes from dominating later-layer models.

---

# 1. Scope and non-goals

## 1.1 Primary use case

Use OPEN when:

- one outcome is substantively privileged;
- prediction is scientifically meaningful;
- relationships may be nonlinear;
- predictors may be redundant or overlapping;
- the researcher wants a structured network rather than a flat feature ranking;
- the goal is exploratory or descriptive predictive structure rather than causal identification.

Examples include Anxiety, suicidal ideation, relapse, treatment response, academic persistence, burnout, social isolation, health behavior, or political participation.

## 1.2 Non-goals

OPEN does **not** by itself estimate:

- causal effects;
- mediation;
- causal direction;
- a globally optimal DAG;
- a global conditional-independence graph;
- latent constructs;
- formal psychometric redundancy.

A predictive arrow

\[
X \rightarrow T
\]

means only:

> \(X\) contributes predictive information about target \(T\) under the specified candidate set, learner, loss, validation scheme, and thresholds.

---

# 2. Core terminology

| Term | Definition |
|---|---|
| Root | Prespecified focal outcome \(Y\) |
| Target | Current variable being predicted |
| Candidate | Variable eligible to predict the current target |
| Layer / depth | Shallowest recursive step at which a variable enters the network |
| Broad relevance | Predictive value before requiring uniqueness |
| Unique relevance | Incremental predictive contribution after competing variables remain available |
| Group relevance | Incremental contribution of removing a related predictor group together |
| Proximal | Stable unique predictor of current target |
| Shared/redundant | Group matters jointly while members have little individual unique contribution |
| Screened | Broadly relevant but not uniquely relevant |
| Screened by partner/set | Screened variable belongs to an important dependence group containing one or more uniquely important members |
| Distal | Enters the network at a later recursive layer |
| Unstable | Relationship does not reproduce under whole-pipeline resampling |
| Negligible | Does not meet practical relevance thresholds |
| Visual distance | Plotting-only transform of edge strength |

---

# 3. Data contract

## 3.1 Required dataset structure

Input data must contain:

- one row per independent observational unit for cross-sectional analysis; or one row per observation plus a cluster/person identifier for repeated-measures analysis;
- one focal outcome;
- candidate predictors;
- optional metadata describing variable types and exclusions.

Recommended metadata table:

| field | description |
|---|---|
| `name` | variable name |
| `role` | `root`, `candidate`, `id`, `time`, `exclude` |
| `type` | `continuous`, `ordinal`, `binary`, `categorical`, `count` |
| `construct` | optional construct label |
| `scale_items` | optional source-item identifiers |
| `time_index` | optional measurement wave/time |
| `exclude_reason` | optional |
| `theory_group` | optional investigator-defined redundancy group |

## 3.2 Required exclusions

The implementation must exclude:

- row identifiers;
- subject identifiers as predictors;
- train/test fold labels;
- variables derived directly from the target in a way that creates deterministic leakage;
- target scale items when the target is the total score, unless the analysis is explicitly item-level;
- duplicated columns;
- variables measured after the outcome when prospective prediction is claimed.

Cross-sectional analyses may include consequences of the target, but interpretation must remain predictive.

## 3.3 Overlapping psychometric scores

If two scale totals share items, metadata should flag this, for example:

```text
measurement_overlap = true
```

These variables may remain in an exploratory analysis, but the output must flag the overlap because predictive redundancy may be mechanically induced.

---

# 4. Outcome types and default loss functions

Let

\[
L(y,\hat y)
\]

denote a loss function.

Reference defaults:

| outcome | default loss | baseline predictor |
|---|---|---|
| continuous | MSE | training mean |
| binary | log loss | training prevalence |
| multiclass | multinomial log loss | class proportions |
| count | Poisson deviance if appropriate | training mean |
| ordinal | ordinal log loss if supported | marginal category probabilities |
| survival | prespecified survival loss | null survival model |

The loss function must be fixed before network construction. All importance quantities for a given target use the same loss.

---

# 5. Prediction engine

## 5.1 Learner-agnostic design

OPEN must not hard-code Random Forest importance as the edge definition.

The software should define a generic predictor interface:

```python
class Predictor:
    def tune(self, X, y, cv, loss): ...
    def fit(self, X, y, params): ...
    def predict(self, X): ...
    def clone(self): ...
```

Supported learners may include:

- elastic net;
- generalized additive models;
- Random Forests;
- Extra Trees;
- gradient boosting;
- XGBoost;
- CatBoost;
- Super Learner / stacked ensemble.

## 5.2 Recommended reference learner library

For an initial implementation supporting continuous and binary outcomes:

### Continuous
- Elastic Net
- Random Forest Regressor
- HistGradientBoostingRegressor

### Binary
- Elastic-net logistic regression
- Random Forest Classifier
- HistGradientBoostingClassifier

Optional external plugins: XGBoost, CatBoost, GAM libraries.

## 5.3 Learner selection policies

Support three modes:

- `fixed`: one learner for the entire analysis;
- `per_target`: choose the best learner for each local target using inner CV;
- `ensemble`: combine learners using stacking.

Recommended research default: `per_target` or `ensemble`, if computationally feasible.

---

# 6. Preprocessing and leakage control

All learned preprocessing must occur **inside the training portion** of every resampling split.

This includes:

- imputation;
- scaling;
- category encoding;
- feature transformations;
- dependence grouping;
- hyperparameter tuning.

No test-fold information may determine redundancy groups or model tuning.

Recommended flow:

```text
raw training data
    ↓
imputation
    ↓
encoding / scaling
    ↓
dependence estimation
    ↓
hyperparameter tuning
    ↓
full + ablated model fitting
    ↓
held-out prediction
```

---

# 7. Resampling architecture

There are two separate purposes:

1. estimate local predictive quantities without training-set optimism;
2. estimate **whole-pipeline structural stability**.

## 7.1 Inner performance estimation

Within each pipeline run, use \(K\)-fold CV or repeated sample splitting to estimate risk.

Reference development default:

```yaml
performance_cv:
  folds: 5
  repeats: 2
```

For clustered/repeated data, use grouped folds.

## 7.2 Hyperparameter tuning

Tune only on training data.

Reference default:

```yaml
tuning_cv:
  folds: 3
```

## 7.3 Whole-pipeline stability resampling

Repeat the **entire recursive procedure** over \(B\) resamples.

Reference defaults:

```yaml
stability:
  method: subsample
  fraction: 0.80
  n_resamples_dev: 100
  n_resamples_final: 500
```

For longitudinal/multilevel data, resample persons/clusters rather than rows.

---

# 8. Baseline predictive signal

For current target \(T\), let \(R_0(T)\) be held-out baseline risk and \(R_{\mathrm{full}}(T)\) the held-out full-model risk.

Define:

\[
G_T = R_0(T)-R_{\mathrm{full}}(T).
\]

Define relative total gain:

\[
g_T = \frac{G_T}{R_0(T)}.
\]

This ratio is defined only when `R_0(T) > 0`. For the supported continuous/MSE path, if `R_0(T) = 0`, the training-mean baseline already has zero held-out error and the target stops with `LOW_TOTAL_GAIN`. A target is eligible for relevance normalization only when `g_T >= tau_gain`; this implies `G_T > 0`. Do not add an epsilon to a risk-scale denominator.

A target is eligible for expansion only when:

\[
g_T \ge \tau_{\mathrm{gain}}.
\]

Prototype default:

```text
tau_gain = 0.02
```

This is an engineering default, not a universal scientific constant. Final cutoffs should be calibrated by simulation.

---

# 9. Broad predictive relevance

For candidate \(X_j\), fit a model using only \(X_j\):

\[
f_j:T\sim X_j.
\]

Let held-out risk be \(R_j(T)\).

Define:

\[
B_{j\rightarrow T} = R_0(T)-R_j(T).
\]

Normalize:

\[
b_{j\rightarrow T} = \frac{B_{j\rightarrow T}}{G_T}.
\]

Compute normalized broad relevance only for targets that pass the total-gain stop; their `G_T` is positive.

Broad relevance asks:

> Does \(X_j\) predict \(T\) in a broad/marginal sense?

It does **not** define the main proximal edge.

Reference threshold:

```text
tau_b = 0.05
```

In research-grade mode, low \(B_j\) must not automatically exclude a predictor from LOCO analysis, because suppressor or interaction-dependent variables can have low marginal relevance but nonzero conditional relevance.

---

# 10. Unique conditional predictive relevance

For each candidate \(X_j\), refit the target model without \(X_j\).

Let \(R_{-j}(T)\) be held-out risk.

Define:

\[
U_{j\rightarrow T} = R_{-j}(T)-R_{\mathrm{full}}(T).
\]

Normalize:

\[
u_{j\rightarrow T} = \frac{U_{j\rightarrow T}}{G_T}.
\]

Compute normalized unique relevance only for targets that pass the total-gain stop; retain signed raw and normalized values, including negative values.

Interpretation:

> \(u_j\) is the fraction of the full model's predictive gain lost when \(X_j\) is unavailable.

Negative \(U_j\) values may occur through sampling variation or regularization. Retain them in diagnostic output but treat them as zero for visual edge strength.

Prototype threshold:

```text
tau_u = 0.05
```

A variable is provisionally individually strong when:

\[
u_{j\rightarrow T}\ge\tau_u.
\]

---

# 11. Candidate redundancy-pair generation

Redundancy is evaluated **before individual weak variables are discarded**.

## 11.1 Dependence estimator interface

```python
class DependenceEstimator:
    def fit(self, X): ...
    def pair_score(self, a, b) -> float: ...
```

## 11.2 Reference default for typical psych scale data

For mostly continuous/ordinal scale-level variables:

\[
D_{ab}=|\rho^{\mathrm{Spearman}}_{ab}|.
\]

Candidate pair:

\[
(a,b)\in\mathcal P_T
\quad\text{if}\quad
D_{ab}\ge\tau_{\mathrm{dep}}.
\]

Prototype default:

```text
tau_dep = 0.70
```

Sensitivity values: `0.60, 0.70, 0.80`.

For mixed data, support plugins such as latent Gaussian/copula dependence, polychoric/tetrachoric correlation, normalized mutual information, or user-specified groups.

The dependence step is candidate generation, not final inference.

---

# 12. Pair/group conditional predictive relevance

For each candidate dependence pair \((a,b)\), refit the model with both removed.

Let \(R_{-(a,b)}(T)\) be held-out risk.

Define:

\[
U_{\{a,b\}\rightarrow T}
=
R_{-(a,b)}(T)-R_{\mathrm{full}}(T).
\]

Normalize:

\[
u_{\{a,b\}\rightarrow T}
=
\frac{U_{\{a,b\}\rightarrow T}}{G_T}.
\]

Under the fold-local selection policy, calculate both `U_{a,b}` and its reference gain on the same held-out rows from folds that selected `(a,b)`. If that pair-specific reference gain is not positive, retain the raw contrast and report normalized group relevance as undefined; the pair cannot meet the group threshold.

Prototype threshold:

```text
tau_group = 0.10
```

A pair is provisionally group-important if:

\[
u_{\{a,b\}\rightarrow T}\ge\tau_{\mathrm{group}}.
\]

---

# 13. Pair classification: exact decision rules

For candidate pair \((a,b)\), define:

\[
I_a=\mathbf 1(u_a\ge\tau_u),
\]

\[
I_b=\mathbf 1(u_b\ge\tau_u),
\]

\[
I_{ab}=\mathbf 1(u_{ab}\ge\tau_{\mathrm{group}}).
\]

## 13.1 Pair not group-important

If \(I_{ab}=0\), do not classify the pair as redundant or partner-screening. Classify members individually later.

## 13.2 Shared/redundant pair

If:

\[
I_{ab}=1,
\]

and:

\[
I_a=0,\qquad I_b=0,
\]

classify:

```text
PAIR_SHARED_REDUNDANT
```

Interpretation: the pair matters jointly, but neither member retains substantial unique contribution.

## 13.3 One member screens the other

If \(I_{ab}=1\) and exactly one member is individually strong, e.g.:

\[
I_a=1,\qquad I_b=0,
\]

then:

- \(a\) remains provisionally `PROXIMAL_UNIQUE`;
- if \(b_b\ge\tau_b\), classify \(b\) as `SCREENED_BY_PARTNER(a)`;
- if \(b_b<\tau_b\), classify \(b\) as negligible within this pair.

This resolves chain structures such as:

\[
Stress\rightarrow Depression\rightarrow Anxiety.
\]

At the Anxiety target, Depression may retain high unique relevance while Stress has low unique relevance. Even if the pair is group-important, it is **not** labeled redundant because the information is asymmetric.

## 13.4 Both individually strong

If \(I_{ab}=1\) and:

\[
I_a=1,\qquad I_b=1,
\]

retain both as provisional proximal predictors. Do not label the pair redundant.

---

# 14. Higher-order shared groups

## 14.1 Shared-pair graph

Construct:

\[
H_T=(V,E_{shared}),
\]

where an edge exists when a pair is classified `PAIR_SHARED_REDUNDANT`.

Connected components of size \(\ge2\) are candidate shared groups.

## 14.2 Group validation

For component:

\[
G=\{X_{g_1},\ldots,X_{g_m}\},
\]

fit a model with all members removed.

Define:

\[
U_{G\rightarrow T}=R_{-G}(T)-R_{\mathrm{full}}(T),
\]

\[
u_{G\rightarrow T}=\frac{U_G}{G_T}.
\]

Validate `SHARED_GROUP` only if:

1. \(u_G\ge\tau_{group}\);
2. every member satisfies \(u_j<\tau_u\).

If a merged component fails, retain supported pairwise shared relations rather than forcing the whole component into one group.

## 14.3 Mixed strong/weak group

If a dependence component contains individually strong members:

\[
S=\{j\in G:u_j\ge\tau_u\},
\]

then the group is not a shared group.

Instead:

- members of \(S\) remain proximal;
- weak members with \(b_j\ge\tau_b\) are `SCREENED_BY_SET(S)`;
- weak members with \(b_j<\tau_b\) are negligible.

---

# 15. Optional substitution/complementarity statistic

For tested group \(G\):

\[
C_G=U_G-\sum_{j\in G}U_j.
\]

Interpretation:

- \(C_G>0\): consistent with substitution/shared information;
- \(C_G\approx0\): approximately additive on the selected loss scale;
- \(C_G<0\): possible complementarity, interaction, suppression, or refit adaptation.

This is diagnostic only and must not override classification rules. Negative \(C_G\) must not be called redundancy.

---

# 16. Individual classification after group analysis

## 16.1 Precedence order

1. `PROXIMAL_UNIQUE`
2. `SHARED_GROUP_MEMBER`
3. `SCREENED_BY_PARTNER`
4. `SCREENED_BY_SET`
5. `SCREENED_GENERAL`
6. `UNSTABLE`
7. `NEGLIGIBLE`

Group logic is evaluated before generic screened classification.

## 16.2 Proximal unique

If:

\[
u_j\ge\tau_u,
\]

classify provisionally `PROXIMAL_UNIQUE` unless whole-pipeline stability later fails.

## 16.3 Shared group member

If \(j\) belongs to a validated shared group, classify `SHARED_GROUP_MEMBER`, even though \(u_j<\tau_u\).

## 16.4 Screened generally

If a variable is not already assigned a stronger group-specific class and:

\[
b_j\ge\tau_b
\]

but:

\[
u_j<\tau_u,
\]

classify `SCREENED_GENERAL`.

## 16.5 Negligible

If neither broad nor unique criteria are met, classify `NEGLIGIBLE`.

---

# 17. Which nodes are promoted into the next layer?

## 17.1 Promote

The following become nodes in the next layer:

- `PROXIMAL_UNIQUE` variables;
- **every member** of a validated `SHARED_GROUP`.

## 17.2 Do not directly promote

The following remain unassigned and available as predictors for later recursive targets:

- `SCREENED_BY_PARTNER`;
- `SCREENED_BY_SET`;
- `SCREENED_GENERAL`;
- `NEGLIGIBLE`;
- unstable candidates.

A screened variable such as Stress can therefore fail to enter Layer 1 for Anxiety but remain eligible for Layer 2 when Depression becomes a target.

## 17.3 Why every shared-group member is promoted

Do not create a synthetic composite and do not choose an arbitrary representative.

Every member remains its own substantive variable. The shared group is a **virtual visual grouping object**, not a modeled target.

---

# 18. Recursive expansion: exact core rule

The core network expands **only into previously unassigned variables**.

Maintain:

```python
assigned_depth: dict[str, int]
```

Initialize:

```python
assigned_depth[root] = 0
```

After the root analysis:

- proximal unique nodes receive depth 1;
- members of shared groups receive depth 1.

## 18.1 Layer-synchronous expansion

Suppose current depth is \(d\), with target set \(\mathcal T_d\).

At the **start of the layer**, define:

\[
\mathcal U_d=\{\text{all variables not yet assigned any depth}\}.
\]

For every target \(T\in\mathcal T_d\), use the same candidate snapshot:

\[
\mathcal C_T=\mathcal U_d.
\]

Therefore:

- root is excluded;
- ancestors are excluded;
- current-layer peers are excluded;
- shared-group peers are excluded;
- already assigned shallower variables cannot leak backward.

Run all local analyses for depth \(d\) before assigning any new depths.

After all targets are evaluated:

- union newly promoted nodes;
- assign each depth \(d+1\);
- retain all qualifying parent edges from depth-\(d\) targets.

This prevents processing order from determining depth.

---

# 19. Multiple parents and shallowest depth

A node may qualify for multiple targets in the same preceding layer.

Example:

\[
X\rightarrow D\rightarrow Y
\]

and:

\[
X\rightarrow L\rightarrow Y.
\]

Then \(X\) receives depth 2 and both qualifying edges may be retained.

If a node already has a shallower depth, discovering it later does not change that depth.

Within a network run:

\[
Depth(X)=\min\{\text{qualifying recursive depths}\}.
\]

Later discoveries involving assigned nodes may be stored as diagnostic cross-links but cannot drive further core recursion.

---

# 20. Stopping rules

Stop expansion from target \(T\) if any condition holds:

1. \(g_T<\tau_{gain}\);
2. no variable or group meets promotion criteria;
3. no unassigned candidates remain;
4. maximum depth reached;
5. insufficient effective sample size.

Prototype default:

```text
max_depth = 3
```

A shallow default is recommended because uncertainty compounds with depth.

---

# 21. Whole-pipeline stability

The entire recursive procedure is the object of stability analysis.

For resample \(b=1,\ldots,B\):

1. resample observations or clusters;
2. rerun preprocessing;
3. rebuild dependence candidates;
4. rerun root analysis;
5. rebuild shared groups;
6. assign Layer 1;
7. recursively expand later layers;
8. reclassify every relationship;
9. save the full graph.

No layer is held fixed across resamples.

---

# 22. Stability statistics

## 22.1 Node-depth probability

\[
\pi_j(d)=\frac{1}{B}\sum_{b=1}^{B}\mathbf 1\{Depth_b(j)=d\}.
\]

Overall node selection probability:

\[
\pi_j=\sum_{d\ge1}\pi_j(d).
\]

## 22.2 Edge-class probability

For class \(c\):

\[
\pi_{a\rightarrow t}^{(c)}
=
\frac{1}{B}
\sum_{b=1}^{B}
\mathbf 1\{(a\rightarrow t)\text{ receives class }c\}.
\]

## 22.3 Shared co-membership probability

Do not define stability using exact group IDs.

For pair \((a,b)\):

\[
\pi_{shared}(a,b)
=
\frac{1}{B}
\sum_{b'=1}^{B}
\mathbf 1\{a,b\text{ are in the same validated shared group}\}.
\]

## 22.4 Screened-by probability

\[
\pi_{screened\ by}(a,m)
=
\frac{1}{B}
\sum_{b=1}^{B}
\mathbf 1\{a\text{ is classified as screened by }m\}.
\]

## 22.5 Path probability

For path \(X\rightarrow M\rightarrow Y\), calculate the empirical fraction of resamples containing the entire path.

Do not multiply edge probabilities.

---

# 23. Final consensus network

## 23.1 Inclusion rule

Reference rule:

```text
pi_min = 0.70
```

A node, edge, or shared relation is included in the primary consensus graph when its whole-pipeline stability is at least \(\pi_{min}\).

Thresholds \(\tau_u,\tau_b,\tau_{group}\) operate inside each resample. Stability operates across resamples.

## 23.2 Final displayed depth

For a stable node:

\[
d^*=\arg\max_d\pi_j(d).
\]

If tied, use the shallower depth.

## 23.3 Final magnitude

For each retained edge report:

- mean raw \(U\) among runs in which the target was evaluated;
- median raw \(U\);
- mean normalized \(u\);
- median normalized \(u\);
- whole-pipeline stability;
- number/proportion of runs in which the target was reached.

The inclusion decision is not made by a confidence interval. Intervals are descriptive.

---

# 24. Held-out loss differences and descriptive uncertainty

For held-out observation \(i\):

\[
D_i=L(y_i,\hat y_{i,-j})-L(y_i,\hat y_{i,full}).
\]

Then:

\[
\widehat U_j=\frac{1}{n_{test}}\sum_iD_i.
\]

Repeated folds/resamples may generate an empirical interval.

The primary network decision is made by **practical predictive thresholds plus whole-pipeline stability**, not by whether a confidence interval excludes zero.

---

# 25. Visualization specification

Visual distance is a **display encoding only**. It is not causal, probabilistic, information-theoretic, or compositional.

## 25.1 Layer layout

- Root: depth 0
- Proximal nodes/shared-group members: depth 1
- Next recursive nodes: depth 2
- etc.

Broad radial position is determined by depth.

## 25.2 Edge thickness

Unique edge:

\[
thickness\propto\max(u_{j\rightarrow T},0).
\]

Shared-group visual edge:

\[
thickness\propto u_{G\rightarrow T}.
\]

## 25.3 Edge opacity

\[
opacity\propto\pi_{edge}.
\]

## 25.4 Screened edges

Use dashed/secondary lines. Only show screened relations when \(b_j\ge\tau_b\) and screened-class stability exceeds the display threshold.

## 25.5 Shared groups

Visualize a validated shared group using:

- a hull or bracket around members;
- one virtual group-to-target edge weighted by \(u_G\);
- separate member nodes that remain individual recursion targets.

Do not draw one thick unique edge per shared member.

## 25.6 Within-layer proximity

Within each radial layer, stronger parent-child relations may be plotted closer.

Example:

\[
d^{viz}_{jT}
=
d_{min}+(d_{max}-d_{min})(1-\tilde s_{jT}),
\]

where \(\tilde s_{jT}\in[0,1]\) is rescaled display strength.

No analytical claim is made from these distances.

---

# 26. Optional diagnostic cross-links

After the core graph is complete, an optional diagnostic pass may examine:

- within-layer relations;
- backward predictive relations;
- relations among already assigned nodes.

These cross-links:

- do not change depth;
- do not drive recursion;
- are stored separately from core edges;
- must be visually distinguished.

This keeps the core graph acyclic by construction without implying causal acyclicity.

---

# 27. Cross-sectional interpretation

Use language such as:

- proximal predictive neighborhood;
- distal predictive layer;
- screened predictive relation;
- shared predictive group.

Avoid causal language such as upstream cause, mediator, or causal parent unless supported by a separate causal design.

A consequence of \(Y\) can appear in Layer 1.

Example:

\[
Anxiety\rightarrow Sleep\ Disturbance
\]

can still produce strong cross-sectional prediction:

\[
Sleep\ Disturbance\rightarrow Anxiety
\]

in OPEN. The OPEN arrow is predictive.

---

# 28. Longitudinal extension

For temporal data, define:

\[
Y_{t+1}=f(Y_t,X_{1t},\ldots,X_{pt}).
\]

Then:

\[
U_{X_t\rightarrow Y_{t+1}}
\]

means that \(X_t\) improves prospective prediction of future \(Y\) beyond competing predictors and prior \(Y\).

## 28.1 Temporal candidate rules

A predictor may only point forward in time. For target at \(t+1\), candidates may come from \(t\) or earlier according to a prespecified lag window.

Do not allow future variables to predict past targets.

---

# 29. Repeated-measures / multilevel data

For nested observations:

- split by person, not row;
- resample persons, not rows;
- prevent observations from the same person appearing in train and test simultaneously.

Support separate modes for between-person, within-person, and prospective within-person networks.

---

# 30. Missing data

All missing-data operations must be train-fold local.

Supported policies:

- `native`: learner-native missing-value handling;
- `impute`: fold-local imputation;
- `multiple_imputation`: run OPEN across imputed datasets and aggregate structure/magnitude.

The first implementation may begin with fold-local single imputation and add multiple imputation later.

---

# 31. Practical threshold configuration

Prototype defaults:

```yaml
thresholds:
  min_total_gain_relative: 0.02
  broad_relative: 0.05
  unique_relative: 0.05
  group_relative: 0.10
  dependence_pair: 0.70
  stability: 0.70
```

These are implementation defaults, not validated universal cutoffs.

Before publication, calibrate thresholds using simulations matching realistic \(n\), \(p\), correlations, reliability, noise, and learner family.

---

# 32. Research-grade versus fast mode

## 32.1 Research-grade mode

- no broad pre-screening before LOCO;
- all candidate single-variable ablations;
- all candidate dependence-pair group ablations;
- whole pipeline rerun under several hundred resamples;
- ablated models retuned where computationally feasible;
- threshold sensitivity analysis.

## 32.2 Fast mode

- tune once per target;
- freeze hyperparameters for ablated refits;
- optionally restrict LOCO to top broad candidates **plus all members of redundancy candidate pairs**;
- fewer resamples;
- shallow depth.

Fast mode must be labeled approximate.

---

# 33. Hyperparameter policy

## `retune_ablation = true`

Tune each full, individual-ablation, and group-ablation model separately.

This more closely approximates best achievable risk under each feature set.

## `retune_ablation = false`

Tune once for the full local target and reuse that hyperparameter configuration for ablated refits.

This is faster and appropriate for prototyping and large simulation grids.

Store the mode in analysis metadata.

---

# 34. Complete local-target pseudocode

```python
def analyze_target(data, target, candidates, config, split_plan):
    # 1. Baseline and full predictive signal
    R0 = cv_risk_baseline(data, target, split_plan, config.loss)
    full = fit_evaluate_model(
        data=data,
        target=target,
        predictors=candidates,
        split_plan=split_plan,
        config=config,
    )
    R_full = full.risk

    G = R0 - R_full
    if R0 <= 0:
        return LocalResult(stop=True, reason="LOW_TOTAL_GAIN")

    g_rel = G / R0

    if g_rel < config.thresholds.min_total_gain_relative:
        return LocalResult(stop=True, reason="LOW_TOTAL_GAIN")

    # 2. Broad relevance
    broad = {}
    for j in candidates:
        R_j = fit_evaluate_model(
            data=data,
            target=target,
            predictors=[j],
            split_plan=split_plan,
            config=config,
        ).risk
        B_j = R0 - R_j
        broad[j] = B_j / G

    # 3. Predictor-side dependence candidates
    dep = fit_dependence_estimator(
        data=data,
        predictors=candidates,
        split_plan=split_plan,
        config=config,
    )

    candidate_pairs = [
        (a, b)
        for a, b in all_pairs(candidates)
        if dep[a, b] >= config.thresholds.dependence_pair
    ]

    # 4. Pair group-ablation BEFORE individual pruning
    pair_group = {}
    for a, b in candidate_pairs:
        R_minus_pair = fit_evaluate_model(
            data=data,
            target=target,
            predictors=[x for x in candidates if x not in {a, b}],
            split_plan=split_plan,
            config=config,
        ).risk
        U_pair = R_minus_pair - R_full
        pair_group[(a, b)] = U_pair / G

    # 5. Individual LOCO
    unique = {}
    for j in candidates:
        R_minus_j = fit_evaluate_model(
            data=data,
            target=target,
            predictors=[x for x in candidates if x != j],
            split_plan=split_plan,
            config=config,
        ).risk
        U_j = R_minus_j - R_full
        unique[j] = U_j / G

    # 6. Pair classification
    pair_classes = classify_pairs(
        broad=broad,
        unique=unique,
        pair_group=pair_group,
        thresholds=config.thresholds,
    )

    # 7. Merge shared pairs and validate higher-order groups
    shared_components = connected_components_of_shared_pairs(pair_classes)
    validated_groups = []

    for G_members in shared_components:
        R_minus_G = fit_evaluate_model(
            data=data,
            target=target,
            predictors=[x for x in candidates if x not in G_members],
            split_plan=split_plan,
            config=config,
        ).risk
        u_G = (R_minus_G - R_full) / G

        if (
            u_G >= config.thresholds.group_relative
            and all(
                unique[j] < config.thresholds.unique_relative
                for j in G_members
            )
        ):
            validated_groups.append(
                SharedGroup(members=G_members, u_group=u_G)
            )

    # 8. Final local classification
    classes = classify_variables(
        candidates=candidates,
        broad=broad,
        unique=unique,
        pair_classes=pair_classes,
        shared_groups=validated_groups,
        thresholds=config.thresholds,
    )

    # 9. Promotion set
    promoted = {
        j for j, c in classes.items()
        if c.kind == "PROXIMAL_UNIQUE"
    }

    for group in validated_groups:
        promoted.update(group.members)

    return LocalResult(
        stop=False,
        target=target,
        baseline_risk=R0,
        full_risk=R_full,
        total_gain=G,
        total_gain_relative=g_rel,
        broad=broad,
        unique=unique,
        pair_group=pair_group,
        pair_classes=pair_classes,
        shared_groups=validated_groups,
        classes=classes,
        promoted=promoted,
    )
```

---

# 35. Complete recursive-network pseudocode

```python
def build_network_once(data, root, all_variables, config):
    assigned_depth = {root: 0}
    parents = {}
    local_results = {}
    core_edges = []
    shared_visual_edges = []
    queue = {0: [root]}

    for depth in range(config.max_depth):
        targets = queue.get(depth, [])
        if not targets:
            break

        # Snapshot unassigned predictors once per layer.
        unassigned = [
            v for v in all_variables
            if v not in assigned_depth
        ]

        if not unassigned:
            break

        next_nodes = set()

        for target in targets:
            result = analyze_target(
                data=data,
                target=target,
                candidates=unassigned,
                config=config,
                split_plan=config.split_plan,
            )
            local_results[target] = result

            if result.stop:
                continue

            for j, cls in result.classes.items():
                if cls.kind == "PROXIMAL_UNIQUE":
                    core_edges.append(
                        Edge(
                            source=j,
                            target=target,
                            relation="PROXIMAL_UNIQUE",
                            u=result.unique[j],
                            b=result.broad[j],
                        )
                    )
                    next_nodes.add(j)
                    parents.setdefault(j, set()).add(target)

            for group in result.shared_groups:
                shared_visual_edges.append(
                    SharedEdge(
                        members=group.members,
                        target=target,
                        u_group=group.u_group,
                    )
                )
                for j in group.members:
                    next_nodes.add(j)
                    parents.setdefault(j, set()).add(target)

            # Screened variables are stored but not promoted.
            # They remain unassigned and can enter a later layer.

        # Assign depths synchronously after all targets at this layer.
        for node in next_nodes:
            if node not in assigned_depth:
                assigned_depth[node] = depth + 1

        queue[depth + 1] = sorted(
            node for node in next_nodes
            if assigned_depth.get(node) == depth + 1
        )

    return NetworkRun(
        root=root,
        assigned_depth=assigned_depth,
        parents=parents,
        local_results=local_results,
        core_edges=core_edges,
        shared_visual_edges=shared_visual_edges,
    )
```

---

# 36. Whole-pipeline stability pseudocode

```python
def fit_stable_open(data, root, variables, config):
    runs = []

    for b in range(config.stability.n_resamples):
        sample = resample_units(
            data,
            method=config.stability.method,
            fraction=config.stability.fraction,
            cluster=config.cluster_id,
            seed=config.seed + b,
        )

        run = build_network_once(
            data=sample,
            root=root,
            all_variables=variables,
            config=config,
        )
        runs.append(run)

    stability = aggregate_runs(runs, config)
    consensus = build_consensus_graph(stability, config)

    return OPENResult(
        runs=runs,
        stability=stability,
        consensus=consensus,
    )
```

---

# 37. Repository architecture

Recommended structure:

```text
open/
├── README.md
├── pyproject.toml
├── configs/
│   ├── default.yaml
│   ├── fast.yaml
│   └── research.yaml
├── src/
│   └── open/
│       ├── __init__.py
│       ├── config.py
│       ├── metadata.py
│       ├── preprocessing.py
│       ├── splits.py
│       ├── learners/
│       │   ├── base.py
│       │   ├── elastic_net.py
│       │   ├── random_forest.py
│       │   ├── boosting.py
│       │   └── ensemble.py
│       ├── dependence/
│       │   ├── base.py
│       │   ├── spearman.py
│       │   ├── copula.py
│       │   └── user_groups.py
│       ├── risk.py
│       ├── relevance.py
│       ├── grouping.py
│       ├── classify.py
│       ├── local_target.py
│       ├── recurse.py
│       ├── stability.py
│       ├── consensus.py
│       ├── graph.py
│       ├── visualization.py
│       ├── diagnostics.py
│       └── io.py
├── simulations/
│   ├── generators.py
│   ├── scenarios/
│   │   ├── linear.py
│   │   ├── chain.py
│   │   ├── redundancy.py
│   │   ├── complementarity.py
│   │   ├── collider.py
│   │   ├── latent_confounder.py
│   │   └── measurement_error.py
│   └── evaluate.py
├── tests/
│   ├── test_relevance.py
│   ├── test_grouping.py
│   ├── test_classification.py
│   ├── test_recursion.py
│   ├── test_stability.py
│   ├── test_consensus.py
│   └── test_synthetic_scenarios.py
└── examples/
    ├── anxiety_example.ipynb
    └── longitudinal_example.ipynb
```

---

# 38. Core software objects

## 38.1 Configuration

```python
@dataclass
class ThresholdConfig:
    min_total_gain_relative: float = 0.02
    broad_relative: float = 0.05
    unique_relative: float = 0.05
    group_relative: float = 0.10
    dependence_pair: float = 0.70
    stability: float = 0.70
```

## 38.2 Relationship classification

```python
class RelationKind(str, Enum):
    PROXIMAL_UNIQUE = "proximal_unique"
    SHARED_GROUP_MEMBER = "shared_group_member"
    SCREENED_BY_PARTNER = "screened_by_partner"
    SCREENED_BY_SET = "screened_by_set"
    SCREENED_GENERAL = "screened_general"
    UNSTABLE = "unstable"
    NEGLIGIBLE = "negligible"
```

## 38.3 Node result

```python
@dataclass
class NodeResult:
    name: str
    depth: int | None
    depth_probabilities: dict[int, float]
    selection_probability: float
```

## 38.4 Edge result

```python
@dataclass
class EdgeResult:
    source: str
    target: str
    relation: RelationKind
    broad_raw: float | None
    broad_norm: float | None
    unique_raw: float | None
    unique_norm: float | None
    stability: float
    evaluated_probability: float
    screened_by: tuple[str, ...] | None
```

## 38.5 Shared group result

```python
@dataclass
class SharedGroupResult:
    target: str
    members: tuple[str, ...]
    group_raw: float
    group_norm: float
    pairwise_comembership: dict[tuple[str, str], float]
    stability_summary: float
```

## 38.6 Risk result

The P1 risk engine returns one immutable `RiskResult` for one target, loss, split plan, and model specification. Model variants may be evaluated on a subset of planned folds (as with D-002 pair ablations); only evaluated folds appear in `folds`, and the result records that subset explicitly.

```python
@dataclass(frozen=True)
class ObservationLoss:
    row_position: int
    observed: float
    predicted: float
    loss: float

@dataclass(frozen=True)
class FoldRisk:
    repeat_index: int
    fold_index: int
    predictors: tuple[str, ...]
    observations: tuple[ObservationLoss, ...]
    risk: float  # mean loss over this fold's held-out observations
    fitted_parameters: dict[str, object] | None

@dataclass(frozen=True)
class RiskResult:
    target: str
    candidate_set: tuple[str, ...]
    predictors: tuple[str, ...]
    learner: str | None  # None for the training-mean baseline
    loss: str
    split_plan_id: str  # stable fingerprint of the serialized SplitPlan
    split_seed: int
    folds: tuple[FoldRisk, ...]
    aggregate_risk: float  # pooled mean of all recorded held-out losses
    outer_model_fit_count: int
    tuning_model_fit_count: int
```

The stable observation key for paired comparisons is `(repeat_index, fold_index, row_position)`. The `SplitPlan` identified by `split_plan_id` remains the source of train/test assignments; row positions in results identify the held-out observations and preserve input-order provenance. `aggregate_risk` weights each recorded held-out observation equally, including repeated out-of-fold appearances in repeated CV. Fold means are retained so alternative aggregations can be inspected without refitting. For the training-mean baseline, `learner` is `None`, `predictors` is empty, and both fit counts are zero; each fold prediction is the mean target value from that fold's training rows.

`outer_model_fit_count` counts model fits on outer training folds. `tuning_model_fit_count` counts estimator fits performed by inner tuning. Their sum is the reported model-fit count. `fitted_parameters` records the parameters used for that fold, or `None` for the baseline and unfitted cases. Paired full/ablation comparisons must have matching split IDs and identical held-out observation keys for every compared fold.

---

# 39. Output files

## `nodes.csv`

```text
node
display_label
depth
selection_probability
depth_probability_1
depth_probability_2
...
```

## `edges.csv`

```text
source
target
relation
broad_raw
broad_norm
unique_raw
unique_norm
stability
evaluated_probability
screened_by
```

## `shared_groups.csv`

```text
target
group_id
members
group_raw
group_norm
```

## `shared_comembership.csv`

```text
target
node_a
node_b
shared_probability
```

## `local_targets.csv`

```text
target
depth
baseline_risk
full_risk
total_gain
relative_total_gain
learner
n_candidates
stop_reason
```

## `analysis_metadata.json`

Store:

- package version;
- git commit;
- configuration;
- seed;
- learner versions;
- preprocessing;
- loss;
- thresholds;
- resampling design;
- variable metadata hash.

---

# 40. Example configuration file

```yaml
analysis:
  root: Anxiety
  max_depth: 3
  random_seed: 20261004

outcome:
  type: continuous
  loss: mse

learner:
  mode: per_target
  library:
    - elastic_net
    - random_forest
    - hist_gradient_boosting
  retune_ablation: false

preprocessing:
  missing: median_mode
  standardize_linear: true
  one_hot_categorical: true

dependence:
  method: spearman
  absolute: true

thresholds:
  min_total_gain_relative: 0.02
  broad_relative: 0.05
  unique_relative: 0.05
  group_relative: 0.10
  dependence_pair: 0.70
  stability: 0.70

performance_cv:
  folds: 5
  repeats: 2
  grouped: false

tuning_cv:
  folds: 3

stability:
  method: subsample
  fraction: 0.80
  n_resamples: 500

visualization:
  show_screened: true
  show_unstable: false
  distance_mode: visual_strength
  edge_width: unique
  edge_opacity: stability
```

---

# 41. Unit tests required before methodological testing

## 41.1 Relevance tests

Verify numerically:

- \(U_j=R_{-j}-R_{full}\);
- \(U_G=R_{-G}-R_{full}\);
- normalization uses \(G_T\);
- negative values remain in raw output.

## 41.2 Classification tests

### Shared case

```text
u_pair = .40
u_a = .01
u_b = .02
```

Expected: `PAIR_SHARED_REDUNDANT`.

### Partner-screened case

```text
u_pair = .40
u_a = .25
u_b = .01
b_b = .20
```

Expected:

```text
a = PROXIMAL_UNIQUE
b = SCREENED_BY_PARTNER(a)
```

### Both proximal

```text
u_pair = .40
u_a = .20
u_b = .18
```

Expected: both `PROXIMAL_UNIQUE`.

## 41.3 Recursion tests

Verify:

- root is always depth 0;
- a node is assigned only once;
- depth is shallowest qualifying depth;
- all targets at one layer use the same unassigned candidate snapshot;
- root never enters Layer 2 candidate set;
- current-layer peers do not enter each other's core expansion models;
- screened variables remain unassigned and eligible later;
- every shared-group member is promoted;
- virtual shared-group objects are never modeled as targets.

## 41.4 Stability tests

Verify:

- every resample rebuilds dependence groups;
- exact group IDs are not used for stability;
- pairwise co-membership is correct;
- path probability is empirical rather than a product of edge probabilities.

---

# 42. Synthetic scenario tests

## 42.1 Direct unique predictor

Generate:

\[
Y=1.0X_1+\epsilon.
\]

Expected: \(X_1\) proximal; noise negligible.

## 42.2 Redundant substitutes

Generate:

\[
X_2=X_1+\eta
\]

with small \(\eta\), and:

\[
Y=X_1+\epsilon.
\]

Expected:

- pair group importance substantial;
- individual \(U_1,U_2\) small;
- shared group retained;
- both members promoted.

## 42.3 Chain

Generate:

\[
S\rightarrow D\rightarrow A.
\]

Expected at root \(A\):

- \(D\) proximal;
- \(S\) broad but weak unique;
- \(S\) screened by \(D\), not redundant with \(D\).

Expected for Layer-1 target \(D\):

- \(S\) proximal;
- final structure \(S\rightarrow D\rightarrow A\).

## 42.4 Two direct predictors

Generate:

\[
Y=\beta_1X_1+\beta_2X_2+\epsilon
\]

with low \(corr(X_1,X_2)\).

Expected: both proximal; no shared group.

## 42.5 Complementarity / interaction

Generate:

\[
Y=X_1X_2+\epsilon.
\]

Expected: group diagnostic may indicate non-additivity; negative \(C_G\) must not be labeled redundancy.

The core redundancy candidate rule can miss independent interacting variables when predictor-side dependence is low. Exhaustive interaction discovery is not a core objective.

## 42.6 Consequence of root

Generate:

\[
Y\rightarrow X.
\]

Expected: \(X\) may appear proximal; interpretation remains predictive.

## 42.7 Collider

Generate:

\[
Y\rightarrow C\leftarrow Z.
\]

Evaluate conditional-prediction artifacts.

## 42.8 Latent common cause

Generate:

\[
U\rightarrow X,\qquad U\rightarrow Y.
\]

Expected: predictive edge may appear; no causal claim.

## 42.9 Measurement error

Generate a true screening variable \(M^*\), then a noisy observed \(M\). Evaluate whether poorer reliability produces residual unique importance in other predictors.

---

# 43. Simulation evaluation metrics

Across simulations, report:

- proximal-edge precision;
- proximal-edge recall;
- shared-group detection rate;
- screened-variable detection rate;
- incorrect redundancy classification rate;
- mean absolute depth error;
- false discovery rate;
- whole-pipeline selection stability;
- predictive risk;
- calibration where applicable;
- runtime;
- number of model fits.

For chain simulations, separately report:

- probability of correct \(S\rightarrow D\rightarrow A\) reconstruction;
- probability of incorrectly calling \(\{S,D\}\) redundant at the root.

---

# 44. Benchmark comparisons

Recommended comparators:

- zero-order correlation ranking;
- flat ML importance;
- elastic-net feature selection;
- EBICglasso;
- Gaussian graphical model;
- Graphical Random Forests;
- Bayesian-network structure learning;
- GENIE3-style all-node tree network;
- copula graphical model when relevant.

The question is not whether OPEN beats every method on every metric. The question is whether it reliably estimates the target-centered predictive objects it was designed to estimate.

---

# 45. Sensitivity analyses

Every empirical paper should vary at least:

- \(\tau_{dep}\);
- \(\tau_u\);
- \(\tau_b\);
- \(\tau_{group}\);
- \(\pi_{min}\);
- learner family;
- maximum depth;
- missing-data strategy.

Suggested development grid:

```text
tau_dep:   [0.60, 0.70, 0.80]
tau_u:     [0.025, 0.05, 0.10]
tau_b:     [0.025, 0.05, 0.10]
tau_group: [0.05, 0.10, 0.20]
pi_min:    [0.60, 0.70, 0.80]
```

Final defaults should be chosen from operating characteristics rather than convenience.

---

# 46. Computational accounting

A local target may require:

- 1 baseline model;
- 1 full model;
- \(p\) univariate broad models;
- \(p\) individual ablations;
- \(q\) candidate pair ablations;
- \(g\) higher-order group ablations;
- inner tuning operations.

Whole-pipeline resampling multiplies this by \(B\).

Log:

```text
n_model_fits
n_targets_evaluated
n_pair_ablations
n_group_ablations
elapsed_time
peak_memory
```

---

# 47. Recommended optimization strategy

For initial repo development:

1. continuous outcomes first;
2. fixed learner first;
3. no hyperparameter retuning per ablation;
4. pairwise redundancy first;
5. depth \(\le2\);
6. 50–100 stability resamples;
7. validate synthetic scenarios;
8. then add higher-order groups;
9. then learner abstraction;
10. then binary and longitudinal outcomes;
11. then final-scale stability resampling.

Do not build all extensions before validating chain and redundancy logic.

---

# 48. Optional advanced extensions

These are not part of the minimum viable method.

## 48.1 Conditional permutation / randomization

Can provide stronger conditional-null inference when the predictor distribution can be modeled credibly.

## 48.2 Knockoffs

Can add FDR-controlled selection when predictor-distribution modeling is defensible.

## 48.3 Copula dependence

Can replace Spearman candidate grouping for mixed/ordinal data.

## 48.4 Shapley/SAGE broad relevance

Can replace simple univariate \(B_j\) when a richer total-participation measure is desired.

## 48.5 Partial Information Decomposition

Can be explored as a theoretical extension for explicit unique/redundant/synergistic information decomposition.

None of these are required for core OPEN.

---

# 49. Reporting language

Recommended:

> Depression was a stable proximal predictor of Anxiety, accounting for a meaningful fraction of the full model's out-of-sample predictive gain.

> Stress showed broad predictive relevance to Anxiety but negligible unique contribution after competing predictors were available and was therefore classified as screened at the root.

> Stress subsequently entered the network at Layer 2 as a stable predictor of Depression.

> Depression and Loneliness formed a shared predictive group for Anxiety: their joint removal substantially degraded prediction, whereas individual removals produced relatively small losses.

Avoid causal language unless a separate design justifies it.

---

# 50. Minimum manuscript reporting standard

For every local target report:

- target name;
- depth;
- candidate set;
- baseline risk;
- full-model risk;
- total relative gain \(g_T\);
- learner;
- broad relevance \(b_j\);
- unique relevance \(u_j\);
- tested group relevance \(u_G\);
- classification;
- stability probability;
- number of resamples in which the target was evaluated.

For the complete network report:

- root;
- max depth;
- node selection probabilities;
- depth distributions;
- edge-class probabilities;
- shared co-membership probabilities;
- screening-partner probabilities;
- thresholds;
- learner configuration;
- resampling procedure;
- sensitivity analysis.

---

# 51. Conceptual comparison with existing frameworks

## EBICglasso

Primary question:

> Which variables remain conditionally associated after controlling for the rest?

OPEN:

> Which variables contribute stable out-of-sample predictive information about a chosen outcome, which information is unique or shared, and what predictive structure appears recursively beyond the proximal neighborhood?

## Bayesian networks

Primary question:

> What DAG/probabilistic factorization best represents the joint distribution under the model assumptions?

OPEN:

> What is the layered predictive architecture around a prespecified outcome?

## Flat machine learning

Primary question:

> Which predictors matter for this prediction model?

OPEN adds redundancy protection, shared-versus-unique classification, screened relations, recursive expansion, depth, and whole-pipeline structural stability.

## Markov-blanket methods

Primary question:

> What local set makes the target conditionally independent of the remaining variables?

OPEN shares target-centeredness but defines relevance through held-out predictive performance and continues outward recursively.

---

# 52. Elevator pitch

> **OPEN turns "what predicts my outcome?" from a flat feature-importance problem into a recursive, redundancy-aware predictive network that distinguishes what is uniquely proximal, what is shared, what is screened, and what becomes relevant farther from the focal outcome.**

Or:

> **EBICglasso estimates conditional association, Bayesian networks estimate probabilistic structure, and OPEN estimates the layered predictive architecture surrounding a researcher-defined outcome.**

---

# 53. Foundational methodological traditions

The framework is a synthesis. Relevant foundations include:

- tree-based network inference and GENIE3;
- graphical random forests;
- local Markov-blanket discovery;
- variable-importance and LOCO-style predictive contribution;
- permutation and conditional permutation importance;
- stability selection;
- psychometric redundancy analysis;
- ARACNE's distinction between direct-looking and screened/indirect relationships;
- copula graphical modeling for mixed/non-Gaussian dependence.

These literatures should be cited for context and contrast without making every adjacent method an implementation dependency.

---

# 54. Implementation milestones

## Milestone 1 — Local target engine

Deliver baseline/full risk, \(B_j\), \(U_j\), pair dependence, \(U_{ab}\), and classification.

Synthetic tests: direct effect, redundant pair, chain.

## Milestone 2 — Recursion

Deliver layer-synchronous expansion, shallowest-depth assignment, shared-group promotion, and screened-node persistence.

## Milestone 3 — Stability

Deliver whole-pipeline resampling, edge-class probability, depth probability, shared co-membership probability, and empirical path probability.

## Milestone 4 — Visualization

Deliver radial layer layout, predictive visual distance, shared-group hulls, stability opacity, and screened dashed edges.

## Milestone 5 — Simulation suite

Deliver all scenarios in Section 42.

## Milestone 6 — Learner abstraction

Add elastic net, RF, boosting, and per-target model selection.

## Milestone 7 — Applied psychological example

Produce a reproducible notebook, config, graph, machine-readable outputs, and sensitivity analysis.

---

# 55. Final algorithm in compact form

For each stability resample:

\[
\boxed{
\text{Root target}
\rightarrow
\text{total predictive gain}
\rightarrow
\text{broad relevance}
\rightarrow
\text{dependence-pair generation}
}
\]

then:

\[
\boxed{
\text{pair/group ablation}
\rightarrow
\text{individual LOCO}
\rightarrow
\text{ordered classification}
}
\]

then:

\[
\boxed{
\text{promote proximal + all shared-group members}
\rightarrow
\text{leave screened nodes unassigned}
\rightarrow
\text{expand into unassigned nodes}
}
\]

Repeat layer by layer, then aggregate across complete pipeline runs:

\[
\boxed{
\text{edge stability}
+
\text{depth stability}
+
\text{shared co-membership}
+
\text{screening stability}
}
\]

to produce the final consensus network.

The authoritative implementation rules are:

1. **Group-ablate before pruning individuals.**
2. **Redundancy requires symmetric lack of individual unique contribution.**
3. **Asymmetric important groups imply screening, not redundancy.**
4. **All members of a validated shared group are promoted.**
5. **Screened variables remain unassigned and eligible for later layers.**
6. **Core recursion only considers previously unassigned variables.**
7. **Depth is the shallowest qualifying recursive depth.**
8. **Whole-pipeline resampling determines structural stability.**
9. **Distance is visualization only.**
10. **Predictive direction is not causal direction.**

These rules are the behavioral contract for the first repository implementation.
