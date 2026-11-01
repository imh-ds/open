"""Run steps 2-17 from docs/tutorial.html in order.

From the repository root, after installing .[visualization]:
    python examples/tutorial/step_by_step.py

Writes the depth-1 and depth-2 business exports, separate psychology networks,
psychology panels and overlay, and the complete psychology-output export to
the working directory. Optional settings lessons are not run here.
"""

import pandas as pd

data = pd.read_csv("examples/tutorial/data/business_customer_renewal.csv")

print(data.shape)
print(data.head())

print(data.dtypes)
print(data.isna().sum())

outcome = "renewal_intent"

predictors = [
    "adoption", "setup_quality", "training_pct", "support_rating",
    "response_speed", "success_contacts", "price_fairness", "industry_fit",
    "tenure_months", "company_size_log",
]

from open.api import DiscoveryConfig

config = DiscoveryConfig(
    max_depth=1,
    folds=10,
    interactions=False,
    seed=42,
)

explore_fraction = 0.5
split_seed = 42

correction = "holm"
alpha = 0.05

from open.api import analyze

result = analyze(
    data,
    outcome=outcome,
    predictors=predictors,
    engine="ols",
    config=config,
    explore_fraction=explore_fraction,
    split_seed=split_seed,
    correction=correction,
    alpha=alpha,
)

print(result.summary())

discovery = result.discovery

print(discovery.links[
    ["depth", "source", "target", "kind", "pass_rate", "tier", "r2_median"]
].to_string(index=False))

print(result.confirmation[
    ["source", "target", "kind", "n", "p", "p_adjusted", "confirmed", "note"]
].to_string(index=False))

print(result.fit[
    ["target", "n", "network_size", "r2_network_explore", "r2_network"]
].to_string(index=False))

from open.api import plot_discovery

plot_discovery(
    discovery,
    path="renewal-depth1.png",
    confirmation=result.confirmation,
    confirmation_fit=result.fit,
)

plot_discovery(
    discovery,
    path="renewal-depth1-kept.png",
    confirmation=result.confirmation,
    confirmation_fit=result.fit,
    show_flagged=False,
)

from open.api import save_split

saved_to = save_split(result, "renewal-depth1-output", plot=False)
print(saved_to)

from dataclasses import replace

deeper_config = replace(config, max_depth=2)

deeper_result = analyze(
    data,
    outcome=outcome,
    predictors=predictors,
    engine="ols",
    config=deeper_config,
    explore_fraction=explore_fraction,
    split_seed=split_seed,
    correction=correction,
    alpha=alpha,
)

print(deeper_result.summary())
print(deeper_result.discovery.nodes.to_string(index=False))
print(deeper_result.discovery.stops)

save_split(deeper_result, "renewal-depth2-output", show_flagged=False)

psych = pd.read_csv("examples/tutorial/data/psychology_wellbeing.csv")
print(psych.shape)
print(psych.head())

outcomes = ["wellbeing", "anxiety"]

psych_predictors = [name for name in psych.columns if name not in outcomes]
print(psych_predictors)

batch = analyze(
    psych,
    outcomes=outcomes,
    predictors=psych_predictors,
    engine="ols",
    config=deeper_config,
    explore_fraction=explore_fraction,
    split_seed=split_seed,
    correction=correction,
    alpha=alpha,
)

wellbeing_result = batch.by_outcome["wellbeing"]
print(wellbeing_result.summary())

plot_discovery(
    wellbeing_result.discovery,
    path="psychology-wellbeing.png",
    confirmation=wellbeing_result.confirmation,
    confirmation_fit=wellbeing_result.fit,
    show_flagged=False,
)

print(wellbeing_result.discovery.nodes.to_string(index=False))
print(wellbeing_result.discovery.stops)
print(wellbeing_result.fit.to_string(index=False))

anxiety_result = batch.by_outcome["anxiety"]
print(anxiety_result.summary())

plot_discovery(
    anxiety_result.discovery,
    path="psychology-anxiety.png",
    confirmation=anxiety_result.confirmation,
    confirmation_fit=anxiety_result.fit,
    show_flagged=False,
)

print(anxiety_result.discovery.nodes.to_string(index=False))
print(anxiety_result.discovery.stops)
print(anxiety_result.fit.to_string(index=False))

from open.api import GraphStyle, plot_outcomes

batch_style = GraphStyle(show_flagged=False)
plot_outcomes(batch, path="psychology-panels.png", style=batch_style)

plot_outcomes(
    batch,
    layout="combined",
    path="psychology-overlay.png",
    style=batch_style,
)

print(batch.confirmation.to_string(index=False))
print(batch.fit.to_string(index=False))

from open.api import save_batch

save_batch(batch, "psychology-output", plot="combined", style=batch_style)
