"""Regenerate the network figures used by the documentation."""

from pathlib import Path

import pandas as pd

from opennet.api import GraphStyle, analyze, plot_network, plot_outcomes
from opennet.explore import ExploreConfig


ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = Path(__file__).with_name("data")
OUTPUT = ROOT / "docs" / "assets" / "tutorial"


def main() -> None:
    business = pd.read_csv(DATA_DIR / "business_customer_renewal.csv")
    outcome = "renewal_intent"
    predictors = [column for column in business.columns if column != outcome]
    business_result = analyze(
        business,
        outcome=outcome,
        predictors=predictors,
        config=ExploreConfig(max_depth=2, interactions=False, seed=42),
        split_seed=42,
    )

    psych = pd.read_csv(DATA_DIR / "psychology_wellbeing.csv")
    outcomes = ["wellbeing", "anxiety"]
    predictors = [column for column in psych.columns if column not in outcomes]
    batch = analyze(
        psych,
        outcomes=outcomes,
        predictors=predictors,
        config=ExploreConfig(max_depth=2, interactions=False, seed=42),
        split_seed=42,
    )
    style = GraphStyle(
        spacing=1.2,
        node_size=1.2,
        label_size=1.1,
        fit_label_size=1.35,
        show_flagged=False,
        show_title=False,
        show_r2_legend=False,
        show_footnote=False,
    )
    OUTPUT.mkdir(parents=True, exist_ok=True)
    plot_network(
        business_result.exploration,
        path=OUTPUT / "business-renewal-network.png",
        confirmation=business_result.confirmation,
        confirmation_fit=business_result.fit,
        style=style,
    )
    for outcome in outcomes:
        split = batch.by_outcome[outcome]
        plot_network(
            split.exploration,
            path=OUTPUT / f"psychology-{outcome}-network.png",
            confirmation=split.confirmation,
            confirmation_fit=split.fit,
            style=style,
        )
    plot_outcomes(
        batch,
        layout="combined",
        path=OUTPUT / "psychology-combined-network.png",
        style=style,
    )


if __name__ == "__main__":
    main()
