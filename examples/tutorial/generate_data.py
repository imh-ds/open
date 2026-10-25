"""Regenerate the fictional datasets used by docs/quickstart.md."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


N = 1_200
BUSINESS_SEED = 20_261_024
PSYCHOLOGY_SEED = 20_261_025
DATA_DIR = Path(__file__).with_name("data")


def _z(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    return (values - values.mean()) / values.std()


def _score(values: np.ndarray, spread: float = 0.9) -> np.ndarray:
    """Map a latent score to a bounded, 1-to-7 survey-style measure."""
    return np.clip(4 + spread * _z(values), 1, 7).round(2)


def make_business_data(seed: int = BUSINESS_SEED, n: int = N) -> pd.DataFrame:
    """Fictional SaaS customer accounts with one renewal-intent outcome."""
    rng = np.random.default_rng(seed)
    industry_fit = _score(rng.normal(size=n))
    training_completion_pct = rng.beta(4, 2, size=n) * 100
    account_tenure_months = rng.integers(1, 73, size=n)
    company_size_log = rng.normal(4.2, 1.1, size=n)

    implementation_quality = _score(
        0.48 * _z(training_completion_pct) + 0.24 * _z(industry_fit) + rng.normal(size=n)
    )
    product_adoption = _score(
        0.56 * _z(implementation_quality) + 0.25 * _z(training_completion_pct)
        + 0.18 * _z(industry_fit) + rng.normal(size=n)
    )
    support_response_speed = _score(rng.normal(size=n))
    contact_rate = np.exp(-0.12 * _z(product_adoption)) * 4.2
    customer_success_contacts = rng.poisson(contact_rate).clip(0, 20)
    support_satisfaction = _score(
        0.50 * _z(support_response_speed) + 0.25 * _z(customer_success_contacts)
        + 0.12 * _z(product_adoption) + rng.normal(size=n)
    )
    price_fairness = _score(0.18 * _z(company_size_log) + rng.normal(size=n))

    renewal_intent = _score(
        0.42 * _z(product_adoption) + 0.26 * _z(support_satisfaction)
        + 0.16 * _z(price_fairness) + 0.12 * _z(industry_fit)
        + 0.08 * _z(account_tenure_months) + rng.normal(size=n),
        spread=1.05,
    )

    return pd.DataFrame({
        "renewal_intent": renewal_intent,
        "adoption": product_adoption,
        "setup_quality": implementation_quality,
        "training_pct": training_completion_pct.round(1),
        "support_rating": support_satisfaction,
        "response_speed": support_response_speed,
        "success_contacts": customer_success_contacts,
        "price_fairness": price_fairness,
        "industry_fit": industry_fit,
        "tenure_months": account_tenure_months,
        "company_size_log": company_size_log.round(2),
    })


def make_psychology_data(seed: int = PSYCHOLOGY_SEED, n: int = N) -> pd.DataFrame:
    """Fictional adult survey with correlated well-being and anxiety outcomes."""
    rng = np.random.default_rng(seed)
    age_years = rng.integers(18, 71, size=n)
    daily_hassles = _score(rng.normal(size=n))
    financial_strain = _score(0.22 * _z(age_years) + rng.normal(size=n))
    social_support = _score(rng.normal(size=n))
    loneliness = _score(-0.43 * _z(social_support) + rng.normal(size=n))
    sleep_duration_hours = np.clip(rng.normal(7.1, 0.95, size=n), 4, 10)
    physical_activity_hours = np.clip(rng.lognormal(1.0, 0.55, size=n), 0, 20)
    self_efficacy = _score(0.12 * _z(age_years) + rng.normal(size=n))
    perceived_stress = _score(
        0.38 * _z(daily_hassles) + 0.28 * _z(financial_strain)
        - 0.23 * _z(social_support) + rng.normal(size=n)
    )
    sleep_quality = _score(
        0.35 * _z(sleep_duration_hours) - 0.28 * _z(perceived_stress)
        + rng.normal(size=n)
    )

    # A shared residual component makes the two outcomes realistically related.
    shared_residual = rng.normal(size=n)
    wellbeing_error = 0.45 * shared_residual + np.sqrt(1 - 0.45**2) * rng.normal(size=n)
    anxiety_error = 0.45 * shared_residual + np.sqrt(1 - 0.45**2) * rng.normal(size=n)
    psychological_wellbeing = _score(
        0.32 * _z(social_support) + 0.28 * _z(sleep_quality)
        + 0.28 * _z(self_efficacy) + 0.12 * _z(physical_activity_hours)
        - 0.32 * _z(perceived_stress) - 0.14 * _z(loneliness)
        + wellbeing_error,
        spread=1.05,
    )
    anxiety_symptoms = _score(
        0.40 * _z(perceived_stress) + 0.32 * _z(daily_hassles)
        + 0.18 * _z(financial_strain) + 0.18 * _z(loneliness)
        - 0.20 * _z(sleep_quality) - 0.15 * _z(self_efficacy)
        + anxiety_error,
        spread=1.05,
    )

    return pd.DataFrame({
        "wellbeing": psychological_wellbeing,
        "anxiety": anxiety_symptoms,
        "stress": perceived_stress,
        "hassles": daily_hassles,
        "financial_strain": financial_strain,
        "social_support": social_support,
        "loneliness": loneliness,
        "sleep_quality": sleep_quality,
        "sleep_hours": sleep_duration_hours.round(2),
        "activity_hours": physical_activity_hours.round(2),
        "self_efficacy": self_efficacy,
        "age_years": age_years,
    })


def write_datasets(output_dir: Path = DATA_DIR) -> tuple[Path, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    business_path = output_dir / "business_customer_renewal.csv"
    psychology_path = output_dir / "psychology_wellbeing.csv"
    make_business_data().to_csv(business_path, index=False, float_format="%.3f")
    make_psychology_data().to_csv(psychology_path, index=False, float_format="%.3f")
    return business_path, psychology_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DATA_DIR)
    args = parser.parse_args()
    for path in write_datasets(args.output_dir):
        print(path)


if __name__ == "__main__":
    main()
