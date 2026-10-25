# OPEN tutorial: one business outcome, then two psychology outcomes

This tutorial uses two fictional datasets to show the workflow in two research
settings. The business example has one outcome. The psychology example has two.
Both files are included in the repository, so you can follow along without
creating or changing a dataset.

The data are synthetic and for learning only. Their known construction is not
evidence that OPEN will recover the same links in real research.

## Get set up

From the repository root, install OPEN and its plotting extra so the examples
can save network figures:

```bash
python -m pip install ".[visualization]"
```

The CSVs and data-generation script are in `examples/tutorial/`. Regenerate
them with:

```bash
python examples/tutorial/generate_data.py
```

The script uses fixed random seeds, so it writes the same data each time.

## Example 1: a business question with one outcome

Imagine a fictional software company wants to understand which account
characteristics predict a customer's **renewal-intent score**. Each row is one
customer account. The outcome is a numeric survey-style score from 1 to 7; the
other ten columns describe account adoption, support, perceived value, and
account characteristics.

| Column | Meaning |
|---|---|
| `renewal_intent` | Outcome: account contact's renewal-intent score, 1–7 |
| `product_adoption` | Reported breadth of product use, 1–7 |
| `implementation_quality` | Account's implementation experience, 1–7 |
| `training_completion_pct` | Completed product training, percent |
| `support_satisfaction` | Satisfaction with support, 1–7 |
| `support_response_speed` | Perceived response speed, 1–7; higher is faster |
| `customer_success_contacts` | Number of customer-success contacts |
| `price_fairness` | Perceived fairness of the price, 1–7 |
| `industry_fit` | Fit between the product and account's industry, 1–7 |
| `account_tenure_months` | Account tenure in months |
| `company_size_log` | Log-scaled company size measure |

Load the file and tell OPEN which column is the outcome. The predictor list is
made from all the other columns, so the outcome cannot accidentally be included
as its own predictor:

```python
import pandas as pd
from open.api import ExploreConfig, analyze, save_split

business = pd.read_csv("examples/tutorial/data/business_customer_renewal.csv")
business_outcome = "renewal_intent"
business_predictors = [c for c in business.columns if c != business_outcome]

# Five folds and depth two keep this demonstration relatively quick.
config = ExploreConfig(folds=5, max_depth=2, interactions=False, seed=42)
business_result = analyze(
    business,
    outcome=business_outcome,
    predictors=business_predictors,
    config=config,
    split_seed=42,
)

print(business_result.summary())
save_split(business_result, "tutorial-output/business")
```

`analyze` splits the 1,200 accounts before it explores anything. It builds the
network on the exploration rows, then tests the selected links on the held-out
confirmation rows. The output folder contains the tables, settings, summary,
confirmation results, and graph.

Start with `business_result.summary()`. Solid links passed the exploratory
stability rule; dotted links are weaker leads. A checkmark means a link also
passed confirmation after multiple-testing correction. The fold pass rate is
not a p-value, and an exploratory link without a checkmark is not confirmed.
The arrows show predictive relationships in this analysis, not causal effects.

## Example 2: a psychology question with two outcomes

Now imagine a survey study asking how everyday experiences relate to both
**psychological well-being** and **anxiety-symptom scores**. Each row is one
fictional adult respondent. Both outcomes and the ten predictors are numeric
survey-style values; the two outcome columns use a 1–7 scale. The anxiety score
is not a clinical diagnosis.

| Column | Meaning |
|---|---|
| `psychological_wellbeing` | Outcome: self-reported well-being, 1–7 |
| `anxiety_symptoms` | Outcome: self-reported anxiety symptoms, 1–7 |
| `perceived_stress` | Perceived stress, 1–7 |
| `daily_hassles` | Frequency or burden of daily hassles, 1–7 |
| `financial_strain` | Perceived financial strain, 1–7 |
| `social_support` | Perceived social support, 1–7 |
| `loneliness` | Self-reported loneliness, 1–7 |
| `sleep_quality` | Self-reported sleep quality, 1–7 |
| `sleep_duration_hours` | Typical sleep duration in hours |
| `physical_activity_hours` | Physical activity in hours per week |
| `self_efficacy` | General self-efficacy, 1–7 |
| `age_years` | Age in years |

Pass both outcome names to `analyze`. It returns one network per outcome, while
using the same row split and correction across the outcome set:

```python
from open.api import analyze, plot_outcomes, save_batch

psychology = pd.read_csv("examples/tutorial/data/psychology_wellbeing.csv")
psych_outcomes = ["psychological_wellbeing", "anxiety_symptoms"]
psych_predictors = [c for c in psychology.columns if c not in psych_outcomes]

psych_result = analyze(
    psychology,
    outcomes=psych_outcomes,
    predictors=psych_predictors,
    config=config,
    split_seed=42,
)

print(psych_result.by_outcome["psychological_wellbeing"].summary())
print(psych_result.confirmation.head())
plot_outcomes(psych_result, path="tutorial-output/psychology-panels.png")
save_batch(psych_result, "tutorial-output/psychology")
```

Use `psych_result.by_outcome[name]` to inspect one outcome's full result.
`psych_result.confirmation` and `psych_result.fit` are combined tables with an
outcome column. By default, Holm correction covers testable links across both
outcome networks. In the fake data, the two scores share some unmeasured random
variation, so they are correlated. OPEN still builds and confirms a separate
network for each outcome; it does not estimate a joint model or their error
correlation.

The combined plot is a visual overlay of those separate networks. It does not
combine their estimates. For details about the method, controls and covariates,
confirmation, or graph settings, see the [method guide](method.md) and the
[API guide](api.md).
