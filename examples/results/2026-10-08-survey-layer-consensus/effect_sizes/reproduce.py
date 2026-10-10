"""Recover coefficients from the original survey CV fits; verify saved risks.

Run with Python from the repository root. This is a demonstration analysis,
not a change to OPEN's selection rules or artifact schema.
"""
from pathlib import Path
import hashlib
import json
import sys
import warnings

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT / "src"))
from open.config import load_config
from open.learners.elastic_net import ElasticNetPredictor

OUT = Path(__file__).resolve().parent
BASE = OUT.parent
data = pd.read_csv(BASE / "sample.csv")
config = load_config(BASE / "config.yaml")
meta = json.loads((BASE / "bundle/analysis_metadata.json").read_text())
ledger = json.loads((BASE / "bundle/assessment_ledger.json").read_text())["assessments"]
plans = {p["run_index"]: p for p in meta["plans"]}
edges = pd.read_csv(BASE / "bundle/edges.csv")
retained = {(r.source, r.target): r.selection_frequency for r in edges.itertuples()}
fold_rows, run_rows, performance = [], [], []
max_error = 0.0
captured = []
with warnings.catch_warnings(record=True) as recorded:
    warnings.simplefilter("always")
    for record in ledger:
        assert record["status"] == "completed"
        plan = plans[record["run_index"]]
        sample = data.iloc[plan["sampled_positions"]].reset_index(drop=True)
        target, candidates = record["target"], record["candidates"]
        y = sample[target].to_numpy(float)
        assert np.isfinite(sample.to_numpy(float)).all(), "This analysis requires complete numeric fixtures"
        losses, this_run = [], []
        for split in plan["split_plan"]["splits"]:
            train, test = split["train_indices"], split["test_indices"]
            X_train, y_train = sample.iloc[train][candidates], y[train]
            model = ElasticNetPredictor(config)
            inner = KFold(n_splits=config.tuning_cv.folds, shuffle=True, random_state=plan["seed"])
            params = model.tune(X_train, y_train, inner, "mse")
            model.fit(X_train, y_train, params)
            predicted = model.predict(sample.iloc[test][candidates])
            pipeline = model._pipeline
            scaler = pipeline["preprocessor"]["scaler"]
            regressor = pipeline["regressor"]
            b = regressor.coef_ / scaler.scale_
            intercept = regressor.intercept_ - np.dot(b, scaler.mean_)
            # Verify conversion back to raw predictor units, not just the MSE.
            np.testing.assert_allclose(sample.iloc[test][candidates].to_numpy() @ b + intercept,
                                       predicted, atol=1e-12, rtol=1e-12)
            beta = b * np.std(X_train.to_numpy(), axis=0, ddof=0) / np.std(y_train, ddof=0)
            losses.extend((y[test] - predicted) ** 2)
            for i, source in enumerate(candidates):
                row = dict(target=target, source=source, run=record["run_index"],
                           fold=split["fold"], repeat=split["repeat"],
                           b=float(b[i]), beta=float(beta[i]), alpha=params["alpha"],
                           l1_ratio=params["l1_ratio"])
                fold_rows.append(row)
                this_run.append(row)
        risk = float(np.mean(losses))
        saved = record["local"]["relevance"]["full_risk"]
        error = abs(risk - saved)
        max_error = max(max_error, error)
        assert error < 1e-12, (target, record["run_index"], risk, saved)
        # Each held-out row is represented once in this demo's two outer folds.
        assert len(losses) == len(y) and config.performance_cv.repeats == 1
        variance = float(np.var(y, ddof=0))
        performance.append(dict(target=target, run=record["run_index"],
                                full_mse=risk, outcome_variance=variance, cv_r2=1-risk/variance))
        for source in candidates:
            fits = [r for r in this_run if r["source"] == source]
            raw = record["local"]["relevance"]["unique"][source]["raw"]
            run_rows.append(dict(target=target, source=source, run=record["run_index"],
                b=float(np.median([r["b"] for r in fits])),
                beta=float(np.median([r["beta"] for r in fits])),
                delta_cv_r2=raw/variance))
        print(f"Verified {target}, run {record['run_index']}: error={error:.3g}", flush=True)
    captured = [str(w.message) for w in recorded]
fold_frame, run_frame = pd.DataFrame(fold_rows), pd.DataFrame(run_rows)
summaries = []
for (target, source), group in run_frame.groupby(["target", "source"], sort=True):
    summaries.append(dict(target=target, source=source, retained=(source,target) in retained,
        stability=retained.get((source,target)), assessments=len(group),
        b_median=group.b.median(), beta_median=group.beta.median(),
        beta_p10=group.beta.quantile(.1), beta_p90=group.beta.quantile(.9),
        delta_cv_r2_median=group.delta_cv_r2.median()))
summary = pd.DataFrame(summaries)
fold_frame.to_csv(OUT / "fold_coefficients.csv", index=False)
run_frame.to_csv(OUT / "assessment_effect_sizes.csv", index=False)
summary.to_csv(OUT / "effect_sizes.csv", index=False)
pd.DataFrame(performance).to_csv(OUT / "prediction_performance.csv", index=False)
verification = dict(assessments=len(ledger), reconstructed_outer_models=len(set(
    (r['target'],r['run'],r['repeat'],r['fold']) for r in fold_rows)),
    max_absolute_full_mse_difference=max_error, tolerance=1e-12,
    coefficient_prediction_reconstruction_verified=True,
    sample_sha256=hashlib.sha256((BASE/'sample.csv').read_bytes()).hexdigest(),
    warnings=captured,
    beta_definition="raw b * training SD(X) / training SD(Y); SD uses ddof=0",
    aggregation="median of outer-fold coefficients per assessment, then median across all 20 assessments",
    interval="10th-90th percentiles across assessment medians; descriptive spread, not a confidence interval",
    delta_cv_r2_definition="(ablated MSE - full MSE) / held-out outcome variance; same rows and folds",
    conditioning="original complete candidate sets; never refitted to retained predictors only")
(OUT/'verification.json').write_text(json.dumps(verification,indent=2)+'\n')

labels = {name: name.replace('_',' ').capitalize() for name in data.columns}
labels['wellbeing']='Well-being'
shown = summary[summary.retained].sort_values(['target','source']).reset_index(drop=True)
fig, ax = plt.subplots(figsize=(10,5))
for i, row in shown.iterrows():
    ax.plot([row.beta_p10,row.beta_p90],[i,i],color='#4f7cac',linewidth=3)
    ax.scatter(row.beta_median,i,color='#183a5c',s=55,zorder=3)
ax.axvline(0,color='gray',linewidth=1)
ax.set_yticks(range(len(shown)),[f"{labels[r.source]} → {labels[r.target]}" for r in shown.itertuples()])
ax.invert_yaxis()
ax.set_xlabel('Standardized coefficient β (predicted outcome SD per predictor SD)')
ax.set_title('Synthetic survey: coefficients for retained predictive links\nDots: median; lines: 10th–90th percentile spread, not confidence intervals')
ax.grid(axis='x',alpha=.2)
fig.tight_layout()
fig.savefig(OUT/'standardized_coefficients.png',dpi=180)
plt.close(fig)
lines = ['# Synthetic survey effect sizes','',
    'Recovered from the original 70% layer-consensus analysis. No selection rules or network structure changed.',
    '', '## How to read the numbers','',
    '**Standardized β:** a one-SD higher predictor corresponds to β SD higher/lower predicted outcome, holding the other local candidates fixed. These are regularized elastic-net coefficients, not OLS coefficients or causal effects.',
    '', '**Raw b:** predicted outcome points per one predictor point. These synthetic scales are not validated questionnaire instruments.',
    '', '**ΔCV R²:** the loss in cross-validated R² after removing that predictor and refitting with the original fixed ablation hyperparameters. This is a dimensionless unique predictive contribution, not the square of β. Contributions need not add to total R².',
    '', 'β and b are medians of per-assessment fold medians over all 20 assessments, including assessments where the relationship did not qualify. The percentile spread describes variation across overlapping samples; it is not a confidence interval.',
    '', 'Cohen’s d is a standardized difference between groups; these predictors are continuous, so no artificial high/low split or conversion to d was made.',
    '', '## Retained relationships','',
    '| Predictor → outcome | Raw b | Standardized β | β 10th–90th percentile | ΔCV R² |',
    '| --- | ---: | ---: | ---: | ---: |']
for r in shown.itertuples():
    lines.append(f'| {labels[r.source]} → {labels[r.target]} | {r.b_median:+.3f} | {r.beta_median:+.3f} | {r.beta_p10:+.3f} to {r.beta_p90:+.3f} | {r.delta_cv_r2_median:+.4f} |')
lines += ['', '![Coefficient plot](standardized_coefficients.png)', '', '## All predictors of Well-being','',
    '| Predictor | Standardized β | ΔCV R² | Retained direct link? |','| --- | ---: | ---: | --- |']
for r in summary[summary.target=='wellbeing'].itertuples():
    lines.append(f'| {labels[r.source]} | {r.beta_median:+.3f} | {r.delta_cv_r2_median:+.4f} | {"Yes" if r.retained else "No"} |')
lines += ['', '## Overall prediction and conditioning','']
for target, group in pd.DataFrame(performance).groupby('target'):
    lines.append(f'- {labels[target]}: median local cross-validated R² = **{group.cv_r2.median():.4f}**.')
lines += ['', 'The Well-being coefficients adjust for all five original predictors, even though only three direct links passed the stability threshold. Sleep Quality, Social Support, and Physical Activity models each adjust for both Perceived Stress and Work Demands. Coefficients depend on these adjustment sets; they are not a single structural equation model and should not be multiplied into indirect causal effects.',
    '', 'R² here uses the held-out outcome variance. This differs slightly from the existing baseline-relative score, whose denominator is the error from training-fold mean predictions. Both are invariant to changing outcome measurement units, but neither guarantees comparability across different populations, measurement reliabilities, or model specifications.',
    '', '## Verification','',
    f'All {len(ledger)} original assessments and {verification["reconstructed_outer_models"]} outer models were reconstructed. Maximum absolute difference from saved full-model MSE: {max_error:.3g} (required < 1e-12). Raw coefficients also reproduce pipeline predictions to numerical precision. Captured warnings: {len(captured)}.',
    '', '[Detailed coefficients](effect_sizes.csv), [assessment-level values](assessment_effect_sizes.csv), [fold coefficients and tuning parameters](fold_coefficients.csv), [prediction performance](prediction_performance.csv), [verification](verification.json), [reproduction script](reproduce.py).',
    '', 'Method references: [standardized coefficients](https://easystats.github.io/parameters/articles/standardize_parameters_effsize.html), [Cohen’s d](https://easystats.github.io/effectsize/reference/cohens_d.html), [elastic-net coefficients](https://scikit-learn.org/1.7/modules/generated/sklearn.linear_model.ElasticNet.html).',
    '', 'Synthetic demonstration only. No post-selection confidence intervals, population effect claims, or independent validation of the discovered network are provided.']
(OUT/'REPORT.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
print(summary.to_string(index=False))
print(json.dumps(verification,indent=2))
