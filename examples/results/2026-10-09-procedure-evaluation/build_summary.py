"""Rebuild the development overview from saved evidence; never fit models."""
import json
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter
import numpy as np


def main():
    base = Path(__file__).resolve().parent
    names = ('noise', 'direct', 'redundant', 'u_shape', 'interaction')
    labels = ('Noise', 'Direct', 'Redundant', 'U shape', 'Interaction')
    rows = []
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.8), sharey=True)
    for ax, engine in zip(axes, ('elastic_net', 'gbm')):
        values = []
        for name in names:
            d = json.loads((base/'pilot'/f'{name}-{engine}'/'evaluation.json').read_text(encoding='utf-8'))
            scores = d['scores']
            reduced = scores['reduced']['prediction_skill'] if scores else np.nan
            full = scores['full']['prediction_skill'] if scores else np.nan
            values.append((reduced, full))
            rows.append((name, engine, d['status'], reduced, full))
        x = np.arange(5)
        ax.bar(x-.18, [v[0] for v in values], .36, label='OPEN retained', color='#3278a8')
        ax.bar(x+.18, [v[1] for v in values], .36, label='All candidates', color='#bc8641')
        ax.axhline(0, color='gray', linewidth=.8)
        ax.set_xticks(x, labels)
        ax.set_title(engine.replace('_', ' ').title())
        ax.yaxis.set_major_formatter(PercentFormatter(1))
        ax.grid(axis='y', alpha=.2)
    axes[0].set_ylabel('Squared-error reduction versus training mean')
    fig.suptitle('Development pilot: one 80-person dataset per scenario\n'
                 'Three outer folds; three stability assessments per target', fontsize=11)
    handles, legend_labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, legend_labels, loc='lower center', ncol=2, fontsize=9)
    fig.tight_layout(rect=(0, .07, 1, 1))
    fig.savefig(base/'pilot_prediction.png', dpi=150)
    plt.close(fig)
    lines = ['# Complete-procedure evaluation: development evidence', '',
        'This checks implementation on predefined synthetic examples. It does not '
        'calibrate defaults, uncertainty, error rates or causal pathways.', '',
        '[Protocol](protocol.md) · [Software verification](verification.json) · '
        '[Independent evidence audit](audit.json)', '',
        '## Well-being survey example', '']
    survey = json.loads((base/'survey'/'evaluation.json').read_text(encoding='utf-8'))
    lines += [f'Status: **{survey["status"]}**; 180 participants, three outer folds, '
        'elastic net, ten assessments per scheduled target and 70% retention.', '',
        '| Model | RMSE | Predictive R² | Squared-error reduction vs baseline |',
        '| --- | ---: | ---: | ---: |']
    if survey['scores']:
        for model, label in (('baseline', 'Training mean'), ('reduced', 'OPEN retained predictors'),
                             ('full', 'All five candidates')):
            s = survey['scores'][model]
            lines.append(f'| {label} | {s["rmse"]:.3f} | {s["r_squared"]:.3f} | '
                         f'{s["prediction_skill"]:.1%} |')
    else:
        lines.append('Combined scores withheld: an evaluation fold failed.')
    lines += ['', '[Unified survey report and graphs](survey/REPORT.md). '
        'It includes the full-data descriptive reference network and root effects '
        'separately from held-out prediction. Good root prediction does not confirm indirect links.', '',
        '## Predefined pilot', '',
        '| Scenario | Engine | Status | Retained predictor skill | All-candidate skill |',
        '| --- | --- | --- | ---: | ---: |']
    for name, engine, status, reduced, full in rows:
        lines.append(f'| {name} | {engine} | {status} | {reduced:.1%} | {full:.1%} |')
    lines += ['', 'Zero means equal prediction to the training-mean baseline; negative means worse. '
        'A value of 50% means half as much squared prediction error, not a 50% path effect.', '',
        '![Pilot prediction comparison](pilot_prediction.png)', '',
        'The direct and redundant examples retained useful predictors. Both engines retained '
        'no predictors in the noise example. Elastic net missed the U shape; GBM retained its '
        'signal and predicted well in this dataset.', '',
        '**Important limitation:** the interaction-only GBM example predicted well with both '
        'original candidates, but the retained predictor sets were empty, empty, and just `b`. '
        'At B=3, two selections out of three (66.7%) fail the 70% cutoff. Some local fits '
        'also had insufficient predictive gain. Retained-model prediction was worse than the '
        'baseline, although the full model reduced squared error by 52.6%. This is an observed '
        'selection limitation in this small pilot, not a calculation exception. More runs '
        'reduce resampling noise but do not guarantee recovery; no thresholds were retuned.', '',
        'Tree models can learn interactions when both inputs are supplied. The recursive '
        'selection procedure can nevertheless discard those inputs. Independent interaction '
        'discovery and default calibration need dedicated scientific evaluation.', '',
        '## Reproduction and remaining gate', '',
        'Run `examples/run_procedure_evaluation.py --output NEW_DIRECTORY` from the checkout '
        'with the recorded dependencies; then run `examples/audit_procedure_evaluation.py NEW_DIRECTORY`. '
        'This script rebuilds only the overview from the saved outputs. Source fingerprints '
        'and per-network revision records identify the executed implementation.', '',
        'All attempted folds, warnings, training-only networks, selected sets, tuning parameters '
        'and predictions are saved. The final scientific gate needs a separately frozen '
        'larger scenario/grid/confirmation protocol, including placement and shared-class '
        'errors, interactions, measurement error, missingness and computational cost. '
        'P6 remains unaccepted.', '']
    (base/'REPORT.md').write_text('\n'.join(lines), encoding='utf-8')


if __name__ == '__main__':
    main()
