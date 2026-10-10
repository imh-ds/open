"""Researcher-facing report with separate evaluation and descriptive evidence."""
from dataclasses import dataclass, fields
import csv
import io
import json
from pathlib import Path
import tempfile

import numpy as np

from open.effects import compare_target_engines, save_engine_comparison
from open.evaluation import ProcedureEvaluation, evaluate_procedure
from open.io import _publish_no_replace
from open.layer_artifacts import portable
from open.layer_io import write_layer_outputs
from open.layer_stability import fit_layer_consensus
from open.layer_types import LayeredOPENResult
from open.splits import make_split_plan


@dataclass(frozen=True)
class ResearcherAnalysis:
    evaluation: ProcedureEvaluation
    reference: LayeredOPENResult | None
    effects: dict | None
    supplement_failures: tuple[str, ...] = ()


def analyze_researcher(data, root, variables, config, *, outer_folds=5,
                       outer_seed=None, include_effects=True, effect_bins=10):
    """Evaluate first, then produce explicitly descriptive full-data supplements.

    Root effects use the full original candidate set and the fixed engine.
    They do not estimate the effects of the selected reduced prediction model.
    """
    if type(include_effects) is not bool or type(effect_bins) is not int or effect_bins < 2:
        raise ValueError('include_effects must be boolean and effect_bins an integer at least two')
    evaluation = evaluate_procedure(data, root, variables, config,
                                    outer_folds=outer_folds, outer_seed=outer_seed)
    frame = data.loc[:, list(variables)].reset_index(drop=True).copy()
    reference = effects = None
    failures = []
    try:
        reference = fit_layer_consensus(frame, root, variables, config)
    except Exception as exc:
        failures.append(f'Reference network: {type(exc).__name__}: {str(exc)[:2000]}')
    candidates = tuple(n for n in variables if n != root)
    if include_effects and candidates:
        try:
            effects = compare_target_engines(frame, root, candidates, config,
                make_split_plan(frame, config), engines=config.learner.library,
                joint_groups=tuple(r.members for layer in reference.layers if layer.depth == 0
                    for t in layer.evidence for r in t.relations if r.qualifies and r.members)
                    if reference is not None else (),
                bins=effect_bins, adaptive_bins=True)
        except Exception as exc:
            failures.append(f'Root effects: {type(exc).__name__}: {str(exc)[:2000]}')
    return ResearcherAnalysis(evaluation, reference, effects, tuple(failures))


def _text(value):
    return str(value).replace('|', '\\|').replace('\n', ' ').replace('\r', ' ')


def _number(value):
    return 'unavailable' if value is None else f'{value:.3f}'


def render_researcher_report(analysis, *, plot_files=()):
    evaluation, reference = analysis.evaluation, analysis.reference
    root = evaluation.config.analysis.root
    completed = sum(f.status == 'completed' for f in evaluation.folds)
    lines = [f'# OPEN researcher report: {_text(root)}', '',
        f'Complete-procedure evaluation: **{evaluation.status}**; {completed}/{len(evaluation.folds)} folds completed.', '',
        'This evaluates root prediction, not SEM global fit or the correctness of indirect pathways.', '',
        f'Engine: `{evaluation.config.learner.library[0]}`. Retention threshold: '
        f'{evaluation.config.thresholds.stability:.0%}. Independent participant rows: {evaluation.n_rows}.', '',
        '## Does the selected predictor set predict new participants?', '',
        'Each fold discovers its own complete network using training participants only. '
        'Retained direct predictors (including joint members) are then separately tuned/refitted. '
        'The all-candidate comparator is separately tuned on the same training participants. '
        'No held-out participants enter discovery, preprocessing or tuning.', '']
    if evaluation.scores is None:
        lines += ['Combined prediction scores are **withheld** because at least one planned fold failed.', '']
    else:
        lines += ['| Predictor | RMSE (outcome units) | R² | Prediction skill |',
                  '| --- | ---: | ---: | ---: |']
        for name, label in [('baseline', 'Training-mean baseline'), ('reduced', 'OPEN retained predictors'),
                            ('full', 'All original candidates')]:
            score = evaluation.scores[name]
            lines.append(f'| {label} | {_number(score["rmse"])} | {_number(score["r_squared"])} | '
                         f'{_number(score["prediction_skill"])} |')
        lines += ['', 'Lower RMSE means smaller prediction errors. R² compares squared errors with '
            'the variation of the pooled held-out outcomes. Prediction skill compares errors with '
            'the training-mean baseline: 0 means equal performance; negative means worse; '
            '0.30 means 30% less squared prediction error. It is not a path effect size.', '']
    lines += ['| Fold | Status | Retained direct predictors | Failure |', '| --- | --- | --- | --- |']
    for fold in evaluation.folds:
        names = ', '.join(fold.selected_predictors) or ('none; training mean' if fold.status == 'completed' else 'unavailable')
        lines.append(f'| {fold.fold} | {fold.status} | {_text(names)} | {_text(fold.failure_message or "—")} |')
    lines += ['', 'Fold predictor sets can differ. These are training-fold discoveries, not one consensus '
        'graph assembled from their arrows. Settings must be fixed before this evaluation; selecting '
        'settings after inspecting these scores needs new confirmation data.', '',
        '## What does the descriptive reference network show?', '',
        'The reference network is discovered separately using all supplied participants. '
        'Its relation frequencies are conditional on finalized earlier layers. '
        'This is descriptive evidence, not independent confirmation of its structure.', '']
    if reference is None:
        lines += ['Reference network unavailable.', '']
    else:
        lines += [f'Reference status: **{reference.status}**.', '',
            '| Relation | Retained / planned | Frequency | Margin above threshold |',
            '| --- | ---: | ---: | ---: |']
        for layer in reference.layers:
            for target in layer.evidence:
                for relation in target.relations:
                    if not layer.finalized or not relation.qualifies:
                        continue
                    source = ' + '.join(relation.members) if relation.members else relation.source
                    count = relation.frequency
                    margin = count.probability-reference.config.thresholds.stability
                    lines.append(f'| {_text(source)} → {_text(target.target)}'
                        f'{" (joint)" if relation.members else ""} | {count.selected}/{count.planned} | '
                        f'{count.probability:.1%} | {margin*100:.1f} percentage points |')
        lines += ['', 'A small margin means the link is close to the chosen display/retention cutoff. '
            'The cutoff is not a confidence level or false-discovery guarantee. '
            'An omitted relation does not establish independence.', '',
            '| Local target | Completed / planned | Median baseline MSE | Median full MSE |',
            '| --- | ---: | ---: | ---: |']
        for layer in reference.layers:
            for target in layer.evidence:
                locals_ = [a.local for a in target.assessments if a.local is not None]
                baseline = float(np.median([v.relevance.baseline_risk for v in locals_])) if locals_ else None
                full = float(np.median([v.relevance.full_risk for v in locals_])) if locals_ else None
                lines.append(f'| {_text(target.target)} | {target.completed_count}/{len(target.assessments)} | '
                             f'{_number(baseline)} | {_number(full)} |')
        lines += ['', 'These local CV scores are conditional on discovered candidates, not complete-procedure '
            'validation. Targets can have different observed participants and different units; '
            'do not average their MSEs into a network fit score.', '']
    lines += ['## How large and what shape are the root predictive associations?', '']
    if analysis.effects is None:
        lines += ['Root effect supplement: **not assessed or unavailable**.', '']
    else:
        lines += ['These descriptive root effects use the **full candidate set** on the full supplied dataset. '
            'They are separate from the retained-predictor model evaluated above.', '',
            '| Predictor | Unique ΔCV R² | Q25 → Q75 (outcome SD) | Curve magnitude (outcome SD) |',
            '| --- | ---: | ---: | ---: |']
        for effect in analysis.effects['engines'][0]['effects']:
            lines.append(f'| {_text(effect["source"])} | {_number(effect["delta_cv_r2"])} | '
                f'{_number(effect["standardized_contrast"])} | {_number(effect["standardized_curve_sd"])} |')
        lines += ['', 'ΔCV R² describes the prediction lost when removing one predictor; it can include '
            'interaction contributions. Q25 → Q75 describes a signed curve change. Curve magnitude '
            'describes overall average-curve variation and can remain positive for a U shape. '
            'None is a causal effect, Cohen’s d, or an additive share of explained variance. '
            'Unavailable effects reflect insufficient support or other recorded limitations.', '',
            'Percentile points, adjacent contrasts, joint contributions and support evidence are in '
            '[the effect supplement](effects/comparison.json).', '']
    lines += ['## What still needs checking?', '',
        '- Threshold sensitivity: **not assessed by this workflow**; reconstruct the full network at other cutoffs.',
        '- Sampling uncertainty and calibrated false-selection rates: **not assessed**.',
        '- Same-depth peer relations and interaction-specific effect curves: **not assessed**.',
        '- Scope: independent cross-sectional scale-level data with a continuous observed root. '
        'Grouped, longitudinal and categorical-outcome support is unavailable.',
        '- Scientific validation: development examples do not establish reliable defaults or minimum sample sizes.', '']
    for failure in analysis.supplement_failures:
        lines += [f'Supplement failure: {_text(failure)}', '']
    for name in plot_files:
        lines += [f'![{name}]({name})', '']
    lines += ['Evidence: [evaluation.json](evaluation.json), [predictions.csv](predictions.csv). '
              'Training-fold network bundles are saved in `fold_N/`; the full-data reference is in `reference/`.', '']
    return '\n'.join(lines)


def write_researcher_outputs(analysis, output_dir, *, metadata, plots=False):
    """Atomically publish portable evidence and a plain-language report once."""
    evaluation = analysis.evaluation
    if metadata.root != evaluation.config.analysis.root or metadata.candidates != tuple(
            n for n in evaluation.variables if n != metadata.root):
        raise ValueError('metadata does not match evaluation variables/root')
    destination = Path(output_dir)
    if destination.exists():
        raise FileExistsError(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=f'.{destination.name}.', dir=destination.parent) as temporary:
        stage = Path(temporary)
        folds = []
        predictions = []
        for fold in evaluation.folds:
            record = {f.name: portable(getattr(fold, f.name)) for f in fields(fold) if f.name != 'network'}
            record['network_bundle'] = f'fold_{fold.fold}' if fold.network is not None else None
            folds.append(record)
            predictions.extend(dict(fold=fold.fold, **portable(p)) for p in fold.predictions)
            if fold.network is not None:
                write_layer_outputs(fold.network, stage/f'fold_{fold.fold}', metadata=metadata)
        payload = dict(schema='open.procedure_evaluation', schema_version=1,
            status=evaluation.status, config=portable(evaluation.config), variables=list(evaluation.variables),
            data_sha256=evaluation.data_hash, n_rows=evaluation.n_rows, outer_seed=evaluation.outer_seed,
            folds=folds, predictions=predictions, scores=portable(evaluation.scores),
            supplement_failures=list(analysis.supplement_failures),
            inference='root_prediction_not_structural_validation')
        (stage/'evaluation.json').write_text(json.dumps(payload, indent=2, ensure_ascii=False, allow_nan=False)+'\n', encoding='utf-8')
        handle = io.StringIO(newline='')
        writer = csv.DictWriter(handle, fieldnames=('fold', 'row_position', 'observed', 'baseline', 'reduced', 'full'), lineterminator='\n')
        writer.writeheader()
        writer.writerows(predictions)
        (stage/'predictions.csv').write_text(handle.getvalue(), encoding='utf-8')
        if analysis.reference is not None:
            write_layer_outputs(analysis.reference, stage/'reference', metadata=metadata)
        if analysis.effects is not None:
            save_engine_comparison(analysis.effects, stage/'effects')
        plot_files = []
        if plots:
            import matplotlib.pyplot as plt
            if evaluation.scores is not None:
                fig, ax = plt.subplots(figsize=(6, 5))
                y = [p['observed'] for p in predictions]
                ax.scatter(y, [p['reduced'] for p in predictions], alpha=.6)
                ax.plot([min(y), max(y)], [min(y), max(y)], color='gray', linestyle='--')
                ax.set(xlabel=f'Observed {metadata.root}', ylabel=f'Held-out predicted {metadata.root}',
                       title='Complete-procedure prediction: each participant held out')
                fig.tight_layout()
                fig.savefig(stage/'prediction.png', dpi=150)
                plt.close(fig)
                plot_files.append('prediction.png')
            if analysis.reference is not None:
                from open.visualization import load_visualization_data, plot_open_network
                fig = plot_open_network(load_visualization_data(stage/'reference'))
                fig.savefig(stage/'reference_network.png', dpi=150)
                plt.close(fig)
                plot_files.append('reference_network.png')
            if analysis.effects is not None:
                fig, ax = plt.subplots(figsize=(7, 5))
                for effect in analysis.effects['engines'][0]['effects']:
                    points = effect['percentile_points']
                    ax.plot([p['percentile'] for p in points],
                            [p['standardized_ale'] if p['standardized_ale'] is not None else np.nan for p in points],
                            marker='o', label=effect['source'])
                ax.axhline(0, color='gray', linewidth=.8)
                ax.set(xlabel='Predictor percentile', ylabel='Centered ALE (outcome SD)',
                       title='Descriptive root curves: full candidate set')
                ax.legend(fontsize=8)
                fig.tight_layout()
                fig.savefig(stage/'root_curves.png', dpi=150)
                plt.close(fig)
                plot_files.append('root_curves.png')
        (stage/'REPORT.md').write_text(render_researcher_report(analysis, plot_files=plot_files), encoding='utf-8')
        _publish_no_replace(stage, destination)
    return destination/'REPORT.md'
