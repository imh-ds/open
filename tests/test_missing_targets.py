"""Regression coverage for incomplete predictors becoming recursive targets."""
from dataclasses import replace
import csv
import json

import numpy as np
import pandas as pd
import pytest

import open.analysis as analysis
import open.grouping as grouping
import open.recurse as recurse
from open.config import (AnalysisConfig, OpenConfig, PerformanceCVConfig,
                         StabilityConfig, ThresholdConfig, TuningCVConfig)
from open.graph import StopReason
from open.io import write_open_outputs
from open.learners.elastic_net import ElasticNetPredictor
from open.metadata import validate_metadata
from open.risk import evaluate_risk
from open.splits import SplitFold, SplitPlan, make_split_plan
from open.stability import fit_stable_open


def chain():
    rng = np.random.default_rng(271828)
    s = rng.normal(size=80)
    d = s + .7 * rng.normal(size=80)
    a = d + .3 * rng.normal(size=80)
    data = pd.DataFrame({'A': a, 'D': d, 'S': s})
    data.loc[0, 'D'] = np.nan
    config = OpenConfig(
        AnalysisConfig('A', 27, max_depth=2),
        thresholds=ThresholdConfig(unique_relative=.15),
        performance_cv=PerformanceCVConfig(2, 1),
        tuning_cv=TuningCVConfig(2),
        stability=StabilityConfig(fraction=1., n_resamples=2),
    )
    return data, config


def test_missing_targets_keep_paired_original_positions_and_training_means():
    data = pd.DataFrame({'y': [0., np.nan, 2., 3., 4., 5., 6., 7.],
                         'x': np.arange(8, dtype=float), 'z': np.arange(8)[::-1]},
                        index=np.arange(100, 108))
    plan = SplitPlan(8, 2, 1, 17, (
        SplitFold(0, 0, (4, 5, 6, 7), (0, 1, 2, 3)),
        SplitFold(0, 1, (0, 1, 2, 3), (4, 5, 6, 7)),
    ))
    common = dict(target='y', candidate_set=('x', 'z'), split_plan=plan)
    baseline = evaluate_risk(data, predictors=(), **common)
    full = evaluate_risk(data, predictors=('x', 'z'), predictor=ElasticNetPredictor(),
                         tuning_folds=2, **common)
    params = {(f.repeat_index, f.fold_index): f.fitted_parameters for f in full.folds}
    ablated = evaluate_risk(data, predictors=('x',), predictor=ElasticNetPredictor(),
                            fitted_parameters=params, **common)
    def keys(result):
        return [(f.repeat_index, f.fold_index, r.row_position)
                for f in result.folds for r in f.observations]
    assert keys(baseline) == keys(full) == keys(ablated)
    assert [key[2] for key in keys(full)] == [0, 2, 3, 4, 5, 6, 7]
    assert all(r.predicted == pytest.approx(5.5) for r in baseline.folds[0].observations)
    assert all(r.predicted == pytest.approx(5/3) for r in baseline.folds[1].observations)
    assert baseline.aggregate_risk == pytest.approx(np.mean(
        [r.loss for f in baseline.folds for r in f.observations]))
    assert baseline.split_plan_id == full.split_plan_id == ablated.split_plan_id


def test_real_recursive_chain_with_missing_promoted_target():
    data, config = chain()
    run = recurse.build_network_once(data, 'A', tuple(data), config)
    assert dict(run.assigned_depth) == {'A': 0, 'D': 1, 'S': 2}
    assert {(e.source, e.target) for e in run.core_edges} == {('D', 'A'), ('S', 'D')}
    assert run.local_results['D'].n_observed_target_rows == 79
    assert run.local_results['D'].n_missing_target_rows == 1


def test_missing_target_dependence_and_group_models_use_observed_rows(monkeypatch):
    rng = np.random.default_rng(120)
    x = rng.normal(size=80)
    data = pd.DataFrame({'D': x + rng.normal(scale=.15, size=80),
                         'X1': x, 'X2': x + rng.normal(scale=.01, size=80)})
    data.loc[0, 'D'] = np.nan
    config = OpenConfig(AnalysisConfig('A', 27), thresholds=ThresholdConfig(unique_relative=.15),
                        performance_cv=PerformanceCVConfig(2, 1), tuning_cv=TuningCVConfig(2))
    original = grouping.score_dependence
    seen = []
    def score(training_data, *args, **kwargs):
        assert 0 not in training_data.index
        seen.append(tuple(training_data.index))
        return original(training_data, *args, **kwargs)
    monkeypatch.setattr(grouping, 'score_dependence', score)
    result = analysis.analyze_target(data, 'D', ('X1', 'X2'), config, make_split_plan(data, config))
    assert len(seen) == 2
    assert result.promoted == frozenset({'X1', 'X2'})
    assert result.shared_components[0].validated
    for risk in [p.ablation for p in result.pair_ablations.pairs] + [
            component.ablation for component in result.shared_components]:
        assert {r.row_position for f in risk.folds for r in f.observations} == set(range(1, 80))


def test_infeasible_recursive_target_stops_only_its_branch(monkeypatch):
    from test_recursion import result
    data = pd.DataFrame({'y': np.arange(12, dtype=float),
                         'a': [1.] + [np.nan] * 11,
                         'b': np.arange(12, dtype=float), 'c': np.arange(12, dtype=float)})
    config = OpenConfig(AnalysisConfig('y', 41, max_depth=2),
                        performance_cv=PerformanceCVConfig(2, 1), tuning_cv=TuningCVConfig(2))
    calls = []
    def analyze(data, target, candidates, config, plan):
        calls.append(target)
        return result(target, candidates, ('a', 'b') if target == 'y' else ('c',))
    monkeypatch.setattr(recurse, 'analyze_target', analyze)
    run = recurse.build_network_once(data, 'y', tuple(data), config)
    assert calls == ['y', 'b']
    assert run.stop_reasons['a'] is StopReason.INSUFFICIENT_EFFECTIVE_SAMPLE_SIZE
    assert run.assigned_depth['c'] == 2
    assert 'a' not in run.local_results


def test_empty_observed_test_fold_is_rejected_before_any_model_fit(monkeypatch):
    data, config = chain()
    plan = make_split_plan(data, config)
    data.loc[list(plan.splits[1].test_indices), 'D'] = np.nan
    monkeypatch.setattr(analysis, 'evaluate_risk', lambda *a, **kw: pytest.fail('fitting started'))
    with pytest.raises(ValueError, match='observed'):
        analysis.analyze_target(data, 'D', ('S',), config, plan)


def test_missing_recursive_target_stability_and_export(tmp_path):
    data, config = chain()
    result = fit_stable_open(data, 'A', tuple(data), config)
    assert result.batch.completed_runs == result.stability.denominator == 2
    assert result.batch.failed_runs == 0
    metadata = validate_metadata(data.columns,
        [{'name': name, 'role': 'root' if name == 'A' else 'candidate', 'type': 'continuous'}
         for name in data], 'A')
    write_open_outputs(result, tmp_path / 'bundle', metadata=metadata)
    with (tmp_path / 'bundle' / 'local_targets.csv').open() as stream:
        rows = list(csv.DictReader(stream))
    d_rows = [row for row in rows if row['target'] == 'D']
    assert len(d_rows) == 2
    assert all(row['n_observed_target_rows'] == '79' and row['n_missing_target_rows'] == '1'
               for row in d_rows)
    saved = json.loads((tmp_path / 'bundle' / 'analysis_metadata.json').read_text())
    assert saved['run_ledger'][0]['target_stop_reasons']['S'] == 'MAX_DEPTH'
