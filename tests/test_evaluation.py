from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

import open.evaluation as module
from layer_helpers import make_config, make_local


def frame():
    return pd.DataFrame({'y': np.arange(24, dtype=float),
                         'a': np.arange(24, dtype=float), 'b': np.ones(24)},
                        index=[0] * 24)


def discovery(monkeypatch, *, incomplete=False, empty=False):
    calls = []
    def fit(data, root, variables, config):
        calls.append(tuple(data['a'] if 'a' in data else data['y']))
        return SimpleNamespace(status='INCOMPLETE_LAYER' if incomplete else 'COMPLETE',
            parents={} if empty else {'a': ('y',), 'b': ('y',)},
            assigned_depth={'y': 0} if empty else {'y': 0, 'a': 1, 'b': 1})
    monkeypatch.setattr(module, 'fit_layer_consensus', fit)
    return calls


class IdentityPredictor:
    def tune(self, X, y, cv, loss):
        assert all(set(a).isdisjoint(b) for a, b in cv)
        self.training = tuple(X['a'])
        return {'test_parameter': 1}

    def fit(self, X, y, params):
        assert tuple(X['a']) == self.training
        return self

    def predict(self, X):
        assert set(X['a']).isdisjoint(self.training)
        return X['a'].to_numpy()


def test_discovery_tuning_fit_never_receive_test_rows(monkeypatch):
    calls = discovery(monkeypatch)
    monkeypatch.setattr(module, 'make_predictor', lambda c: IdentityPredictor())
    result = module.evaluate_procedure(frame(), 'y', ('y', 'a', 'b'), make_config(), outer_folds=3)
    assert result.status == 'COMPLETE'
    assert len(calls) == 3
    assert sorted(p.row_position for f in result.folds for p in f.predictions) == list(range(24))
    for f, values in zip(result.folds, calls):
        assert values == tuple(frame().iloc[list(f.train_positions)]['a'])
        assert f.selected_predictors == ('a', 'b')  # also covers joint member union
        assert f.reduced_parameters == {'test_parameter': 1}
    assert result.scores['reduced']['rmse'] == 0
    assert result.scores['reduced']['prediction_skill'] == 1
    assert result.scores['reduced']['r_squared'] == 1


def test_incomplete_discovery_keeps_every_fold_and_withholds_scores(monkeypatch):
    discovery(monkeypatch, incomplete=True)
    result = module.evaluate_procedure(frame(), 'y', ('y', 'a', 'b'), make_config(), outer_folds=3)
    assert result.status == 'INCOMPLETE'
    assert len(result.folds) == 3 and result.scores is None
    assert all(f.status == 'failed' and f.network is not None and not f.predictions for f in result.folds)


def test_empty_selection_is_valid_baseline_and_negative_scores_survive(monkeypatch):
    discovery(monkeypatch, empty=True)
    monkeypatch.setattr(module, 'make_predictor', lambda c: IdentityPredictor())
    result = module.evaluate_procedure(frame(), 'y', ('y', 'a', 'b'), make_config(), outer_folds=3)
    assert result.scores['reduced']['prediction_skill'] == 0
    assert result.scores['reduced']['r_squared'] < 0
    assert all(p.reduced == p.baseline for f in result.folds for p in f.predictions)


def test_constant_root_metrics_are_unavailable(monkeypatch):
    discovery(monkeypatch, empty=True)
    d = frame().assign(y=1.)
    # Root-only analysis avoids unnecessary models and is a valid baseline procedure.
    result = module.evaluate_procedure(d, 'y', ('y',), make_config(), outer_folds=3)
    assert result.scores['reduced']['rmse'] == 0
    assert result.scores['reduced']['r_squared'] is None
    assert result.scores['reduced']['prediction_skill'] is None


def test_model_failure_is_preserved_not_silently_omitted(monkeypatch):
    discovery(monkeypatch)
    monkeypatch.setattr(module, 'make_predictor', lambda c: (_ for _ in ()).throw(RuntimeError('fit failed')))
    result = module.evaluate_procedure(frame(), 'y', ('y', 'a', 'b'), make_config(), outer_folds=3)
    assert result.scores is None
    assert all(f.failure_type == 'RuntimeError' for f in result.folds)


@pytest.mark.parametrize('folds', [1, True, 25])
def test_invalid_outer_fold_counts(folds):
    with pytest.raises(ValueError):
        module.evaluate_procedure(frame(), 'y', ('y', 'a'), make_config(), outer_folds=folds)


def test_rejects_historical_scope():
    config = make_config()
    config = replace(config, stability=replace(config.stability, scope='whole_pipeline'))
    with pytest.raises(ValueError, match='layer_consensus'):
        module.evaluate_procedure(frame(), 'y', ('y', 'a'), config)


def test_real_engine_complete_small_pipeline():
    rng = np.random.default_rng(41)
    a = rng.normal(size=32)
    d = pd.DataFrame({'y': 2*a + rng.normal(scale=.1, size=32), 'a': a})
    result = module.evaluate_procedure(d, 'y', ('y', 'a'), make_config(n_resamples=2), outer_folds=2)
    assert result.status == 'COMPLETE'
    assert result.scores['reduced']['prediction_skill'] > .9
    assert result.scores['full']['prediction_skill'] > .9
    assert len(result.data_hash) == 64


def test_mixed_failed_and_successful_folds_withhold_combined_scores(monkeypatch):
    discovery(monkeypatch)
    original = module.fit_layer_consensus
    calls = []
    def fit(*args):
        network = original(*args)
        if not calls:
            network.status = 'INCOMPLETE_LAYER'
        calls.append(network)
        return network
    monkeypatch.setattr(module, 'fit_layer_consensus', fit)
    monkeypatch.setattr(module, 'make_predictor', lambda c: IdentityPredictor())
    result = module.evaluate_procedure(frame(), 'y', tuple(frame()), make_config(), outer_folds=3)
    assert [f.status for f in result.folds] == ['failed', 'completed', 'completed']
    assert result.scores is None
    assert sum(len(f.predictions) for f in result.folds) == 16


def test_finalized_joint_only_relation_enters_reduced_prediction(monkeypatch):
    import open.layer_stability as layers
    monkeypatch.setattr(layers, 'analyze_target',
        lambda d, t, c, config, splits: replace(make_local(t, c, shared=(('a', 'b'),)),
                                               n_observed_target_rows=len(d)))
    monkeypatch.setattr(module, 'make_predictor', lambda c: IdentityPredictor())
    result = module.evaluate_procedure(frame(), 'y', tuple(frame()),
        make_config(n_resamples=1, max_depth=1), outer_folds=3)
    assert result.status == 'COMPLETE'
    for fold in result.folds:
        retained = [r for r in fold.network.layers[0].evidence[0].relations if r.qualifies]
        assert len(retained) == 1 and retained[0].relation == 'SHARED_GROUP'
        assert fold.selected_predictors == ('a', 'b')


def test_reduced_and_full_are_independently_tuned(monkeypatch):
    discovery(monkeypatch)
    original = module.fit_layer_consensus
    def fit(*args):
        network = original(*args)
        network.parents = {'a': ('y',)}
        return network
    monkeypatch.setattr(module, 'fit_layer_consensus', fit)
    class WidthPredictor(IdentityPredictor):
        def tune(self, X, y, cv, loss):
            super().tune(X, y, cv, loss)
            return {'width': len(X.columns)}
    monkeypatch.setattr(module, 'make_predictor', lambda c: WidthPredictor())
    result = module.evaluate_procedure(frame(), 'y', tuple(frame()), make_config(), outer_folds=3)
    assert all(f.reduced_parameters['width'] == 1 and f.full_parameters['width'] == 2 for f in result.folds)


@pytest.mark.parametrize('bad', ['shape', 'nonfinite'])
def test_invalid_prediction_is_recorded_as_failure(monkeypatch, bad):
    discovery(monkeypatch)
    class InvalidPredictor(IdentityPredictor):
        def predict(self, X):
            return np.zeros((len(X), 1)) if bad == 'shape' else np.full(len(X), np.nan)
    monkeypatch.setattr(module, 'make_predictor', lambda c: InvalidPredictor())
    result = module.evaluate_procedure(frame(), 'y', tuple(frame()), make_config(), outer_folds=3)
    assert result.scores is None and all(f.status == 'failed' for f in result.folds)
