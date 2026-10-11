import numpy as np
import pandas as pd
import pytest


class Model:
    def __init__(self, function):
        self.function = function
        self.rows = 0

    def predict(self, X):
        self.rows += len(X)
        return self.function(X)


def reference():
    rng = np.random.default_rng(19)
    return pd.DataFrame(rng.normal(size=(800,4)),columns=['a','b','noise1','noise2'])


def settings(**kwargs):
    from open.interactions import PairScreenConfig
    return PairScreenConfig(**kwargs)


def screen(model, **kwargs):
    from open.interactions import screen_pair_interactions
    X = kwargs.pop('X',reference())
    return screen_pair_interactions(model,X,tuple(X),outcome_sd=kwargs.pop('sd',1),
                                    config=settings(**kwargs))


def test_additive_nonlinear_effects_do_not_become_interactions():
    model = Model(lambda X: X.a**2 + np.sin(X.b) + 3*X.noise1)
    result = screen(model)
    assert all(p['screen']['status']=='available' for p in result['pairs'])
    assert max(p['screen']['standardized_rms'] for p in result['pairs']) < 1e-12
    assert not any(p['selected_for_detail'] for p in result['pairs'])


def test_pure_product_is_found_without_individual_importance_filter():
    result = screen(Model(lambda X:X.a*X.b),max_details=1)
    selected = [p for p in result['pairs'] if p['selected_for_detail']]
    assert len(selected)==1 and selected[0]['members']==['a','b']
    assert selected[0]['detail']['status']=='available'
    assert selected[0]['detail']['standardized_rms'] > 0
    assert all(p['screen']['standardized_rms'] < 1e-12
               for p in result['pairs'] if p['members']!=['a','b'])


def test_prediction_budget_and_all_planned_pairs_are_preserved():
    model = Model(lambda X:X.a*X.b)
    result = screen(model,prediction_rows=256,max_details=2)
    assert result['planned_pairs']==6 and len(result['pairs'])==6
    assert model.rows==result['prediction_rows']==256
    assert all(p['screen']['status']=='prediction_budget' for p in result['pairs'][1:])
    assert result['pairs'][0]['detail']['status']=='prediction_budget'


def test_pair_cap_is_deterministic_and_unexamined_is_not_zero():
    result = screen(Model(lambda X:X.a*X.b),max_pairs=1)
    assert all(p['screen']['status']=='pair_limit' and p['screen']['standardized_rms'] is None
               for p in result['pairs'][1:])
    X=reference()[['noise2','b','a','noise1']]
    reordered = screen(Model(lambda X:X.a*X.b),X=X,max_pairs=1)
    assert [p['members'] for p in result['pairs']]==[p['members'] for p in reordered['pairs']]


def test_scaling_missing_constants_and_sparse_support():
    one=screen(Model(lambda X:X.a*X.b))
    hundred=screen(Model(lambda X:100*X.a*X.b),sd=100)
    assert one['pairs'][0]['screen']['standardized_rms']==pytest.approx(
        hundred['pairs'][0]['screen']['standardized_rms'])
    X=reference()[['a','b']]
    X.b=X.a
    sparse=screen(Model(lambda X:X.a*X.b),X=X)
    assert sparse['pairs'][0]['screen']['status']=='unsupported_cells'
    assert sparse['pairs'][0]['screen']['standardized_rms'] is None
    X.b=1
    constant=screen(Model(lambda X:X.a),X=X)
    assert constant['pairs'][0]['screen']['status']=='constant_predictor'
    X.b=np.nan
    missing=screen(Model(lambda X:X.a),X=X)
    assert missing['pairs'][0]['screen']['status']=='missing_reference'


def test_constant_outcome_does_not_consume_prediction_budget():
    model=Model(lambda X:X.a*X.b)
    result=screen(model,sd=0)
    assert model.rows==0
    assert all(p['screen']['status']=='constant_outcome' for p in result['pairs'])


@pytest.mark.parametrize('scale',[1e160,1e-170])
def test_large_finite_units_do_not_overflow_rms(scale):
    one=screen(Model(lambda X:X.a*X.b))
    enormous=screen(Model(lambda X:scale*X.a*X.b),sd=scale)
    assert np.isfinite(enormous['pairs'][0]['screen']['raw_rms'])
    assert enormous['pairs'][0]['screen']['standardized_rms']==pytest.approx(
        one['pairs'][0]['screen']['standardized_rms'])


def test_named_estimator_receives_fitted_candidate_order():
    from sklearn.linear_model import LinearRegression
    from open.interactions import screen_pair_interactions
    X=reference()[['a','b']]
    model=LinearRegression().fit(X,2*X.a+3*X.b)
    result=screen_pair_interactions(model,X[['b','a']],('a','b'),
                                   outcome_sd=1,config=settings())
    assert result['pairs'][0]['screen']['standardized_rms'] < 1e-12


@pytest.mark.parametrize('kwargs',[{'max_pairs':True},{'screen_bins':1},
    {'prediction_rows':-1},{'screen_rows':0},{'min_cell':0},{'seed':-1}])
def test_invalid_settings_rejected(kwargs):
    with pytest.raises(ValueError):
        settings(**kwargs)


def test_fold_aggregation_keeps_unsupported_in_planned_denominator():
    from open.interactions import aggregate_pair_screens
    X=reference()[['a','b']]
    complete=screen(Model(lambda X:X.a*X.b),X=X)
    X.b=X.a
    unavailable=screen(Model(lambda X:X.a*X.b),X=X)
    result=aggregate_pair_screens([complete,unavailable])
    pair=result['pairs'][0]
    assert pair['planned_folds']==2 and pair['supported_screen_folds']==1
    assert pair['mean_screen_rms'] is None
    assert pair['status']=='incomplete_support'
