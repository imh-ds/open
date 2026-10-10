"""Real engine integration, fixed ablations, reproducibility and dependency gates."""
from dataclasses import replace
import builtins
import importlib.util

import numpy as np
import pandas as pd
import pytest

from open.config import AnalysisConfig, OpenConfig, LearnerConfig, PerformanceCVConfig, TuningCVConfig
from open.splits import make_split_plan
from open.risk import evaluate_risk


def fixture(engine):
    rng = np.random.default_rng(19)
    x = rng.uniform(-2, 2, 48)
    data = pd.DataFrame({'x': x, 'z': rng.normal(size=48), 'y': 3*x + rng.normal(0,.1,48)})
    config = OpenConfig(AnalysisConfig('y', 17, max_depth=2),
        learner=LearnerConfig(library=(engine,)),
        performance_cv=PerformanceCVConfig(2,1), tuning_cv=TuningCVConfig(2))
    return data, config, make_split_plan(data, config)


@pytest.mark.parametrize('engine', ['elastic_net','random_forest','gbm','lightgbm'])
def test_engine_real_risk_clone_and_fixed_ablation(engine):
    if engine == 'lightgbm' and importlib.util.find_spec('lightgbm') is None:
        pytest.skip('optional LightGBM not installed')
    from open.learners import make_predictor
    data, config, plan = fixture(engine)
    predictor = make_predictor(config)
    common = dict(target='y',candidate_set=('x','z'),split_plan=plan,predictor=predictor,tuning_folds=2)
    full = evaluate_risk(data,predictors=('x','z'),**common)
    params = {(f.repeat_index,f.fold_index):f.fitted_parameters for f in full.folds}
    ablated = evaluate_risk(data,predictors=('z',),fitted_parameters=params,**common)
    replay = evaluate_risk(data,predictors=('x','z'),fitted_parameters=params,**common)
    assert full.learner == engine
    assert full.aggregate_risk == replay.aggregate_risk
    assert full.aggregate_risk < ablated.aggregate_risk
    assert full.tuning_model_fit_count == predictor.tuning_grid_size * 4
    assert ablated.tuning_model_fit_count == 0
    assert [f.fitted_parameters for f in full.folds] == [f.fitted_parameters for f in ablated.folds]
    assert full.split_plan_id == ablated.split_plan_id
    assert all([r.row_position for r in a.observations] == [r.row_position for r in b.observations]
               for a,b in zip(full.folds,ablated.folds))
    assert predictor.clone().engine_name == engine


def test_lightgbm_missing_has_explicit_install_instruction(monkeypatch):
    from open.learners import make_predictor
    data, config, plan = fixture('lightgbm')
    original = builtins.__import__
    def guarded(name,*args,**kwargs):
        if name == 'lightgbm': raise ImportError('not installed')
        return original(name,*args,**kwargs)
    monkeypatch.setattr(builtins,'__import__',guarded)
    with pytest.raises(ImportError,match='lightgbm.*install|install.*lightgbm'):
        make_predictor(config).fit(data[['x','z']],data.y)


@pytest.mark.parametrize('engine', ['random_forest','gbm','lightgbm'])
def test_engine_runs_recursive_layer_consensus(engine):
    if engine == 'lightgbm' and importlib.util.find_spec('lightgbm') is None:
        pytest.skip('optional LightGBM not installed')
    from open.stability import fit_stable_open
    from open.config import StabilityConfig
    data, config, _ = fixture(engine)
    config = replace(config,stability=StabilityConfig(scope='layer_consensus',n_resamples=2))
    result = fit_stable_open(data,'y',tuple(data.columns),config,n_resamples=2)
    assert result.status == 'COMPLETE'
    assert result.assigned_depth['x'] == 1


@pytest.mark.parametrize('library',[(),('unknown',),('elastic_net','gbm'),('gbm', 'gbm')])
def test_exactly_one_supported_engine(library):
    with pytest.raises(ValueError,match='learner.library'):
        LearnerConfig(library=library)


def test_lightgbm_provenance_uses_its_own_distribution(monkeypatch):
    import open.stability as stability
    monkeypatch.setattr(stability.distribution_metadata,'version',lambda name:'version-of-'+name)
    _,config,_=fixture('lightgbm')
    _,dependencies,learners,_=stability._capture_versions(config)
    assert learners['lightgbm']=='version-of-lightgbm'
    assert dependencies['lightgbm']=='version-of-lightgbm'
