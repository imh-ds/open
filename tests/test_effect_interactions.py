import json
import numpy as np
import pandas as pd

from open.config import AnalysisConfig,OpenConfig,PerformanceCVConfig,TuningCVConfig
from open.splits import make_split_plan


def test_effect_workflow_is_opt_in_and_reuses_explanatory_fits(tmp_path):
    from open.effects import compare_target_engines,save_engine_comparison
    from open.interactions import PairScreenConfig
    X=pd.DataFrame(np.random.default_rng(19).normal(size=(160,2)),columns=['a','b'])
    X['y']=X.a*X.b
    config=OpenConfig(AnalysisConfig('y',17),performance_cv=PerformanceCVConfig(2,1),
                      tuning_cv=TuningCVConfig(2))
    plan=make_split_plan(X,config)
    baseline=compare_target_engines(X,'y',('a','b'),config,plan,engines=('gbm',),bins=2)
    examined=compare_target_engines(X,'y',('a','b'),config,plan,engines=('gbm',),bins=2,
        interaction_screen=PairScreenConfig(detail_bins=2))
    one,two=baseline['engines'][0],examined['engines'][0]
    assert 'interactions' not in one
    assert one['full_mse']==two['full_mse']
    for name in ('scored_outer_fit_count','explanatory_refit_count','tuning_model_fit_count'):
        assert one[name]==two[name]
    assert two['interactions']['planned_folds']==2
    assert two['interactions']['additional_model_fits']==0
    save_engine_comparison(examined,tmp_path)
    assert (tmp_path/'interaction_pairs.csv').exists()
    assert (tmp_path/'interaction_cells.csv').exists()
    assert (tmp_path/'INTERACTIONS.md').exists()
    assert json.loads((tmp_path/'comparison.json').read_text())==examined
    save_engine_comparison(baseline,tmp_path)
    assert not (tmp_path/'interaction_pairs.csv').exists()
    assert not (tmp_path/'interaction_cells.csv').exists()
    assert not (tmp_path/'INTERACTIONS.md').exists()
