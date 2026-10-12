import numpy as np
import pandas as pd
import pytest

from open.interactions import PairScreenConfig,screen_pair_interactions


class Product:
    def predict(self,X):return (X.a*X.b).to_numpy()


def config():
    assert 'adaptive_details' in PairScreenConfig.__dataclass_fields__, 'adaptive detail mode missing'
    return PairScreenConfig(adaptive_details=True,max_details=1)


def test_small_reference_coarsens_to_supported_grid():
    X=pd.DataFrame({'a':np.repeat([-2.,-1.,1.,2.],8),'b':np.tile([-2.,-1.,1.,2.],8)})
    fixed=screen_pair_interactions(Product(),X,tuple(X),outcome_sd=1)
    assert fixed['pairs'][0]['detail']['status']=='unsupported_cells'
    adaptive=screen_pair_interactions(Product(),X,tuple(X),outcome_sd=1,config=config())
    detail=adaptive['pairs'][0]['detail']
    assert detail['status']=='available' and detail['realized_bins']==[2,2]
    assert detail['coverage']=='complete'
    assert adaptive['pairs'][0]['screen']==fixed['pairs'][0]['screen']


def test_partial_grid_masks_unobserved_cells_and_keeps_counts():
    cfg=config()
    from open.interactions import _adaptive_detail
    X=pd.DataFrame({'a':np.linspace(-2,2,80),'b':np.linspace(-2,2,80)})
    d=_adaptive_detail(Product(),X,('a','b'),cfg,1.,100000)
    assert d['status']=='available' and d['coverage']=='partial'
    assert d['realized_bins']==[2,2]
    supported=[c for c in d['cells'] if c['supported']]
    unsupported=[c for c in d['cells'] if not c['supported']]
    assert len(supported)==2 and len(unsupported)==2
    assert all(c['count']==0 and c['standardized_rms'] is None for c in unsupported)
    assert d['supported_reference_rows']==80 and d['prediction_rows']==320


def test_adaptive_budget_and_config_validation():
    cfg=config()
    with pytest.raises(ValueError):PairScreenConfig(adaptive_details=1)
    from open.interactions import _adaptive_detail
    X=pd.DataFrame({'a':np.linspace(-2,2,80),'b':np.linspace(-2,2,80)})
    d=_adaptive_detail(Product(),X,('a','b'),cfg,1.,10)
    assert d['status']=='prediction_budget' and d['prediction_rows']==0


def test_partial_report_masks_unsupported_cells(tmp_path):
    from open.interactions import aggregate_pair_screens
    from open.interaction_visualization import save_interaction_report
    X=pd.DataFrame({'a':np.repeat([-2.,-1.,1.,2.],8),'b':np.tile([-2.,-1.,1.,2.],8)})
    assert 'adaptive_details' in PairScreenConfig.__dataclass_fields__
    cfg=PairScreenConfig(adaptive_details=True,max_details=1,detail_rows=10)
    screen=screen_pair_interactions(Product(),X,tuple(X),outcome_sd=1,config=cfg)
    assert screen['pairs'][0]['detail']['coverage']=='partial'
    summary=aggregate_pair_screens([screen]);summary['fold_screens']=[dict(screen,repeat=0,fold=0)]
    save_interaction_report(dict(target='y',engines=[dict(engine='fixture',interactions=summary)]),
        tmp_path,plots=True)
    assert 'Unsupported cells are masked' in (tmp_path/'INTERACTIONS.md').read_text()
