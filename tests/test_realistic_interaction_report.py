import sys
from pathlib import Path
import pytest
import numpy as np
sys.path.insert(0,str(Path(__file__).parents[1]/'examples'))


def modules():
    assert (Path(__file__).parents[1]/'examples/realistic_interaction_report.py').exists(),'realistic report missing'
    import realistic_interaction_report as r
    import realistic_interaction_verify as v
    return r,v


def calibration_rows(r):
    return [dict(id=c['id'],cell=c['cell'],benchmark=c['benchmark'],pairs=[dict(eligible=True,score=.01),dict(eligible=True,score=.02)],
                 result_sha256='a'*64) for c in r.study.d.stage_tasks('calibration')]


def test_threshold_full_coverage_both_pairs_and_independent_check():
    r,v=modules();rows=calibration_rows(r);rows[12]['pairs'][1]['score']=.07
    cut=r.learn_thresholds(rows);assert cut[rows[12]['cell']]==.07 and len(cut)==18
    audit=dict(stage='calibration',manifest='frozen',planned=3600,complete=3600,missing=0,audit_errors=[],rows=rows,
        standalone=dict(status='complete',checked=3600,max_score_difference=0.))
    artifact=v.verify_calibration(audit,'frozen');assert artifact['thresholds']==cut
    with pytest.raises(ValueError):r.learn_thresholds(rows[:-1])
    with pytest.raises(ValueError):r.learn_thresholds(rows[:-1]+[rows[0]])
    with pytest.raises(ValueError):v.verify_calibration(dict(audit,complete=3599),'frozen')
    with pytest.raises(ValueError):v.verify_calibration(dict(audit,standalone={}), 'frozen')


def test_descriptive_interval_and_unknown_denominator():
    r,v=modules();a=r.interval(0,80)
    assert a['low']==0 and abs(a['high']-(1-.025**(1/80)))<1e-12
    assert r.interval(0,0)['rate'] is None
    tasks=r.study.d.stage_tasks('development')[:80]
    rows=[dict(id=c['id'],detected=False,control=False,any_selected=False,displayed=False,shape_correct=None) for c in tasks[:-1]]
    a=r.summarize(rows,tasks)
    assert a['planned']==80 and a['missing']==1 and a['status']=='incomplete'
    assert a['pattern']['rate'] is None and a['display']['n']==79
    assert 'qualification' not in a


def test_scores_recomputed_from_vectors_not_saved_gains():
    r,v=modules()
    comparisons=[dict(observed=[1.,-1.],baseline_predicted=[0.,0.],reference_mse=1.,
        pairs=[dict(predicted=[.5,-.5],matched_predicted=[0.,0.]),dict(predicted=[0.,0.],matched_predicted=[0.,0.])]) for _ in range(3)]
    scores=v.vector_scores(comparisons)
    assert scores[0]['score']==.75 and scores[1]['score']==0.
    assert scores[0]['eligible'] and not scores[1]['eligible']


def test_shape_magnitude_and_truth_blind_display():
    r,v=modules();a,b=np.meshgrid([-1.,-.5,0.,.5,1.],[-1.,-.5,0.,.5,1.],indexing='ij');curve=a*b
    curves=[dict(contrast=curve.tolist(),support=np.ones((5,5),bool).tolist(),training_sd=1.) for _ in range(3)]
    truth=[curve.tolist() for _ in range(3)]
    good=r.curve_metrics(curves,truth,True);bad=r.curve_metrics(curves,[(-curve).tolist()]*3,True)
    assert good['displayed'] and bad['displayed']
    assert good['shape_correct'] and not bad['shape_correct']
    assert good['standardized_rmse']==0 and good['relative_magnitude_error']==0
    assert not r.curve_metrics(curves,truth,False)['displayed']
