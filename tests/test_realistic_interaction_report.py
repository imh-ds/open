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


def test_unresolved_display_kept_in_pattern_denominator():
    r,v=modules();tasks=r.study.d.stage_tasks('development')[:80]
    rows=[dict(id=c['id'],detected=False,control=False,any_selected=False,displayed=False,shape_correct=None) for c in tasks]
    rows[0].update(displayed=True,shape_correct=True);rows[1].update(displayed=True)
    result=r.summarize(rows,tasks)
    assert result['pattern']['n']==2 and result['pattern']['k']==1
    assert result['pattern']['rate'] is None and result['pattern']['low'] is None and result['pattern']['high'] is None
    assert result['pattern']['unknown']==1
    assert result['pattern_assessed']['n']==1 and result['pattern_assessed']['rate']==1.


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


def test_streamed_raw_audit_preserves_missing_and_rejects_duplicate_shard(tmp_path,monkeypatch):
    r,v=modules();m=r.study
    monkeypatch.setattr(m,'MANIFEST',tmp_path/'manifest.json')
    monkeypatch.setattr(m,'sources',lambda:m.old.sources());m.freeze(False)
    c=m.d.tasks()[0];raw=tmp_path/'raw';m.run_task(c,raw)
    m.io.write(raw/'shard-calibration-000.json',dict(id='000',stage='calibration',tasks=m.d.shard_tasks('calibration','000'),manifest=m.identity(),thresholds=None))
    pack=tmp_path/'pack';r.archive.pack(raw,pack)
    audit=r.aggregate('calibration',[pack],tmp_path/'combined')
    assert audit['complete']==1 and audit['missing']==3599 and audit['audit_errors']==[]
    assert len(audit['unknown_attempt_cost_ids'])==3599
    assert audit['known_attempt_cost']['fits']==692 and audit['standalone']['checked']==1
    assert not (tmp_path/'combined'/'proposed-thresholds.json').exists()
    # Add a distinct inventory-valid archive carrying the same logical shard.
    (raw/'extra.txt').write_text('retained extra evidence')
    second=tmp_path/'second';r.archive.pack(raw,second)
    with pytest.raises(ValueError,match='duplicate shard'):r.aggregate('calibration',[pack,second],tmp_path/'duplicate')


def test_no_artifacts_means_unknown_cost_not_zero_evidence(tmp_path,monkeypatch):
    r,v=modules();monkeypatch.setattr(r.study,'freeze',lambda:None)
    monkeypatch.setattr(r.study,'identity',lambda:'test-manifest')
    result=r.aggregate('calibration',[tmp_path/'missing'],tmp_path/'report')
    assert result['missing']==3600 and len(result['unknown_attempt_cost_ids'])==3600


def test_display_and_pattern_series_share_benchmark_color(tmp_path,monkeypatch):
    r,v=modules();import matplotlib.figure
    figures=[];monkeypatch.setattr(matplotlib.figure.Figure,'savefig',lambda fig,*a,**k:figures.append(fig))
    groups=[dict(n=n,reliability=1.,gamma=.1,benchmark='product-b0.2',status='complete',
                 detection=dict(rate=.5),display=dict(rate=.5),pattern=dict(rate=.8)) for n in (150,300,600)]
    r.plots(groups,tmp_path)
    lines=figures[1].axes[0].lines
    assert len(lines)==2 and lines[0].get_color()==lines[1].get_color()
