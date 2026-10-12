import importlib.util,json,sys
from pathlib import Path
import pytest


def module(tmp_path):
    root=Path(__file__).parents[1];p=root/'examples/pair_confidence_study.py'
    assert p.exists(),'confidence stage runner missing'
    sys.path.insert(0,str(root/'examples'))
    s=importlib.util.spec_from_file_location('pair_confidence_study',p)
    m=importlib.util.module_from_spec(s);s.loader.exec_module(m)
    m.MANIFESTS={s:tmp_path/f'{s}.json' for s in ('calibration','confirmation')}
    m.THRESHOLDS=tmp_path/'thresholds.json'
    m.freeze('calibration',False);return m


def test_confirmation_cannot_start_without_frozen_calibration(tmp_path):
    m=module(tmp_path)
    with pytest.raises((ValueError,FileNotFoundError)):m.freeze('confirmation',False)
    m.io.write(m.THRESHOLDS,dict(status='complete',thresholds={'a':0.}))
    with pytest.raises(ValueError):m.freeze('confirmation',False)
    assert len(m.shard_tasks('calibration','049'))==40
    with pytest.raises(ValueError):m.shard_tasks('calibration','050')


def test_partial_checkpoints_cost_and_no_threshold_learning(tmp_path):
    m=module(tmp_path);c=m.rules.tasks('calibration')[0];X,y,_=m.models.generate(c,c['n'],c['seed'])
    a,b=m.splits(c)[0];v=m.models.compare(X,y,a,b,c['split_seed'])
    out=tmp_path/'out';m.io.write(out/c['id']/'comparison-0.json',dict(context=m.context(c),result=v,fitting_seconds=1.,warnings=[]))
    audit=m.aggregate('calibration',out)
    assert audit['complete']==0 and audit['missing']==2000
    assert audit['partial_checkpoint_cost']['fits']==173 and audit['partial_checkpoint_cost']['comparisons']==1
    assert c['id'] in audit['unfinished_attempt_cost_unknown_ids']
    assert m.io.read(out/'learned-thresholds.json')['status']=='unavailable'


def test_foreign_failure_and_shard_identity_are_audit_errors(tmp_path):
    m=module(tmp_path);c=m.rules.tasks('calibration')[0];out=tmp_path/'out'
    m.io.write(out/c['id']/'error.json',dict(task=c,context='wrong source',error='failure'))
    m.io.write(out/'shard-999.json',dict(stage='calibration',id='999',tasks=[],manifest=m.identity('calibration')))
    audit=m.aggregate('calibration',out)
    assert len(audit['audit_errors'])==2 and not audit['thresholds_ready']


def test_native_resume_stamp_and_tampering(tmp_path):
    m=module(tmp_path);c=m.rules.tasks('calibration')[100];out=tmp_path/'out'
    v=m.run_task(c,out);audit=m.aggregate('calibration',out)
    assert audit['complete']==1 and audit['fits']==692 and audit['saved_setting_refits']==20
    assert m.run_task(c,out)==v
    m.io.write(out/c['id']/'error.json',dict(task=c,context=m.context(c),error='interrupted earlier attempt'))
    assert len(m.aggregate('calibration',out)['historical_failed_tasks'])==1
    p=out/c['id']/'result.json';z=m.io.read(p);z['comparisons'][0]['pairs'][0]['curve']['contrast'][0][0]+=.1
    m.io.write(p,z);m.io.write(out/c['id']/'verified.json',m.verification(c,out/c['id']))
    assert m.aggregate('calibration',out)['complete']==0


def test_threshold_provenance_and_strict_complete_identity_validation(tmp_path):
    m=module(tmp_path);planned=m.rules.tasks('calibration')
    rows=[dict(id=c['id'],cell=c['cell'],scenario=c['scenario'],score=0.,result_sha256='a'*64) for c in planned]
    rule=m.make_rule(rows);m.io.write(m.THRESHOLDS,rule)
    m.THRESHOLDS.write_bytes(m.THRESHOLDS.read_bytes().replace(b'\r\n',b'\n'))
    m.freeze('confirmation',False);m.freeze('confirmation')
    data=m.THRESHOLDS.read_bytes().replace(b'\r\n',b'\n')
    m.THRESHOLDS.write_bytes(data.replace(b'\n',b'\r\n'))
    m.freeze('confirmation')
    rule['thresholds'][planned[0]['cell']]=.1;m.io.write(m.THRESHOLDS,rule)
    with pytest.raises(ValueError):m.freeze('confirmation')


def test_error_and_missing_confirmation_cannot_qualify(tmp_path):
    m=module(tmp_path);planned=m.rules.tasks('calibration')
    rows=[dict(id=c['id'],cell=c['cell'],scenario=c['scenario'],score=0.,result_sha256='a'*64) for c in planned]
    m.io.write(m.THRESHOLDS,m.make_rule(rows));m.freeze('confirmation',False)
    audit=m.aggregate('confirmation',tmp_path/'out')
    assert not audit['null_qualified'] and audit['missing']==5200
    assert all(c['upper_bound']==1. for c in audit['null_cells'])


def test_complete_null_gate_is_separate_from_missing_alternatives_and_control_errors(tmp_path):
    m=module(tmp_path);planned=m.rules.tasks('confirmation')
    null=[c for c in planned if c['scenario'] in ('noise','additive')]
    rows=[dict(**c,pair=p,qualifies=False,pattern_consistent=False,truth_cosine=None)
          for c in null for p in ('x1:x2','x1:x3')]
    c=next(c for c in planned if c['scenario']=='mixed')
    rows.extend([dict(**c,pair=p,qualifies=True,pattern_consistent=True,truth_cosine=1.) for p in ('x1:x2','x1:x3')])
    complete={t['id'] for t in null}|{c['id']}
    summary=m.confirmation_summary(rows,[],complete)
    assert summary['null_qualified']
    signal=next(s for s in summary['signal_cells'] if (s['cell'],s['scenario'],s['b'])==(c['cell'],'mixed',c['b']))
    assert signal['control_detected']==1 and signal['unknown']==29 and not signal['provisional_good_detection']
    assert signal['control_lower']==1 and signal['control_upper']==30
    complete.remove(null[0]['id']);rows=[r for r in rows if r['id']!=null[0]['id']]
    assert not m.confirmation_summary(rows,[],complete)['null_qualified']
