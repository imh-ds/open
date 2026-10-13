"""Tampered saved evidence must fail without refitting a model."""
from pathlib import Path
import copy,importlib.util,itertools
import numpy as np
import pytest

PATH=Path(__file__).resolve().parents[1]/'examples/results/2026-10-13-prospective-interaction-signal/recompute_vectors.py'

def auditor():
    spec=importlib.util.spec_from_file_location('signal_auditor',PATH)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module);return module

def curve_record():
    grid=np.array([-1.2815516,-.5244005,0.,.5244005,1.2815516])
    x=np.repeat(np.array(list(itertools.product(grid,grid))),5,axis=0)
    y=np.linspace(-1,1,len(x));sd=float(np.std(y));surface=grid[:,None]*grid[None,:]*sd
    return x,y,dict(predicted=surface.tolist(),contrast=(surface/sd).tolist(),training_sd=sd,
        support=np.ones((5,5),bool).tolist(),support_counts=np.full((5,5),5).tolist(),grid=grid.tolist())

def test_correct_curve_and_pattern_are_reconstructed():
    m=auditor();x,y,c=curve_record();m.verify_curve(c,x,y,(0,1))
    pattern=m.pattern([c,c,c],1.)
    assert pattern['common_supported']==16 and pattern['consistent'] and pattern['truth_cosine']==pytest.approx(1.)

@pytest.mark.parametrize('field',['predicted','contrast','support','training_sd'])
def test_curve_tampering_is_rejected(field):
    m=auditor();x,y,c=curve_record()
    if field=='support':c[field][0][0]=False
    elif field=='training_sd':c[field]*=2
    else:c[field][0][0]+=1
    with pytest.raises((AssertionError,ValueError)):m.verify_curve(c,x,y,(0,1))

def test_opposite_shape_cannot_receive_pattern_support():
    m=auditor();_,_,c=curve_record();reverse=copy.deepcopy(c);reverse['contrast']=(-np.array(c['contrast'])).tolist()
    p=m.pattern([c,c,reverse],1.)
    assert not p['consistent'] and p['min_cosine']==pytest.approx(-1.) and p['truth_cosine']==pytest.approx(-1.)

def test_no_common_support_stays_unresolved():
    m=auditor();_,_,c=curve_record();c['support']=np.zeros((5,5),bool).tolist()
    p=m.pattern([c,c,c],1.)
    assert p['status']=='unresolved_support' and p['truth_cosine'] is None and not p['consistent']

def test_false_tuning_winner_or_training_size_is_rejected():
    m=auditor();params=[dict(curves__n_knots=k,ridge__alpha=a) for k,a in itertools.product((3,5),(.1,1.,10.,100.))]
    trace=dict(candidate_parameters=params,fold_losses=[[1.,1.,1.] for p in params],selected_index=0,inner_training_n=[133,133,134])
    m.verify_trace(trace,params[0],False,200,615)
    wrong=copy.deepcopy(trace);wrong['selected_index']=1
    with pytest.raises(AssertionError):m.verify_trace(wrong,params[1],False,200,615)
    wrong=copy.deepcopy(trace);wrong['inner_training_n']=[200,200,200]
    with pytest.raises(AssertionError):m.verify_trace(wrong,params[0],False,200,615)

def summaries():
    tasks=[dict(id=f'case-{i}',cell=f'cell-{i//4}',benchmark=f'benchmark-{i%4}') for i in range(24)]
    rows=copy.deepcopy(tasks);strata=[dict(cell=c['cell'],benchmark=c['benchmark']) for c in tasks]
    return tasks,rows,strata,copy.deepcopy(strata)

def test_duplicate_dataset_summary_is_rejected():
    m=auditor();tasks,rows,strata,conditions=summaries();rows.append(copy.deepcopy(rows[0]))
    with pytest.raises(AssertionError):m.verify_summary_structure(tasks,rows,strata,conditions)

@pytest.mark.parametrize('damage',['missing','duplicate','foreign'])
def test_incomplete_or_foreign_condition_grid_is_rejected(damage):
    m=auditor();tasks,rows,strata,conditions=summaries()
    if damage=='missing':conditions=conditions[:1];strata=strata[:1]
    elif damage=='duplicate':conditions[-1]=copy.deepcopy(conditions[0])
    else:conditions[-1]['cell']='foreign'
    with pytest.raises(AssertionError):m.verify_summary_structure(tasks,rows,strata,conditions)

def test_condition_matching_uses_identity_not_zip_position():
    m=auditor();tasks,rows,strata,conditions=summaries();conditions.reverse()
    dataset_map,stratum_map,condition_map=m.verify_summary_structure(tasks,rows,strata,conditions)
    assert len(dataset_map)==len(stratum_map)==len(condition_map)==24
    for key,r in stratum_map.items():assert (condition_map[key]['cell'],condition_map[key]['benchmark'])==key
