"""Independent saved-vector arithmetic audit. No project model/decision imports or fits.

Requires NumPy, pandas, SciPy and scikit-learn (KFold indices only).
Run from any directory; input defaults to this preserved result directory.
"""
from pathlib import Path
import argparse,hashlib,itertools,json,zipfile
import numpy as np
import pandas as pd
from scipy.stats import beta
from sklearn.model_selection import KFold

GRID=np.array([-1.2815516,-.5244005,0.,.5244005,1.2815516])
IDENTITY='5a383f67f965c36ec9ffd07175e8153940d6b8844d5e021e25620c87862f5f4a'
SELECTION='0686c0d6a8fb92e6b034a0a74e7b7e02f3331583e5c13b438f2fb41d7616f79f'
THRESHOLD='755a6fe65f908c8b3f61a6ca4fa2d5aa5622a912519b53238b470773dbecbf17'

def fp(v):return hashlib.sha256(json.dumps(v,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()
def sha(data):return hashlib.sha256(data).hexdigest()
def close(a,b):np.testing.assert_allclose(a,b,rtol=1e-10,atol=1e-12)
def optional(value,saved):
    if value is None:assert pd.isna(saved)
    else:close(value,saved)

def verify_summary_structure(tasks,datasets,strata,conditions):
    expected={c['id']:c for c in tasks};keys={(c['cell'],c['benchmark']) for c in tasks}
    assert len(expected)==len(tasks) and len(keys)==24
    assert len(datasets)==len(expected) and len({r['id'] for r in datasets})==len(datasets)
    assert {r['id'] for r in datasets}==set(expected)
    for r in datasets:assert (r['cell'],r['benchmark'])==(expected[r['id']]['cell'],expected[r['id']]['benchmark'])
    maps=[]
    for records in (strata,conditions):
        mapping={(r['cell'],r['benchmark']):r for r in records}
        assert len(records)==len(mapping)==24 and set(mapping)==keys
        maps.append(mapping)
    return {r['id']:r for r in datasets},maps[0],maps[1]

def generated(c,n,seed):
    rng=np.random.default_rng(seed);x=rng.normal(size=(n,c['p']));rho=c['rho'];coefficient=c['b']
    x[:,1]=rho*x[:,0]+np.sqrt(1-rho*rho)*x[:,1]
    a,b=x[:,0],x[:,1];product=coefficient*(a*b-rho)
    base=a+.6*b if c['scenario']=='mixed' else np.zeros(n)
    variance=(1.36+1.2*rho if c['scenario']=='mixed' else 0.)+coefficient**2*(1+rho*rho)
    truth=base+product;y=truth+rng.normal(scale=np.sqrt(variance*2/3),size=n)
    additive=base+coefficient*rho/(1+rho*rho)*((a*a-1)+(b*b-1))
    return x,y,additive,truth

def verify_trace(t,params,joint,train_n,seed):
    candidates=([dict(curves__joint_weight=w,curves__n_knots=k,ridge__alpha=a)
        for w,k,a in itertools.product((1.,10.,100.),(3,5),(.1,1.,10.,100.))] if joint else
        [dict(curves__n_knots=k,ridge__alpha=a) for k,a in itertools.product((3,5),(.1,1.,10.,100.))])
    assert t['candidate_parameters']==candidates
    loss=np.asarray(t['fold_losses']);assert loss.shape==(len(candidates),3) and np.isfinite(loss).all() and (loss>=0).all()
    winner=int(np.argmin(loss.mean(axis=1)));assert t['selected_index']==winner and params==candidates[winner]
    assert t['inner_training_n']==[len(a) for a,b in KFold(3,shuffle=True,random_state=seed).split(range(train_n))]

def verify_curve(curve,xtrain,ytrain,pair):
    assert curve['grid']==GRID.tolist()
    prediction=np.asarray(curve['predicted']);assert prediction.shape==(5,5) and np.isfinite(prediction).all()
    sd=float(np.std(ytrain));assert sd>0;close(sd,curve['training_sd'])
    anchored=prediction-prediction[:,[2]]-prediction[[2],:]+prediction[2,2]
    close(anchored/sd,curve['contrast'])
    a,b=np.meshgrid(GRID,GRID,indexing='ij');x=xtrain[:,list(pair)]
    counts=((np.abs(x[:,0,None,None]-a)<=.5)&(np.abs(x[:,1,None,None]-b)<=.5)).sum(axis=0)
    mask=(counts>=5)&(counts[:,[2]]>=5)&(counts[[2],:]>=5)&(counts[2,2]>=5)
    np.testing.assert_array_equal(counts,curve['support_counts']);np.testing.assert_array_equal(mask,curve['support'])

def pattern(curves,truth_b):
    a=np.asarray([c['contrast'] for c in curves]);support=np.asarray([c['support'] for c in curves])
    assert a.shape==(3,5,5) and support.shape==(3,5,5) and support.dtype==bool and np.isfinite(a).all()
    mask=support.all(axis=0);mask[2,:]=False;mask[:,2]=False;count=int(mask.sum())
    result=dict(common_supported=count,consistent=False,min_cosine=None,truth_cosine=None)
    if count<4:return dict(result,status='unresolved_support')
    vectors=a[:,mask];norms=np.linalg.norm(vectors,axis=1)
    if (norms<1e-8).any():return dict(result,status='unresolved_magnitude')
    similarity=[float(np.clip(vectors[i]@vectors[j]/(norms[i]*norms[j]),-1,1)) for i,j in ((0,1),(0,2),(1,2))]
    minimum=min(similarity);truth_cosine=None
    if truth_b is not None:
        truth=(truth_b*GRID[:,None]*GRID[None,:])[mask];norm=float(np.linalg.norm(truth))
        if norm>=1e-8:truth_cosine=min(float(np.clip(v@truth/(s*norm),-1,1)) for v,s in zip(vectors,norms))
    return dict(result,status='assessed',min_cosine=minimum,consistent=minimum>=.8,truth_cosine=truth_cosine)

def verify_case(c,v,raw,environment,context,saved,threshold):
    identity=c['id'];assert v['task']==c and v['context']==context and v['status']=='complete'
    assert json.loads(raw[identity+'/receipt.json'])==dict(task=c,context=context,environment=environment)
    verified=json.loads(raw[identity+'/verified.json'])
    names={'receipt.json','result.json','timing.json','attempts.json'}|{f'comparison-{i}.json' for i in range(4)}
    assert verified['context']==context and verified['saved_setting_refits']==20 and set(verified['digests'])==names
    for name,digest in verified['digests'].items():assert sha(raw[identity+'/'+name])==digest
    ledger=json.loads(raw[identity+'/attempts.json']);assert ledger['context']==context and ledger['attempts']
    assert all(t['status']=='complete' and not t['unknown_fit_cost'] and not t['unknown_verification_cost'] for t in ledger['attempts'])
    training=generated(c,c['n'],c['seed']);independent=generated(c,1000,c['test_seed'])
    full=[np.concatenate([a,b]) for a,b in zip(training,independent)]
    splits=[(a.tolist(),b.tolist()) for a,b in KFold(3,shuffle=True,random_state=c['split_seed']).split(range(c['n']))]
    splits.append((list(range(c['n'])),list(range(c['n'],c['n']+1000))))
    assert len(v['comparisons'])==len(v['population'])==4
    numerator=np.zeros((2,2));denominator=0.;positive=[0,0];delta=0.;training_sizes=set();curve_checks=0
    for j,(f,(train,test)) in enumerate(zip(v['comparisons'],splits)):
        assert f['train_indices']==train and f['assessment_indices']==test and not set(train)&set(test)
        assert len(f['pairs'])==2 and [r['pair'] for r in f['pairs']]==[[0,1],[0,2]]
        checkpoint=json.loads(raw[identity+f'/comparison-{j}.json'])
        assert checkpoint['context']==context and checkpoint['result']==f and checkpoint['population']==v['population'][j]
        x,y,ideal_additive,ideal_joint=training if j<3 else full
        observed=np.asarray(f['observed']);np.testing.assert_array_equal(observed,y[test])
        reference=float(np.square(observed-y[train].mean()).mean());baseline=float(np.square(observed-np.asarray(f['baseline_predicted'])).mean())
        close([reference,baseline],[f['reference_mse'],f['baseline_mse']]);assert reference>0 and f['fits']==173 and f['seed']==c['split_seed']
        verify_trace(f['baseline_selection'],f['baseline_parameters'],False,len(train),c['split_seed'])
        if j<3:training_sizes.update(f['baseline_selection']['inner_training_n'])
        for pos,pair in enumerate(f['pairs']):
            verify_trace(pair['selection'],pair['parameters'],True,len(train),c['split_seed'])
            assert pair['matched_parameters']=={k:pair['parameters'][k] for k in ('curves__n_knots','ridge__alpha')}
            joint=float(np.square(observed-np.asarray(pair['predicted'])).mean());matched=float(np.square(observed-np.asarray(pair['matched_predicted'])).mean())
            gains=np.array([(baseline-joint)/reference,(matched-joint)/reference,(baseline-matched)/reference])
            recorded=np.array([pair['overall_gain'],pair['matched_gain'],pair['marginal_gain']]);close(gains,recorded);delta=max(delta,float(abs(gains-recorded).max()))
            close([joint,matched],[pair['joint_mse'],pair['matched_additive_mse']])
            verify_curve(pair['curve'],x[train],y[train],pair['pair']);curve_checks+=1
            if j<3:numerator[pos]+=(np.array([baseline,matched])-joint)*len(observed);positive[pos]+=int(matched>joint)
        population=v['population'][j];close(population['additive_predicted'],ideal_additive[test]);close(population['joint_predicted'],ideal_joint[test])
        am=float(np.square(observed-ideal_additive[test]).mean());jm=float(np.square(observed-ideal_joint[test]).mean())
        close([am,jm,(am-jm)/reference],[population['additive_mse'],population['joint_mse'],population['gap']])
        if j<3:denominator+=len(observed)*reference
    metrics=[]
    for pos,name in enumerate(('x1:x2','x1:x3')):
        overall,matched=numerator[pos]/denominator;eligible=bool(overall>0 and positive[pos]>=2)
        score=max(0.,matched) if eligible else 0.;above=score>threshold;record=saved.loc[(identity,name)]
        close([overall,matched,score,threshold],[record.cv_overall,record.cv_matched,record.score,record.fixed_threshold])
        assert eligible==record.eligible and positive[pos]==record.positive_folds and above==record.above_threshold
        curves=[f['pairs'][pos]['curve'] for f in v['comparisons'][:3]];p=pattern(curves,c['b'] if pos==0 else None)
        assert p['status']==record.pattern_status and p['consistent']==record.pattern_consistent and p['common_supported']==record.common_supported
        optional(p['min_cosine'],record.pattern_cosine);optional(p['truth_cosine'],record.truth_cosine)
        metrics.append((above,p))
    displayed=bool(metrics[0][0] and metrics[0][1]['status']=='assessed' and metrics[0][1]['consistent'])
    true_cosine=metrics[0][1]['truth_cosine']
    row=dict(id=identity,cell=c['cell'],benchmark=c['benchmark'],control=bool(metrics[1][0]),detected=bool(metrics[0][0]),displayed=displayed,
        shape_correct=bool(displayed and true_cosine is not None and true_cosine>=.8))
    return row,dict(pair_risks=8,population_risks=4,curves=curve_checks,max_gain_difference=delta,inner_training_sizes=sorted(training_sizes),
        fits=sum(t['fits'] for t in ledger['attempts']),refits=sum(t['verification_refits'] for t in ledger['attempts']))

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--input',type=Path,default=Path(__file__).resolve().parent)
    parser.add_argument('--output',type=Path);args=parser.parse_args();folder=args.input;output=args.output or folder/'independent-vectors';output.mkdir(parents=True,exist_ok=True)
    root=Path(__file__).resolve().parents[3]
    manifest=json.loads((root/'examples/study_manifests/prospective-interaction.json').read_text());assert fp(manifest)==IDENTITY
    selection=json.loads((root/'examples/study_manifests/prospective-interaction-selection.json').read_text());assert fp(selection)==SELECTION
    null=json.loads((root/'examples/study_manifests/prospective-interaction-null-audit.json').read_text());assert fp(null)==selection['stage_a_audit'] and all(r['pass'] for r in null['strata'])
    threshold_data=(root/'examples/study_manifests/pair-confidence-thresholds.json').read_bytes();assert sha(threshold_data.replace(b'\r\n',b'\n'))==THRESHOLD==manifest['thresholds_sha256']
    thresholds=json.loads(threshold_data)['thresholds'];audit=json.loads((folder/'actions-audit.json').read_text())
    assert audit['manifest']==IDENTITY and audit['selection']==SELECTION and audit['complete']==audit['planned']==10800 and not audit['missing_ids'] and not audit['audit_errors'] and not audit['skipped_ids']
    selected=set(selection['selected_ids']);tasks=[c for c in manifest['tasks'] if c['id'] in selected];assert len(tasks)==10800
    saved=pd.read_csv(folder/'actions-case-metrics.csv').set_index(['id','pair'],verify_integrity=True);assert len(saved)==21600
    anchors={r['path'].replace(chr(92),'/').split('/')[-2].rsplit('-',1)[-1]:r for r in audit['merge']['archives']}
    assert set(anchors)=={f'{i:03d}' for i in range(135)}
    rows=[];seen=set();files=pair_risks=population_risks=curves=cost=refits=0;delta=0.;sizes={};shards=set()
    for path in sorted((folder/'raw').glob('shard-*.zip')):
        shard=path.stem.split('-')[-1];assert shard in anchors and shard not in shards;shards.add(shard)
        receipt=json.loads(path.with_suffix('.json').read_text());assert path.stat().st_size==receipt['bytes']==anchors[shard]['bytes'] and sha(path.read_bytes())==receipt['sha256']==anchors[shard]['sha256']
        with zipfile.ZipFile(path) as archive:
            names={r['path'] for r in receipt['inventory']};assert len(names)==len(archive.namelist())==receipt['files'] and set(archive.namelist())==names
            raw={r['path']:archive.read(r['path']) for r in receipt['inventory']}
        for r in receipt['inventory']:assert len(raw[r['path']])==r['bytes'] and sha(raw[r['path']])==r['sha256'];files+=1
        selected_tasks=tasks[int(shard)*80:(int(shard)+1)*80]
        assert json.loads(raw[f'shard-signal-{shard}.json'])==dict(id=shard,stage='signal',tasks=selected_tasks,manifest=IDENTITY,selection=SELECTION)
        for c in selected_tasks:
            assert c['id'] not in seen;seen.add(c['id']);context=fp(dict(task=c,manifest=IDENTITY,selection=SELECTION))
            row,m=verify_case(c,json.loads(raw[c['id']+'/result.json']),raw,manifest['environment'],context,saved,thresholds[c['cell']]);rows.append(row)
            pair_risks+=m['pair_risks'];population_risks+=m['population_risks'];curves+=m['curves'];cost+=m['fits'];refits+=m['refits'];delta=max(delta,m['max_gain_difference'])
            sizes.setdefault(c['cell'],set()).update(m['inner_training_sizes'])
        print('Independently checked shard',shard,flush=True)
    assert seen==selected and len(shards)==135
    assert cost==audit['known_attempt_cost']['fits'] and refits==audit['known_attempt_cost']['verification_refits']
    saved_rows,stratum_map,condition_map=verify_summary_structure(tasks,
        json.loads((folder/'actions-dataset-metrics.json').read_text()),audit['strata'],audit['conditions'])
    assert all(row==saved_rows[row['id']] for row in rows)
    bounds=[];conditions=[]
    for key,r in stratum_map.items():
        c=condition_map[key]
        group=[v for v in rows if (v['cell'],v['benchmark'])==(r['cell'],r['benchmark'])];assert len(group)==450
        passes={}
        for name,key,alpha,target,upper in [('control','control',.02/36,.05,True),('detection','detected',.02/24,.8,False),('shape','shape_correct',.01/24,.9,False)]:
            values=group if name!='shape' else [v for v in group if v['displayed']];n=len(values);k=sum(v[key] for v in values)
            if not n:bound=None
            elif upper:bound=1. if k==n else float(beta.ppf(1-alpha,k+1,n-k))
            else:bound=0. if not k else float(beta.ppf(alpha,k,n-k+1))
            passed=bound is not None and (bound<=target if upper else bound>=target);passes[name]=passed
            assert (k,n,passed)==(r[name]['k'],r[name]['n'],r[name]['pass']);optional(bound,r[name]['bound'])
            bounds.append(dict(cell=r['cell'],benchmark=r['benchmark'],endpoint=name,k=k,n=n,bound=bound,passed=passed))
        predictive=passes['control'] and passes['detection'];pattern_pass=predictive and passes['shape']
        assert predictive==c['predictive_support'] and pattern_pass==c['curve_pattern_support']
        conditions.append(dict(cell=r['cell'],benchmark=r['benchmark'],predictive=predictive,pattern=pattern_pass))
    assert len(bounds)==72 and len(conditions)==24
    assert audit['global_predictive_support']==all(c['predictive'] for c in conditions) and audit['global_curve_pattern_support']==all(c['pattern'] for c in conditions)
    check=dict(datasets=len(rows),shards=len(shards),raw_files=files,pair_risk_checks=pair_risks,population_risk_checks=population_risks,curve_checks=curves,
        max_gain_difference=delta,bounds_checked=len(bounds),qualified=sum(c['pattern'] for c in conditions),inner_training_sizes={k:sorted(v) for k,v in sizes.items()},
        known_fitting_calls=cost,known_verification_refits=refits,model_fits_performed=0,manifest=IDENTITY,selection=SELECTION,
        scope='Independent reconstruction of outcomes, indices, saved tuning winners, prediction risks, diagnostic population vectors, curve contrasts/support/cosines, decisions and bounds. Saved model predictions are inputs; this does not independently retrain models or establish causal or effect-size accuracy.')
    pd.DataFrame(rows).to_csv(output/'dataset-metrics.csv',index=False);pd.DataFrame(bounds).to_csv(output/'bounds.csv',index=False)
    (output/'check.json').write_text(json.dumps(check,indent=2)+'\n');print(json.dumps(check),flush=True)
if __name__=='__main__':main()
