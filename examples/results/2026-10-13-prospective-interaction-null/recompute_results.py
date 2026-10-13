"""Independent no-fit audit of preserved null shards, candidate counts and CP gates.
Requires numpy/pandas/scipy/scikit-learn; imports no project decision/model modules.
"""
from pathlib import Path
import argparse,json,hashlib,zipfile,itertools
import numpy as np
import pandas as pd
from scipy.stats import beta
from sklearn.model_selection import KFold


def fp(v):return hashlib.sha256(json.dumps(v,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()
def sha(b):return hashlib.sha256(b).hexdigest()
def assertclose(a,b):np.testing.assert_allclose(a,b,rtol=1e-10,atol=1e-12)
def generate(c,n,seed):
 r=np.random.default_rng(seed);x=r.normal(size=(n,c['p']));x[:,1]=c['rho']*x[:,0]+np.sqrt(1-c['rho']**2)*x[:,1]
 truth=np.zeros(n) if c['scenario']=='noise' else x[:,0]+.6*(x[:,1]**2-1)
 return truth+r.normal(scale=1. if c['scenario']=='noise' else np.sqrt(1.72*2/3),size=n)
def trace(t,params,joint):
 candidates=([dict(curves__joint_weight=w,curves__n_knots=k,ridge__alpha=a) for w,k,a in itertools.product((1.,10.,100.),(3,5),(.1,1.,10.,100.))] if joint else
 [dict(curves__n_knots=k,ridge__alpha=a) for k,a in itertools.product((3,5),(.1,1.,10.,100.))])
 assert t['candidate_parameters']==candidates
 loss=np.asarray(t['fold_losses']);assert loss.shape==(len(candidates),3) and np.isfinite(loss).all() and (loss>=0).all()
 i=int(np.argmin(loss.mean(axis=1)));assert t['selected_index']==i and candidates[i]==params


def check_structure(c,v,verified):
 assert len(v['comparisons'])==4
 splits=[(a.tolist(),b.tolist()) for a,b in KFold(3,shuffle=True,random_state=c['split_seed']).split(range(c['n']))]
 splits.append((list(range(c['n'])),list(range(c['n'],c['n']+1000))))
 for f,(train,test) in zip(v['comparisons'],splits):
  assert f['train_indices']==train and f['assessment_indices']==test and not set(train)&set(test)
  assert len(f['pairs'])==2 and [r['pair'] for r in f['pairs']]==[[0,1],[0,2]]
 names={'receipt.json','result.json','timing.json','attempts.json'}|{f'comparison-{i}.json' for i in range(4)}
 assert set(verified['digests'])==names

def check_threshold(manifest,data):
 expected='755a6fe65f908c8b3f61a6ca4fa2d5aa5622a912519b53238b470773dbecbf17'
 assert manifest['thresholds_sha256']==expected and sha(data.replace(b'\r\n',b'\n'))==expected

def check_selection(selection,audit,manifest):
 cells=audit['passed_cells'];signals=[c for c in manifest['tasks'] if c['stage']=='signal']
 expected=dict(schema='open.prospective_selection.v1',manifest=fp(manifest),stage_a_audit=fp(audit),cells=cells,
   selected_ids=[c['id'] for c in signals if c['cell'] in cells],skipped_ids=[c['id'] for c in signals if c['cell'] not in cells],
   skip_reason='both null-class gates not passed',multiplicity=dict(false=36,detection=24,shape=24))
 assert selection==expected

def archive_anchors(audit):
 anchors={}
 for r in audit['merge']['archives']:
  shard=r['path'].replace(chr(92),'/').split('/')[-2].rsplit('-',1)[-1]
  assert shard not in anchors;anchors[shard]=r
 assert set(anchors)=={f'{i:03d}' for i in range(180)}
 return anchors


def main():
 parser=argparse.ArgumentParser();parser.add_argument('--input',type=Path,required=True);parser.add_argument('--output',type=Path,required=True);a=parser.parse_args();a.output.mkdir(parents=True,exist_ok=True)
 manifest=json.loads((a.input/'frozen-manifest.json').read_text());identity=fp(manifest)
 audit=json.loads((a.input/'actions-audit.json').read_text());thresholds=json.loads((a.input/'fixed-thresholds.json').read_text())['thresholds']
 check_threshold(manifest,(a.input/'fixed-thresholds.json').read_bytes())
 check_selection(json.loads((a.input/'actions-proposed-selection.json').read_text()),audit,manifest)
 anchors=archive_anchors(audit)
 expected={c['id']:c for c in manifest['tasks'] if c['stage']=='null'};assert len(expected)==14400 and audit['manifest']==identity
 assert identity=='5a383f67f965c36ec9ffd07175e8153940d6b8844d5e021e25620c87862f5f4a'
 rows=[];seen=set();files=pair_checks=population_checks=0;known_fits=known_refits=0;maximum_delta=0.;shard_ids=set()
 for path in sorted((a.input/'raw').glob('shard-*.zip')):
  receipt=json.loads(path.with_suffix('.json').read_text());assert path.stat().st_size==receipt['bytes']
  assert sha(path.read_bytes())==receipt['sha256']
  anchor=anchors[path.stem.rsplit('-',1)[-1]];assert receipt['sha256']==anchor['sha256'] and receipt['bytes']==anchor['bytes']
  with zipfile.ZipFile(path) as z:
   assert set(z.namelist())=={r['path'] for r in receipt['inventory']} and len(z.namelist())==receipt['files']
   data={r['path']:z.read(r['path']) for r in receipt['inventory']}
   for r in receipt['inventory']:assert len(data[r['path']])==r['bytes'] and sha(data[r['path']])==r['sha256'];files+=1
  stamps=[name for name in data if name.startswith('shard-') and name.endswith('.json')];assert len(stamps)==1
  stamp=json.loads(data[stamps[0]]);shard=stamp['id'];assert shard not in shard_ids;shard_ids.add(shard)
  tasks=list(expected.values())[int(shard)*80:(int(shard)+1)*80];assert stamp==dict(id=shard,stage='null',tasks=tasks,manifest=identity,selection=None)
  for c in tasks:
   i=c['id'];assert i not in seen;seen.add(i);context=fp(dict(task=c,manifest=identity,selection=None))
   v=json.loads(data[i+'/result.json']);assert v['task']==c and v['context']==context and v['status']=='complete'
   received=json.loads(data[i+'/receipt.json']);assert received==dict(task=c,context=context,environment=manifest['environment'])
   verified=json.loads(data[i+'/verified.json']);check_structure(c,v,verified);assert verified['context']==context and verified['saved_setting_refits']==20
   for name,digest in verified['digests'].items():assert sha(data[i+'/'+name])==digest
   attempts=json.loads(data[i+'/attempts.json']);assert attempts['context']==context
   assert all(t['status']=='complete' and not t['unknown_fit_cost'] and not t['unknown_verification_cost'] for t in attempts['attempts'])
   known_fits+=sum(t['fits'] for t in attempts['attempts']);known_refits+=sum(t['verification_refits'] for t in attempts['attempts'])
   training=generate(c,c['n'],c['seed']);independent=generate(c,1000,c['test_seed']);full=np.concatenate([training,independent])
   numerators=np.zeros((2,2));denominator=0.;positive=[0,0]
   for j,f in enumerate(v['comparisons']):
    cp=json.loads(data[i+f'/comparison-{j}.json']);assert cp['context']==context and cp['result']==f and cp['population']==v['population'][j]
    all_y=training if j<3 else full;observed=np.asarray(f['observed']);np.testing.assert_array_equal(observed,all_y[f['assessment_indices']])
    ref=float(np.square(observed-all_y[f['train_indices']].mean()).mean());baseline=float(np.square(observed-np.asarray(f['baseline_predicted'])).mean());assertclose([ref,baseline],[f['reference_mse'],f['baseline_mse']])
    assert f['fits']==173 and f['seed']==c['split_seed'];trace(f['baseline_selection'],f['baseline_parameters'],False)
    for pos,pair in enumerate(f['pairs']):
     trace(pair['selection'],pair['parameters'],True);assert pair['matched_parameters']=={key:pair['parameters'][key] for key in ('curves__n_knots','ridge__alpha')}
     joint=float(np.square(observed-np.asarray(pair['predicted'])).mean());matched=float(np.square(observed-np.asarray(pair['matched_predicted'])).mean())
     gains=np.array([(baseline-joint)/ref,(matched-joint)/ref,(baseline-matched)/ref]);saved=np.array([pair['overall_gain'],pair['matched_gain'],pair['marginal_gain']]);assertclose(gains,saved);maximum_delta=max(maximum_delta,float(abs(gains-saved).max()));pair_checks+=1
     assertclose([joint,matched],[pair['joint_mse'],pair['matched_additive_mse']])
     if j<3:numerators[pos]+=(np.array([baseline,matched])-joint)*len(observed);positive[pos]+=int(matched>joint)
    pop=v['population'][j];am=float(np.square(observed-np.asarray(pop['additive_predicted'])).mean());jm=float(np.square(observed-np.asarray(pop['joint_predicted'])).mean());assertclose([am,jm,(am-jm)/ref],[pop['additive_mse'],pop['joint_mse'],pop['gap']]);population_checks+=1
    if j<3:denominator+=len(observed)*ref
   scores=[]
   for pos in range(2):
    overall,matched=numerators[pos]/denominator;score=max(0.,matched) if overall>0 and positive[pos]>=2 else 0.;scores.append(score)
   rows.append(dict(id=i,cell=c['cell'],benchmark=c['benchmark'],control=bool(max(scores)>thresholds[c['cell']]),pair0_score=scores[0],pair1_score=scores[1]))
  print('Verified shard',shard,flush=True)
 assert seen==set(expected) and shard_ids=={f'{i:03d}' for i in range(180)}
 frame=pd.DataFrame(rows);saved=pd.DataFrame(json.loads((a.input/'actions-dataset-metrics.json').read_text()));assert dict(zip(frame.id,frame.control))==dict(zip(saved.id,saved.control))
 strata=[];passed=[]
 for cell in dict.fromkeys(c['cell'] for c in expected.values()):
  good=True
  for benchmark in ('noise','additive'):
   part=frame[(frame.cell==cell)&(frame.benchmark==benchmark)];assert len(part)==1200;k=int(part.control.sum());upper=1. if k==1200 else float(beta.ppf(1-.02/36,k+1,1200-k));qualifies=upper<=.05;good=good and qualifies
   original=next(r for r in audit['strata'] if r['cell']==cell and r['benchmark']==benchmark);assert original['k']==k and original['n']==1200 and original['pass']==qualifies;assertclose(original['bound'],upper)
   strata.append(dict(cell=cell,benchmark=benchmark,k=k,n=1200,observed_rate=k/1200,upper=upper,qualified=qualifies))
  if good:passed.append(cell)
 assert passed==audit['passed_cells'] and known_fits==audit['known_attempt_cost']['fits'] and known_refits==audit['known_attempt_cost']['verification_refits']
 check=dict(datasets=len(seen),shards=len(shard_ids),raw_files=files,pair_risk_checks=pair_checks,population_risk_checks=population_checks,max_score_difference=maximum_delta,
            false_alarm_datasets=int(frame.control.sum()),known_fitting_calls=known_fits,known_verification_refits=known_refits,passed_cells=passed,model_fits_performed=0,manifest=identity)
 frame.to_csv(a.output/'independent-dataset-metrics.csv',index=False);pd.DataFrame(strata).to_csv(a.output/'independent-null-bounds.csv',index=False)
 (a.output/'independent-check.json').write_text(json.dumps(check,indent=2)+'\n');print(json.dumps(check),flush=True)
if __name__=='__main__':main()
