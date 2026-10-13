from pathlib import Path
import json,hashlib
import numpy as np
import pandas as pd
from scipy.stats import beta
base=Path(__file__).resolve().parent;folder=base;root=Path(__file__).resolve().parents[3]
audit=json.loads((folder/'actions-audit.json').read_text());rows=json.loads((folder/'actions-dataset-metrics.json').read_text())
manifest=json.loads((root/'examples/study_manifests/prospective-interaction.json').read_text());selection=json.loads((root/'examples/study_manifests/prospective-interaction-selection.json').read_text())
fp=lambda v:hashlib.sha256(json.dumps(v,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()
assert fp(manifest)==audit['manifest'] and fp(selection)==audit['selection']
null_audit=json.loads((root/'examples/study_manifests/prospective-interaction-null-audit.json').read_text())
assert fp(null_audit)==selection['stage_a_audit'] and len(null_audit['strata'])==12 and all(r['pass'] for r in null_audit['strata'])
assert audit['complete']==audit['planned']==10800 and not audit['audit_errors'] and not audit['missing_ids'] and not audit['skipped_ids']
assert len(rows)==len({r['id'] for r in rows})==10800 and {r['id'] for r in rows}==set(selection['selected_ids'])
pairs=pd.read_csv(folder/'actions-case-metrics.csv');assert len(pairs)==21600 and not pairs.duplicated(['id','pair']).any()
threshold_file=root/'examples/study_manifests/pair-confidence-thresholds.json'
assert hashlib.sha256(threshold_file.read_bytes().replace(b'\r\n',b'\n')).hexdigest()==manifest['thresholds_sha256']
thresholds=json.loads(threshold_file.read_text())['thresholds']
np.testing.assert_allclose(pairs.fixed_threshold,pairs.cell.map(thresholds),rtol=0,atol=1e-15)
expected_score=np.where((pairs.cv_overall>0)&(pairs.positive_folds>=2),np.maximum(0.,pairs.cv_matched),0.)
np.testing.assert_allclose(expected_score,pairs.score,rtol=1e-12,atol=1e-14)
assert ((pairs.score>pairs.fixed_threshold)==pairs.above_threshold).all()
pairs=pairs.set_index(['id','pair'],verify_integrity=True)
for r in rows:
    true=pairs.loc[(r['id'],'x1:x2')];control=pairs.loc[(r['id'],'x1:x3')]
    assert r['detected']==bool(true.above_threshold) and r['control']==bool(control.above_threshold)
    display=bool(true.above_threshold and true.pattern_status=='assessed' and true.pattern_consistent)
    assert r['displayed']==display and r['shape_correct']==bool(display and true.truth_cosine>=.8)
results=[]
for r,c in zip(audit['strata'],audit['conditions']):
    group=[v for v in rows if (v['cell'],v['benchmark'])==(r['cell'],r['benchmark'])];assert len(group)==450
    values={}
    for name,key,alpha,target,upper in [('control','control',.02/36,.05,True),('detection','detected',.02/24,.8,False),('shape','shape_correct',.01/24,.9,False)]:
        assessed=group if name!='shape' else [v for v in group if v['displayed']];n=len(assessed);k=sum(v[key] for v in assessed)
        bound=None if not n else (1. if k==n else float(beta.ppf(1-alpha,k+1,n-k))) if upper else (None if not n else 0. if not k else float(beta.ppf(alpha,k,n-k+1)))
        passed=bound is not None and (bound<=target if upper else bound>=target)
        assert (k,n,passed)==(r[name]['k'],r[name]['n'],r[name]['pass'])
        if bound is not None:np.testing.assert_allclose(bound,r[name]['bound'],rtol=0,atol=1e-14)
        values[name]=dict(k=k,n=n,bound=bound,passed=passed)
    predictive=values['control']['passed'] and values['detection']['passed']
    assert predictive==c['predictive_support'] and (predictive and values['shape']['passed'])==c['curve_pattern_support']
    results.append(dict(cell=r['cell'],benchmark=r['benchmark'],**values,predictive=predictive,pattern=c['curve_pattern_support']))
(base/'independent-counts.json').write_text(json.dumps(dict(datasets=10800,pair_rows=21600,verified_bounds=72,conditions=results,qualified=sum(r['predictive'] for r in results)),indent=2)+'\n')
assert audit['global_predictive_support']==all(r['predictive'] for r in results) and audit['global_curve_pattern_support']==all(r['pattern'] for r in results)
print('Verified 10800 dataset rows, 21600 pair rows, 72 bounds; qualified',sum(r['predictive'] for r in results))
