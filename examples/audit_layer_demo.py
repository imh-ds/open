"""Audit saved layer demonstrations without fitting models or executing artifacts."""
import argparse
from collections import Counter
import hashlib
import json
import math
from pathlib import Path
from statistics import median
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
import numpy as np
import pandas as pd
from open.layer_io import load_layer_bundle


def require(condition, message):
    if not condition: raise ValueError(message)


def close(a,b):
    return a is b if a is None or b is None else math.isclose(a,b,rel_tol=0,abs_tol=1e-12)


def audit_demo(output, historical=None):
    output=Path(output)
    bundle=load_layer_bundle(output/'bundle')  # strict structural and numerical reconstruction
    meta=bundle.metadata
    summary=json.loads((output/'summary.json').read_text(encoding='utf-8'))
    sample=(output/'sample.csv').read_bytes()
    require(hashlib.sha256(sample).hexdigest()==summary['sample_sha256'],'sample hash mismatch')
    frame=pd.read_csv(output/'sample.csv')
    b=meta['planned_resamples_per_target']
    require(tuple(frame)==meta['variables'],'sample variables mismatch')
    require(meta['path_frequency_available'] is False,'fabricated path probability')
    fraction=meta['config']['stability']['fraction']
    for plan in meta['plans']:
        expected=tuple(sorted(int(i) for i in np.random.default_rng(plan['seed']).choice(
            len(frame),size=math.floor(len(frame)*fraction),replace=False)))
        require(expected==plan['sampled_positions'],'sample positions differ from seeded plan')
    target_count=0
    for layer in bundle.layers:
        for target in layer['evidence']:
            target_count+=1
            records=[a for a in bundle.assessments if (a['depth'],a['target'])==(layer['depth'],target['target'])]
            require(len(records)==b,'missing assessment')
            completed=[a['local'] for a in records if a['status']=='completed']
            for candidate in target['candidates_evidence']:
                name=candidate['source']
                counts=Counter('LOW_TOTAL_GAIN' if loc['stop'] else
                    loc['classes'][name]['kind'] or 'UNCLASSIFIED' for loc in completed)
                require(counts=={k:v['selected'] for k,v in candidate['classes'].items()},'class recount mismatch')
                require(candidate['promotion']['selected']==sum(name in loc['promoted'] for loc in completed),
                        'promotion recount mismatch')
                for kind in ('broad','unique'):
                    values=[loc['relevance'][kind][name] for loc in completed]
                    raw=[v['raw'] for v in values]
                    base=[v['raw']/loc['relevance']['baseline_risk'] for v,loc in zip(values,completed)
                          if loc['relevance']['baseline_risk']>0]
                    gain=[v['normalized'] for v in values if v['normalized'] is not None]
                    strength=candidate[kind]
                    for prefix,items in [('raw',raw),('baseline_relative',base),('gain_relative',gain)]:
                        require(strength[prefix+'_count']==len(items) and
                            close(strength[prefix+'_median'],median(items) if items else None),'score summary mismatch')
            # Joint retention is exact member identity, independently deduplicated per assessment.
            groups=Counter()
            for loc in completed:
                members={tuple(sorted(p['members'])) for p in loc['retained_pairs']}
                members.update(tuple(sorted(c['members'])) for c in loc['shared_components'] if c['validated'])
                groups.update(members)
            for rel in target['relations']:
                count=rel['frequency']
                if rel['relation']=='SHARED_GROUP':
                    require(groups[rel['members']]==count['selected'],'joint recount mismatch')
                elif rel['relation']=='PROXIMAL_UNIQUE':
                    require(sum(not loc['stop'] and loc['classes'][rel['source']]['kind']=='PROXIMAL_UNIQUE'
                        for loc in completed)==count['selected'],'unique recount mismatch')
                require(count['planned']==b and count['completed']==len(completed),'denominator mismatch')
    require(summary['assessment_count']==len(bundle.assessments),'summary count mismatch')
    historical_checked=False
    max_difference=0.
    if historical is not None:
        historical=Path(historical)
        require((historical/'sample.csv').read_bytes()==sample,'historical sample differs')
        old=json.loads((historical/'recursion_trace.json').read_text(encoding='utf-8'))
        new={a['run_index']:a for a in bundle.assessments if a['target']==meta['root']}
        require(len(old)==len(new)==b,'root opportunities differ')
        for trace in old:
            record=new[trace['run_index']]
            require(record['status']=='completed' and trace['seed']==meta['plans'][record['run_index']]['seed'],
                    'root status/seed mismatch')
            loc=record['local']; previous=trace['targets'][meta['root']]
            require(tuple(previous['input_candidates'])==record['candidates'],'root candidates changed')
            pairs=[(loc['relevance'][key],previous[key]) for key in ('baseline_risk','full_risk')]
            pairs += [(loc['total_gain_relative'],previous['relative_gain'])]
            for name,score in loc['relevance']['unique'].items():
                pairs.extend([(score['raw'],previous['unique_scores'][name]['raw']),
                              (score['normalized'],previous['unique_scores'][name]['gain_relative'])])
                require(loc['classes'][name]['kind']==previous['candidates'][name]['class'],'root class changed')
            for a,c in pairs:
                require(close(a,c),'historical root calculation differs')
                if a is not None: max_difference=max(max_difference,abs(a-c))
        historical_checked=True
    sensitivity=[]
    for item in summary.get('sensitivity',[]):
        if item['directory']=='.': continue
        other=load_layer_bundle(output/item['directory']/'bundle')
        require(other.metadata['plans']==meta['plans'],'sensitivity sample plans differ')
        sensitivity.append({'threshold':item['threshold'],'status':other.metadata['status'],
                            'assessment_count':len(other.assessments)})
    audit={'passed':True,'method':'layer_consensus','sample_sha256':summary['sample_sha256'],
        'planned_per_target':b,'target_count':target_count,'assessment_count':len(bundle.assessments),
        'status':meta['status'],'historical_root_checked':historical_checked,
        'historical_root_max_absolute_difference':max_difference,'absolute_tolerance':1e-12,
        'all_counts_scores_and_assignments_revalidated':True,'sensitivity':sensitivity,
        'no_refitting':True,'path_frequency_available':False}
    (output/'audit.json').write_text(json.dumps(audit,indent=2)+'\n',encoding='utf-8')
    return audit


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--historical',type=Path)
    args=parser.parse_args()
    print(json.dumps(audit_demo(args.output,args.historical),indent=2))
