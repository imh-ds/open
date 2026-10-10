"""Analysis-only demonstration of fixed within-depth peer comparisons.

Reproduce with: python reproduce.py --output NEW_DIRECTORY
Uses existing analysis APIs; does not change recursive scheduling or bundles.
"""
from dataclasses import replace
from pathlib import Path
from importlib.metadata import version
import argparse
import csv
import hashlib
import json
import shutil
import subprocess
import sys
import time
import warnings

import numpy as np
import pandas as pd

ROOT=next(p for p in Path(__file__).resolve().parents if (p/'pyproject.toml').exists())
sys.path.insert(0,str(ROOT/'src'))
from open.config import load_config,LearnerConfig
from open.layer_artifacts import portable
from open.layer_consensus import aggregate_target_assessments
from open.layer_io import load_layer_bundle
from open.layer_stability import make_layer_sample_plans,assess_layer_target
from open.local_target import LocalClass


def run(output):
    output=Path(output)
    if output.exists(): raise FileExistsError('Use a new output directory')
    original=ROOT/'examples/results/2026-10-08-survey-layer-consensus'
    load_layer_bundle(original/'bundle')  # audit the frozen reference
    data=pd.read_csv(original/'sample.csv')
    config=load_config(original/'config.yaml')
    frozen=json.loads((original/'summary.json').read_text())
    depths=frozen['assigned_depth']
    groups={depth:tuple(n for n in data if depths[n]==depth) for depth in sorted(set(depths.values())) if depth>0}
    plans=make_layer_sample_plans(data,config,n_resamples=20)
    output.mkdir(parents=True)
    shutil.copy2(original/'sample.csv',output/'sample.csv')
    shutil.copy2(__file__,output/'reproduce.py')
    pd.DataFrame({'name':list(depths),'frozen_depth':list(depths.values())}).to_csv(output/'frozen_depths.csv',index=False)
    rows,ledger=[],[]
    started=time.perf_counter()
    captured_warnings=[]
    for engine in ('elastic_net','gbm'):
        run_config=replace(config,learner=LearnerConfig(library=(engine,)))
        for depth,peers in groups.items():
            for target in peers:
                candidates=tuple(n for n in peers if n!=target)
                print(f'{engine}: depth {depth}, {target} ~ {candidates}',flush=True)
                with warnings.catch_warnings(record=True) as captured:
                    warnings.simplefilter('always')
                    assessments=assess_layer_target(data,depth=depth,target=target,candidates=candidates,plans=plans,config=run_config)
                captured_warnings.extend(dict(category=w.category.__name__,message=str(w.message)) for w in captured)
                evidence=aggregate_target_assessments(assessments,planned_count=20,pi_min=config.thresholds.stability)
                assert evidence.complete,[(a.status,a.failure_message) for a in assessments if a.status!='completed']
                ledger.append(dict(engine=engine,evidence=portable(evidence)))
                for candidate in evidence.candidates_evidence:
                    frequency=candidate.classes.get('PROXIMAL_UNIQUE')
                    selected=frequency.selected if frequency else 0
                    # Independent recount from actual saved local classes.
                    count=sum(not a.local.stop and a.local.classes[candidate.source].kind is LocalClass.PROXIMAL_UNIQUE for a in assessments)
                    assert count==selected
                    relative=[a.local.relevance.unique[candidate.source].raw/a.local.relevance.baseline_risk for a in assessments if a.local.relevance.baseline_risk>0]
                    rows.append(dict(engine=engine,depth=depth,source=candidate.source,target=target,
                        candidates=';'.join(candidates),selected=selected,planned=20,completed=len(assessments),
                        frequency=selected/20,retained=selected>0 and selected/20>=config.thresholds.stability,
                        median_unique_gain_over_baseline=float(np.median(relative)),
                        class_counts=json.dumps({k:v.selected for k,v in candidate.classes.items()},sort_keys=True)))
                print('  complete:',[(r['source'],r['selected']) for r in rows if r['engine']==engine and r['target']==target],flush=True)
    (output/'assessment_evidence.json').write_text(json.dumps(ledger,indent=2,allow_nan=False)+'\n',encoding='utf-8')
    pd.DataFrame(rows).to_csv(output/'peer_edges.csv',index=False)
    # Recover original retained unique links, never relabel the GBM overlay as
    # a GBM-discovered base network.
    layers=json.loads((original/'bundle/layer_evidence.json').read_text())['layers']
    original_edges=[]
    for layer in layers:
        for target in layer['evidence']:
            for relation in target['relations']:
                if relation['qualifies'] and relation['relation']=='PROXIMAL_UNIQUE':
                    original_edges.append((relation['source'],relation['target']))
    labels={n:n.replace('_',' ').title() for n in data}
    labels['wellbeing']='Well-being'
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    positions=dict(wellbeing=(.5,.05),sleep_quality=(.15,.48),social_support=(.5,.48),physical_activity=(.85,.48),
                   perceived_stress=(.23,.88),work_demands=(.7,.88))
    fig,ax=plt.subplots(figsize=(11,7))
    ax.set_xlim(0,1);ax.set_ylim(-.02,1.07);ax.axis('off')
    for source,target in original_edges:
        ax.annotate('',xy=positions[target],xytext=positions[source],
            arrowprops=dict(arrowstyle='->',color='#aab2ba',lw=1.7,shrinkA=38,shrinkB=38),zorder=1)
    retained=[r for r in rows if r['engine']=='elastic_net' and r['retained']]
    for row in retained:
        source,target=row['source'],row['target']
        ax.annotate('',xy=positions[target],xytext=positions[source],
            arrowprops=dict(arrowstyle='->',color='#ba5728',lw=2.4,shrinkA=39,shrinkB=39,
                           connectionstyle='arc3,rad=.2'),zorder=2)
    for name,position in positions.items():
        ax.text(*position,labels[name].replace(' ','\n'),ha='center',va='center',fontsize=11,
            bbox=dict(boxstyle='round,pad=.4',fc='#eaf1f7' if name!='wellbeing' else '#d6e7f2',ec='#38536b'),zorder=3)
    ax.set_title('Same-depth demonstration: frozen elastic-net reference\nGray = original links; orange = retained peer predictions\nArrow = predictor → target, not causal direction',fontsize=13)
    ax.text(.02,1.0,'Depth 2',fontsize=10,color='#4b6478')
    ax.text(.02,.58,'Depth 1',fontsize=10,color='#4b6478')
    ax.text(.02,.13,'Depth 0',fontsize=10,color='#4b6478')
    fig.tight_layout();fig.savefig(output/'peer_overlay.png',dpi=160);plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(13,6),sharex=True)
    for ax,engine in zip(axes,('elastic_net','gbm')):
        current=[r for r in rows if r['engine']==engine]
        names=[labels[r['source']]+' → '+labels[r['target']] for r in current]
        ax.barh(names,[r['frequency'] for r in current],color=['#ba5728' if r['retained'] else '#c5cdd4' for r in current])
        ax.axvline(.7,color='#38536b',ls='--',label='70% threshold')
        for i,r in enumerate(current): ax.text(r['frequency']+.015,i,f"{r['selected']}/20",va='center',fontsize=9)
        ax.set_xlim(0,1.18);ax.set_xlabel('Unique-link selection / 20 planned assessments')
        ax.set_title(engine+' on fixed reference layers');ax.invert_yaxis()
    fig.suptitle('Peer comparison stability: eight directed comparisons per engine\nCandidate sets exclude other depths and Well-being; no causal ordering inferred',fontsize=12)
    fig.tight_layout(rect=(0,0,1,.91));fig.savefig(output/'peer_frequencies.png',dpi=160);plt.close(fig)
    lines=['# Within-depth predictive assessment demonstration','',
        'Synthetic data only. Existing 180-row survey and audited 70% elastic-net reference layers are frozen. This standalone analysis uses existing APIs; automatic peer scheduling has not been added to the recursive engine.',
        '', '## Fixed comparisons','',
        'Depth 1: each of Social Support, Sleep Quality and Physical Activity is predicted by the other two. Depth 2: Stress is predicted by Work Demands, and Work Demands by Stress. Well-being and variables from other depths are excluded from these peer models. These are conditional on the peer candidate set only: relationships can reflect common upstream influences.',
        '', 'Both engines use the same twenty seeds 613–632, 144-row samples (80% without replacement), two outer folds and two inner tuning folds. Elastic net searches 25 tuning candidates; GBM searches four, so this is not an equal-budget engine benchmark. A unique edge qualifies when the existing local PROXIMAL_UNIQUE class appears in at least 14/20 completed assessments. All candidate comparisons are reported, including nonretained results. Joint relations/classes remain in the full assessment evidence; a joint relation is not converted into individual unique arrows. The CSV median_unique_gain_over_baseline column is the median signed increase in held-out MSE after removing a predictor, divided by the same assessment baseline MSE; it is not an ALE magnitude or a causal effect.',
        '', '## Results','', '| Within-depth prediction | Elastic net | GBM |','| --- | ---: | ---: |']
    for row in [r for r in rows if r['engine']=='elastic_net']:
        other=next(r for r in rows if r['engine']=='gbm' and (r['source'],r['target'])==(row['source'],row['target']))
        lines.append(f"| {labels[row['source']]} → {labels[row['target']]} | {row['selected']}/20 {'retained' if row['retained'] else 'not retained'} | {other['selected']}/20 {'retained' if other['retained'] else 'not retained'} |")
    lines+=['','![Peer frequencies](peer_frequencies.png)','','## Illustrative overlay','',
        'The overlay combines original elastic-net edges with elastic-net peer evidence, leaving all discovery depths unchanged. GBM frequencies are a separate fixed-layer diagnostic: the reference layers were not rediscovered using GBM. A reciprocal pair is shown as two predictive arrows, never resolved into a causal direction by comparing scores.',
        '', '![Peer overlay](peer_overlay.png)', '', '## Interpretation','',
        'A retained peer arrow means its source helped predict the peer target with the other peer variables accounted for, consistently under the existing selection rule. It does not show that one variable causes another. A chain drawn in the overlay is a series of separately supported predictions; its indirect effect and full-path stability have not been estimated. Do not multiply curve magnitudes, contrasts or selection frequencies to obtain an indirect effect.',
        '', 'The original discovery used the same data: this is an exploratory post-discovery illustration, not independent confirmation of layer placement. Testing more peer pairs introduces additional selection opportunities; the existing 70% threshold is not a calibrated multiple-testing correction.',
        '', 'RF/GBM/LightGBM can learn feature interactions without manual product terms, but that does not guarantee recovery of every interaction. Tree size, tuning, noise and sample size matter. First-order percentile ALE curves average across the other predictors and can hide interaction structure. Assessing peer predictability is different from quantifying an interaction in the Well-being model.',
        '', '[Peer edge table](peer_edges.csv), [assessment evidence](assessment_evidence.json), [audit](audit.json), [reproduction script](reproduce.py), [sample](sample.csv), [original network report](../2026-10-08-survey-layer-consensus/REPORT.md).']
    (output/'REPORT.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    audit=dict(analysis='exploratory_fixed_layer_peer_demo',automatic_recursion_changed=False,
        engine_revision=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
        input_sha256=hashlib.sha256((original/'sample.csv').read_bytes()).hexdigest(),
        frozen_depths=depths,config=portable(config),engines=['elastic_net','gbm'],
        engine_configs={engine:portable(replace(config,learner=LearnerConfig(library=(engine,)))) for engine in ('elastic_net','gbm')},
        versions={package:version(package) for package in ('numpy','pandas','scikit-learn','matplotlib')},
        plans=portable(plans),assessment_count=sum(len(e['evidence']['assessments']) for e in ledger),
        directed_candidate_comparisons=len(rows),warnings=captured_warnings,elapsed_seconds=time.perf_counter()-started,
        unique_selection_counts_independently_recounted=True,
        every_target_completed_all_twenty=True,inference='predictive_not_causal_or_mediated_effect')
    (output/'audit.json').write_text(json.dumps(audit,indent=2,allow_nan=False)+'\n',encoding='utf-8')
    print(json.dumps({k:v for k,v in audit.items() if k not in ('plans','config')},indent=2),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',required=True,type=Path)
    args=parser.parse_args()
    run(args.output)
