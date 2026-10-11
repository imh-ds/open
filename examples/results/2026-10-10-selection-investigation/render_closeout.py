"""Render descriptive findings from audited evidence; no fits or policy changes."""
import gzip
import json
from pathlib import Path
import sys
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

evidence = Path(sys.argv[1])
output = Path(__file__).parent
audit = json.loads((evidence/'audit.json').read_text())
assert (audit['planned'], audit['complete'], audit['incomplete'], audit['audit_errors']) == (74,74,0,0)
continuation = audit['continuation']
assert continuation['status'] == 'complete' and continuation['audited_checkpoints'] == 300
assert continuation['structural']['path_recovered'] and not continuation['structural']['extra_edges']
models = {(r['task']['case'], r['task']['scope'], r['task']['variant'], r['task']['inputs']):r
          for r in audit['diagnostics'] if r['task']['kind']=='model'}
def row(case, variant, inputs):
    return models[case,'full',variant,inputs]
def value(case, variant, inputs, scope='test'):
    return row(case,variant,inputs)[scope]['skill']
cases = ['interaction-n80-p5','interaction-n300-p5','interaction-n300-p15']
variants = [('gbm_shipped','all','GBM: all inputs'),('gbm_shipped','signal','GBM: known pair only'),
            ('gbm_deeper','all','Deeper GBM: all inputs'),('rf_shipped','all','RF: all inputs'),
            ('linear','product','Known product: linear control')]
colors = ['#305c84','#56a086','#b08c4d','#8774aa','#a8c0ce']
plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False})
fig, axes = plt.subplots(2,2,figsize=(14,10),layout='constrained')
fig.suptitle('Selection investigation: what these fixed examples showed\nDevelopment diagnosis; no minimum-N or policy validation',fontsize=16)
ax = axes[0,0]; x = np.arange(3); width=.15
for i,(variant,inputs,label) in enumerate(variants):
    ax.bar(x+(i-2)*width,[value(c,variant,inputs)*100 for c in cases],width,label=label,color=colors[i])
ax.set_xticks(x,['80 people\n5 inputs','300 people\n5 inputs','300 people\n15 inputs'])
ax.axhline(0,color='#555',lw=.8); ax.set_ylim(-23,80)
ax.set_ylabel('Prediction error reduction vs mean (%)')
ax.set_title('Interaction: irrelevant inputs made learning harder')
ax.legend(fontsize=8,loc='upper left',ncol=2)
ax.text(.02,.02,'Each condition uses a different dataset/seed.\nKnown inputs are a diagnostic control.',transform=ax.transAxes,fontsize=8)
ax = axes[0,1]; case='chain-n80-p5'
for i,(inputs,label,color) in enumerate([('signal','Real predictor x1','#305c84'),('signal_noise','x1 + noise x4','#b08c4d')]):
    vals=[value(case,'gbm_shipped',inputs,scope)*100 for scope in ('cv','test')]
    bars=ax.bar(np.arange(2)+(i-.5)*.3,vals,.3,label=label,color=color)
    ax.bar_label(bars,fmt='%.1f%%',padding=3)
ax.set_xticks([0,1],['Nested CV','Fixed independent test'])
ax.set_ylim(0,75); ax.set_ylabel('Prediction error reduction vs mean (%)')
ax.set_title('Small chain: noise helped CV, hurt test prediction')
ax.legend(loc='upper left',fontsize=9)
ax = axes[1,0]
loco=sorted([r for r in audit['diagnostics'] if r['task']['kind']=='loco'],key=lambda r:r['task']['run'])
for name,label,color in [('fixed_ablation','Fixed full-model settings','#305c84'),('retuned_ablation','Retuned reduced model','#b08c4d')]:
    ax.plot([r['task']['run'] for r in loco],[r['contrasts'][name]['baseline_relative']*100 for r in loco],
            'o-',label=label,color=color)
ax.axhline(0,color='#555',lw=.8);ax.set_ylim(-1,15);ax.set_xticks(range(10))
ax.set_xlabel('Original subsample ID (overlapping samples)')
ax.set_ylabel('CV contribution of noise x4 (% of baseline error)')
ax.set_title('Retuning did not remove the apparent noise contribution')
ax.legend(fontsize=9)
payload=json.loads(gzip.decompress((evidence/'continuation/chain-n300-p15/network.json.gz').read_bytes()))
relations=[r for l in payload['layer_evidence.json']['layers'] for t in l['evidence'] for r in t['relations'] if r['qualifies']]
assert {(r['source'],r['target']) for r in relations} == {('x1','y'),('x2','x1'),('x3','x2')}
ax=axes[1,1];ax.axis('off');ax.set_xlim(-.2,3.3);ax.set_ylim(-.8,1)
positions={'x3':0,'x2':1,'x1':2,'y':3}
for name,pos in positions.items():
    ax.text(pos,.2,name,ha='center',va='center',fontsize=13,
        bbox=dict(boxstyle='round,pad=.5',fc='#e4edf5' if name!='y' else '#305c84',ec='#305c84'),
        color='white' if name=='y' else '#26455f')
    ax.text(pos,-.05,f"Depth {3-pos}",ha='center',fontsize=9)
for r in relations:
    start,end=positions[r['source']],positions[r['target']]
    ax.annotate('',(end-.16,.2),(start+.16,.2),arrowprops=dict(arrowstyle='->',lw=2,color='#305c84'))
    f=r['frequency'];ax.text((start+end)/2,.43,f"{f['selected']}/{f['planned']}",ha='center',fontsize=10)
ax.set_title('Interrupted chain completed: all 3 intended links')
ax.text(1.5,-.3,'300 people / 15 candidate inputs\n300 completed assessments; no extra links',ha='center',fontsize=11)
ax.text(1.5,-.6,'Predictive traversal; arrows do not establish causal direction.',ha='center',fontsize=8)
fig.savefig(output/'investigation-overview.png',dpi=150)
plt.close(fig)

lines=['# Selection investigation: findings','',
    '[Actions run 38116199436](https://github.com/imh-ds/open/actions/runs/38116199436) succeeded: 96/96 jobs.',
    '**All 74 diagnostics completed. The interrupted chain also completed and recovered all 3 intended links, with no extra links.**','',
    'The saved evidence was independently audited locally, including data/split identities, arithmetic and deterministic saved-setting model refits.',
    'No production selection rule, threshold, model grid or historical denominator changed.','',
    '![Diagnostic comparisons and completed recursive chain](investigation-overview.png)','',
    '## What the investigation indicates','',
    '### The recursive engine finished correctly in this example','',
    'The original 300-person/15-input chain now retains x1->y, x2->x1 and x3->x2 at the intended depths.',
    'The continuation has 300 complete assessments: 120 reused and 180 new on Actions. Each target used 100 planned assessments before the next layer advanced.',
    'Original pilot completion remains historically 11/12; this is a separately recorded successful continuation. It does not validate arbitrary networks.','',
    '| Predictive link | Selected / planned |','|---|---:|']
for r in sorted(relations,key=lambda r:positions[r['target']],reverse=True):
    f=r['frequency'];lines.append(f"| {r['source']} -> {r['target']} | {f['selected']}/{f['planned']} |")
lines+=['',f"The retained-input refit reduced test prediction error by {continuation['scores']['retained']['skill']*100:.1f}% versus predicting the training mean; the all-input refit reduced it by {continuation['scores']['full']['skill']*100:.1f}%.",
    'Those percentages are baseline-relative prediction skill, not a causal effect or standardized association coefficient.','',
    '### Interaction misses were also a model-learning problem','',
    'For all three original interaction datasets, restricting GBM to the two known signal inputs improved both nested-CV and fixed-test skill.',
    'The known product-feature linear control predicted well, showing usable signal in these synthetic examples.',
    'This supports difficulty learning the interaction amid nuisance inputs under the tested settings; it does not isolate a population cause or supply an automatic input filter.',
    'The expanded-depth GBM grid and RF improved some all-input comparisons but did not uniformly solve the problem.','',
    '| Original discovery dataset | GBM all-input CV | GBM known-pair CV | GBM all-input test | GBM known-pair test | Product-control test |',
    '|---|---:|---:|---:|---:|---:|']
for c in cases:
    vals=[value(c,'gbm_shipped','all','cv'),value(c,'gbm_shipped','signal','cv'),
          value(c,'gbm_shipped','all'),value(c,'gbm_shipped','signal'),value(c,'linear','product')]
    lines.append('| '+c+' | '+' | '.join(f'{v*100:.1f}%' for v in vals)+' |')
lines+=['','These are matched input comparisons within each dataset. Different conditions have different seeds, so the table is not a controlled estimate of N or input-count effects.',
    'Knowing which inputs truly matter is a diagnostic privilege in a simulation; real users do not have that information.',
    'The earlier 49/100 link promotions at N300/p5 remain a selection-loss observation; useful full-model prediction does not guarantee stable link retention.','',
    '### The stable noise link was not explained away by ablation retuning','',
    'The small-chain x4 noise input had previously been retained in 89/100 assessments.',
    'Across the first 10 matched original samples, its apparent CV contribution stayed positive in 10/10 even when the reduced model was separately retuned.',
    f"Mean contribution was {np.mean([r['contrasts']['fixed_ablation']['baseline_relative'] for r in loco])*100:.2f}% of baseline error with fixed settings and {np.mean([r['contrasts']['retuned_ablation']['baseline_relative'] for r in loco])*100:.2f}% with retuning.",
    'These overlapping samples are not 10 independent population replications.',
    f"For the full small-chain discovery sample, GBM with x1 alone had CV skill {value(case,'gbm_shipped','signal','cv')*100:.1f}% and test skill {value(case,'gbm_shipped','signal')*100:.1f}%. Adding x4 gave CV skill {value(case,'gbm_shipped','signal_noise','cv')*100:.1f}% but lower test skill {value(case,'gbm_shipped','signal_noise')*100:.1f}%.",
    'This pattern is consistent with an apparent benefit specific to this discovery dataset that failed on new test participants. It is not evidence that x4 is a real population predictor.',
    'A 70% retention threshold describes repeatability within one dataset; this example shows that repeatability can retain a noise input.','',
    '## Limits and next step','',
    'No calculation or recursion error was identified by these checks. Statistical reliability remains unresolved: model-learning limitations and false links still need fresh independent multi-seed validation.',
    'Do not lower the threshold, switch engines, adopt the diagnostic grid, or establish a minimum N from these previously inspected development cases.',
    'The audit independently refits saved parameters; it does not repeat tuning searches to verify winning settings or inner tuning scores.',
    'All 64 model comparisons include their two discovery scopes; the table above summarizes full scope only. See REPORT-from-actions.md and audit.json for every planned result.',
    'Execution/checksum evidence is in execution.json. Finalization elapsed time excludes distributed assessment work.']
(output/'REPORT.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
