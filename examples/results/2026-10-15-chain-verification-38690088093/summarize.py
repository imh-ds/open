"""Render applied interpretation from the completed independent audit; no fits."""
from pathlib import Path
import csv
import json
from collections import Counter
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

BASE=Path(__file__).resolve().parent
r=json.loads((BASE/'audit/audit.json').read_text())
assert r['complete']==110 and not r['integrity_errors']
assert r['model_refits']==0 and r['audit_model_fits']==0
rows=r['cases']
fields=['case','n','exact_network','incorrect_connection','missing_links','extra_links','correct_depths','assessments','observation_checks']
with (BASE/'case-results.csv').open('w',newline='',encoding='utf-8') as f:
    writer=csv.DictWriter(f,fieldnames=fields);writer.writeheader()
    for row in rows:
        s=row['structure']
        writer.writerow(dict(case=row['spec']['id'],n=row['spec']['n'],exact_network=s['exact'],incorrect_connection=s['incorrect'],
            missing_links=len(s['missing']),extra_links=len(s['extra']),correct_depths=all(s['depth_matches'].values()),
            assessments=row['assessments'],observation_checks=row['observation_checks']))
checks=sum(v['observation_checks'] for v in rows)
assessments=sum(v['assessments'] for v in rows)
warnings=sum(v['warnings'] for v in rows)
fig,axes=plt.subplots(2,1,figsize=(10,6))
lines=['# Known-chain verification completed','',
       'The independent verification is complete. No models were refitted.','',
       '|People per dataset|Datasets checked|Entire chain and depths correct|Datasets with an extra connection|Qualified|',
       '|---|---:|---:|---:|---|']
for ax,n in zip(axes,(80,200)):
    cell=r['cells'][f'chain-n{n}'];subset=[v for v in rows if v['spec']['n']==n]
    links=Counter(tuple(e) for v in subset for e in v['edges'])
    lines.append(f"|{n}|{cell['complete']}/55|{cell['recovery']['count']}/55|{cell['incorrect']['count']}/55|{cell['qualified']}|")
    ax.set(xlim=(0,10),ylim=(0,2.6));ax.axis('off')
    ax.text(.1,2.2,f'{n} people: {cell["recovery"]["count"]}/55 exact networks',fontsize=14,weight='bold')
    for x,label in ((1.2,'X1\nDepth 2'),(4.7,'X2\nDepth 1'),(8.2,'Y\nOutcome')):
        ax.text(x,1.2,label,ha='center',va='center',fontsize=12,bbox=dict(boxstyle='round,pad=.5',fc='#e6f3f0',ec='#226454'))
    for start,end,edge in ((1.8,4.1,('X1','X2')),(5.3,7.6,('X2','Y'))):
        ax.annotate('',xy=(end,1.2),xytext=(start,1.2),arrowprops=dict(arrowstyle='-|>',lw=2.5,color='#226454'))
        ax.text((start+end)/2,1.55,f'{links[edge]}/55 datasets',ha='center',fontsize=11)
    nuisance=sum(bool(v['structure']['nuisance_promoted']) for v in subset)
    ax.text(4.7,.3,f'Irrelevant variables retained: {nuisance}/55 datasets',ha='center',fontsize=11,color='#555555')
fig.suptitle('Recovery of the known synthetic chain',fontsize=17)
fig.text(.5,.015,'Counts refer to separate datasets, not the 100 resamples inside each analysis.',ha='center',fontsize=10)
fig.tight_layout(rect=(0,.05,1,.96));fig.savefig(BASE/'chain-recovery.png',dpi=160);plt.close(fig)
lines+=['','![Recovered chain](chain-recovery.png)','',
'## What this means simply','',
'The method found the direct connection X2 -> Y and then followed it backward to find X1 -> X2. It put X2 and X1 at the intended depths and tested the irrelevant variables without needing to retain them.',
'',
'This complements the earlier noise-only check: the noise check asked whether it invents connections, and this check asked whether it can recover connections that exist. These are separate benchmarks; their denominators are not pooled.',
'',
'## What it does not establish','',
'These datasets contain a fairly strong, simple linear chain with continuous Gaussian variables and two irrelevant covariates. This supports the shipped ElasticNet recursive workflow under those conditions. It does not show that 80 people is enough for weaker effects, many predictors, measurement error, interactions or correlated/redundant predictors. It does not validate causal direction in real observational data.',
'',
'The 110 correlated-predictor reference cases remain pending. They are the recommended next scientific validation before treating this as broad first-release evidence; no additional study was launched.',
'',
'## Verification and retained evidence','',
f'All {assessments:,} target-resample assessment blocks and {checks:,} recorded observation-loss checks passed the independent replay, with {warnings} recorded warnings and no integrity errors. No new model fits or Actions dispatches were used for recovery.',
'',
'The original Actions run 38690088093 remains failed because its 32-minute verification step timed out. This separate completed local audit supplies the scientific endpoint; it does not rewrite the workflow status.',
'',
'For each sample size, the frozen pass criteria require at least 50/55 exact networks and at most 1/55 datasets with an incorrect connection. Exact 95% binomial intervals are marginal per condition, not a general error guarantee.']
for n in (80,200):
    c=r['cells'][f'chain-n{n}']
    lines.append(f"N={n}: exact-recovery 95% interval {c['recovery']['lower']:.1%} to {c['recovery']['upper']:.1%}; incorrect-connection interval {c['incorrect']['lower']:.1%} to {c['incorrect']['upper']:.1%}.")
lines+=['','Raw inputs, controls, runtime records and SHA256 fingerprints are stored alongside the audit. Per-case verification caches permit resuming without repeating completed checks. See README.md for reproduction.']
(BASE/'interpretation.md').write_text('\n'.join(lines)+'\n',encoding='utf-8',newline='\n')
print('Rendered completed interpretation, case CSV and chain-recovery graph.')
