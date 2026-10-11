"""Summarize frozen development evidence; does not fit or nominate new methods."""
import hashlib
import json
from pathlib import Path
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import t

output = Path(__file__).parent
evidence = Path(sys.argv[1])
audit = json.loads((output/'audit-from-actions.json').read_text())
local = json.loads((evidence/'audit.json').read_text())
assert audit == local
assert (audit['planned'],audit['complete'],audit['missing'],audit['audit_errors']) == (960,960,0,[])
assert audit['development_nominees'] == ['rf_shipped']
assert not audit['failed_tasks']
manifest = json.loads((output/'manifest.json').read_text())
assert manifest == json.loads((output.parents[1]/'study_manifests/fresh-learning.json').read_text())
records = [json.loads((evidence/task['id']/'result.json').read_text()) for task in manifest['tasks']]
assert len(records) == len({r['task']['id'] for r in records}) == 960
cells = {(c['scenario'],c['n'],c['p'],c['variant']):c for c in audit['cells']}
points = [(80,5),(300,5),(300,15)]
variants = ['gbm_shipped','gbm_deeper','rf_shipped','benchmark']
labels = ['Current GBM','Broader GBM grid','Random forest','Known-function benchmark']
colors = ['#285b83','#c08337','#318a69','#8c769b']

def values(scenario,variant):
    return [cells[scenario,n,p,variant]['mean_test_skill']*100 for n,p in points]

def interval(values):
    v = np.asarray(values,float)
    mean = float(v.mean())
    margin = float(t.ppf(.975,len(v)-1)*v.std(ddof=1)/np.sqrt(len(v)))
    return mean,mean-margin,mean+margin

plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False})
fig,axes = plt.subplots(2,2,figsize=(13,9),layout='constrained')
fig.suptitle('Fresh datasets: learning interactions remains difficult\n20 independent datasets per point; development evidence',fontsize=16)
for ax,scenario,title in zip(axes.flat,['interaction','additive','chain','noise'],
        ['Pure interaction','Ordinary additive relationship','Chain: predicting the root outcome','Noise only: no real signal']):
    for variant,label,color in zip(variants,labels,colors):
        y = values(scenario,variant)
        error = [interval(cells[scenario,n,p,variant]['test_skill_values'])[2]*100-y[i]
                 for i,(n,p) in enumerate(points)]
        ax.errorbar(np.arange(3),y,yerr=error,marker='o',label=label,color=color,capsize=3)
    ax.axhline(0,color='#888888',lw=.8)
    ax.set_xticks(np.arange(3),['80 people\n5 inputs','300 people\n5 inputs','300 people\n15 inputs'])
    ax.set_title(title)
    ax.set_ylabel('Reduction in squared error vs mean guess (%)')
    ax.grid(axis='y',alpha=.15)
axes[0,0].legend(fontsize=9)
fig.savefig(output/'learning-overview.png',dpi=160)
plt.close(fig)

rows = []
for c in audit['cells']:
    mean,low,high = interval(c['test_skill_values'])
    diff,dlo,dhi = interval(c['paired_differences'])
    rows.append(dict(scenario=c['scenario'],n=c['n'],p=c['p'],variant=c['variant'],
        independent_datasets=c['complete'],mean_test_skill=mean,
        mean_test_skill_ci_low=low,mean_test_skill_ci_high=high,
        positive_test_skill=c['positive_skill_lower_bound_count'],
        positive_test_skill_wilson_low=c['positive_skill_wilson_bounds'][0],
        positive_test_skill_wilson_high=c['positive_skill_wilson_bounds'][1],
        paired_mean_difference=diff,paired_mean_ci_low=dlo,paired_mean_ci_high=dhi))
pd.DataFrame(rows).to_csv(output/'cell-summary.csv',index=False)
execution = json.loads((output/'execution.json').read_text(encoding='utf-8-sig'))
assert execution['headSha'] == '31f85727a5e93528a4bec4970764bd0b4365f797'
jobs = execution['jobs']
assert len(jobs) == 122 and all(j['conclusion']=='success' for j in jobs)
def stamp(value):
    return pd.Timestamp(value).timestamp()
events = [(stamp(j['startedAt']),1) for j in jobs]+[(stamp(j['completedAt']),-1) for j in jobs]
active = peak = 0
for _,delta in sorted(events):
    active += delta
    peak = max(peak,active)
wall = (max(stamp(j['completedAt']) for j in jobs)-min(stamp(j['startedAt']) for j in jobs))/60
minutes = sum((stamp(j['completedAt'])-stamp(j['startedAt']))/60 for j in jobs)
folds = sum(len(r['folds']) for r in records)
cv_rows = sum(len(f['test']) for r in records for f in r['folds'])
warnings = sum(len(r['warnings'])+sum(len(f['warnings']) for f in r['folds']) for r in records)
cv_positive = {scenario:{variant:sum(r['cv']['skill']>0 for r in records
    if r['task']['case']['scenario']==scenario and r['task']['variant']==variant)
    for variant in variants} for scenario in ('noise','interaction')}
checksum = hashlib.sha256((output/'actions-evidence.zip').read_bytes()).hexdigest()
receipt = dict(run_id=38133443512,source_commit=execution['headSha'],
    archive_bytes=(output/'actions-evidence.zip').stat().st_size,archive_sha256=checksum,
    complete=960,planned=960,independent_datasets=240,outer_fold_fits=folds,
    cv_prediction_rows=cv_rows,test_prediction_rows=960*2000,
    unique_test_participants=240*2000,recorded_warnings=warnings,
    local_aggregate_matches_actions=True,peak_allocated_jobs=peak,
    summed_job_minutes=minutes,workflow_wall_minutes=wall,
    cv_positive_counts=cv_positive,
    local_audit_scope='Regenerated data/splits, declared-grid and training-size checks, arithmetic and identities. Saved-setting fold/final refits were performed on Actions; tuning winners were not independently retuned.')
(output/'receipt.json').write_text(json.dumps(receipt,indent=2)+'\n',encoding='utf-8')
lines = ['# Fresh independent learner investigation: results','',
    'All 960/960 planned method fits completed across 240 fresh datasets. All 122 Actions jobs succeeded. '
    'The local data/split/arithmetic audit passed with zero errors and exactly reproduced the Actions audit. '
    f'There were {warnings} recorded fit warnings.','',
    '## What this means','',
    'The interaction-learning concern reproduced across independent datasets. Random forests improved prediction '
    'relative to the current GBM and met the frozen development nomination rule. They did not eliminate the '
    'difficulty. Increasing sample size helped at five inputs; adding ten nuisance inputs at N300 substantially '
    'reduced interaction performance. This pattern is specific to the tested strong pure-interaction simulations.','',
    'The known-function benchmark was given the correct product x1*x2. It recovered about 59–60% error reduction '
    'in every interaction cell, demonstrating usable signal. The tree models received all candidate predictors '
    'without supplied product terms. The benchmark therefore provides a privileged diagnostic, not a fair '
    'automated feature-discovery competitor or a requirement that users supply interaction terms.','',
    'Test skill is the reduction in squared prediction error relative to guessing the discovery-sample mean. '
    'For example, 28% means 28% less squared error; negative values mean worse than that guess. '
    'These numbers describe whole-model prediction, not path strengths or coefficients.','',
    '| Interaction condition | Current GBM | Broader GBM | Random forest | Known-product benchmark |',
    '|---|---:|---:|---:|---:|']
for n,p in points:
    scores = [f"{cells['interaction',n,p,v]['mean_test_skill']*100:.1f}%" for v in variants]
    lines.append(f"|N{n}, {p} inputs|"+'|'.join(scores)+'|')
lines += ['',
    'With N80/p5, GBM beat the mean guess in 10/20 independent datasets and RF in 17/20. '
    'With N300/p15, GBM did so in 13/20; broader GBM and RF did so in 20/20. '
    'Beating a mean guess is a weak adequacy threshold: RF still recovered only 16.5% error reduction '
    'where the known-product benchmark achieved 59.7%. Even 20/20 observed successes has a pointwise Wilson '
    '95% lower bound of about 84%; it does not certify a 90% reliability standard.','',
    'For ordinary additive relationships and root prediction in chains, all methods beat the mean guess '
    'in all tested datasets. Current GBM averaged about 49% at N80 and 55–56% at N300. RF differences '
    'were small: approximately +2–3 percentage points at N80 and −0.4 to −1.1 points at N300. '
    'These chain results assess root prediction only; the recursive network was not reconstructed.','',
    'In noise-only data, all tree models were worse than the mean guess on all 60 independent test datasets '
    'per method. RF nevertheless had positive nested-CV skill in 3/60 noise datasets; none had positive '
    'independent-test skill. This illustrates sampling uncertainty in CV, but does not estimate the '
    'network false-link rate: no network selection was run here.','',
    '![Mean predictive performance](learning-overview.png)','',
    'Error bars are descriptive, pointwise 95% t intervals for mean skill over 20 independent datasets. '
    'They are not simultaneous confidence bounds for a searched method recommendation. '
    'See cell-summary.csv for all cells, Wilson success intervals and paired mean differences.','',
    '## Frozen development decision','',
    'RF improved interaction mean skill over shipped GBM by 7.1 and 13.9 percentage points at the two '
    'N300 points, exceeding the prespecified 5-point threshold at both. Its largest additive/chain loss '
    'was 1.1 points, within the 5-point allowance. Thus RF is a candidate for the next structural study, '
    'not a validated new default. Broader GBM improved interactions by 3.1 and 9.9 points at N300, '
    'so it did not meet both required thresholds.','',
    'Next: compare current GBM with RF on fresh independent datasets using the actual B100 layer-consensus '
    'pipeline. Measure correct and false links, predictive loss after selection, and complete recursive '
    'path recovery. Freeze that structural protocol and budget before dispatch. Minimum sample-size guidance '
    'and any production-default change require further disjoint confirmation. No new study is dispatched by this closeout.','',
    '## Execution and retained evidence','',
    f'The workflow took {wall:.1f} wall minutes, peaked at {peak} allocated concurrent jobs, and used '
    f'{minutes:.1f} summed job minutes ({minutes/60:.2f} runner hours), including setup/transfer/audit. '
    'These are not CPU time or a controlled speedup estimate.','',
    f'The archive retains all {folds:,} outer-fold records and 960 final prediction sets. '
    f'There are {cv_rows:,} CV prediction rows and 1,920,000 independent-test prediction rows, '
    'but only 480,000 unique test participants: each of four methods predicts the same test people. '
    'CV repetitions likewise reuse participants. There are 20 independent datasets per cell, not 100 '
    'resampling replications or millions of independent study replicates.','',
    'Every task on Actions verified fold and final predictions by deterministic refits at saved settings. '
    'The local repeated audit regenerated data/splits, checked declared grid membership, tuning fit sizes, '
    'identities and arithmetic. It did not repeat all tree refits or independently retune to verify tuning winners.','',
    '[Actions run](https://github.com/imh-ds/open/actions/runs/38133443512), '
    '[frozen protocol](../../fresh_learning_protocol.md), '
    '[original Actions report](REPORT-from-actions.md).','',
    f'Raw evidence: actions-evidence.zip, {receipt["archive_bytes"]:,} bytes; SHA256 `{checksum}`. '
    'Archive CRC verification passed. Manifest, execution metadata, receipt and original Actions audit '
    'are preserved here. The ZIP retains complete per-task receipts, fold checkpoints, results, verification '
    'receipts and independent-test predictions; it survives Actions artifact expiry.']
(output/'REPORT.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
print(json.dumps(receipt,indent=2))
