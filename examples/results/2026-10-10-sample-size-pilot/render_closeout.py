import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import numpy as np

import sys
base = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(__file__).parent
rows = {p.parent.name:json.loads(p.read_text()) for p in base.glob('*/result.json')}
assert len(rows) == 12
scenarios = ('noise','additive','interaction','chain')
points = ((80,5),(300,5),(300,15))
colors = ['#d9ece3','#f4cccc','#ffe4b5','#e2e5e9']
values, labels = [], []
for scenario in scenarios:
    vr, lr = [], []
    for n,p in points:
        r = rows[f'{scenario}-n{n}-p{p}']
        s = r['structural']
        if r['status'] != 'complete':
            v,label = 3,'Budget stop\nNo final result'
        elif s['false_root_inputs']:
            v,label = 2,'Extra noise link\n2 chain links missed'
        elif not s['root_recovered']:
            v,label = 1,'Both interaction\nlinks missed'
        else:
            v,label = 0,('Correctly empty' if scenario == 'noise' else
                'All 3 chain links' if scenario == 'chain' else 'Correct root link')
        vr.append(v);lr.append(label)
    values.append(vr);labels.append(lr)
fig,axes = plt.subplots(2,1,figsize=(11,11),gridspec_kw={'height_ratios':[1,1.35]},layout='constrained')
fig.suptitle('Sample-size feasibility pilot: 12 individual examples\nOne dataset per condition; these are not reliability estimates',fontsize=15)
ax = axes[0]
ax.imshow(values,cmap=ListedColormap(colors),vmin=0,vmax=3,aspect='auto')
ax.set_xticks(range(3),['80 people / 5 predictors','300 people / 5 predictors','300 people / 15 predictors'])
ax.set_yticks(range(4),['Noise only','Linear relationship','Pure interaction','Three-step chain'])
for i in range(4):
    for j in range(3): ax.text(j,i,labels[i][j],ha='center',va='center',fontsize=11,color='#263238')
ax.set_title('Selected links compared with the declared simulation targets',pad=12)
for spine in ax.spines.values(): spine.set_visible(False)
ax.tick_params(length=0)
names = [f'{s}-n{n}-p{p}' for s in scenarios for n,p in points]
mins = [rows[name]['elapsed_seconds']/60 for name in names]
ax = axes[1]
ax.barh(range(12),mins,color=['#bd7330' if rows[name]['status']!='complete' else '#3979a4' for name in names])
ax.set_yticks(range(12),names);ax.invert_yaxis()
ax.set_xlim(0,108);ax.set_xlabel('Case computation time (minutes; four assessment workers)')
ax.axvline(90,color='#bd7330',linestyle='--',linewidth=1)
for i,m in enumerate(mins): ax.text(m+1,i,f'{m:.1f}',va='center',fontsize=10)
ax.set_title('11 cases finished; the 15-predictor chain reached the 90-minute soft budget',pad=12)
ax.spines[['top','right']].set_visible(False)
fig.savefig(base/'pilot-overview.png',dpi=150)
plt.close(fig)

lines = ['# Sample-size feasibility pilot: findings','',
    'GitHub Actions run [38099118007](https://github.com/imh-ds/open/actions/runs/38099118007) finished successfully.',
    '**11 of 12 datasets completed. The 300-person, 15-predictor chain stopped at the soft time budget.**',
    'The successful workflow includes explicit incomplete-case reporting; it does not mean every analysis completed.','',
    '## What the examples showed','',
    '- All three linear examples retained the intended predictor without an extra root link.',
    '- All three noise examples retained no links.',
    '- All three pure-interaction examples retained neither required predictor.',
    '- The 300-person, five-predictor chain recovered all three expected links and depths.',
    '- The 80-person chain recovered its required direct predictor, also retained an unrelated predictor, and missed both indirect links.',
    '- The 300-person, 15-predictor chain is incomplete. It is not a statistical failure or a validated recovery.','',
    '![Pilot overview and runtime](pilot-overview.png)','',
    '## Prediction and structure are different','',
    'In the 300-person, five-predictor interaction example, the all-input model had independent-test skill .172,',
    'but each required predictor was promoted in only 49/100 assessments; 51 assessments stopped for low gain.',
    'Both fell below the unchanged 70% retention requirement, leaving an empty selected model with skill zero.',
    'In the other interaction examples the all-input model itself predicted poorly: skill âˆ’.090 at N80/p5',
    'and .007 at N300/p15. Those cases are not evidence of a strong fitted model whose links were merely filtered out.',
    'The population oracle root R-squared is .60 in the signal scenarios. Independent-test oracle skills vary',
    'with test draws. The interaction problem requires examining learning behavior as well as sample size.','',
    'The unrelated x4 predictor in the 80-person chain was promoted in 89/100 assessments and retained.',
    'This is a false root link under the declared data generator. Repeated subsamples reuse the same dataset:',
    'stability does not guarantee a genuine population relationship. Its cause has not been diagnosed here.','',
    '## Case-level evidence','',
    '| Case | Complete | Full-model test skill | Selected-model test skill | Oracle test skill | Seconds |',
    '|---|---|---:|---:|---:|---:|']
for name in names:
    r = rows[name];sc = r['scores']
    scores = [f"{sc[k]['skill']:.3f}" if sc else 'unavailable' for k in ('full','retained','oracle')]
    lines.append(f"| {name} | {r['status']} | {' | '.join(scores)} | {r['elapsed_seconds']:.1f} |")
lines += ['', '## Feasibility and limits','',
    'Case computation totaled 571.9 minutes (9.53 runner hours), excluding setup, artifact transfer and audit.',
    'The full workflow ran about four hours with three concurrent case jobs.',
    'Completed five-predictor root jobs took approximately 14â€“26 minutes; 15-predictor root jobs took 80â€“86 minutes.',
    'The completed five-predictor chains took 47 and 63 minutes. Increasing concurrent case jobs shortens scheduling',
    'delay but does not make an individual recursive case fit inside its deadline.','',
    'Local model/tuning fits used 51â€“52/34â€“35 people for N80 and 192/128 people for N300.',
    'The incomplete chain preserved 116 assessments: 100 at the root and 16 in the next comparison.',
    'Overall, 1,616 assessment checkpoints and 22,000 independent test predictions were saved;',
    'no test predictions or final structural scores were accepted for the incomplete case.',
    'Recorded warning count: zero. OS worker process-lifetime memory peaks were about 132â€“134 MB;',
    'these cannot be summed into a simultaneous whole-job memory estimate.','',
    'This pilot has one dataset per condition, one engine, one fold configuration and clean continuous data.',
    'Sample size, predictor count and seed differ across examples; these are not controlled estimates of factor effects.',
    'It establishes no minimum N, reliable false-positive rate, optimal fold count or complete scientific validation.','',
    '## Recommended next step','',
    'Before dispatching a large guidance grid, investigate the pure-interaction learning/selection behavior and',
    'the stable false root link using targeted, recorded development diagnostics. Use fresh multi-dataset',
    'confirmation for any proposed engine/grid/selection change. Separately budget a checkpointed completion',
    'of the unfinished chain if needed; do not silently rerun or lower B100.',
    'The larger sample-size guidance study remains pending. Production settings were unchanged.','',
    'The original Actions report is preserved as [REPORT-from-actions.md](REPORT-from-actions.md).',
    'See [audit.json](audit.json), the frozen [manifest.json](manifest.json), and each case directory for raw evidence.']
(base/'REPORT.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
