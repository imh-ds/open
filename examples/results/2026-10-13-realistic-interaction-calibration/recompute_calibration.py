"""Descriptive summaries of frozen null calibration, without new fits/cutoffs."""
from pathlib import Path
import json,sys
import numpy as np
ROOT=next(p for p in Path(__file__).resolve().parents if (p/'examples/realistic_interaction_study.py').is_file())
sys.path.insert(0,str(ROOT/'examples'))
import realistic_interaction_study as study
import realistic_interaction_verify as verify
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def main():
    out=ROOT/'examples/results/2026-10-13-realistic-interaction-calibration'
    ci=ROOT/'.superpowers/realistic-calibration-download/.superpowers/realistic-interaction'
    audit=study.io.read(out/'ci-audit.json' if (out/'ci-audit.json').exists() else ci/'audit.json');cutoffs=verify.verify_calibration(audit,study.identity())['thresholds']
    tasks={c['id']:c for c in study.d.stage_tasks('calibration')};rows=audit['rows'];summary=[]
    for cell in dict.fromkeys(c['cell'] for c in tasks.values()):
        cases=[r for r in rows if r['cell']==cell];task=tasks[cases[0]['id']]
        positive=sum(any(p['score']>0 for p in r['pairs']) for r in cases)
        maximum=max(cases,key=lambda r:max(p['score'] for p in r['pairs']))
        summary.append(dict(cell=cell,n=task['n'],gamma=task['gamma'],reliability=task['reliability'],observed_rho=task['rho'],
            positive_before_cutoff=positive,planned=200,rate_before_cutoff=positive/200,cutoff=cutoffs[cell],
            maximum_id=maximum['id'],maximum_null_class=maximum['benchmark'],
            exceeds_own_learned_cutoff=sum(any(p['score']>cutoffs[cell] for p in r['pairs']) for r in cases)))
    assert sum(r['positive_before_cutoff'] for r in summary)==1437 and all(r['exceeds_own_learned_cutoff']==0 for r in summary)
    study.io.write(out/'calibration-summary.json',dict(contexts=summary,positive_before_cutoff=1437,planned=3600,
        interpretation='Before-cutoff frequency is not the final-rule false-positive rate. No exceedance in calibration is guaranteed by construction, not independent validation.'))
    study.pd.DataFrame(summary).to_csv(out/'calibration-summary.csv',index=False)
    fig,axes=plt.subplots(1,2,figsize=(10,4.7),sharey=False)
    vmax=max(cutoffs.values())*100
    for ax,reliability in zip(axes,(1.,.8)):
        grid=np.array([[next(r['cutoff'] for r in summary if r['n']==n and r['gamma']==g and r['reliability']==reliability)*100
                        for n in (150,300,600)] for g in (.1,.3,.5)])
        im=ax.imshow(grid,cmap='YlOrRd',vmin=0,vmax=vmax)
        ax.set_xticks(range(3),(150,300,600));ax.set_yticks(range(3),[f'{g:.1f} / {reliability*g:.2f}' for g in (.1,.3,.5)])
        ax.set_xlabel('People per dataset');ax.set_title(f'Predictor reliability {reliability:.1f}')
        for i in range(3):
            for j in range(3):ax.text(j,i,f'{grid[i,j]:.2f}',ha='center',va='center',color='white' if grid[i,j]>vmax*.6 else 'black')
    axes[0].set_ylabel('Latent / observed predictor correlation')
    fig.suptitle('Calibration: the evidence bar generally falls with larger samples')
    fig.colorbar(im,ax=axes.ravel().tolist(),label='Cutoff: normalized predictive-gain score × 100',fraction=.035,pad=.035)
    fig.subplots_adjust(left=.13,right=.84,bottom=.18,top=.80,wspace=.28)
    fig.text(.5,.04,'Each cell is the largest null score; these are screening cutoffs, not effect sizes or verified false-positive rates.',ha='center',fontsize=9)
    fig.savefig(out/'calibration-cutoffs.png',dpi=160);plt.close(fig)
    print('Descriptive18-context summaries and graph generated; no fitting')


if __name__=='__main__':main()
