"""Create an applied summary figure and case table from the audited results."""
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

BASE=Path(__file__).resolve().parent


if __name__=='__main__':
    result=json.loads((BASE/'audit/audit.json').read_text())
    assert result['complete']==result['planned']==110 and not result['integrity_errors']
    with (BASE/'cases.csv').open('w',newline='',encoding='utf-8') as stream:
        writer=csv.writer(stream)
        writer.writerow(['dataset','participants','execution','complete','any_false_finding',
                         'retained_arrows','assessments','observation_checks','warnings'])
        for row in result['cases']:
            writer.writerow([row['spec']['id'],row['spec']['n'],
                'continued' if 'continuation_manifest' in row else 'original',
                row['status']=='complete',row['structure']['incorrect'],len(row['edges']),
                row['assessments'],row['observation_checks'],row['warnings']])
    fig,ax=plt.subplots(figsize=(7,4.6))
    for i,n in enumerate((80,200)):
        cell=result['cells'][f'noise-n{n}']; bound=cell['incorrect']
        rate=100*bound['count']/cell['planned']
        ax.errorbar(i,rate,yerr=[[rate-100*bound['lower']],[100*bound['upper']-rate]],
                    fmt='o',color='#225ea8',markersize=9,capsize=7,linewidth=2)
        ax.annotate(f"{bound['count']}/55 datasets ({rate:.1f}%)",(i,rate),
                    xytext=(0,12),textcoords='offset points',ha='center',fontsize=11)
    ax.axhline(10,color='#a63603',linestyle='--',linewidth=1.4,label='Prespecified upper-bound ceiling: 10%')
    ax.set(xticks=(0,1),xticklabels=('80 participants','200 participants'),xlim=(-.6,1.6),
           ylim=(-.6,12),yticks=(0,2,4,6,8,10,12),ylabel='Datasets with any false finding (%)',
           title='Noise-only validation: 110 completed datasets')
    ax.spines[['top','right']].set_visible(False)
    ax.legend(loc='upper center',frameon=False,fontsize=9)
    fig.text(.5,.025,'Dots: observed rates. Whiskers: exact 95% intervals per condition.\n'
             'Passing this check does not establish detection of real links.',ha='center',fontsize=9)
    fig.tight_layout(rect=(0,.1,1,1))
    fig.savefig(BASE/'noise-check.png',dpi=160)
    plt.close(fig)
