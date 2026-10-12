"""Rebuild the descriptive figure from retained dataset-level summaries."""
import json
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

root=Path(__file__).resolve().parent
a=json.loads((root/'audit.json').read_text())
d=json.loads((root/'findings-data.json').read_text())
cells=[c for c in a['cells'] if c['true_complete'] is not None]
fig,axes=plt.subplots(1,2,figsize=(15,8),layout='constrained')
labels=[]
for i,c in enumerate(cells):
    rows=[r for r in d['per_dataset'] if (r['scenario'],r['n'],r['p'])==(c['scenario'],c['n'],c['p'])]
    complete=[r['true_pair']['complete_gain'] for r in rows if r['true_pair']['complete_gain'] is not None]
    conditional=[r['true_pair']['conditional_gain'] for r in rows if r['true_pair']['complete_gain'] is None and r['true_pair']['conditional_gain'] is not None]
    for values,color,marker,offset,label in [(complete,'#245e9c','o',-.1,'All three folds assessed'),(conditional,'#d78820','x',.12,'Only selected folds assessed')]:
        clipped=np.clip(values,-.3,.8)
        axes[0].scatter(clipped,[i+offset]*len(values),c=color,marker=marker,s=26,label=label if i==0 else None)
        for v,x in zip(values,clipped):
            if v!=x:axes[0].annotate(f'{v:.2f}',(x,i+offset),xytext=(7,5),textcoords='offset points',fontsize=8,color=color)
    axes[1].barh(i-.16,c['true_fixed_details'],height=.3,color='#bfc8d1',label='Fixed grid' if i==0 else None)
    axes[1].barh(i+.16,c['true_adaptive_details'],height=.3,color='#245e9c',label='Adaptive grid' if i==0 else None)
    name={'pure':'Pure joint effect','mixed':'Separate + joint effects','correlated_mixed':'Correlated inputs + joint effect'}[c['scenario']]
    labels.append(f"{name}\nN={c['n']}, p={c['p']}; complete gains {c['true_complete']}/10")
for ax in axes:
    ax.set_yticks(range(len(cells)),labels);ax.invert_yaxis();ax.legend(loc='lower right',fontsize=8)
axes[0].axvline(0,color='black',linewidth=.8);axes[0].set_xlim(-.35,.8)
axes[0].set_xlabel('True-pair held-out skill improvement\nValues below -0.30 appear at the left boundary with exact labels')
axes[1].set_xlim(0,30);axes[1].set_xlabel('True-pair detail views supported / 30 planned folds per condition')
fig.suptitle('Completed development study: prediction checks and plot availability\nPositive gain is provisional; missing comparisons remain unknown',fontsize=14)
fig.savefig(root/'FINDINGS.png',dpi=150);plt.close(fig)
