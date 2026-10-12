from pathlib import Path
import json,numpy as np,pandas as pd,matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
here=Path(__file__).resolve().parent;r=pd.read_csv(here/'case_metrics.csv');f=pd.read_csv(here/'fold_metrics.csv');modes=('v2','research','conservative');labels=('Production v2','Minimum CV','Restrained rule');colors=('#708090','#2166ac','#c78b36')
fig,axes=plt.subplots(1,2,figsize=(11,4.7),layout='constrained')
ax=axes[0];data=f[(f.n==80)&(f.pair=='x1:x2')];counts=[int(((data['mode']==m)&(data.gain<-1)).sum()) for m in modes];bars=ax.bar(labels,counts,color=colors);ax.bar_label(bars);ax.set_ylim(0,4);ax.set_ylabel('Outer folds with incremental skill below -1');ax.set_title('Extreme small-N failures decreased\n180 outer folds per method, N=80');ax.tick_params(axis='x',labelrotation=15)
ax=axes[1];data=r[(r.n>=300)&(r.rho==.7)&(r.pair=='x1:x2')];x=np.arange(2)
for i,(m,label,color) in enumerate(zip(modes,labels,colors)):
 counts=[int(((data['mode']==m)&(data.scenario==s)&(data.independent_gain>.02)).sum()) for s in ('pure','mixed')];bars=ax.bar(x+(i-1)*.24,counts,.24,label=label,color=color);ax.bar_label(bars)
ax.set_xticks(x,('Pure interaction','Main effects + interaction'));ax.set_ylim(0,18);ax.set_ylabel('Independent-test gain > .02, out of 15');ax.set_title('Useful interaction gains were also lost\nCorrelation .7; tested N/p settings with N >=300');ax.legend(loc='upper right',fontsize=8)
fig.suptitle('Fresh development: a trade-off, not a satisfactory default\nFixed x1:x2 comparisons; descriptive cutoffs do not certify interactions')
fig.savefig(here/'findings.png',dpi=160);plt.close(fig)
