from pathlib import Path
import numpy as np,pandas as pd,matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
here=Path(__file__).resolve().parent;r=pd.read_csv(here/'case_metrics.csv');target=r[r.pair=='x1:x2'];fig,axes=plt.subplots(1,2,figsize=(12,4.9),layout='constrained');colors=('#c78b36','#2166ac')
ax=axes[0];x=np.arange(2);null=target[target['mode']=='conservative']
for i,(col,label,color) in enumerate(zip(('independent_gain','independent_matched_gain'),('Original overall gain','Matched joint-block gain'),colors)):
 counts=[int(((null.scenario==s)&(null[col]>.02)).sum()) for s in ('noise','additive')];bars=ax.bar(x+(i-.5)*.32,counts,.32,label=label,color=color);ax.bar_label(bars)
ax.set_xticks(x,('Noise only','Individual effects only'));ax.set_ylim(0,8);ax.set_ylabel('Independent gains > .02, out of 60');ax.set_title('Misleading credit removed\nRestrained method; fixed target pair');ax.legend(fontsize=9)
ax=axes[1];data=target[(target['mode']=='research')&(target.n>=300)];x=np.arange(4)
for i,(col,label,color) in enumerate(zip(('independent_gain','independent_matched_gain'),('Original overall gain','Matched joint-block gain'),colors)):
 groups=((0.,'pure'),(0.,'mixed'),(.7,'pure'),(.7,'mixed'));counts=[int(((data.rho==rho)&(data.scenario==scenario)&(data[col]>.02)).sum()) for rho,scenario in groups];bars=ax.bar(x+(i-.5)*.32,counts,.32,label=label,color=color);ax.bar_label(bars)
ax.set_xticks(x,('Pure\nr=0','Mixed\nr=0','Pure\nr=.7','Mixed\nr=.7'));ax.set_ylim(0,18);ax.set_ylabel('Independent gains > .02, out of 15');ax.set_title('Useful interaction gains retained\nMinimum-CV method, tested N >=300 settings');ax.legend(fontsize=9)
fig.suptitle('Fresh development: a clearer interaction score, with remaining reliability limits\nDescriptive cutoff; no significance test or calibrated interaction certificate')
fig.savefig(here/'findings.png',dpi=160);plt.close(fig)
