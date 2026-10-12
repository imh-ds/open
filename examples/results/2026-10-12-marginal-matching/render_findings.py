from pathlib import Path
import pandas as pd,numpy as np,matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
here=Path(__file__).resolve().parent;r=pd.read_csv(here/'metrics.csv');r=r[(r.scenario=='noise')&(r['mode']=='conservative')&(r.original_gain>.02)].sort_values('n')
fig,ax=plt.subplots(figsize=(9,4.6),layout='constrained');x=np.arange(3)
a=ax.bar(x-.18,r.original_gain,.36,color='#c78b36',label='Original comparison');b=ax.bar(x+.18,r.matched_joint_gain,.36,color='#2166ac',label='Interaction contribution with matched settings')
ax.bar_label(a,fmt='%.4f',padding=4);ax.bar_label(b,fmt='%.6f',padding=4,fontsize=9);ax.axhline(0,color='black',lw=.6);ax.set_ylim(-.005,.065);ax.set_xticks(x,[f'N={int(v.n)}, correlation {v.rho:g}' for v in r.itertuples()]);ax.set_ylabel('Prediction gain (same scale as original study)');ax.set_title('Three noise-only cases: the improvement came from the individual-variable fit\nThe interaction itself added essentially nothing');ax.legend(fontsize=9,loc='upper left');fig.savefig(here/'findings.png',dpi=160);plt.close(fig)
