"""Render the controlled-strength comparison and the four qualified conditions."""
from pathlib import Path
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter
folder=Path(__file__).resolve().parent
audit=json.loads((folder/'actions-audit.json').read_text())
moderate=[r for r in audit['strata'] if r['benchmark']=='matched-q0.04']
qualified=[(r,c) for r,c in zip(audit['strata'],audit['conditions']) if c['predictive_support']]
fig,(left,right)=plt.subplots(1,2,figsize=(13,5.6),gridspec_kw={'width_ratios':[1.1,1]})
labels=['300 people · 5 predictors\nDistinct predictors','300 people · 5 predictors\nStrongly correlated predictors','600 people · 5 predictors\nDistinct predictors','600 people · 5 predictors\nStrongly correlated predictors','300 people · 15 predictors\nStrongly correlated predictors']
for i,r in enumerate(moderate):
    value=r['detection']['k']/450;bound=r['detection']['bound'];color='#277c69' if r['detection']['pass'] else '#687b91'
    left.barh(i,value,color=color,height=.57)
    left.plot([bound,value],[i,i],color='#1d2937',lw=2);left.plot(bound,i,'|',color='#1d2937',ms=13)
    left.text(value+.018,i,f'{value:.0%}',va='center',fontsize=11)
left.set_yticks(range(5),labels);left.invert_yaxis();left.axvline(.8,color='#a84532',ls='--',lw=1.3)
left.set_xlim(0,1.11);left.xaxis.set_major_formatter(PercentFormatter(1));left.set_xlabel('True interactions detected')
left.set_title('More data helps; predictor overlap still hurts\nSame distinct contribution: 4% of outcome variance',fontsize=12,pad=16)
left.text(.02,-.28,'Black marks: conservative lower bounds.\nTo qualify, the bound must exceed 80%.',transform=left.transAxes,fontsize=9,color='#445160')
names=[]
for i,(r,c) in enumerate(qualified):
    names.append(r['cell'].replace('n300-p5-rho0','300 people · distinct').replace('n600-p5-rho0.7','600 people · correlated').replace('n600-p5-rho0','600 people · distinct')+'\n'+{'mixed-b0.8':'Strong mixed interaction','matched-q0.04':'Moderate mixed interaction','pure-b0.8':'Pure interaction'}[r['benchmark']])
    value=r['shape']['k']/r['shape']['n'];bound=r['shape']['bound']
    right.plot([bound,value],[i,i],color='#277c69',lw=2);right.plot(value,i,'o',color='#277c69',ms=7)
    right.plot(bound,i,'|',color='#1d2937',ms=12);right.text(value+.012,i,f"{r['shape']['k']}/{r['shape']['n']}",va='center',fontsize=9)
right.set_yticks(range(4),names);right.invert_yaxis();right.set_xlim(.88,1.07);right.xaxis.set_major_formatter(PercentFormatter(1));right.axvline(.9,color='#a84532',ls='--')
right.set_xticks([.9,.95,1.]);right.set_xlabel('Displayed patterns resembling\nthe known pattern');right.set_title('Plots performed well in the 4 qualified settings\nAccuracy is conditional on a plot being displayed',fontsize=12,pad=16)
right.text(.02,-.28,'Relative shape/direction only. Effect-size magnitude\nand complete surfaces are not certified.',transform=right.transAxes,fontsize=9,color='#445160')
for ax in (left,right):
    ax.spines[['top','right']].set_visible(False);ax.grid(axis='x',alpha=.15);ax.set_axisbelow(True)
fig.suptitle('Interaction validation: useful in some conditions, not reliably across all conditions',fontsize=15,y=.99)
fig.subplots_adjust(left=.19,right=.96,wspace=1.1,top=.80,bottom=.27)
fig.savefig(folder/'RESULTS.png',dpi=160,bbox_inches='tight');plt.close(fig)
