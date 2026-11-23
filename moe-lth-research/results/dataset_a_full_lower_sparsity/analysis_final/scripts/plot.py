"""Replot exported scientific tables without rehashing/reloading checkpoints.
DATASET_A_ANALYSIS_OUT selects input analysis; DATASET_A_PLOT_OUT selects a fresh output tree.
Default output: analysis_final/replotted/figures. Existing figures are refused.
"""
import os,json
from pathlib import Path
import numpy as np,pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
I=Path(os.environ.get('DATASET_A_ANALYSIS_OUT',str(Path(__file__).resolve().parents[1])))
O=Path(os.environ.get('DATASET_A_PLOT_OUT',str(I/'replotted')))
E=[8,12,16,24,32];SP=[.5,.55,.6,.65];A=list(range(0,101,10))
TH={'strict':0.,'ppl_1pct':np.log(1.01),'ppl_2pct':np.log(1.02),'ppl_5pct':np.log(1.05)}
fnew=pd.read_csv(I/'canonical_lower_sparsity.csv');ss=pd.read_csv(I/'setting_summary.csv');cs=pd.read_csv(I/'combined_setting_summary.csv')
def csv(*args):pass
# Exact plot inputs saved, all uncertainty is across five paired seeds.
(O/'figures').mkdir(parents=True,exist_ok=True)
plotdata=fnew.groupby(['num_experts','router_age','sparsity']).agg(mean_gap=('ticket_gap','mean'),sd_gap=('ticket_gap','std'),mean_sparse=('sparse_final_validation_loss','mean'),sd_sparse=('sparse_final_validation_loss','std'),mean_dense=('dense_final_validation_loss','mean')).reset_index();csv('plot_curve_data.csv',plotdata)
def save(name):
 if name!='combined_gap_heatmaps':plt.tight_layout()
 for ext in ['pdf','png']:
  p=O/'figures'/f'{name}.{ext}'
  if p.exists():raise FileExistsError(p)
  plt.savefig(p,dpi=180,bbox_inches='tight')
 plt.close()
for kind in ['gap','absolute']:
 fig,axs=plt.subplots(1,5,figsize=(19,3.9))
 for ax,e in zip(axs,E):
  for sp in SP:
   h=plotdata[(plotdata.num_experts==e)&(plotdata.sparsity==sp)];y=h.mean_gap if kind=='gap' else h.mean_sparse;sd=h.sd_gap if kind=='gap' else h.sd_sparse;ax.plot(h.router_age,y,'.-',label=f'{sp:.0%}');ax.fill_between(h.router_age,y-sd,y+sd,alpha=.12)
  if kind=='gap':ax.axhline(0,color='k',lw=.8)
  else:
   h=plotdata[(plotdata.num_experts==e)&(plotdata.sparsity==.5)];ax.plot(h.router_age,h.mean_dense,'k--',label='Dense')
  ax.set(title=f'{e} experts',xlabel='Router age (%)',ylabel='Sparse − dense CE (nats/token)' if kind=='gap' else 'Endpoint CE (nats/token)')
 axs[-1].legend();fig.suptitle('Mean ± sample SD; n=5 source seeds per point');save('ticket_gap_vs_age' if kind=='gap' else 'absolute_endpoints')
csv('plot_heatmap_data.csv',cs)
fig,axs=plt.subplots(1,5,figsize=(19,5));limit=max(abs(cs.mean_gap.min()),abs(cs.mean_gap.max()))
for ax,e in zip(axs,E):
 h=cs[cs.num_experts==e].pivot(index='sparsity',columns='router_age',values='mean_gap');im=ax.imshow(h,aspect='auto',cmap='coolwarm',vmin=-limit,vmax=limit);ax.set_xticks(range(11),A,rotation=90);ax.set_yticks(range(len(h)),[f'{x:.0%}' for x in h.index]);ax.set(title=f'{e} experts',xlabel='Router age (%)');axs[0].set_ylabel('Expert weight sparsity')
fig.subplots_adjust(left=.05,right=.88,wspace=.32,bottom=.16,top=.84);cbax=fig.add_axes([.91,.20,.012,.60]);fig.colorbar(im,cax=cbax,label='Mean sparse − dense CE (nats/token)');fig.suptitle('Mean paired CE gap, n=5 source seeds; zero = dense match',y=.97);save('combined_gap_heatmaps')
fig,axs=plt.subplots(4,5,figsize=(19,12))
for i,name in enumerate(TH):
 for j,e in enumerate(E):
  h=ss[ss.num_experts==e].pivot(index='sparsity',columns='router_age',values=name+'_passing_seeds');ax=axs[i,j];ax.imshow(h,vmin=0,vmax=5,cmap='YlGn');ax.set_xticks(range(11),A,rotation=90);ax.set_yticks(range(4),['50%','55%','60%','65%']);ax.set_title(f'{e}E, {name}')
  for y in range(4):
   for x in range(11):ax.text(x,y,str(int(h.iloc[y,x])),ha='center',va='center',fontsize=8)
fig.suptitle('Matching seeds / 5; strict zero gap and exploratory perplexity tolerances');save('matching_seed_counts')
chosen=ss.sort_values('mean_gap').head(5);seedplot=[]
fig,axs=plt.subplots(1,5,figsize=(19,4))
for ax,r in zip(axs,chosen.to_dict('records')):
 g=fnew[(fnew.num_experts==r['num_experts'])&(fnew.sparsity==r['sparsity'])]
 for s,h in g.groupby('seed'):ax.plot(h.router_age,h.ticket_gap,'.-',label=str(s));seedplot.extend(h.to_dict('records'))
 ax.axhline(0,color='k',lw=.8);ax.axvline(r['router_age'],color='k',ls=':',lw=.8);ax.set(title=f"{r['num_experts']}E, {r['sparsity']:.0%}; selected R{r['router_age']}",xlabel='Router age (%)',ylabel='Paired CE gap')
axs[-1].legend(title='Seed');fig.suptitle('Exploratory strongest mean settings; all ages shown to expose selection');save('strongest_candidates_seed_views');csv('plot_seed_data.csv',seedplot)
# Every architecture/sparsity seed curve supports candid interior-optimum inspection.
fig,axs=plt.subplots(5,4,figsize=(16,16),sharex=True)
for i,e in enumerate(E):
 for j,sp in enumerate(SP):
  ax=axs[i,j]
  for s,h in fnew[(fnew.num_experts==e)&(fnew.sparsity==sp)].groupby('seed'):ax.plot(h.router_age,h.ticket_gap,'.-',label=str(s),alpha=.7)
  ax.axhline(0,color='k',lw=.7);ax.set(title=f'{e}E, {sp:.0%}',xlabel='Router age (%)',ylabel='Paired CE gap')
axs[0,-1].legend(title='Seed');save('all_seed_age_curves')
