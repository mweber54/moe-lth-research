"""Refine the newly generated combined heatmap's colorbar layout only."""
import os
from pathlib import Path
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
O=Path(os.environ.get('DATASET_A_ANALYSIS_OUT',str(Path(__file__).resolve().parents[1])))
cs=pd.read_csv(O/'plot_heatmap_data.csv');E=[8,12,16,24,32];A=list(range(0,101,10));limit=max(abs(cs.mean_gap.min()),abs(cs.mean_gap.max()))
fig,axs=plt.subplots(1,5,figsize=(19,5))
for ax,e in zip(axs,E):
 h=cs[cs.num_experts==e].pivot(index='sparsity',columns='router_age',values='mean_gap');im=ax.imshow(h,aspect='auto',cmap='coolwarm',vmin=-limit,vmax=limit);ax.set_xticks(range(11),A,rotation=90);ax.set_yticks(range(len(h)),[f'{x:.0%}' for x in h.index]);ax.set(title=f'{e} experts',xlabel='Router age (%)')
axs[0].set_ylabel('Expert weight sparsity');fig.subplots_adjust(left=.05,right=.88,wspace=.32,bottom=.16,top=.84);cbax=fig.add_axes([.91,.20,.012,.60]);fig.colorbar(im,cax=cbax,label='Mean sparse − dense CE (nats/token)');fig.suptitle('Mean paired CE gap, n=5 source seeds; zero = dense match',y=.97)
for ext in ['pdf','png']:fig.savefig(O/'figures'/f'combined_gap_heatmaps.{ext}',dpi=180,bbox_inches='tight')
plt.close(fig)
