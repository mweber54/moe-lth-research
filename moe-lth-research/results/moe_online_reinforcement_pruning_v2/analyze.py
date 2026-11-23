"""Analyze only independently verified completed branches."""
import json,statistics,os,sys
from pathlib import Path
try:
 import matplotlib
except ModuleNotFoundError:
 python=Path(__file__).resolve().parents[2]/'.venv'/'bin'/'python'
 os.execv(str(python),[str(python),str(Path(__file__).resolve()),*sys.argv[1:]])
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import status
HERE=Path(__file__).resolve().parent

def main():
 rows=[];steps=('post_mask_250','251','275','500','1000','2500')
 fig,ax=plt.subplots(figsize=(9,5))
 for seed in (7,17,29):
  for order in ('A','B'):
   root=HERE/'branches'/f'seed{seed}_{order}'
   for arm in status.ARMS:
    if status.status(root/arm)!='COMPLETED':raise RuntimeError(f'unverified branch: {root/arm}')
   losses={arm:json.loads((root/arm/'complete.json').read_text())['losses'] for arm in status.ARMS}
   r=losses['random-small'];h=losses['high-reinforcement-small'];u=losses['untouched'];curve=[h[s]-r[s] for s in steps]
   row={'seed':seed,'order':order,'delta':curve[-1],'high_excess_untouched':h['2500']-u['2500'],'random_excess_untouched':r['2500']-u['2500'],'high_minus_random_trajectory':dict(zip(steps,curve)),'high_excess_trajectory':{s:h[s]-u[s] for s in steps},'random_excess_trajectory':{s:r[s]-u[s] for s in steps}};rows.append(row)
   ax.plot(range(len(steps)),curve,marker='o',label=f'{seed} {order}')
 means={str(seed):statistics.mean(r['delta'] for r in rows if r['seed']==seed) for seed in (7,17,29)};values=list(means.values());sd=statistics.stdev(values)
 report={'cells':rows,'seed_means':means,'grand_mean':statistics.mean(values),'seed_mean_sample_sd':sd,'paired_standardized_effect':statistics.mean(values)/sd if sd else None,'cell_directions':{'positive':sum(r['delta']>0 for r in rows),'negative':sum(r['delta']<0 for r in rows),'zero':sum(r['delta']==0 for r in rows)},'seed_directions':{'positive':sum(v>0 for v in values),'negative':sum(v<0 for v in values),'zero':sum(v==0 for v in values)}}
 (HERE/'analysis.json').write_text(json.dumps(report,indent=2)+'\n')
 ax.axhline(0,color='gray',linewidth=.8);ax.set_xticks(range(len(steps)),['immediate','251','275','500','1000','2500']);ax.set_xlabel('Global step');ax.set_ylabel('Validation CE: high minus random');ax.legend(ncol=3);fig.tight_layout();fig.savefig(HERE/'paired_differences.png',dpi=180)
 print(json.dumps(report,indent=2))
if __name__=='__main__':main()
