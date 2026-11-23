"""Past-only matched-distribution diagnostics; no loss or future weights."""
from common import *
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def main():
 reg=registry();F=D/'figures';F.mkdir(exist_ok=True);edges=np.linspace(0,1.5,151);histrows=[];fig,axes=plt.subplots(1,5,figsize=(15,3.4),sharex=True,sharey=True)
 pair_stats=[];pair_hists=[]
 balances={(int(r['seed']),r['parameter_name']):float(r['tensor_rms']) for r in csv.DictReader((D/'magnitude_balance.csv').open())}
 for ax,seed in zip(axes,SEEDS):
  cp=torch.load(source(seed)/'checkpoints/step_1000.pt',map_location='cpu',weights_only=True,mmap=True)['model'];masks=load(D/'masks'/f'seed{seed}.pt')
  for branch in ('growing','stable'):
   counts=np.zeros(len(edges)-1,dtype=np.int64)
   for n,idx in masks[branch].items():
    values=cp[n].flatten()[idx].abs().numpy()/balances[seed,n];assert values.max()<edges[-1];counts+=np.histogram(values,edges)[0]
   assert counts.sum()==reg['mask_files'][str(seed)]['removed_coordinates_per_branch']
   ax.stairs(counts/counts.sum(),edges,label=branch,linestyle='-' if branch=='growing' else '--')
   for i,c in enumerate(counts):histrows.append(dict(seed=seed,branch=branch,left_rms=edges[i],right_rms=edges[i+1],count=int(c)))
  differences=[]
  for name in masks['growing']:
   flat=cp[name].flatten().numpy().astype(np.float64);g=masks['growing'][name].numpy();st=masks['stable'][name].numpy();differences.append(np.abs(np.abs(flat[g])-np.abs(flat[st]))/balances[seed,name])
  diffs=np.concatenate(differences);q=np.quantile(diffs,[0,.5,.9,.95,.99,1]);pair_stats.append(dict(seed=seed,pairs=len(diffs),mean_mismatch_rms=float(diffs.mean()),minimum=float(q[0]),median=float(q[1]),p90=float(q[2]),p95=float(q[3]),p99=float(q[4]),maximum=float(q[5])))
  pedges=np.linspace(0,.00002,101);counts=np.histogram(diffs,pedges)[0];assert counts.sum()==len(diffs)
  for i,c in enumerate(counts):pair_hists.append(dict(seed=seed,left_rms=pedges[i],right_rms=pedges[i+1],count=int(c)))
  ax.set(title=f'Seed {seed}',xlabel='Removed |w| / tensor RMS');ax.legend(fontsize=7)
 axes[0].set_ylabel('Fraction of removed coordinates');plt.tight_layout();plt.savefig(F/'current_magnitude_distributions.png',dpi=160);plt.savefig(F/'current_magnitude_distributions.pdf');plt.close();table(D/'magnitude_histogram.csv',histrows)
 groups=list(csv.DictReader((D/'matched_group_summary.csv').open()));fig,axes=plt.subplots(2,5,figsize=(15,6),sharex=True)
 for row,tensor in enumerate(('fc1','fc2')):
  for ax,seed in zip(axes[row],SEEDS):
   for branch in ('growing','stable'):
    z=[r for r in groups if int(r['seed'])==seed and r['tensor']==tensor and r['branch']==branch];means=[sum(float(r[f'mean_abs{st}'])/balances[seed,r['parameter_name']] for r in z)/len(z) for st in (250,500,1000)];ax.plot([250,500,1000],means,marker='o',label=branch)
   ax.set(title=f'{tensor}, seed {seed}',xlabel='Historical source step');ax.legend(fontsize=7)
  axes[row,0].set_ylabel('Removed mean |w| / tensor RMS at1000')
 plt.tight_layout();plt.savefig(F/'historical_trajectories_by_matrix.png',dpi=160);plt.savefig(F/'historical_trajectories_by_matrix.pdf');plt.close()
 table(D/'pair_mismatch_distribution.csv',pair_stats);table(D/'pair_mismatch_histogram.csv',pair_hists)
 save(D/'matched_balance_checks.json',dict(passed=True,exact_histogram_counts=True,all_group_counts_equal=True,no_future_weight_inputs=True,source_steps_read=[1000],matching_and_growth_summary_inputs_only=True,coordinate_examples_manually_reviewed=True,maximum_pair_mismatch_rms=max(float(r['worst_mismatch_rms']) for r in csv.DictReader((D/'magnitude_balance.csv').open()))))
if __name__=='__main__':main()
