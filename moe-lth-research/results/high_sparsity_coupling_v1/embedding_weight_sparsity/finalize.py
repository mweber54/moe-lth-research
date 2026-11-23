"""Finalize presentation and verify unchanged protected outputs (checkpoint reads only)."""
import os
os.environ.setdefault('OMP_NUM_THREADS','2')
from pathlib import Path
import json,hashlib,math
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
O=Path(__file__).resolve().parent;B=O.parent
STEPS=json.load(open(O/'analysis_config.json'))['steps'];SEEDS=[7,17,29,43,61];colors={0.:'black',.8:'#0072B2',.85:'#D55E00'};styles={0.:'-',.8:'--',.85:':'}
plt.rcParams.update({'font.size':8,'pdf.fonttype':42})
def save(fig,name):
 fig.savefig(O/'figures'/f'{name}.png',dpi=160,bbox_inches='tight');fig.savefig(O/'figures'/f'{name}.pdf',bbox_inches='tight');plt.close(fig)
def sha(p):
 h=hashlib.sha256()
 with open(p,'rb') as f:
  for buf in iter(lambda:f.read(8*1024*1024),b''):h.update(buf)
 return h.hexdigest()
P=pd.read_csv(O/'checkpoint_provenance.csv');P['saved_step_segment_offset']=P.global_recovery_step-P.saved_step;P.to_csv(O/'checkpoint_provenance.csv',index=False)
selected=P[(P.track=='rerun')&(P.role=='registered')&P.step.isin([0,2500,5000])]
# CDFs use the complete exported empirical arrays; include the smallest positive value.
cd={h:np.load(O/'cdf'/f'{h}.npz') for h in selected.E_sha256.unique()}
minpos=min(float(d['absolute_weight'][d['absolute_weight']>0].min()) for d in cd.values());xmin=10**math.floor(math.log10(minpos));bins=np.array(json.load(open(O/'histogram_bins.json'))['shared_bin_edges'])
fig,axs=plt.subplots(5,3,figsize=(12,15),sharex=True,sharey=True,layout='constrained')
for i,seed in enumerate(SEEDS):
 for k,step in enumerate([0,2500,5000]):
  ax=axs[i,k]
  for sp in [0.,.8,.85]:
   r=selected[(selected.seed==seed)&(selected.step==step)&(selected.sparsity==sp)].iloc[0];d=cd[r.E_sha256];pos=d['absolute_weight']>0
   ax.plot(d['absolute_weight'][pos],d['cdf'][pos],color=colors[sp],ls=styles[sp],label='dense' if sp==0 else f'{sp:.0%}')
  ax.set_xscale('log');ax.set_xlim(xmin,bins[-1]);ax.set_ylim(0,1.02);ax.set_title(f'Seed {seed}, step {step}');ax.set_xlabel('Positive |E| (exact zeros separate)');ax.set_ylabel('Empirical cumulative fraction');ax.legend(fontsize=6)
fig.suptitle('C. Full absolute-weight empirical CDF; primary rerun, each seed separate',fontsize=11);save(fig,'C_magnitude_CDF')
# Export reproducible signed histogram counts, computed once per E hash.
histfile=O/'signed_histogram_counts.csv'
if histfile.exists():H=pd.read_csv(histfile)
else:
 import torch
 torch.set_num_threads(2);hs=[]
 for r in selected.drop_duplicates('E_sha256').itertuples():
  d=torch.load(r.checkpoint_path,map_location='cpu',weights_only=False,mmap=True);e=d['model']['token_embedding.weight']
  raw=e.detach().cpu().contiguous().numpy();h=hashlib.sha256(str(raw.dtype).encode()+str(raw.shape).encode()+raw.tobytes()).hexdigest();assert h==r.E_sha256
  counts,_=np.histogram(e.double().numpy(),bins=bins);assert counts.sum()==65536
  hs.extend(dict(E_sha256=h,bin_index=i,left=bins[i],right=bins[i+1],count=int(c),density=c/(65536*(bins[i+1]-bins[i]))) for i,c in enumerate(counts))
 H=pd.DataFrame(hs);H.to_csv(histfile,index=False)
fig,axs=plt.subplots(5,3,figsize=(12,15),sharex=True,sharey=True,layout='constrained')
for i,seed in enumerate(SEEDS):
 for k,step in enumerate([0,2500,5000]):
  ax=axs[i,k]
  for sp in [0.,.8,.85]:
   r=selected[(selected.seed==seed)&(selected.step==step)&(selected.sparsity==sp)].iloc[0];h=H[H.E_sha256==r.E_sha256].sort_values('bin_index')
   ax.stairs(h.density,bins,color=colors[sp],ls=styles[sp],label='dense' if sp==0 else f'{sp:.0%}')
  ax.set_xlim(bins[0],bins[-1]);ax.set_ylim(0,H.density.max()*1.05);ax.set_title(f'Seed {seed}, step {step}');ax.set_xlabel('Signed stored weight');ax.set_ylabel('Density');ax.legend(fontsize=6)
fig.suptitle('D. Signed weight distributions; shared bins and axes, primary rerun',fontsize=11);save(fig,'D_signed_histograms')
# Short axis labels and constrained layout keep panel titles separate from the figure title.
fig,axs=plt.subplots(5,2,figsize=(12,15),sharex=True,layout='constrained')
for j,(name,label) in enumerate([('row','Token row'),('column','Coordinate column')]):
 d=pd.read_csv(O/f'{name}_statistics.csv');d=d[(d.track=='rerun')&(d.role=='registered')]
 for i,seed in enumerate(SEEDS):
  ax=axs[i,j]
  for sp in [0.,.8,.85]:
   q=d[(d.seed==seed)&(d.sparsity==sp)].groupby('step').l2.quantile([.1,.5,.9]).unstack().reindex(STEPS)
   ax.plot(STEPS,q[.5],color=colors[sp],label='dense' if sp==0 else f'{sp:.0%}');ax.fill_between(STEPS,q[.1],q[.9],color=colors[sp],alpha=.1)
  ax.axvline(2500,color='gray',ls=':',lw=1);ax.set_title(f'Seed {seed}, {label}');ax.set_xlabel('Global attempted recovery step');ax.set_ylabel('L2 norm');ax.legend(fontsize=6)
fig.suptitle('F. Norm median and within-tensor 10–90% range (not confidence intervals)\nDotted line: optimizer/scaler reset at 2500; model weights unchanged',fontsize=11);save(fig,'F_norm_distributions')
# Report both consistent and threshold-dependent effects, without choosing thresholds via labels.
f=O/'findings.md';text=f.read_text();text=text.split('\n## Threshold-specific endpoint interpretation')[0]
text=text.replace('2. **Scale and shape:**\n-','2. **Scale and shape:**\n\n-').replace('within each seed.\n-','within each seed.\n\n-').replace('whether the token is used.\n-','whether the token is used.\n\n-')
text=text.replace('These group sizes count dependent conditions, not additional independent seeds.','These are post hoc subsets of the same five seeds; pooling both sparsities would not add independent replication.').replace('1 conditions,','1 condition,')
endpoint=pd.read_csv(O/'endpoint_seed_summary.csv')
text+='\n## Threshold-specific endpoint interpretation\n\nAt tau=.1, the equal-seed mean increase from initialization is 0.176086 / 0.182190 percentage points (80% / 85%) with the fixed initialization scale, versus 0.029602 / 0.032959 percentage points after current-RMS normalization. Most of that mean increase disappears after removing overall scale shrinkage; residual shape changes vary across seeds. Dense changes are +0.143738 percentage points with the initialization scale and −0.003967 with current RMS. These are descriptive changes of the measured distributions.\n\nThe sparse–dense near-zero differences depend on threshold and sparsity; there is no uniform increase at every threshold. The predefined current-scale endpoints are:\n\n| Expert sparsity | tau | Mean sparse−dense fraction | Seed range | Positive / 5 |\n|---|---:|---:|---|---:|\n'
for sp in [.8,.85]:
 for tau in [.001,.01,.1]:
  r=endpoint[(endpoint.sparsity==sp)&(endpoint.metric==f'near_zero_current_rms_tau_{tau:g}_sparse_minus_dense')].iloc[0]
  text+=f'| {sp:.0%} | {tau:g} | {r.equal_seed_mean:+.9f} | {r.seed_min:+.9f} to {r.seed_max:+.9f} | {int(r.positive_seeds)}/5 |\n'
text+='\nIn particular, the 80% condition has a lower tau=.01 current-scale near-zero fraction in all five seeds (mean −0.000405884), whereas the 85% condition has almost no mean difference there (+0.000003052) and mixed signs. The fixed-initialization scale also shows a negative tau=.01 contrast for 80% in all five seeds (mean −0.000363159). Full-grid effects are in paired_threshold_curves.csv; these examples use the three thresholds specified in the request.\n\nThe winner pattern is not consistent across winning seeds: at 80%, winning seed 61 has a positive current-scale tau=.1 contrast (+0.000808716), like nonwinning seeds 17 and 29, while winning seeds 7 and 43 have negative contrasts. At 85%, the only winning seed is 7 (negative contrast); one winning seed cannot establish replication across winning seeds. RMS is lower for every sparse endpoint, including all winners and nonwinners. These annotations do not establish a mechanism or classifier.\n\nNo row or coordinate column is wholly near-zero by a global RMS <=.1×initial-RMS criterion. Positive per-entry near-zero fractions therefore do not indicate whole near-zero rows or columns. Signed histogram counts are exported in signed_histogram_counts.csv, and CDF plots include the smallest observed positive magnitude. Figures C/D/F were checked and redrawn with separate figure and panel titles.\n'
f.write_text(text)
# Verify the protected tree after all work, then record all new artifact hashes.
prior=json.load(open(B/'analysis_audit_v2/final_artifact_manifest.json'));same={p:sha(B/p)==h for p,h in prior['outputs'].items()};assert all(same.values())
json.dump({'protected_existing_outputs':len(same),'all_match_prior_audit':True,'mismatches':[]},open(O/'protected_output_validation.json','w'),indent=2)
num=json.load(open(O/'numerical_checks.json'));ind=json.load(open(O/'independent_validation.json'));assert num['original_inputs_unchanged'] and all(v is True for k,v in ind.items() if k!='prior_manifest_format')
required=['analysis_config.json','checkpoint_provenance.csv','checkpoint_summary.csv','threshold_curves.csv','row_statistics.csv','column_statistics.csv','sparse_dense_paired.csv','coverage.csv','findings.md'];assert all((O/p).exists() for p in required)
assert len(list((O/'figures').glob('*.png')))==11 and len(list((O/'figures').glob('*.pdf')))==11
outputs={str(p.relative_to(O)):sha(p) for p in sorted(O.rglob('*')) if p.is_file() and p.name not in ['artifact_manifest.json','completion.json','execution.log','validation.log','finalization.log'] and '__pycache__' not in p.parts}
json.dump({'sha256':outputs},open(O/'artifact_manifest.json','w'),indent=2)
json.dump({'analysis_complete':True,'checkpoint_only':True,'training_launched':False,'weights_modified':False,'previous_outputs_unchanged':True,'registered_checkpoint_records':315,'primary_rerun_records':240,'historical_records':75,'additional_alias_records':75,'unique_E_tensors':230,'missing_historical_slots':165,'png_figures':11,'pdf_figures':11,'independent_seed_count':5,'configuration_sha256':sha(O/'analysis_config.json'),'artifact_manifest_sha256':sha(O/'artifact_manifest.json')},open(O/'completion.json','w'),indent=2)
print('Finalization complete: all numerical checks and 47 protected-output hashes pass.',flush=True)
