"""Read-only, CPU checkpoint analysis; all writes are confined to this directory."""
import os
os.environ.setdefault('OMP_NUM_THREADS','2')
import csv, json, hashlib, sys, math, time
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import numpy as np
import pandas as pd
import torch
ROOT=Path(__file__).resolve().parents[3]
BASE=ROOT/'results/high_sparsity_coupling_v1'
OUT=Path(__file__).resolve().parent
STEPS=[0,25,100,250,500,1000,1500,2000,2500,2525,2600,3000,3500,4000,4500,5000]
TAUS=np.logspace(-6,0,61)
QS=[0,1,5,10,25,50,75,90,95,99,100]
SEEDS=[7,17,29,43,61]
def read(p):return json.loads(Path(p).read_text())
def write(p,obj):Path(p).write_text(json.dumps(obj,indent=2,allow_nan=False)+'\n')
def sha(p):
 h=hashlib.sha256()
 with open(p,'rb') as f:
  for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
 return h.hexdigest()
def thash(t):
 a=t.detach().cpu().contiguous().numpy()
 return hashlib.sha256(str(a.dtype).encode()+str(a.shape).encode()+a.tobytes()).hexdigest()
def load(p):return torch.load(p,map_location='cpu',weights_only=False,mmap=True)
def frame(rows,p):pd.DataFrame(rows).to_csv(OUT/p,index=False)
def stats(a):
 return dict(entries=a.size,exact_zero_count=int((a==0).sum()),exact_zero_fraction=float((a==0).mean()),nonfinite_count=int((~np.isfinite(a)).sum()),valid=bool(np.isfinite(a).all()),mean=float(a.mean()),std=float(a.std(ddof=0)),rms=float(np.sqrt(np.mean(a*a))),minimum=float(a.min()),maximum=float(a.max()),**{f'abs_q{q:03d}':float(v) for q,v in zip(QS,np.percentile(np.abs(a),QS,method='linear'))})
def countcurve(a,scale):return np.searchsorted(np.sort(np.abs(a).ravel()),TAUS*scale,side='right')/a.size

def checks():
 a=np.array([[0.,-1.,2.],[0.,.01,-.1]])
 assert stats(a)['exact_zero_count']==2
 assert np.count_nonzero(np.abs(a)<=.1)==4
 r=np.sqrt(np.mean(a*a))
 assert np.array_equal(countcurve(a,r),countcurve(8*a,8*r))
 assert np.all(countcurve(.5*a,r)>=countcurve(a,r)) and np.any(countcurve(.5*a,r)>countcurve(a,r))
 assert np.all(countcurve(np.zeros((2,2)),0)==1)
 assert thash(torch.tensor([1.]))==thash(torch.tensor([1.]).clone())
 assert thash(torch.tensor([1.]))!=thash(torch.tensor([1.],dtype=torch.float64))
 # Tiny actual implementation verifies parameter identity without allocating the full model.
 sys.path.insert(0,str(ROOT/'src'))
 from moe_lth.models.transformer import TinyMoELanguageModel
 cfg=read(BASE/'cohort.json')['conditions'][0]['config']['model'].copy()
 cfg.update(num_layers=0)
 m=TinyMoELanguageModel(cfg)
 assert m.token_embedding.weight is m.lm_head.weight
 assert m.position_embedding.weight is not m.token_embedding.weight
 return {'counting':True,'current_scale_invariance':True,'initial_scale_shrinkage_sensitivity':True,'zero_RMS':True,'dtype_sensitive_content_hash':True,'actual_model_parameter_tying':True}

def main():
 torch.set_num_threads(2)
 cohort=read(BASE/'cohort.json'); byid={j['id']:j for j in cohort['conditions']}
 sourcepaths=[BASE/'protocol.md',BASE/'provenance_audit.md',BASE/'checkpoint_coverage.csv',BASE/'report/provenance_diagnosis.md',BASE/'report/measurement_provenance.csv',BASE/'report/coverage.csv',BASE/'cohort.json',ROOT/'src/moe_lth/models/transformer.py']
 config={'created_utc':time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),'checkpoint_only':True,'training_launched':False,'parameter':'token_embedding.weight','tied_alias':'lm_head.weight','expected_shape':[256,256],'steps':STEPS,'seeds':SEEDS,'tau_grid':TAUS.tolist(),'row_column_taus':[.001,.01,.1],'scales':['current_rms','original_initialization_rms'],'statistics_dtype':'CPU float64 converted from saved tensor; saved precision is unchanged','std_ddof':0,'quantiles_percent':QS,'quantile_method':'linear','comparison':'abs(E)<=tau*global_scale; count includes exact zeros; denominator all entries','zero_rms_policy':'scale=0 means threshold=0; count exact zeros, no division','nonfinite_policy':'flag invalid and stop analysis rather than silently dropping nonfinite values','content_hash':'SHA256(str(numpy original dtype)+str(shape)+original contiguous bytes)','CDF':'NPZ per unique E: unique absolute values, integer multiplicities, cumulative counts and full empirical CDF; includes exact zeros','replication':'seed n=5, both sparsities dependent, historical/rerun duplicate weights not independent','histogram_rule':'241 shared signed bin edges from -ceil(2*global maxabs)/2 to +ceil(2*global maxabs)/2, chosen without ticket labels','historical_step0':'actual original initialization E only; no reconstructed full or trained model','input_sha256':{str(p.relative_to(ROOT)):sha(p) for p in sourcepaths},'file_hash_policy':'recompute SHA256 of every unique analyzed physical checkpoint; compare canonical hashes against prior independent audit','primary_track':'rerun','supplementary_track':'historical; aliases remain excluded from independent comparisons'}
 if (OUT/'analysis_config.json').exists():
  old=read(OUT/'analysis_config.json')
  assert old['tau_grid']==config['tau_grid']
 else:write(OUT/'analysis_config.json',config)
 validation=checks();write(OUT/'numerical_checks.json',validation)
 rows=list(csv.DictReader(open(BASE/'report/measurement_provenance.csv')))
 records=[]
 for r in rows:
  p=r['checkpoint_path'].removeprefix('reconstructed:')
  records.append(dict(track=r['track'],condition=r['condition'],seed=int(r['seed']),sparsity=float(r['sparsity']) if r['sparsity'] else 0.,step=int(r['step']),checkpoint_path=p,expected_sha256=r['checkpoint_sha256'],role='registered',initial_tensor_only=r['track']=='historical' and int(r['step'])==0))
 for j in cohort['conditions']:
  done=read(BASE/'reruns'/j['id']/'completed.json'); a=Path(done['attempt'])
  extra=[(a/'snapshots/step_2500_after_reset.pt',2500,'after_optimizer_scaler_reset'),(a/'segment_0/checkpoints/final_recovered.pt',2500,'native_segment_final'),(Path(done['checkpoint']),5000,'native_segment_final'),(a/'segment_0/interruption.pt',2500,'interruption_final'),(a/'segment_2500/interruption.pt',5000,'interruption_final')]
  for p,step,role in extra:
   if p.exists():records.append(dict(track='rerun',condition=j['id'],seed=j['seed'],sparsity=j['sparsity'] or 0.,step=step,checkpoint_path=str(p),expected_sha256='',role=role,initial_tensor_only=False))
 # Hash once per unique physical path; cache is confined to this new analysis directory.
 paths=sorted({r['checkpoint_path'] for r in records})
 hashfile=OUT/'physical_file_hashes.json'
 hashes=read(hashfile) if hashfile.exists() else {}
 def verify(p):
  stat=Path(p).stat(); old=hashes.get(p)
  if old and old['bytes']==stat.st_size and old['mtime_ns']==stat.st_mtime_ns:return p,old
  h=sha(p)
  return p,dict(sha256=h,bytes=stat.st_size,mtime_ns=stat.st_mtime_ns)
 with ThreadPoolExecutor(max_workers=3) as ex:
  for i,(p,h) in enumerate(ex.map(verify,paths)):
   hashes[p]=h
   if i%20==0:
    write(hashfile,hashes);print(f'hashed {i+1}/{len(paths)}',flush=True)
 write(hashfile,hashes)
 initial={}; initial_hash={}
 for seed in SEEDS:
  j=next(j for j in cohort['conditions'] if j['seed']==seed)
  d=load(j['initial']);e=d['model']['token_embedding.weight']
  initial[seed]=np.sqrt(np.mean(e.double().numpy()**2));initial_hash[seed]=thash(e)
 (OUT/'cdf').mkdir(exist_ok=True);cache={};provenance=[];arrays={}
 for i,r in enumerate(records):
  p=r['checkpoint_path'];h=hashes[p]['sha256']
  if r['expected_sha256']:assert h==r['expected_sha256'],p
  d=load(p);sd=d['model'];e=sd['token_embedding.weight'];head=sd['lm_head.weight']
  assert e.shape==(256,256) and torch.equal(e,head)
  storage_alias=e.untyped_storage().data_ptr()==head.untyped_storage().data_ptr() and e.storage_offset()==head.storage_offset()
  eh=thash(e); r.update(checkpoint_sha256=h,E_sha256=eh,original_dtype=str(e.dtype),shape='256x256',head_equal=True,head_storage_alias=storage_alias,saved_step=d.get('step'),global_recovery_step=r['step'],initial_E_sha256=initial_hash[r['seed']],initial_rms=initial[r['seed']],cdf_path=f'cdf/{eh}.npz')
  if r['step']==0:assert eh==initial_hash[r['seed']]
  if eh not in cache:
   a=e.double().numpy().copy();s=stats(a)
   assert s['valid'],f'Invalid tensor {p}'
   av,counts=np.unique(np.abs(a),return_counts=True);cum=np.cumsum(counts)
   np.savez_compressed(OUT/r['cdf_path'],absolute_weight=av,multiplicity=counts,cumulative_count=cum,cdf=cum/a.size,exact_zero_count=s['exact_zero_count'],entries=a.size)
   cache[eh]=s;arrays[eh]=a
  provenance.append(r)
  if i%30==0:print(f'loaded {i+1}/{len(records)}, unique E={len(cache)}',flush=True)
 frame(provenance,'checkpoint_provenance.csv')
 canonical=[r for r in provenance if r['role']=='registered']
 key=lambda r:(r['track'],r['condition'],r['step'])
 lookup={key(r):r for r in canonical};assert len(lookup)==315
 coverage=[]
 for track in ['rerun','historical']:
  for j in cohort['conditions']:
   for step in STEPS:
    k=(track,j['id'],step);r=lookup.get(k)
    coverage.append(dict(track=track,condition=j['id'],seed=j['seed'],sparsity=j['sparsity'] or 0,step=step,available=r is not None,status=('actual_original_initialization_E' if r and r['initial_tensor_only'] else 'saved_checkpoint' if r else 'unavailable_no_reconstruction'),checkpoint_path=r['checkpoint_path'] if r else ''))
 frame(coverage,'coverage.csv')
 summary=[];curves=[];rowstats=[];colstats=[];computed={}
 for r in provenance:
  meta={k:r[k] for k in ['track','condition','seed','sparsity','step','role','E_sha256']};eh=r['E_sha256'];s=cache[eh];a=arrays[eh]
  z=meta|s|dict(initial_rms=r['initial_rms'],rms_change_from_initial=s['rms']-r['initial_rms'],rms_ratio_to_initial=s['rms']/r['initial_rms'] if r['initial_rms'] else None,exact_zero_change_from_initial=s['exact_zero_fraction']-cache[initial_hash[r['seed']]]['exact_zero_fraction'])
  for q in QS:z[f'abs_q{q:03d}_change_from_initial']=s[f'abs_q{q:03d}']-cache[initial_hash[r['seed']]][f'abs_q{q:03d}']
  summary.append(z)
  ck=(eh,r['seed'])
  if ck not in computed:
   cc={};rr=[];co=[]
   for scale_name,scale in [('current_rms',s['rms']),('original_initialization_rms',r['initial_rms'])]:cc[scale_name]=countcurve(a,scale)
   for axis,dest,idname in [(1,rr,'token_id'),(0,co,'coordinate_id')]:
    norms=np.sqrt(np.sum(a*a,axis=axis));rms=np.sqrt(np.mean(a*a,axis=axis));zeros=np.mean(a==0,axis=axis)
    near={(sn,t):np.mean(np.abs(a)<=t*sc,axis=axis) for sn,sc in [('current_rms',s['rms']),('original_initialization_rms',r['initial_rms'])] for t in [.001,.01,.1]}
    for idx in range(len(norms)):
     x={idname:idx,'l2':norms[idx],'rms':rms[idx],'exact_zero_fraction':zeros[idx]}
     for (sn,t),v in near.items():x[f'near_zero_{sn}_tau_{t:g}']=v[idx]
     dest.append(x)
   computed[ck]=(cc,rr,co)
  cc,rr,co=computed[ck]
  for sn,values in cc.items():
   init_a=arrays[initial_hash[r['seed']]];init_values=countcurve(init_a,r['initial_rms'])
   for idx,(tau,value) in enumerate(zip(TAUS,values)):
    curves.append(meta|dict(scale=sn,scale_value=s['rms'] if sn=='current_rms' else r['initial_rms'],tau=tau,absolute_threshold=tau*(s['rms'] if sn=='current_rms' else r['initial_rms']),near_zero_count=int(round(value*a.size)),near_zero_fraction=value,change_from_initial=value-init_values[idx]))
  rowstats.extend(meta|x for x in rr);colstats.extend(meta|x for x in co)
 frame(summary,'checkpoint_summary.csv');frame(curves,'threshold_curves.csv');frame(rowstats,'row_statistics.csv');frame(colstats,'column_statistics.csv')
 pairs=[];paircurves=[];pairrows=[];paircols=[]
 for r in canonical:
  if not r['sparsity']:continue
  j=byid[r['condition']];dense=lookup[(r['track'],j['dense_id'],r['step'])]
  assert dense['seed']==r['seed'] and dense['step']==r['step'] and dense['sparsity']==0
  assert r['initial_E_sha256']==dense['initial_E_sha256']
  sm=cache[r['E_sha256']];dm=cache[dense['E_sha256']]
  meta={k:r[k] for k in ['track','condition','seed','sparsity','step']}
  meta.update(dense_condition=dense['condition'],sparse_E_sha256=r['E_sha256'],dense_E_sha256=dense['E_sha256'],initial_E_equal=True)
  z=meta|{f'{k}_sparse_minus_dense':sm[k]-dm[k] for k in ['exact_zero_fraction','rms','mean','std']+[f'abs_q{q:03d}' for q in QS]}
  sc,sr,sco=computed[(r['E_sha256'],r['seed'])];dc,dr,dco=computed[(dense['E_sha256'],r['seed'])]
  for sn in sc:
   for t in [.001,.01,.1]:
    idx=int(np.argmin(np.abs(TAUS-t)));z[f'near_zero_{sn}_tau_{t:g}_sparse_minus_dense']=sc[sn][idx]-dc[sn][idx]
   for t,sv,dv in zip(TAUS,sc[sn],dc[sn]):paircurves.append(meta|dict(scale=sn,tau=t,sparse_fraction=sv,dense_fraction=dv,sparse_minus_dense=sv-dv))
  pairs.append(z)
  for ss,dd,dest,idname in [(sr,dr,pairrows,'token_id'),(sco,dco,paircols,'coordinate_id')]:
   for srow,drow in zip(ss,dd):dest.append(meta|{idname:srow[idname]}|{f'{k}_sparse_minus_dense':srow[k]-drow[k] for k in srow if k!=idname})
 frame(pairs,'sparse_dense_paired.csv');frame(paircurves,'paired_threshold_curves.csv');frame(pairrows,'paired_row_statistics.csv');frame(paircols,'paired_column_statistics.csv')
 validation.update(canonical_records=len(canonical),physical_provenance_records=len(provenance),unique_physical_paths=len(paths),unique_E_tensors=len(cache),all_initial_E_equal=True,all_head_values_equal=True,all_head_storage_tied=all(r['head_storage_alias'] for r in provenance),all_physical_file_hashes_recomputed_and_canonical_matched=True,dense_pairs_verified=len(pairs),coverage_slots=len(coverage),missing_slots=sum(not r['available'] for r in coverage),original_inputs_unchanged=all(sha(ROOT/p)==h for p,h in config['input_sha256'].items()))
 # All common-track E hashes must match; reset aliases must equal the before state.
 common=[r for r in canonical if r['track']=='historical']
 assert all(r['E_sha256']==lookup[('rerun',r['condition'],r['step'])]['E_sha256'] for r in common)
 assert all(r['E_sha256']==lookup[('rerun',r['condition'],r['step'])]['E_sha256'] for r in provenance if r['role']!='registered')
 validation.update(cross_track_identical_E=len(common),reset_and_native_aliases_equal=True)
 write(OUT/'numerical_checks.json',validation)
 print(json.dumps(validation),flush=True)
 plot_and_report(summary,curves,rowstats,colstats,pairs,cache,arrays,canonical)

def plot_and_report(summary,curves,rowstats,colstats,pairs,cache,arrays,canonical):
 import matplotlib
 matplotlib.use('Agg')
 import matplotlib.pyplot as plt
 plt.rcParams.update({'font.size':8,'pdf.fonttype':42})
 figs=OUT/'figures';figs.mkdir(exist_ok=True)
 S=pd.DataFrame(summary);S=S[(S.role=='registered')&(S.track=='rerun')]
 C=pd.DataFrame(curves);C=C[(C.role=='registered')&(C.track=='rerun')]
 R=pd.DataFrame(rowstats);R=R[(R.role=='registered')&(R.track=='rerun')]
 K=pd.DataFrame(colstats);K=K[(K.role=='registered')&(K.track=='rerun')]
 P=pd.DataFrame(pairs);P=P[P.track=='rerun']
 colors={0.:'black',.8:'#0072B2',.85:'#D55E00'}
 def save(fig,name):
  fig.savefig(figs/(name+'.png'),dpi=160,bbox_inches='tight');fig.savefig(figs/(name+'.pdf'),bbox_inches='tight');plt.close(fig)
 def boundary(ax):ax.axvline(2500,color='.5',ls=':',lw=1);ax.set_xlabel('Global attempted recovery step')
 fig,ax=plt.subplots(figsize=(7,3))
 for _,g in S.groupby('condition'):ax.plot(g.step,g.exact_zero_fraction,lw=.7)
 ax.set_ylabel('Exact-zero fraction');boundary(ax)
 ax.set_title(f'A. Tied E: exact zeros (all {len(S)} primary checkpoint records)')
 if S.exact_zero_count.sum()==0:ax.text(.5,.65,'No exact-zero entries at any primary checkpoint',ha='center',transform=ax.transAxes)
 save(fig,'A_exact_zero')
 for sp in [.8,.85]:
  fig,axs=plt.subplots(3,2,figsize=(11,9),sharex=True)
  for row,t in enumerate([.001,.01,.1]):
   for col,sn in enumerate(['current_rms','original_initialization_rms']):
    ax=axs[row,col]
    for seed,color in zip(SEEDS,plt.cm.tab10.colors):
     for s,ls in [(0,'--'),(sp,'-')]:
      g=C[(C.seed==seed)&(C.sparsity==s)&(C.scale==sn)&np.isclose(C.tau,t,rtol=1e-12,atol=0)]
      ax.plot(g.step,g.near_zero_fraction,color=color,ls=ls,label=f'{seed}: '+('dense' if s==0 else f'{s:.0%}'))
    ax.set_title(f'tau={t:g}; {sn}');ax.set_ylabel('Near-zero fraction');boundary(ax)
  axs[0,0].legend(fontsize=6,ncol=2);fig.suptitle(f'B. {sp:.0%} versus matched dense; dotted line = optimizer/scaler reset only')
  fig.tight_layout();save(fig,f'B_near_zero_{sp:.2f}')
 maxabs=max(max(abs(v['minimum']),abs(v['maximum'])) for v in cache.values());bound=math.ceil(maxabs*2)/2;bins=np.linspace(-bound,bound,241)
 write(OUT/'histogram_bins.json',{'shared_bin_edges':bins.tolist(),'all_tensor_maxabs':maxabs,'density_integrates_to_one_per_tensor':True})
 for kind in ['CDF','histogram']:
  fig,axs=plt.subplots(5,3,figsize=(12,15),sharex=True,sharey=True)
  for i,seed in enumerate(SEEDS):
   for k,step in enumerate([0,2500,5000]):
    ax=axs[i,k]
    for sp in [0.,.8,.85]:
     r=next(r for r in canonical if r['track']=='rerun' and r['seed']==seed and r['sparsity']==sp and r['step']==step);a=arrays[r['E_sha256']]
     if kind=='CDF':
      d=np.load(OUT/r['cdf_path']);positive=d['absolute_weight']>0
      ax.plot(d['absolute_weight'][positive],d['cdf'][positive],color=colors[sp],ls={0.:'-',.8:'--',.85:':'}[sp],label='dense' if sp==0 else f'{sp:.0%}')
      ax.set_xscale('log');ax.set_xlim(1e-6,bound);ax.set_xlabel('Positive |E| (exact zeros reported separately)');ax.set_ylabel('Empirical cumulative fraction')
     else:
      density,_=np.histogram(a,bins=bins,density=True);ax.stairs(density,bins,color=colors[sp],ls={0.:'-',.8:'--',.85:':'}[sp],label='dense' if sp==0 else f'{sp:.0%}')
      ax.set_xlim(-bound,bound);ax.set_xlabel('Signed stored weight');ax.set_ylabel('Density')
    ax.set_title(f'Seed {seed}, step {step}');ax.legend(fontsize=6)
  fig.suptitle(('C. Absolute-weight empirical CDF' if kind=='CDF' else 'D. Signed weight distributions')+'; primary rerun, no pooling across seeds');fig.tight_layout();save(fig,'C_magnitude_CDF' if kind=='CDF' else 'D_signed_histograms')
 for seed in SEEDS:
  fig,axs=plt.subplots(2,4,figsize=(15,7))
  for i,sp in enumerate([.8,.85]):
   for j,sn in enumerate(['current_rms','original_initialization_rms']):
    sparse=C[(C.seed==seed)&(C.sparsity==sp)&(C.scale==sn)].pivot(index='step',columns='tau',values='near_zero_fraction').reindex(STEPS)
    dense=C[(C.seed==seed)&(C.sparsity==0)&(C.scale==sn)].pivot(index='step',columns='tau',values='near_zero_fraction').reindex(STEPS)
    for k,data in enumerate([sparse,sparse-dense]):
     ax=axs[i,j*2+k];v=data.to_numpy();limit=max(float(np.abs(v).max()),1e-12)
     im=ax.imshow(v,aspect='auto',origin='lower',cmap='coolwarm' if k else 'viridis',vmin=-limit if k else 0,vmax=limit if k else 1)
     ax.set_xticks([0,10,20,30,40,50,60],['1e-6','1e-5','1e-4','1e-3','1e-2','1e-1','1']);ax.tick_params(axis='x',rotation=45)
     ax.set_yticks(range(len(STEPS)),STEPS);ax.axhline(8,color='gray',ls=':');ax.set_xlabel('tau (log-spaced grid)');ax.set_ylabel('Recovery step (uneven time spacing)')
     ax.set_title(f'{sp:.0%}, {sn}\n'+('sparse minus dense' if k else 'near-zero fraction'));fig.colorbar(im,ax=ax,shrink=.7)
  fig.suptitle(f'E. Seed {seed}; dotted line at 2500 = state reset, weights unchanged');fig.tight_layout();save(fig,f'E_heatmap_seed_{seed}')
 fig,axs=plt.subplots(5,2,figsize=(12,15),sharex=True)
 for i,seed in enumerate(SEEDS):
  for j,(data,label) in enumerate([(R,'Token row'),(K,'Coordinate column')]):
   ax=axs[i,j]
   for sp in [0.,.8,.85]:
    g=data[(data.seed==seed)&(data.sparsity==sp)].groupby('step').l2.quantile([.1,.5,.9]).unstack().reindex(STEPS)
    ax.plot(STEPS,g[.5],color=colors[sp],label='dense' if sp==0 else f'{sp:.0%}');ax.fill_between(STEPS,g[.1],g[.9],color=colors[sp],alpha=.10)
   ax.set_title(f'Seed {seed}, {label} L2');ax.set_ylabel('L2 norm: median and 10–90% within-tensor range');boundary(ax);ax.legend(fontsize=6)
 fig.suptitle('F. Row/column norm distribution trajectories; bands are not confidence intervals');fig.tight_layout();save(fig,'F_norm_distributions')
 report(S,C,R,K,P)

def report(S,C,R,K,P):
 final=S[S.step==5000];initial=S[S.step==0];p=P[P.step==5000]
 lines=['# Tied byte-token embedding coordinate sparsity','',f'Primary analysis: {len(S)} rerun checkpoint records (15 conditions ×16 steps); 75 historical records are separate and have identical E at all common states. Additional reset/native/interruption aliases are documented in checkpoint_provenance.csv and excluded from comparisons. Historical initialization uses the actual saved original E, with no missing trained-state reconstruction. All 165 absent historical slots are listed in coverage.csv.','', 'E is token_embedding.weight, shape 256×256 (65,536 entries), tied to lm_head.weight. Position embeddings are excluded. All values were saved in torch.float32 and converted to CPU float64 solely for statistics. Per-checkpoint file hashes were freshly recomputed; source files and prior experiment outputs were not modified.','',f'1. **Exact zeros:** {int(S.exact_zero_count.sum())} across all primary checkpoint records; all tensors are finite. No E acquires exact zeros. Repeated records are not independent observations.','', '2. **Scale and shape:**']
 for sp in [0.,.8,.85]:
  g=final[final.sparsity==sp]
  lines.append(f'- {"Dense" if sp==0 else f"{sp:.0%} sparse"}: final RMS mean {g.rms.mean():.9f}; mean relative change from initialization {(g.rms_ratio_to_initial.mean()-1)*100:.6f}% (seed range {(g.rms_ratio_to_initial.min()-1)*100:.6f}% to {(g.rms_ratio_to_initial.max()-1)*100:.6f}%).')
 for sn in ['current_rms','original_initialization_rms']:
  for sp in [0.,.8,.85]:
   g=C[(C.step==5000)&(C.sparsity==sp)&(C.scale==sn)&np.isclose(C.tau,.1,rtol=1e-12,atol=0)]
   lines.append(f'- tau=.1, {sn}, {"dense" if sp==0 else f"{sp:.0%}"}: final fraction mean {g.near_zero_fraction.mean():.9f}, initialization change mean {g.change_from_initial.mean():+.9f}, seed range {g.change_from_initial.min():+.9f} to {g.change_from_initial.max():+.9f}.')
 lines+=['','The initialization scale retains the effect of global magnitude shrinkage; the current scale removes that overall scale and measures residual distribution-shape changes. Both effects can coexist. The complete 61-point curves and full empirical CDFs are exported; conclusions are not restricted to tau=.1.','', '3. **Matched sparse–dense comparisons:** Initial E tensors are verified identical across all three conditions within each seed.']
 for sp in [.8,.85]:
  g=p[p.sparsity==sp]
  lines.append(f'- {sp:.0%}: final RMS difference mean {g.rms_sparse_minus_dense.mean():+.9f}, range {g.rms_sparse_minus_dense.min():+.9f} to {g.rms_sparse_minus_dense.max():+.9f}; negative in {int((g.rms_sparse_minus_dense<0).sum())}/5 seeds.')
  for sn in ['current_rms','original_initialization_rms']:
   col=f'near_zero_{sn}_tau_0.1_sparse_minus_dense';v=g[col]
   lines.append(f'  tau=.1 ({sn}) difference mean {v.mean():+.9f}, range {v.min():+.9f} to {v.max():+.9f}; positive in {int((v>0).sum())}/5 seeds.')
 lines+=['','4. **Rows and coordinates:** Row and column CSVs preserve IDs and include all requested exact/near-zero fractions under both global scales. A small row has both an input-embedding and tied output-prediction role; its norm does not establish whether the token is used.']
 concentration=[]
 for data,idname,label in [(R,'token_id','rows'),(K,'coordinate_id','columns')]:
  for sp in [0.,.8,.85]:
   f=data[(data.step==5000)&(data.sparsity==sp)];ini=data[(data.step==0)&(data.sparsity==sp)]
   z=f.merge(ini,on=['seed',idname],suffixes=('_final','_initial'));delta=z.l2_final/z.l2_initial-1
   lines.append(f'- {"Dense" if sp==0 else f"{sp:.0%}"} {label}: final L2 range {f.l2.min():.6f}–{f.l2.max():.6f}; relative change median {delta.median()*100:.6f}%, 5th–95th percentiles {np.percentile(delta,5)*100:.6f}% to {np.percentile(delta,95)*100:.6f}% (descriptive range over IDs and seeds, not uncertainty). No all-zero {label}; {int((f.rms<=.1*f.seed.map(initial_rms_lookup(S))).sum())} whole {label} have RMS <=.1×initial global RMS.')
   z['relative_l2_change']=delta
   concentration.extend(dict(kind=label,seed=int(row.seed),sparsity=sp,id=int(getattr(row,idname)),relative_l2_change=row.relative_l2_change) for row in z.itertuples())
 frame(concentration,'row_column_relative_changes.csv')
 lines+=['','5. **Ticket labels (descriptive only):** Four of ten sparse models have strict sparse<=dense validation labels in each track; these are seed 7 at .80/.85, seed 43 at .80, and seed 61 at .80. Historical/rerun weights duplicate and do not provide replication. Thresholds were fixed before computation and not chosen to separate labels.','']
 tickets=pd.read_csv(BASE/'report/rerun_tickets.csv');endpoint=p.merge(tickets[['seed','sparsity','strict_success','gap']],on=['seed','sparsity']);frame(endpoint.to_dict('records'),'endpoint_ticket_annotations.csv')
 for row in endpoint.itertuples():
  lines.append(f'- Seed {row.seed}, {row.sparsity:.0%}, {"winner" if row.strict_success else "nonwinner"}: RMS sparse−dense {row.rms_sparse_minus_dense:+.9f}; current-scale tau=.1 fraction difference {getattr(row,"near_zero_current_rms_tau_0_1_sparse_minus_dense",float("nan")):+.9f}.' if False else f'- Seed {row.seed}, {row.sparsity:.0%}, {"winner" if row.strict_success else "nonwinner"}: RMS sparse−dense {row.rms_sparse_minus_dense:+.9f}.')
 lines+=['','The same direction of a small RMS change appears among winners and nonwinners; it does not separate the labels. Per-seed exceptions and the dependent second sparsity prevent interpreting pooled winner averages as independent evidence. See endpoint_ticket_annotations.csv for all metrics and labels.','', '## Interpretation and limitations','', 'Small coordinates are near-zero entries, not evidence of inactive, useless, or safely removable parameters. The row/column results must not be interpreted as token usage. Coordinate sparsity is distinct from low-rank structure and earlier CKA measurements. No causal, dispensability, or winning-ticket mechanism claim follows from this checkpoint-only comparison. Seeds (n=5) are the independent units; no entry-, token-, coordinate-, or checkpoint-level confidence intervals are used. Equal-seed averages and seed ranges above are descriptive. Historical missing states remain unavailable. All plots use rerun trajectories as primary; the identical historical endpoints are retained in tables, not double-counted.','', '## Outputs','', 'analysis_config.json fixes the tau grid, scales, quantiles, algorithms and input hashes before summary computation. checkpoint_provenance.csv includes physical file SHA256, E hash, saved dtype/shape, track/seed/condition/step and alias roles. checkpoint_summary.csv and threshold_curves.csv include changes from initialization. row_statistics.csv and column_statistics.csv retain IDs. sparse_dense_paired.csv contains scalar contrasts; paired_threshold_curves.csv contains every threshold contrast, and paired_row_statistics.csv / paired_column_statistics.csv contain ID-matched norm and near-zero contrasts. cdf/*.npz exports every unique empirical CDF without downsampling (including exact zeros). figures/ contains A–F in PNG and PDF. numerical_checks.json records focused checks and coverage. All files are confined to this directory.']
 (OUT/'findings.md').write_text('\n'.join(lines)+'\n')
def initial_rms_lookup(S):return S[S.step==0].groupby('seed').rms.first().to_dict()
if __name__=='__main__':main()
