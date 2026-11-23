"""Read-only input audit and seed-paired endpoint analysis. No training entrypoints.
Run: OMP_NUM_THREADS=2 .venv-highres/bin/python results/dataset_a_full_lower_sparsity/analysis_final/scripts/analyze.py
Outputs are exclusively inside analysis_final. Existing output files are refused.
"""
import os,sys,json,hashlib,itertools,math
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import numpy as np,pandas as pd,torch
from scipy import stats
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parents[4]; B=ROOT/'results/dataset_a_full_lower_sparsity'; O=Path(os.environ.get('DATASET_A_ANALYSIS_OUT',str(B/'analysis_final'))); O.mkdir(parents=True,exist_ok=True); AD=ROOT/'results/final_dataset_a'
sys.path.insert(0,str(ROOT/'src'))
from moe_lth.pruning.router_age import state_dict_hash,parameter_group
from moe_lth.models import TinyMoELanguageModel
from moe_lth.pruning.magnitude_prune import expert_local_magnitude_masks
E=[8,12,16,24,32]; S=[7,17,29,43,61]; A=list(range(0,101,10)); SP=[.5,.55,.6,.65]
TH={'strict':0.,'ppl_1pct':math.log(1.01),'ppl_2pct':math.log(1.02),'ppl_5pct':math.log(1.05)}
torch.set_num_threads(2)
def read(p):return json.loads(Path(p).read_text())
def sha(p):
 with Path(p).open('rb') as h:return hashlib.file_digest(h,'sha256',_bufsize=8*1024*1024).hexdigest()
def csv(n,x):
 p=O/n
 if p.exists():raise FileExistsError(p)
 pd.DataFrame(x).to_csv(p,index=False)
def write(n,x):
 p=O/n
 if p.exists():raise FileExistsError(p)
 p.write_text(x)
def load(p):return torch.load(p,map_location='cpu',weights_only=False)
def group(x,k):return {n:v for n,v in x.items() if parameter_group(n)==k}
checks=[]
def check(scope,name,ok,detail=''):checks.append(dict(scope=scope,check=name,passed=bool(ok),detail=str(detail)))
def holm(p):
 p=np.asarray(p);idx=np.argsort(p);out=np.empty(len(p));out[idx]=np.minimum(1,np.maximum.accumulate((len(p)-np.arange(len(p)))*p[idx]));return out
f=pd.read_csv(B/'manifests/run_manifest.csv');lo=f[f.job_type=='sparse'];de=f[f.job_type=='dense']
keys=['num_experts','seed','router_age','sparsity'];expected=set(itertools.product(E,S,A,SP))
check('grid','exact_sparse_product',set(map(tuple,lo[keys].values))==expected and len(lo)==1100)
check('grid','unique_condition_ids',f.condition_id.is_unique)
check('grid','completion_status',len(de)==275 and lo.status.value_counts().to_dict()=={'COMPLETE':1040,'REUSED':60} and de.status.eq('REUSED').all())
seal=read(B/'extension_seal.json');hashrows=[]
# Full byte hashing, no reliance on historical size/mtime caches.
hash_jobs=[(base/rel,digest) for section,base in [('code',ROOT),('immutable',B),('inputs',Path('/'))] for rel,digest in seal[section].items()]
def audit_hash(job):
 p,digest=job
 try:actual=sha(p);ok=actual==digest;detail=actual
 except Exception as ex:ok=False;detail=str(ex)
 check('seal',str(p),ok,detail)
 return dict(path=str(p),expected_sha256=digest,actual_sha256=detail,passed=ok)
with ThreadPoolExecutor(max_workers=4) as pool:
 for i,z in enumerate(pool.map(audit_hash,hash_jobs)):
  hashrows.append(z)
  if i%100==0:print('sealed hash audit',i,'/',len(hash_jobs),flush=True)
print('sealed inputs/code hashed',flush=True)
sources=read(B/'runtime_sources.json');old=pd.read_csv(AD/'canonical/dataset_a_canonical.csv');dense=pd.read_csv(AD/'canonical/dataset_a_dense.csv').set_index('condition_id');src=pd.read_csv(AD/'canonical/dataset_a_sources.csv').set_index(['num_experts','seed'])
cp=pd.read_csv(AD/'inputs/source_checkpoint_manifest.csv').set_index(['num_experts','seed','router_age']);batch=read(AD/'inputs/canonical_data_hashes.json')
for name in ['canonical/dataset_a_canonical.csv','canonical/dataset_a_dense.csv','canonical/dataset_a_sources.csv','inputs/source_checkpoint_manifest.csv','DATASET_A_FROZEN.json']:
 check('original',name,sha(AD/name)==read(AD/'dataset_a_output_hashes.json')[name])
check('original','frozen_PASS',read(AD/'dataset_a_finalization_status.json')['status']=='PASS')
check('original','inventory',len(old)==1650 and not old.duplicated(keys).any() and set(old.sparsity)=={.7,.75,.8,.85,.9,.95} and old.condition_valid.all())
dense_records=[]
for r in de.to_dict('records'):
 p=Path(r['artifact_path']);pr=read(p/'provenance.json');summ=read(p/'summary.json');d=dense.loc[r['condition_id']];rec=pr['original_record']
 check(r['condition_id'],'table_reference',pr['reference_type']=='finalized_dataset_a_table_endpoint' and pr['table_sha256']==sha(AD/'canonical/dataset_a_dense.csv') and abs(summ['final_validation_loss']-d['final'])<1e-12 and int(rec['architecture'])==r['num_experts'] and int(rec['seed'])==r['seed'] and int(rec['router_age'])==r['router_age'] and abs(float(rec['final'])-d['final'])<1e-12)
 dense_records.append(dict(**r,dense_final_validation_loss=d['final'],reference_provenance=pr['reference_type'],historical_raw_path=d['path'],raw_checkpoint_available=(Path(d['path'])/'checkpoints/final_recovered.pt').exists(),reference_table_sha256=pr['table_sha256']))
csv('dense_references.csv',dense_records)
maskinfo={};initialhash={};routerhash={}
for sid,info in sources.items():
 e=int(info['config']['model']['num_experts']);s=int(info['config']['seed']);cfg=info['config'];init=load(info['checkpoints']['0']['path']);final=load(info['checkpoints']['100']['path']);model=TinyMoELanguageModel(cfg['model']);model.load_state_dict(final['model'])
 check(sid,'source_config',cfg['training']['steps']==2500 and cfg['training']['learning_rate']==.0003 and cfg['training']['weight_decay']==.01 and cfg['training']['grad_clip']==1 and cfg['training']['batch_size']==128 and cfg['training']['precision']=='fp16' and cfg['data']['seq_len']==128 and cfg['model']['top_k']==1 and cfg['model']['capacity_factor']==1.25 and cfg['routing']['aux_loss_weight']==.1)
 for age,c in info['checkpoints'].items():
  ck=load(c['path']);key=(e,s,int(age));routerhash[key]=state_dict_hash(group(ck['model'],'router'))
  check(sid,'checkpoint_R'+age,ck['step']==25*int(age) and c['sha256']==cp.loc[key,'sha256'])
 for sp,rec in info['masks'].items():
  sp=float(sp);m=torch.load(rec['learned_path'],map_location='cpu',weights_only=True);generated=expert_local_magnitude_masks(model,sp);total=sum(v.numel() for v in m.values());keep=sum(int(v.sum()) for v in m.values());mh=state_dict_hash(m)
  check(sid,f'mask_{sp}',mh==rec['learned_state_hash']==state_dict_hash(generated) and all('.moe.experts.' in n and n.endswith('.weight') for n in m))
  maskinfo[e,s,sp]=dict(actual_sparsity=1-keep/total,mask_keep_count=keep,mask_pruned_count=total-keep,prunable_expert_weight_count=total,total_model_parameter_count=sum(p.numel() for p in model.parameters()),retained_model_parameter_count=sum(p.numel() for p in model.parameters())-total+keep,mask_hash=mh,mask_artifact_path=rec['learned_path'])
  base={n:v*m[n] if n in m else v for n,v in init['model'].items()};initialhash[e,s,sp]=(state_dict_hash(group(base,'expert')),state_dict_hash(group(base,'shared')))
 del model,init,final,ck,base,m,generated
 print('source/masks',sid,flush=True)
rows=[]
def audit_recovery(r):
 cid=r['condition_id'];p=Path(r['artifact_path']);e,s,a,sp=[r[k] for k in keys];key=(e,s,a);mi=maskinfo[e,s,sp];d=dense.loc[r['dense_condition_id']];issues=[]
 try:
  meta=read(p/'metadata.json');res=read(p/'validated_result.json');done=read(p/'pilot_completed.json');summ=read(p/'summary.json');curve=[json.loads(x) for x in (p/'metrics.jsonl').read_text().splitlines()];ex=read(p/'checkpoint_exposure.json')[-1];snap=read(p/'pilot_sparsity_checks.json')
  check(cid,'identity_budget',done['condition_id']==cid and meta['condition']==res['condition']==cid and meta['seed']==s and meta['router_step']==25*a and meta['router_age_percent']==a and meta['sparsity']==sp and meta['source_attempts']==meta['recovery_attempts']==meta['recovery_steps']==meta['optimizer_step_attempts']==2500 and meta['router_mode']=='trainable' and meta['mask_type']=='learned')
  check(cid,'initialization',meta['expert_state_hash']==initialhash[e,s,sp][0] and meta['shared_state_hash']==initialhash[e,s,sp][1] and meta['initial_router_state_hash']==routerhash[key] and meta['mask_hash']==res['mask_hash']==mi['mask_hash'] and meta['router_checkpoint']==sources[r['source_id']]['checkpoints'][str(a)]['path'])
  check(cid,'data_optimizer_eval',meta['training_batch_sequence_hash']==batch[str(s)]['train'] and meta['validation_batch_sequence_hash']==batch[str(s)]['validation'] and meta['optimizer']=='fresh_AdamW' and meta['optimizer_state_entries_at_start']==0 and meta['scheduler']=='none' and meta['precision']=='fp16' and meta['recovery_eval_interval']==50 and meta['integrity_checks_passed'] and meta['trainable_router_gradient_observed'])
  sf=res['final_validation_loss'];df=d['final'];check(cid,'finite_endpoint_agreement',np.isfinite(sf) and [x['step'] for x in curve]==list(range(0,2501,50)) and all(np.isfinite(x['loss']) for x in curve) and max(abs(sf-z) for z in [done['final_validation_loss'],summ['final_validation_loss'],curve[-1]['loss']])<1e-10 and abs(res['dense_reference_loss']-df)<1e-10 and int(d['architecture'])==e and int(d['seed'])==s and int(d['router_age'])==a)
  check(cid,'applied_updates',meta['optimizer_steps_applied']+meta['amp_overflow_step_count']==2500 and ex['applied_updates']==meta['optimizer_steps_applied'] and ex['attempted_updates']==2500 and {x['step'] for x in snap}=={0,2500} and all(x['masked_weights_zero'] and x['masked_optimizer_moments_zero'] and x['mask_hash']==mi['mask_hash'] for x in snap))
  ckpath=p/'checkpoints/final_recovered.pt';actual=sha(ckpath);check(cid,'checkpoint_bytes',actual==r['artifact_sha256']==res['final_checkpoint_sha256']==done['final_checkpoint_sha256']);hashrows.append(dict(path=str(ckpath),expected_sha256=r['artifact_sha256'],actual_sha256=actual,passed=actual==r['artifact_sha256']))
  ck=load(ckpath);m=torch.load(mi['mask_artifact_path'],map_location='cpu',weights_only=True)
  check(cid,'checkpoint_finite_mask_config',ck['step']==2500 and all(bool(torch.isfinite(v).all()) for v in ck['model'].values()) and all(not bool(ck['model'][n][~v.bool()].any()) for n,v in m.items()) and ck['config']['training']['learning_rate']==.0003 and ck['config']['training']['weight_decay']==.01 and ck['config']['seed']==s and ck['config']['model']['num_experts']==e)
  del ck,m
  rows.append(dict(**r,**mi,source_checkpoint_step=25*a,source_checkpoint_hash=cp.loc[key,'sha256'],router_checkpoint_hash=routerhash[key],source_applied_updates=src.loc[(e,s),'training_steps_applied'],recovery_applied_updates=meta['optimizer_steps_applied'],amp_skipped_updates=meta['amp_overflow_step_count'],sparse_final_validation_loss=sf,dense_final_validation_loss=df,ticket_gap=sf-df,perplexity_ratio=math.exp(sf-df),reference_provenance='finalized_dataset_a_table_endpoint',data_hash=meta['training_batch_sequence_hash'],validation_hash=meta['validation_batch_sequence_hash'],optimizer='fresh_AdamW',learning_rate=.0003,weight_decay=.01,grad_clip=1.,precision='fp16',sparse_artifact_path=str(p),protocol_version='Dataset A 2500/2500 native trainable router',analysis_origin='lower_sparsity'))
 except Exception as ex:check(cid,'available_artifacts',False,repr(ex));rows.append(dict(**r,artifact_exception=repr(ex)))
with ThreadPoolExecutor(max_workers=4) as pool:
 for i,_ in enumerate(pool.map(audit_recovery,lo.to_dict('records'))):
  if i%25==0:print('recovery audit',i,'/',len(lo),flush=True)
rows.sort(key=lambda x:(x['num_experts'],x['seed'],x['router_age'],x['sparsity']))
checks.sort(key=lambda x:(x['scope'],x['check']))
hashrows.sort(key=lambda x:x['path'])
csv('input_hash_verification.csv',hashrows);csv('verification_checks.csv',checks)
failed=[x for x in checks if not x['passed']];fnew=pd.DataFrame(rows);fnew['condition_valid']=[not any(x['scope']==c for x in failed) for c in fnew.condition_id];csv('canonical_lower_sparsity.csv',fnew)
# Original compatibility is checked against every new matched source/seed/age and dense endpoint.
lookup=fnew.drop_duplicates(['num_experts','seed','router_age']).set_index(['num_experts','seed','router_age'])
compatible=True
for r in old.to_dict('records'):
 k=(r['num_experts'],r['seed'],r['router_age']);n=lookup.loc[k]
 compatible &= r['source_checkpoint_hash']==n.source_checkpoint_hash and r['router_checkpoint_hash']==n.router_checkpoint_hash and r['data_hash']==n.data_hash and r['validation_hash']==n.validation_hash and r['source_attempt_budget']==r['recovery_attempt_budget']==2500 and abs(r['dense_final_validation_loss']-n.dense_final_validation_loss)<1e-10
check('original','per_row_compatibility',compatible)
old['analysis_origin']='original_finalized';old['reference_provenance']='finalized_dataset_a_table_endpoint';old['perplexity_ratio']=np.exp(old.ticket_gap)
combined=pd.concat([fnew,old],ignore_index=True,sort=False);check('combined','unique_scientific_keys',not combined.duplicated(keys).any());csv('canonical_combined.csv',combined)
csv('coverage_audit.csv',lo.groupby(['num_experts','sparsity','status']).size().reset_index(name='count'))
# Statistics use five independent source seeds; ages and sparsities are correlated repeated measures.
def summarize(frame):
 out=[]
 for k,g in frame.groupby(['num_experts','router_age','sparsity']):
  g=g.sort_values('seed');x=g.ticket_gap.to_numpy();n=len(x);mean=x.mean();sd=x.std(ddof=1);se=sd/np.sqrt(n);r=dict(zip(['num_experts','router_age','sparsity'],k),n=n,mean_gap=mean,sd_gap=sd,ci95_lower=mean-stats.t.ppf(.975,n-1)*se,ci95_upper=mean+stats.t.ppf(.975,n-1)*se,upper95_one_sided=mean+stats.t.ppf(.95,n-1)*se,min_leave_one_seed_out_mean=min(np.delete(x,i).mean() for i in range(n)),max_leave_one_seed_out_mean=max(np.delete(x,i).mean() for i in range(n)))
  r.update({f'gap_seed_{int(s)}':v for s,v in zip(g.seed,x)})
  for name,tol in TH.items():r[name+'_passing_seeds']=int((x<=tol).sum());r[name+'_p_one_sided']=stats.t.cdf((mean-tol)/se,n-1) if se else float(mean>=tol)
  out.append(r)
 out=pd.DataFrame(out)
 for name in TH:out[name+'_p_holm']=holm(out[name+'_p_one_sided'])
 # Extra joint family protects simultaneous claims across all four tolerance families.
 joint=holm(out[[n+'_p_one_sided' for n in TH]].to_numpy().ravel()).reshape(len(out),4)
 for j,name in enumerate(TH):out[name+'_p_holm_joint']=joint[:,j]
 return out
ss=summarize(fnew);cs=summarize(combined);csv('setting_summary.csv',ss);csv('combined_setting_summary.csv',cs)
counts=[]
for name,tol in TH.items():
 for dims in [[],['num_experts'],['sparsity'],['router_age'],['seed'],['num_experts','sparsity'],['num_experts','router_age','sparsity']]:
  groups=[('all',fnew)] if not dims else fnew.groupby(dims)
  for k,g in groups:
   h=g[g.ticket_gap<=tol];row=dict(threshold=name,tolerance_nats=tol,grouping='+'.join(dims) or 'overall',group=str(k),matches=len(h),denominator=len(g),unique_source_seeds=h[['num_experts','seed']].drop_duplicates().shape[0],unique_seed_values=h.seed.nunique(),contributing_settings=h[['num_experts','router_age','sparsity']].drop_duplicates().shape[0]);counts.append(row)
csv('ticket_candidate_counts.csv',counts)
rep=[]
for name,tol in TH.items():
 rep.append(dict(threshold=name,seed_matches=int((fnew.ticket_gap<=tol).sum()),seed_denominator=1100,all_five=int((ss[name+'_passing_seeds']==5).sum()),at_least_four=int((ss[name+'_passing_seeds']>=4).sum()),at_least_three=int((ss[name+'_passing_seeds']>=3).sum()),setting_denominator=220,upper95_below=int((ss.upper95_one_sided<tol).sum()),holm_220=int((ss[name+'_p_holm']<.05).sum()),holm_joint_880=int((ss[name+'_p_holm_joint']<.05).sum())))
csv('replicated_candidate_counts.csv',rep)
# All age pairs and adjacent tested sparsities: exact within-source contrasts.
contrast=[]
for (e,sp),g in fnew.groupby(['num_experts','sparsity']):
 pivot=g.pivot(index='seed',columns='router_age',values='ticket_gap');sparse=g.pivot(index='seed',columns='router_age',values='sparse_final_validation_loss');den=g.pivot(index='seed',columns='router_age',values='dense_final_validation_loss')
 for a,b in itertools.combinations(A,2):
  x=pivot[a]-pivot[b];se=x.std(ddof=1)/np.sqrt(5);contrast.append(dict(num_experts=e,sparsity=sp,age_a=a,age_b=b,mean_gap_difference=x.mean(),sd=x.std(ddof=1),ci95_lower=x.mean()-stats.t.ppf(.975,4)*se,ci95_upper=x.mean()+stats.t.ppf(.975,4)*se,p_two_sided=stats.ttest_1samp(x,0).pvalue,mean_sparse_difference=(sparse[a]-sparse[b]).mean(),mean_dense_difference=(den[a]-den[b]).mean(),seeds_a_better=int((x<0).sum()),**{f'difference_seed_{s}':x.loc[s] for s in S}))
contrast=pd.DataFrame(contrast);contrast['p_holm_family_1100']=holm(contrast.p_two_sided);csv('paired_age_contrasts.csv',contrast)
sc=[]
for (e,a),g in combined.groupby(['num_experts','router_age']):
 p=g.pivot(index='seed',columns='sparsity',values='ticket_gap');sp=list(p.columns)
 for u,v in zip(sp[:-1],sp[1:]):
  x=p[v]-p[u];se=x.std(ddof=1)/np.sqrt(5);sc.append(dict(num_experts=e,router_age=a,sparsity_low=u,sparsity_high=v,mean_difference=x.mean(),sd=x.std(ddof=1),ci95_lower=x.mean()-stats.t.ppf(.975,4)*se,ci95_upper=x.mean()+stats.t.ppf(.975,4)*se,p=stats.ttest_1samp(x,0).pvalue,**{f'difference_seed_{s}':x.loc[s] for s in S}))
sc=pd.DataFrame(sc);sc['p_holm']=holm(sc.p);csv('paired_sparsity_contrasts.csv',sc)
best=[]
for (e,sp),g in fnew.groupby(['num_experts','sparsity']):
 means=g.groupby('router_age')[['ticket_gap','sparse_final_validation_loss']].mean();a=int(means.ticket_gap.idxmin());b=int(means.sparse_final_validation_loss.idxmin());sw=g.loc[g.groupby('seed').ticket_gap.idxmin()].set_index('seed').router_age;loo=[]
 for s in S:loo.append(int(g[g.seed!=s].groupby('router_age').ticket_gap.mean().idxmin()))
 p=g.pivot(index='seed',columns='router_age',values='ticket_gap');interior=[]
 if a not in [0,100]:
  interior=[dict(boundary=z,mean=(p[a]-p[z]).mean(),seeds_better=int((p[a]<p[z]).sum()),p=stats.ttest_1samp(p[a]-p[z],0).pvalue) for z in [0,100]]
 best.append(dict(num_experts=e,sparsity=sp,best_gap_age=a,best_sparse_age=b,mean_best_gap=means.loc[a,'ticket_gap'],seed_winner_ages=json.dumps({int(k):int(v) for k,v in sw.items()}),seeds_selecting_mean_winner=int((sw==a).sum()),leave_one_out_winner_ages=json.dumps(loo),boundary_comparisons=json.dumps(interior)))
csv('exploratory_age_selection.csv',best)
high=[]
for (e,a),g in cs.groupby(['num_experts','router_age']):
 for name in TH:
  for count in [3,4,5]:
   h=g[g[name+'_passing_seeds']>=count];high.append(dict(num_experts=e,router_age=a,threshold=name,criterion=f'at_least_{count}_seeds',highest_observed_tested_sparsity=h.sparsity.max() if len(h) else np.nan))
  for kind in ['p_one_sided','p_holm','p_holm_joint']:
   h=g[g[name+'_'+kind]<.05];high.append(dict(num_experts=e,router_age=a,threshold=name,criterion=kind,highest_observed_tested_sparsity=h.sparsity.max() if len(h) else np.nan))
 for s in S:
  h=combined[(combined.num_experts==e)&(combined.router_age==a)&(combined.seed==s)]
  for name,tol in TH.items():high.append(dict(num_experts=e,router_age=a,threshold=name,criterion=f'individual_seed_{s}',highest_observed_tested_sparsity=h.loc[h.ticket_gap<=tol,'sparsity'].max()))
csv('highest_observed_sparsity.csv',high)
# Compatible earlier random controls; validate separately, never pool into learned grid.
prior=pd.read_csv(ROOT/'results/dataset_b_lower_sparsity/manifests/run_manifest.csv');randomrows=[]
for r in prior[(prior.job_type=='sparse')&(prior.mask_type=='random')&(prior.arm=='A')].to_dict('records'):
 try:
  p=Path(r['artifact_path']);m=read(p/'metadata.json');z=read(p/'validated_result.json');done=read(p/'pilot_completed.json');n=fnew[(fnew.num_experts==32)&(fnew.seed==r['seed'])&(fnew.router_age==r['router_age'])&(fnew.sparsity==r['sparsity'])].iloc[0]
  ms=sources[r['source_id']]['masks'][str(float(r['sparsity']))];oldsrc=read(ROOT/'results/dataset_b_lower_sparsity/runtime_sources.json')[r['source_id']];rm=oldsrc['masks'][str(float(r['sparsity']))];randommask=torch.load(rm['random_path'],map_location='cpu',weights_only=True);lm=torch.load(ms['learned_path'],map_location='cpu',weights_only=True)
  ok=r['status']=='COMPLETE' and m['source_attempts']==m['recovery_steps']==2500 and m['router_mode']=='trainable' and m['training_batch_sequence_hash']==n.data_hash and m['validation_batch_sequence_hash']==n.validation_hash and m['initial_router_state_hash']==n.router_checkpoint_hash and abs(z['dense_reference_loss']-n.dense_final_validation_loss)<1e-10 and sha(p/'checkpoints/final_recovered.pt')==done['final_checkpoint_sha256']==r['artifact_sha256'] and state_dict_hash(randommask)==m['mask_hash']==rm['random_state_hash'] and set(randommask)==set(lm) and all(int(randommask[k].sum())==int(lm[k].sum()) for k in lm)
  randomrows.append(dict(condition_id=r['condition_id'],num_experts=32,seed=r['seed'],router_age=r['router_age'],sparsity=r['sparsity'],valid=ok,random_loss=z['final_validation_loss'],learned_loss=n.sparse_final_validation_loss,random_minus_learned=z['final_validation_loss']-n.sparse_final_validation_loss,random_gap=z['final_validation_loss']-n.dense_final_validation_loss,artifact_path=str(p)))
 except Exception as ex:randomrows.append(dict(condition_id=r['condition_id'],valid=False,exception=repr(ex)))
csv('random_controls.csv',randomrows)
rr=pd.DataFrame(randomrows);rs=[]
if len(rr) and rr.valid.all():
 for (a,sp),g in rr.groupby(['router_age','sparsity']):
  x=g.random_minus_learned;se=x.std(ddof=1)/np.sqrt(len(x));rs.append(dict(router_age=a,sparsity=sp,n=len(x),mean_random_minus_learned=x.mean(),ci95_lower=x.mean()-stats.t.ppf(.975,len(x)-1)*se,ci95_upper=x.mean()+stats.t.ppf(.975,len(x)-1)*se,p=stats.ttest_1samp(x,0).pvalue,random_strict_matches=int((g.random_gap<=0).sum())))
 rs=pd.DataFrame(rs);rs['p_holm']=holm(rs.p);csv('random_control_summary.csv',rs)
# Exact plot inputs saved, all uncertainty is across five paired seeds.
(O/'figures').mkdir(exist_ok=True)
plotdata=fnew.groupby(['num_experts','router_age','sparsity']).agg(mean_gap=('ticket_gap','mean'),sd_gap=('ticket_gap','std'),mean_sparse=('sparse_final_validation_loss','mean'),sd_sparse=('sparse_final_validation_loss','std'),mean_dense=('dense_final_validation_loss','mean')).reset_index();csv('plot_curve_data.csv',plotdata)
def save(name):
 plt.tight_layout()
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
 h=cs[cs.num_experts==e].pivot(index='sparsity',columns='router_age',values='mean_gap');im=ax.imshow(h,aspect='auto',cmap='coolwarm',vmin=-limit,vmax=limit);ax.set_xticks(range(11),A,rotation=90);ax.set_yticks(range(len(h)),[f'{x:.0%}' for x in h.index]);ax.set(title=f'{e} experts',xlabel='Router age (%)',ylabel='Expert weight sparsity')
fig.colorbar(im,ax=axs.ravel().tolist(),label='Mean gap; zero = dense match',shrink=.7);fig.subplots_adjust(right=.9);save('combined_gap_heatmaps')
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
failed=[x for x in checks if not x['passed']];csv('final_verification_checks.csv',checks)
status='PASS' if not failed else 'FAIL'
write('audit_report.md',f'''# Coverage and provenance audit

Validation: **{status}**. {len(checks)} checks; {len(failed)} failures. Exact grid: 1,100 sparse conditions, 1,040 COMPLETE + 60 REUSED; 275 unique table-backed dense references. No conditions omitted. See verification CSVs for per-condition evidence and any exceptions.

All sealed code, immutable runtime inputs, source checkpoints, masks and prior reused artifacts were SHA-256 rehashed from bytes. Every sparse final checkpoint was rehashed, loaded on CPU, checked for finite tensors, step 2500, masked zeros and config. Completion/result/summary/metrics endpoints were compared. Masks were independently regenerated from source-final magnitudes without writing masks. Source R0 expert/shared hashes and requested checkpoint router hashes were independently recomputed and matched to metadata. Snapshot records confirm masked weights and optimizer moments at 0/2500; historical optimizer states are not saved and cannot be independently reconstructed. Runtime assertions are supporting evidence, not new training.

The code actually ranks the concatenated weight tensors **within each expert**, rather than separately per tensor (some design prose says per-tensor magnitude). Random controls preserve each resulting tensor's retained count. Only expert `.weight` tensors are pruned; biases/shared/router parameters remain. Realized counts and retained whole-model counts are in canonical_lower_sparsity.csv.

Dense endpoints and provenance rows match the SHA-256 guarded finalized dense table. Their historical checkpoint hashes are provenance only; raw dense checkpoints are not reverified. Available-file flags in dense_references.csv do not imply verification. Original higher sparsities are 70,75,80,85,90,95%; 1,650 rows, frozen PASS, guarded table hashes and row-by-row source/router/data/validation/budget/dense compatibility: {compatible}. Combined table has 2,750 unique scientific cells. Original raw sparse checkpoints were not reaudited; their prior finalized audit is the provenance basis.

Applied recovery updates range {fnew.recovery_applied_updates.min()}–{fnew.recovery_applied_updates.max()}, despite 2,500 attempts. Source applied counts are retained from finalized sources (NA remains unavailable). AMP skip counts are independently cross-checked against exposure summaries, not equated with attempted steps. Metadata uses a Dataset B protocol label for the common worker, but arm A is verified as 2500/2500.

Fresh AdamW, constant LR 0.0003, weight decay 0.01, grad clip 1, fp16, batch 128, byte sequence length 128, native top-1 capacity 1.25, auxiliary weight 0.1. The same seed-indexed training/validation batch hashes are checked; evaluation is 32 materialized validation batches every 50 attempts. `_evaluate_condition_loss` in src/moe_lth/experiments/run_router_age_recovery.py uses token-normalized PyTorch cross_entropy (natural-log CE); exp(gap) is therefore the endpoint perplexity ratio. Evaluation loss excludes auxiliary optimization loss.

No prospective ticket endpoint tolerance was found in the inspected extension design, runner README, protocol and finalized analysis. scripts/full_lower_sparsity/README.md states: “this package does not define a post-hoc ticket tolerance.” results/dataset_b_lower_sparsity/design.json has dense_matching_tolerance=null and says: “Do not select a matching tolerance after inspecting these results.” The existing within_5pct/within_10pct time-to-threshold metrics are recovery-progress loss thresholds, not prospective endpoint perplexity ticket criteria. Strict ≤0 is the user-requested endpoint count; 1/2/5% perplexity thresholds are exploratory.

Failures: {json.dumps(failed,indent=2)}
''')
arch=fnew.groupby('num_experts').ticket_gap.agg(['mean','min','max']);spar=fnew.groupby('sparsity').ticket_gap.agg(['mean','min','max']);bc=pd.DataFrame(best);strictarch=fnew.assign(match=fnew.ticket_gap<=0).groupby('num_experts').match.sum();oldcount=int((old.ticket_gap<=0).sum());rs_text=pd.DataFrame(rs).to_string(index=False) if len(rs) else 'No valid matched random controls.'
write('findings.md',f'''# Dataset A lower-sparsity findings

1. Validation **{status}**: {len(failed)} failed checks; exact 1,100/1,100 sparse coverage and 275 unique matched dense endpoints. Dense verification is table-backed; historical raw dense checkpoints are not claimed verified.
2. Strict endpoint matches: **{rep[0]['seed_matches']}/1,100**.
3. Strict matches in all five seeds: **{rep[0]['all_five']}/220 settings**.
4. Threshold sensitivity (1/2/5% worse perplexity, exploratory):

```
{pd.DataFrame(rep).to_string(index=False)}
```

5. Unadjusted one-sided 95% upper-bound counts, Holm within each 220-setting threshold family and stricter joint 880-test family are above. These quantify mean-gap evidence under seed-level t assumptions; descriptive endpoint matches are not equivalence discoveries.
6. This measures **router-conditioned dense-matching sparse recovery**, or lottery-ticket candidates. Trained router checkpoints and source-final learned mask selection differ from initialization-only original LTH. No endpoint-only result establishes geometry, causal mechanism, equal expert exposure or theoretical novelty.

## Observations

Architecture averages across the balanced observed grid (descriptive; 220 correlated runs per architecture):

```
{arch.to_string()}
Strict matches by architecture:
{strictarch.to_string()}
```

Sparsity averages (descriptive):

```
{spar.to_string()}
```

Mean-gap winning ages, absolute sparse-loss winning ages and five-seed winner stability:

```
{bc.drop(columns='boundary_comparisons').to_string(index=False)}
```

Absolute-loss and gap mean winners disagree in {int((bc.best_gap_age!=bc.best_sparse_age).sum())}/20 architecture/sparsity curves. Interior mean-gap winners occur in {int((~bc.best_gap_age.isin([0,100])).sum())}/20 curves; selecting these ages on these same results is exploratory. All seed curves, leave-one-seed-out winning ages and candidate-versus-boundary comparisons are retained. An interior minimum of the mean alone does not establish a reproducible interior optimum.

Original 70–95% sweep has {oldcount}/1,650 strict seed matches. The compatible combined grid contains ten tested sparsities at each age; combined_setting_summary.csv and combined_gap_heatmaps show the connection. highest_observed_sparsity.csv reports maxima separately for every architecture/age, seed threshold, replicated count and inferential criterion. NA means no tested sparsity qualifies. No monotonicity assumption, interpolation or extrapolation is used.

## Statistical evidence and precision

Each setting uses five paired sparse-minus-dense gaps, sample SD (ddof=1), two-sided 95% t interval and one-sided 95% upper bound (df=4). Tests of mean gap below tolerance use p=t_CDF((mean−tolerance)/SE); rejection supports a mean below that tolerance under independent identically distributed source seeds and approximately normal paired gaps. Five seeds give low precision and weak diagnostics; seeds are not additional independent samples when reused across ages/sparsities. Dense endpoints are stored once and their repetition never increases n. Nonsignificance is not proof of equivalence.

paired_age_contrasts.csv contains all 55 within-seed age contrasts for each of 20 architecture/sparsity curves (1,100 tests), sparse and dense decomposition, five differences and two-sided intervals. Holm is applied over all 1,100 contrasts. {int((contrast.p_two_sided<.05).sum())} contrasts are unadjusted p<.05; {int((contrast.p_holm_family_1100<.05).sum())} survive Holm. The sparse−dense decomposition distinguishes relative improvement from shared absolute improvement. paired_sparsity_contrasts.csv compares adjacent measured sparsities within seed, with a separate {len(sc)}-test Holm family. These are exploratory post-sweep comparisons. Confidence intervals displayed are pointwise, not simultaneous; SD shading is dispersion, not a confidence band.

Mean-setting tests use Holm at alpha .05 for each threshold's 220 settings; joint Holm over all 880 tests is also provided. Combined-grid tests use separate 550-setting per-threshold and 2,200-test joint families. A strict seed-run count does not imply a supported negative population mean. Per-setting leave-one-seed-out mean ranges and selection stability expose single-seed influence. Inference concerns variation over these source seeds and this fixed validation corpus, not independent validation datasets or all possible architectures.

## Separate random-mask control evidence

{len(rr)} earlier arm-A random conditions, limited to 32 experts, R0/R100 and 50–65% sparsity; validation pass count {int(rr.valid.sum()) if len(rr) else 0}. Exact source/seed/router/data/budget/dense matching and per-tensor retained-count matching are checked. No other architecture or age is covered. Positive random−learned means learned masks do better:

```
{rs_text}
```

Separate Holm family of eight paired tests. These controls cannot support full-grid random-mask generalization.

## Interpretation and smallest follow-ups

Endpoint evidence can support router-conditioned recovery where the descriptive and uncertainty criteria agree. Router-dependent differences may reflect optimization, routing or capacity effects, but endpoint CE cannot identify those mechanisms. A selected best checkpoint carries selection bias; confirm a small fixed set of strongest candidate settings against fixed boundary ages on fresh source seeds and an independent held-out evaluation corpus before making discoveries. If an interior optimum survives paired and multiplicity checks, test only that fixed age and its two neighboring measured ages plus boundaries; otherwise prioritize candidate replication. A learned-vs-random follow-up needs only the selected architecture/age/sparsity and exactly count-matched masks, rather than a new full sweep. No follow-up was launched; Dataset B remains deferred.

## Reproduction and artifacts

Run scripts/analyze.py from repository root as documented in its docstring. It refuses existing generated outputs; use a fresh analysis output tree for a rerun. Inputs are read-only and no training entrypoint is invoked. Exact plot inputs are plot_curve_data.csv, plot_heatmap_data.csv, plot_seed_data.csv and canonical_lower_sparsity.csv. Figures are PDF and PNG in figures/. Input hashes and per-condition checks are retained. See audit_report.md for protocol and provenance limitations.
''')
write('analysis_metadata.json',json.dumps(dict(validation=status,failures=len(failed),thresholds=TH,python=sys.version,numpy=np.__version__,pandas=pd.__version__,torch=torch.__version__,script_sha256=sha(__file__),input_manifest_sha256=sha(B/'manifests/run_manifest.csv')),indent=2))
print(status,pd.DataFrame(rep).to_string(index=False),flush=True)
