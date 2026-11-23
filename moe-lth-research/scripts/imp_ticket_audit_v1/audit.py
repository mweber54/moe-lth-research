"""Read-only historical audit; all outputs confined to a new audit directory.
Run: OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 .venv/bin/python scripts/imp_ticket_audit_v1/audit.py
"""
from pathlib import Path
import csv,json,hashlib,os,itertools,math
import numpy as np
import torch
from scipy import stats
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parents[2]; os.chdir(ROOT)
OUT=ROOT/'results/imp_ticket_audit_v1'; OUT.mkdir(exist_ok=True)
torch.set_num_threads(1)
def read(p):
    with open(p) as f:return list(csv.DictReader(f))
def write(name,rows):
    fields=list(dict.fromkeys(k for r in rows for k in r))
    with (OUT/name).open('w') as f:
        w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(rows)
def sha(p):
    h=hashlib.sha256()
    with open(p,'rb') as f:
        for b in iter(lambda:f.read(4*1024*1024),b''):h.update(b)
    return h.hexdigest()
def statehash(d):
    h=hashlib.sha256()
    for n in sorted(d):h.update(n.encode());h.update(d[n].detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()
def load(p):return torch.load(p,map_location='cpu',weights_only=True)
def curve(p):
    return {int(r['step']):float(r['loss']) for line in open(p) if (r:=json.loads(line)) and 'loss' in r}
def finite(x):return x is not None and np.isfinite(x)
# Broad artifact inventory, no recursive symlink traversal; explicitly referenced paths are checked below.
art=[]; docs=[]
for base in ('results','logs','experiments'):
    for root,dirs,files in os.walk(base,followlinks=False):
        dirs[:]=[d for d in dirs if d!='imp_ticket_audit_v1']
        for n in files:
            p=Path(root)/n
            if p.suffix in ('.pt','.pth','.npz','.csv','.json','.jsonl','.md'):
                art.append(dict(path=str(p),bytes=p.stat().st_size,kind=p.suffix,cohort=p.parts[1] if len(p.parts)>1 else '',symlink=p.is_symlink()))
                if p.suffix=='.md':docs.append(dict(path=str(p),sha256=sha(p)))
write('repository_artifacts.csv',art);write('research_documents.csv',docs)
raw=read('results/imp_80_85/final_recovery_metrics.csv'); manifest=read('results/imp_80_85/imp_mask_manifest.csv')
mm={r['mask_hash']:r for r in manifest}
rows=[]; inv=[]; matrices=[]; shared=[]; spectra=[]; configurations=[]; masksamples={}; within=[]
for ip,group in itertools.groupby(sorted(raw,key=lambda r:r['original_initialization_checkpoint']),key=lambda r:r['original_initialization_checkpoint']):
    group=list(group); payload=load(ip); state=payload['model'];config=payload['config']; initsha=sha(ip)
    assert payload['step']==0
    e=int(group[0]['num_experts']);seed=int(group[0]['seed'])
    print('measuring',e,seed,flush=True)
    configurations.append(dict(architecture=f'E{e}',source_seed=seed,config=config,checkpoint=ip,step=payload['step']))
    for n,t in state.items():
        if '.experts.' not in n:
            a=t.numpy().astype(np.float64);shared.append(dict(architecture=f'E{e}',source_seed=seed,parameter=n,group='router' if '.router.' in n else 'shared',shape=str(list(a.shape)),mean=float(a.mean()),std=float(a.std()),rms=float(np.sqrt(np.mean(a*a))),positive_fraction=float((a>0).mean())))
    prev=None
    for r in group:
        mrow=mm[r['mask_hash']];mp=mrow['mask_path'];mask=load(mp)
        maskhash=statehash(mask); assert maskhash==r['mask_hash'];assert initsha==r['initial_sha256']
        s=float(r['sparsity']); densepath=Path(r['matched_dense_checkpoint']); dp=densepath.parent.parent/'metrics.jsonl'
        sc=curve(r['curves_path']);dc=curve(dp);assert 2500 in sc and 2500 in dc
        dm_path=densepath.parent.parent/'metadata.json'; dm=json.load(open(dm_path)) if dm_path.exists() else {}
        match=all(dm.get(k)==r[k] for k in ('training_batch_sequence_hash','validation_batch_sequence_hash'))
        # Dense metadata fields are mandatory for valid pair labels.
        dense=float(dc[2500]); sparse=float(sc[2500]);assert abs(dense-float(r['dense_validation_ce']))<1e-8; assert abs(sparse-float(r['final_validation_loss']))<1e-8
        ident=dict(architecture=f'E{e}',source_seed=seed,mask_id=maskhash,initialization_id=initsha,sparsity=s,discovery_run=f'dataset-a-imp-r0-v1/E{e}/seed{seed}',recovery_run=r['curves_path'])
        rr=dict(ident,protocol=r['protocol'],budget=2500,rewind_step=0,absolute_sparse_ce=sparse,absolute_dense_ce=dense,sparse_minus_dense_ce=sparse-dense,relative_ce_gap=(sparse-dense)/dense,dense_pair_hashes_match=match,endpoint_curves_verified=True,strict_original_lth_status='unestablished_no_untouched_test_or_equivalence',validation_endpoint_at_or_better_dense=(sparse<=dense) if match else '',within_5pct_screen=((sparse-dense)/dense<=.05) if match else '',initial_sparse_ce=sc[0],initial_dense_ce=dc[0],immediate_mask_damage=sc[0]-dc[0],ce_step50=sc.get(50),ce_step250=sc.get(250),ce_step500=sc.get(500),early_reduction_250=sc[0]-sc[250],attempted_updates=int(r['optimizer_step_attempts']),applied_updates=int(r['optimizer_steps_applied']),dense_checkpoint=str(densepath),mask_path=mp,initialization_path=ip,train_hash=r['training_batch_sequence_hash'],validation_hash=r['validation_batch_sequence_hash'])
        for k in ('routing_entropy_initial','routing_entropy_final','mean_selected_probability_initial','mean_selected_probability_final','router_parameter_drift_normalized_final','early_mean_validation_loss'):rr[k]=float(r[k])
        gp=Path(r['curves_path']).parent/'gradient_stats/gradient_stats.jsonl'
        with gp.open() as f:g=json.loads(next(f))
        for k in ('expert_grad_norm','router_grad_norm','shared_grad_norm','grad_norm'):rr['first_logged_'+k]=g.get(k)
        rr['gradient_weight_alignment']='unavailable_in_aggregate_gradient_log'
        rr['first_gradient_timing']='first_training_attempt; not a pretraining-independent measurement'
        sub=[];sample={}
        for n,m in mask.items():
            a=state[n].numpy();b=m.numpy(); assert b.dtype==bool and b.shape==a.shape
            v=a[b];p=a[~b];absflat=np.abs(a).ravel();den=float(np.sqrt(np.mean(a*a)))
            degrees0=b.mean(0);degrees1=b.mean(1)
            # Initial magnitude ranks evaluated against full matrix distribution, ties assigned midrank.
            sorted_abs=np.sort(absflat);ret=np.abs(v)
            ranks=(np.searchsorted(sorted_abs,ret,'left')+np.searchsorted(sorted_abs,ret,'right'))/(2*len(sorted_abs))
            out=dict(ident,parameter=n,layer=n.split('.')[1],expert=n.split('.experts.')[1].split('.')[0],matrix=n.split('.')[-2],rows=a.shape[0],cols=a.shape[1],n=a.size,keep=int(b.sum()),sparsity=float(1-b.mean()),mean=float(a.mean()),std=float(a.std()),rms=den,fan_in_scaled_rms=den*np.sqrt(a.shape[1]),retained_abs_mean=float(np.abs(v).mean()),pruned_abs_mean=float(np.abs(p).mean()),retained_abs_over_pruned=float(np.abs(v).mean()/np.abs(p).mean()),retained_abs_over_matrix_rms=float(np.abs(v).mean()/den),retained_positive_fraction=float((v>0).mean()),pruned_positive_fraction=float((p>0).mean()),retained_magnitude_rank_mean=float(ranks.mean()),retained_rms_over_matrix_rms=float(np.sqrt(np.mean(v*v))/den),retained_energy_fraction=float(np.sum(v.astype(float)**2)/np.sum(a.astype(float)**2)),row_density_cv=float(degrees1.std()/degrees1.mean()),col_density_cv=float(degrees0.std()/degrees0.mean()),zero_rows=int((degrees1==0).sum()),zero_cols=int((degrees0==0).sum()))
            matrices.append(out);sub.append(out)
            idx=np.linspace(0,a.size-1,min(2048,a.size),dtype=int);sample[n]=b.ravel()[idx].copy()
            if out['expert']=='0':
                # Full singular spectrum for prespecified expert 0 in each layer/matrix; no outcome selection.
                sv=np.linalg.svd(a,compute_uv=False); sm=np.linalg.svd(a*b,compute_uv=False)
                for label,x in [('dense_initial',sv),('masked_initial',sm)]:
                    spectra.append(dict(ident,parameter=n,state=label,scope='expert0_each_layer_only',spectral_norm=float(x[0]),spectral_norm_dimension_scaled=float(x[0]/(den*(np.sqrt(a.shape[0])+np.sqrt(a.shape[1])))),stable_rank=float(np.sum(x*x)/(x[0]**2)),smallest_singular=float(x[-1])))
        for k in ('retained_abs_over_pruned','retained_positive_fraction','retained_magnitude_rank_mean','retained_rms_over_matrix_rms','row_density_cv','col_density_cv','retained_energy_fraction'):
            rr[k]=float(np.mean([x[k] for x in sub]))
        rr['matrix_density_sd']=float(np.std([1-x['sparsity'] for x in sub]));rr['actual_sparsity']=1-sum(x['keep'] for x in sub)/sum(x['n'] for x in sub)
        exp={}
        for x in sub:exp.setdefault((x['layer'],x['expert']),[]).append(x)
        rr['expert_density_sd']=float(np.std([sum(t['keep'] for t in z)/sum(t['n'] for t in z) for z in exp.values()]))
        inv.append(dict(ident,mask_path=mp,mask_exists=True,mask_tensor_hash_verified=True,mask_file_sha256=sha(mp),initialization_path=ip,initialization_file_hash_verified=True,initial_step=0,rewind_step=0,mask_parent_hash=mrow['parent_mask_hash'],discovery_source_checkpoint=mrow['trained_checkpoint'],discovery_source_exists=Path(mrow['trained_checkpoint']).is_file(),final_checkpoint=r['trained_checkpoint'],final_checkpoint_exists=Path(r['trained_checkpoint']).is_file(),dense_checkpoint=str(densepath),dense_checkpoint_exists=densepath.is_file(),dense_pair_hashes_match=match,train_hash=rr['train_hash'],validation_hash=rr['validation_hash'],initializer='PyTorch module defaults; no custom reinitialization',schedule=r['schedule'],discovery_randomness='source seed resets before trajectory; deterministic tie policy; no independent discovery replicate'))
        if prev:
            inter=sum(int((mask[n]&prev[1][n]).sum()) for n in mask);union=sum(int((mask[n]|prev[1][n]).sum()) for n in mask)
            within.append(dict(architecture=f'E{e}',source_seed=seed,sparsity_a=prev[0],sparsity_b=s,jaccard=inter/union,scope='exact_within_seed_nested_masks; differing_sparsities'))
        prev=(s,mask);masksamples[(e,seed,s)]=sample;rows.append(rr)
write('comparison.csv',rows);write('provenance.csv',inv);write('matrix_initialization_mask_statistics.csv',matrices);write('shared_router_initialization.csv',shared);write('singular_values_summary.csv',spectra);write('within_seed_mask_overlap.csv',within)
(OUT/'checkpoint_configs.json').write_text(json.dumps(configurations,indent=2))
over=[]
for (ea,sa,spa),a in masksamples.items():
    for (eb,sb,spb),b in masksamples.items():
        if ea!=eb or spa!=spb or sa>=sb:continue
        inter=sum(np.count_nonzero(a[n]&b[n]) for n in a);union=sum(np.count_nonzero(a[n]|b[n]) for n in a)
        over.append(dict(architecture=f'E{ea}',seed_a=sa,seed_b=sb,sparsity=spa,sampled_coordinates=sum(len(v) for v in a.values()),jaccard=inter/union,iid_density_reference=(1-spa)/(1+spa),scope='systematic_coordinate_sample_unaligned; descriptive_only; expert_and_hidden_neuron_permutations_unresolved'))
write('cross_seed_mask_overlap.csv',over)
# Preserve supplemental tables without pretending their historical controls prove protocol matching.
for src,dst in [('results/imp_extended_recovery/sparse_dense_comparison.csv','extended_recovery.csv'),('results/imp_mask_information_v2/results.csv','mask_control_endpoints.csv'),('results/imp_mask_information_v2/paired_effects.csv','within_seed_mask_effects.csv'),('results/imp_recovery_order_stability_v1/historical_compatibility.csv','order_study_compatibility.csv'),('results/imp_recovery_order_stability_v1/historical_endpoint_readouts.csv','historical_fp32_readouts.csv')]:
    write(dst,[dict(x,source_table=src) for x in read(src)])
# Cohort-level uncertainty always uses seed units, no repeated-sparsity pooling.
summ=[];assoc=[];effects=[]
features=['retained_abs_over_pruned','retained_positive_fraction','retained_magnitude_rank_mean','retained_rms_over_matrix_rms','row_density_cv','col_density_cv','matrix_density_sd','initial_sparse_ce','immediate_mask_damage','ce_step250','ce_step500','routing_entropy_initial','router_parameter_drift_normalized_final','first_logged_expert_grad_norm']
for (arch,s),g in itertools.groupby(sorted(rows,key=lambda x:(x['architecture'],x['sparsity'])),key=lambda x:(x['architecture'],x['sparsity'])):
    g=list(g);x=np.array([r['sparse_minus_dense_ce'] for r in g]);n=len(g);ci=stats.t.ppf(.975,n-1)*x.std(ddof=1)/np.sqrt(n)
    summ.append(dict(architecture=arch,sparsity=s,n_source_seeds=n,mean_gap=x.mean(),sd=x.std(ddof=1),ci95_low=x.mean()-ci,ci95_high=x.mean()+ci,at_or_better_dense=sum(r['validation_endpoint_at_or_better_dense'] is True for r in g),within5pct=sum(r['within_5pct_screen'] is True for r in g)))
    for f in features:
        z=np.array([r[f] if finite(r.get(f)) else np.nan for r in g],float)
        if not np.isfinite(z).all() or np.ptp(z)==0:continue
        rho=stats.spearmanr(z,x).statistic
        # Descriptive leave-one-seed-out prediction increment beyond dense CE, initial sparse CE and CE250.
        base=np.array([[r['absolute_dense_ce'],r['initial_sparse_ce'],r['ce_step250']] for r in g]);y=x
        def loo(X):
            errors=[]
            for i in range(n):
                ix=np.arange(n)!=i;mu=X[ix].mean(0);sd=X[ix].std(0);sd[sd==0]=1
                tr=(X[ix]-mu)/sd;te=(X[i]-mu)/sd
                # Prespecified ridge 1, intercept unpenalized, no hyperparameter search.
                tr=np.column_stack([np.ones(n-1),tr]);te=np.r_[1,te];pen=np.diag([0]+[1]*(tr.shape[1]-1));coef=np.linalg.solve(tr.T@tr+pen,tr.T@y[ix]);errors.append((te@coef-y[i])**2)
            return np.mean(errors)
        bm=loo(base);am=bm if f in ('initial_sparse_ce','ce_step250') else loo(np.column_stack([base,z]))
        assoc.append(dict(architecture=arch,sparsity=s,n_source_seeds=n,feature=f,spearman_rho=rho,baseline_loo_mse=bm,augmented_loo_mse=am,loo_mse_improvement=bm-am,inference='exploratory_n5_many_features_no_validated_predictor'))
        win=np.array([r['validation_endpoint_at_or_better_dense'] is True for r in g]);
        effects.append(dict(architecture=arch,sparsity=s,feature=f,n_endpoint_better=int(win.sum()),n_endpoint_worse=int((~win).sum()),mean_better=float(z[win].mean()) if win.any() else '',mean_worse=float(z[~win].mean()) if (~win).any() else '',difference=float(z[win].mean()-z[~win].mean()) if win.any() and (~win).any() else '',classification='validation_endpoint_screen_not_strict_winners'))
write('seed_level_summary.csv',summ);write('exploratory_predictors.csv',assoc);write('endpoint_group_comparisons.csv',effects)
fig,axes=plt.subplots(1,3,figsize=(12,3.5),sharey=True)
for ax,e in zip(axes,(8,16,32)):
    for s,off,c in [(.8,-.035,'tab:blue'),(.85,.035,'tab:orange')]:
        g=sorted([r for r in rows if r['architecture']==f'E{e}' and r['sparsity']==s],key=lambda r:r['source_seed']);ax.scatter(np.arange(len(g))+off,[r['sparse_minus_dense_ce'] for r in g],label=f'{s:.0%}',c=c)
    ax.axhline(0,c='black',lw=.8);ax.set_xticks(range(5),[7,17,29,43,61]);ax.set_title(f'E{e}, 2500 attempts');ax.set_xlabel('Source seed')
axes[0].set_ylabel('Sparse − matched dense validation CE');axes[-1].legend();fig.tight_layout();fig.savefig(OUT/'matched_gaps.png',dpi=180);plt.close(fig)
fig,axes=plt.subplots(1,3,figsize=(12,3.5))
for ax,f in zip(axes,['retained_magnitude_rank_mean','initial_sparse_ce','ce_step250']):
    for e,c in [(8,'tab:blue'),(16,'tab:orange'),(32,'tab:green')]:
        g=[r for r in rows if r['architecture']==f'E{e}' and r['sparsity']==.8];ax.scatter([r[f] for r in g],[r['sparse_minus_dense_ce'] for r in g],label=f'E{e}',c=c)
    ax.axhline(0,c='black',lw=.8);ax.set_xlabel(f)
axes[0].set_ylabel('Sparse − dense CE');axes[-1].legend();fig.tight_layout();fig.savefig(OUT/'candidate_properties.png',dpi=180);plt.close(fig)
checks=dict(rows=len(rows),independent_architecture_seed_groups=len(configurations),all_dense_pairs_hash_match=all(r['dense_pair_hashes_match'] for r in rows),all_mask_initial_hashes_verified=True,all_curve_endpoints_verified=True,strict_winners='not_established',matrix_rows=len(matrices),artifact_rows=len(art),seed_summaries=summ)
(OUT/'validation.json').write_text(json.dumps(checks,indent=2));print(json.dumps(checks,indent=2))
