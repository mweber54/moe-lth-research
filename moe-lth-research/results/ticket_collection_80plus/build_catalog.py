import csv,json,os,pathlib,collections
root=pathlib.Path.cwd();out=root/'results/ticket_collection_80plus'
def save(name,rows):
 if not rows:return
 with (out/name).open('w',newline='') as f:
  w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
rows=[]
for r in csv.DictReader((root/'results/final_dataset_a/canonical/dataset_a_canonical.csv').open()):
 if float(r['sparsity'])<.8:continue
 p=pathlib.Path(r['sparse_artifact_path']); cp=p/'checkpoints/final_recovered.pt';mask=pathlib.Path(r['mask_artifact_path']); init=pathlib.Path(r['source_artifact_path'])/'checkpoints/step_0.pt'
 rows.append(dict(condition_id=r['condition_id'],num_experts=r['num_experts'],seed=r['seed'],target_sparsity=r['sparsity'],actual_sparsity=r['actual_sparsity'],router_age_percent=r['router_age'],expert_shared_initialization='original_source_step_0',router_checkpoint=r['source_checkpoint_path'],initialization_checkpoint=str(init),initialization_exists=init.exists(),mask_hash=r['mask_hash'],mask_path=str(mask),mask_exists=mask.exists(),final_checkpoint=str(cp),final_checkpoint_exists=cp.exists(),sparse_loss=r['sparse_final_validation_loss'],dense_loss=r['dense_final_validation_loss'],relative_loss_gap_pct=100*float(r['ticket_gap'])/float(r['dense_final_validation_loss']),dense_artifact_path=r['dense_artifact_path'],validated=r['condition_valid']))
save('canonical_recoveries.csv',rows)
groups=collections.defaultdict(list)
for r in rows:groups[r['mask_hash']].append(r)
best=sorted([min(g,key=lambda r:r['relative_loss_gap_pct']) for g in groups.values()],key=lambda r:r['relative_loss_gap_pct'])
save('distinct_masks_best_router.csv',best);save('shortlist_80pct_25_masks.csv',[r for r in best if float(r['target_sparsity'])==.8]);save('within_5pct_dense.csv',[r for r in rows if r['relative_loss_gap_pct']<=5])
# Broader local observations: retain provenance, do not claim canonical validation.
observed=[]
skip={'geometry','logs','figures','trajectories','cache','diagnostic_cache','progress_cache','__pycache__','capture_recovery','amplitude_sources','failed_builds','transformed','schedules','probes','structural_analysis'}
for base,dirs,files in os.walk(root/'results',followlinks=False):
 dirs[:]=[d for d in dirs if d not in skip]
 if 'metadata.json' not in files:continue
 p=pathlib.Path(base)
 try:d=json.loads((p/'metadata.json').read_text());sp=float(d.get('sparsity',0))
 except (ValueError,OSError):continue
 if sp<.8-1e-6 or not d.get('mask_hash'):continue
 cp=p/'checkpoints/final_recovered.pt';summ=p/'summary.json'
 observed.append(dict(family=p.relative_to(root/'results').parts[0],artifact_path=str(p),target_sparsity=sp,mask_hash=d['mask_hash'],mask_type=d.get('mask_type','unspecified'),router_age_percent=d.get('router_age_percent',''),router_mode=d.get('router_mode',''),integrity_checks_passed=d.get('integrity_checks_passed',''),final_checkpoint_exists=cp.exists(),summary_exists=summ.exists()))
save('broader_local_observations.csv',observed)
counts=dict(canonical_recoveries=len(rows),canonical_distinct_masks=len(groups),canonical_checkpoint_exists=sum(r['final_checkpoint_exists'] for r in rows),canonical_initialization_exists=sum(r['initialization_exists'] for r in rows),canonical_mask_exists=sum(r['mask_exists'] for r in rows),canonical_within_5pct=sum(r['relative_loss_gap_pct']<=5 for r in rows),broader_metadata_records=len(observed),broader_observed_unique_masks=len({r['mask_hash'] for r in observed}),union_observed_unique_masks=len(set(groups)|{r['mask_hash'] for r in observed}),broader_family_counts=dict(collections.Counter(r['family'] for r in observed)))
(out/'audit_counts.json').write_text(json.dumps(counts,indent=2)+'\n');print(json.dumps(counts,indent=2),flush=True)
