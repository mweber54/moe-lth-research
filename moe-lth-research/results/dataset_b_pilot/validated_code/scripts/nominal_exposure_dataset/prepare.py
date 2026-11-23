"""Read-only Dataset A compatibility reuse plus immutable pilot manifest."""
from pilot_common import *
import time,shutil,platform,subprocess

def main():
 torch.set_num_threads(2)
 dest=OUT/'manifests/initial_manifest.csv'
 if dest.exists():raise RuntimeError('Pilot already prepared; refuse to rewrite immutable manifest')
 f=pd.read_csv(AD/'canonical/dataset_a_canonical.csv');d=pd.read_csv(AD/'canonical/dataset_a_dense.csv');src=pd.read_csv(AD/'canonical/dataset_a_sources.csv');cp=pd.read_csv(AD/'inputs/source_checkpoint_manifest.csv');mm=pd.read_csv(AD/'inputs/mask_manifest.csv');hashes=read(AD/'dataset_a_input_hashes.json');rows=grid();sources={};inputs={}
 def linked(p):
  p=Path(p)
  if not p.exists():p=Path(str(p).replace('router_conditioned_fixed_budget_dataset/','router_conditioned_fixed_budget_dataset.HOME_BACKUP/'))
  if not p.is_file():raise FileNotFoundError(p)
  old=hashes.get(str(p));st=p.stat()
  if old and old['bytes']==st.st_size and old['mtime_ns']==st.st_mtime_ns:dg=old['sha256']
  else:dg=sha(p)
  inputs[str(p)]=dict(sha256=dg,bytes=st.st_size,mtime_ns=st.st_mtime_ns);return str(p),dg
 for seed in SEEDS:
  source=src[(src.num_experts==32)&(src.seed==seed)].iloc[0];assert source.valid
  c=A.load(Path(source.path)/'checkpoints/step_0.pt')['config'];assert not A.protocol_errors(c,32,seed)
  checkpoints={}
  for age in AGES:
   r=cp[(cp.num_experts==32)&(cp.seed==seed)&(cp.router_age==age)].iloc[0];path,digest=linked(r['path']);assert digest==r.sha256;checkpoints[str(age)]={'path':path,'sha256':digest,'router_state_hash':r.router,'model_hash':r.model_hash}
  masks={}
  for sp in SP:
   r=mm[(mm.num_experts==32)&(mm.seed==seed)&(mm.target_sparsity==sp)].iloc[0];path,digest=linked(r.path);learned=torch.load(path,map_location='cpu',weights_only=True);assert state_dict_hash(learned)==r.mask_hash
   rp=OUT/f'masks/short_seed{seed}_s{sp:.2f}_random.pt';randomized=random_mask(learned,f'source_e32_seed{seed}_T2500',sp);assert check_masks(learned,randomized);tmp=rp.with_suffix('.pt.tmp');torch.save(randomized,tmp);tmp.replace(rp)
   masks[str(sp)]={'learned_path':path,'learned_state_hash':r.mask_hash,'learned_file_sha256':digest,'random_path':str(rp),'random_state_hash':state_dict_hash(randomized),'random_file_sha256':sha(rp),'random_seed':mask_seed(f'source_e32_seed{seed}_T2500',sp),'keep_count':int(r.keep_count),'pruned_count':int(r.pruned_count)}
  sources[str(seed)]={'path':source.path,'checkpoints':checkpoints,'masks':masks,'config':c}
  cc=json.loads(json.dumps(c));cc['training'].update(steps=10000,checkpoint_steps=[0,2500,5000,10000],checkpoint_only_steps=[1250],record_exposure=True,record_train_routes=False,record_rich_routes=False,save_optimizer=False);cc['output_dir']=None;jsave(OUT/f'configs/source_e32_seed{seed}_T10000.json',cc)
 for r in rows:
  cid=r['condition_id'];r.update(status='PENDING',artifact_path=None,artifact_sha256=None,block_reason=None)
  if r['job_type']=='source':continue
  r['mask_identity']=f"{r['source_mask_family']}_s{r['sparsity']:.2f}_{r['mask_type']}" if r['job_type']=='sparse' else 'dense_no_mask'
  if r['arm']=='A' and r['mask_type']!='random':
   if r['job_type']=='dense':matches=d[(d.architecture==32)&(d.seed==r['seed'])&(d.router_age==r['router_age'])];pathcol='path'
   else:matches=f[(f.num_experts==32)&(f.seed==r['seed'])&(f.router_age==r['router_age'])&(f.sparsity==r['sparsity'])];pathcol='sparse_artifact_path'
   if len(matches)==1:
    path=Path(matches.iloc[0][pathcol]);meta=read(path/'metadata.json');summary=read(path/'summary.json');curve=[json.loads(l) for l in (path/'metrics.jsonl').read_text().splitlines()]
    assert meta['recovery_steps']==2500 and meta['seed']==r['seed'] and meta['router_step']==r['router_checkpoint_step'] and meta.get('router_mode','trainable')=='trainable' and meta.get('temperature',1)==1 and meta['optimizer']=='fresh_AdamW'
    assert len(curve)==51 and curve[-1]['step']==2500 and abs(curve[-1]['loss']-summary['final_validation_loss'])<1e-10
    pp,digest=linked(path/'checkpoints/final_recovered.pt');assert digest==matches.iloc[0]['final_checkpoint_sha256' if r['job_type']=='dense' else 'sparse_final_checkpoint_sha256'];r.update(status='REUSED',artifact_path=str(path),artifact_sha256=digest)
   else:r.update(status='BLOCKED_REFERENCE',block_reason='Dataset A shard artifact unavailable; do not rerun or invent historical reference')
 # Random A requires its learned endpoint and dense comparator; all five seeds remain registered.
 lookup={r['condition_id']:r for r in rows}
 for r in rows:
  if r.get('arm')=='A' and r.get('mask_type')=='random':
   learned=lookup[jobid('A',r['seed'],r['router_age'],'sparse',r['sparsity'],'learned')]
   if learned['status']!='REUSED' or lookup[r['dense_condition_id']]['status']!='REUSED':r.update(status='BLOCKED_REFERENCE',block_reason='Await validated A learned/dense artifacts; configured seed is retained')
 cfg={'protocol_version':'dataset-b-nominal-exposure-v1','architectures_configured':{str(e):attempts(e) for e in (8,12,16,24,32)},'launch_experts':[32],'arms':ARMS,'seeds':SEEDS,'router_ages':AGES,'sparsities':SP,'mask_types':{'learned':'expert-local source-final magnitude','random':'exact per-tensor counts matched; one independent draw per source/seed/sparsity'},'recovery_eval_interval':50,'primary_early_auc_window_steps':1250,'schedule':'constant 0.0003; unchanged through 2500','continuation':'fresh initialization; no historical continuation; no short/long prefix consolidation','reuse_tolerance':1e-10,'dense_matching':'same source identity, seed, router absolute step and recovery budget; no dense matching claim based on a post-hoc loss tolerance','diagnostic_steps':[0,2500,10000],'source_extra_absolute_checkpoints':[1250,2500],'gpu_policy':'wait until no external compute process on GPU 0; exclusive pilot flock; no process killing','minimum_free_storage_gib':30,'full_sweep_authorized':False}
 jsave(OUT/'configs/design.json',cfg);jsave(OUT/'manifests/short_sources.json',sources);jsave(OUT/'manifests/input_provenance.json',inputs);csv(dest,rows);csv(OUT/'manifests/run_manifest.csv',rows)
 immutable={str(p.relative_to(OUT)):sha(p) for folder in ['configs','manifests','masks'] for p in (OUT/folder).rglob('*') if p.is_file() and p.name not in ['run_manifest.csv','immutable_hashes.json']};jsave(OUT/'manifests/immutable_hashes.json',immutable)
 jsave(OUT/'audits/preparation.json',{'counts':pd.DataFrame(rows).groupby(['job_type','status']).size().to_dict().__str__(),'existing_A_validated_sparse':sum(r.get('arm')=='A' and r.get('job_type')=='sparse' and r.get('mask_type')=='learned' and r.get('status')=='REUSED' for r in rows),'runtime_environment':{'python':sys.version,'torch':torch.__version__},'git_commit':subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),'source_artifacts_validated':5,'long_source_reuse':'none admitted; historical fixed-budget source inventories contain only 2500-attempt trajectories','timestamp':time.time()})
 print(pd.DataFrame(rows).groupby(['stage','job_type','status']).size().to_string())
if __name__=='__main__':main()
