"""Freeze historical cohort and inspect checkpoint contents before new geometry."""
from common import *
import argparse
CAN=ROOT/'results/final_dataset_a/canonical/dataset_a_canonical.csv'
def main():
 p=argparse.ArgumentParser();p.add_argument('--verify-local',action='store_true');a=p.parse_args()
 if a.verify_local:
  manifest=frozen();records=[]
  for x in manifest['files']:records.append(verify_file(x['path'],x['sha256']))
  save(OUT/'hosts'/f'{os.uname().nodename}.json',dict(host=os.uname().nodename,files=len(records),verified=True,torch=torch.__version__,cuda=torch.cuda.is_available(),unix=time.time()));return
 if (OUT/'freeze.json').exists():raise RuntimeError('Frozen registration exists; refusing overwrite')
 with CAN.open() as f:rows=list(csv.DictReader(f))
 chosen=[r for r in rows if int(r['num_experts'])==32 and int(r['seed']) in SEEDS and r['router_age']=='0' and float(r['sparsity']) in (.8,.85)]
 assert len(chosen)==10 and len({(r['seed'],r['sparsity']) for r in chosen})==10
 files={};coverage=[];jobs=[];tickets=[];histories=[]
 def file(p,h=None):
  x=verify_file(p,h);files[str(p)]=x;return x
 def checkpoint(p,role,seed,h=None):
  x=file(p,h);d=load(p);coverage.append(dict(seed=seed,role=role,**x,step=d.get('step'),keys=','.join(sorted(d)),optimizer='optimizer' in d,scaler='scaler' in d,rng=all(k in d for k in ('python_rng','numpy_rng','torch_rng','cuda_rng')),model='model' in d));return d
 for seed in SEEDS:
  pair=[r for r in chosen if int(r['seed'])==seed];b=pair[0]
  for field in ('source_checkpoint_path','source_checkpoint_hash','dense_artifact_path','dense_final_checkpoint_sha256','data_hash','validation_hash'):
   assert pair[0][field]==pair[1][field],(seed,field)
  init=Path(b['source_checkpoint_path']);d=checkpoint(init,'original_initialization',seed,b['source_checkpoint_hash']);assert d['step']==0
  source=init.parent/'step_2500.pt';checkpoint(source,'source_dense_final_reference',seed)
  config=d['config'];assert config['model']['num_experts']==32
  for key in ('train_path','validation_path'):file(ROOT/config['data'][key])
  root=ROOT/f'results/imp_80_85/trajectories/e32/seed_{seed}'
  for k in range(9):
   st=read(root/f'round_{k:02d}/stage.json');meta=read(root/f'mask_{k+1:02d}.json');file(root/f'round_{k:02d}/stage.json');file(root/f'mask_{k+1:02d}.json');checkpoint(Path(meta['trained_checkpoint']),f'IMP_round_{k:02d}_mask_source',seed,st['checkpoint_sha256']);file(root/f'mask_{k+1:02d}.pt')
   assert st['identity']['round']==k and st['identity']['initial_sha256']==b['source_checkpoint_hash']
   histories.append(dict(seed=seed,round=k,trained_sparsity=st['sparsity'],target=meta['target_sparsity'],mask_hash=meta['mask_hash'],parent_mask_hash=meta['parent_mask_hash'],attempted=st['optimizer_step_attempts'],applied=st['optimizer_steps_applied']))
  dense_source=Path(b['dense_artifact_path'])/'checkpoints/final_recovered.pt';checkpoint(dense_source,'dense_recovery_2500',seed,b['dense_final_checkpoint_sha256'])
  dm=read(ROOT/f'results/imp_extended_recovery/e32/seed_{seed}/dense/extension_manifest.json')
  for sp in (None,.8,.85):
   label='dense' if sp is None else f'final_s{sp:.2f}';ext=ROOT/f'results/imp_extended_recovery/e32/seed_{seed}'/label;em=read(ext/'extension_manifest.json');file(ext/'extension_manifest.json');file(ext/'extended_summary.json');assert em==read(ext/'extended_summary.json')
   assert em['identity']['training_batch_hash']==dm['identity']['training_batch_hash'] and em['identity']['validation_batch_hash']==dm['identity']['validation_batch_hash']
   src=dense_source if sp is None else root/label/'checkpoints/final_recovered.pt';checkpoint(src,label+'_boundary',seed,em['source_checkpoint_sha256'])
   mask=None
   if sp is not None:
    mp=root/f'mask_{8 if sp==.8 else 9:02d}.pt';mm=load_masks(mp);mask=str(mp);assert R._mask_hash(mm)==em['mask_hash'];st=read(root/label/'stage.json');file(root/label/'stage.json');assert st['initial_router_state_hash']==read(root/'round_00/stage.json')['initial_router_state_hash'];assert st['identity']['initial_sha256']==b['source_checkpoint_hash']
   cps={}
   for step in (3000,4000,5000):
    cp=ext/f'checkpoints/step_{step}.pt';payload=checkpoint(cp,f'{label}_recovery_{step}',seed,em.get('step_5000_checkpoint_sha256') if step==5000 else None);assert payload['step']==step and payload['identity']==em['identity'];cps[str(step)]=str(cp)
   metrics=[json.loads(l) for l in (ext/'metrics_absolute.jsonl').read_text().splitlines()];file(ext/'metrics_absolute.jsonl');last=metrics[-1];assert last['absolute_step']==5000 and abs(last['loss']-em['extension_final_validation_cross_entropy'])<1e-12
   jobs.append(dict(id=f'e32_seed{seed}_{label}',seed=seed,sparsity=sp,initial=str(init),source_reference=str(source),boundary=str(src),mask=mask,historical_checkpoints=cps,config=config,canonical_pair=b,dense_id=f'e32_seed{seed}_dense',extension_identity=em['identity']))
   if sp is not None:
    sc=em['extension_final_validation_cross_entropy'];dc=dm['extension_final_validation_cross_entropy'];tickets.append(dict(seed=seed,sparsity=sp,sparse_ce=sc,dense_ce=dc,gap=sc-dc,strict_success=sc<=dc,label_origin='historical_validation',checkpoint_sha256=em['step_5000_checkpoint_sha256']))
 assert len({j['boundary'] for j in jobs if j['sparsity'] is None})==5
 # No geometric outcome has been calculated by this audit.
 manifest=dict(version=1,exploratory_post_hoc=True,seeds=list(SEEDS),num_experts=32,conditions=jobs,files=list(files.values()),canonical_index=file(CAN),independent_unit='source seed',unique_dense=5,created_unix=time.time())
 save(OUT/'cohort.json',manifest);table(OUT/'checkpoint_coverage.csv',coverage);table(OUT/'imp_history.csv',histories);table(OUT/'historical_tickets.csv',tickets)
 save(OUT/'freeze.json',dict(cohort_sha256=sha(OUT/'cohort.json'),protocol_sha256=sha(OUT/'protocol.md'),unix=time.time(),geometry_examined=False))
 print(json.dumps(tickets,indent=2))
if __name__=='__main__':torch.set_num_threads(2);main()
