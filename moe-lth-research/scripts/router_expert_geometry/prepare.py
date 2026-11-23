from gcommon import *
from moe_lth.models import TinyMoELanguageModel
from moe_lth.data import build_dataloaders
from moe_lth.experiments import run_router_age_recovery as R

def prepare():
 OUT.mkdir(parents=True,exist_ok=True)
 if (OUT/'initial_manifest.csv').exists():raise RuntimeError('Already registered; do not overwrite')
 inventory=[];sources=read(ROOT/'results/dataset_b_lower_sparsity/runtime_sources.json')
 for sid,info in sources.items():
  for age,rec in info['checkpoints'].items():
   path=Path(rec['path']); actual=sha(path) if path.is_file() else ''
   inventory.append(dict(source_id=sid,kind='checkpoint',key=age,path=str(path),expected_sha256=rec['sha256'],actual_sha256=actual,verified=actual==rec['sha256']))
  for sp in SP:
   m=info['masks'][str(sp)]
   for kind in ('learned','random'):
    path=Path(m[kind+'_path']);actual=sha(path) if path.is_file() else ''
    inventory.append(dict(source_id=sid,kind=kind+'_mask',key=sp,path=str(path),expected_sha256=m[kind+'_file_sha256'],actual_sha256=actual,verified=actual==m[kind+'_file_sha256']))
   learned=torch.load(m['learned_path'],weights_only=True);rand=torch.load(m['random_path'],weights_only=True)
   assert P.check_masks(learned,rand)
   assert P.state_dict_hash(learned)==m['learned_state_hash'] and P.state_dict_hash(rand)==m['random_state_hash']
   assert P.state_dict_hash(P.random_mask(learned,sid,sp))==m['random_state_hash']
  cfg=info['config'];train,val=build_dataloaders(cfg['data'],128,int(cfg['seed']),reshuffle_each_epoch=False)
  batch=next(iter(train))[0][:8].clone();g=torch.Generator().manual_seed(72419);positions=torch.randperm(batch.numel(),generator=g)[:1024]
  model=TinyMoELanguageModel(cfg['model']);ck=torch.load(info['checkpoints']['100']['path'],map_location='cpu',weights_only=False);model.load_state_dict(ck['model'])
  with protected(model):
   output=model(batch);inputs=[x.reshape(-1,x.shape[-1])[positions].cpu() for x in output.pre_router_hidden_states]
  path=OUT/'probes'/f'{sid}.pt';path.parent.mkdir(exist_ok=True);torch.save(dict(tokens=batch,positions=positions,inputs=inputs),path)
  sampler_g=torch.Generator().manual_seed(int(cfg['seed']));torch.empty((),dtype=torch.int64).random_(generator=sampler_g);global_indices=torch.randperm(len(train.dataset),generator=sampler_g)[:8]
  assert all(torch.equal(train.dataset[int(idx)][0],batch[j]) for j,idx in enumerate(global_indices))
  record=dict(global_training_sequence_indices=global_indices.tolist(),sampler_seed=int(cfg['seed']),selection_seed=72419,sequence_count=8,positions=positions.tolist(),sequence_indices=(positions//batch.shape[1]).tolist(),position_indices=(positions%batch.shape[1]).tolist(),token_ids=batch.tolist(),source_reference=info['checkpoints']['100'],probe_sha256=sha(path),counterfactual_positions=list(range(128)),training_data_sha256=sha(ROOT/cfg['data']['train_path']),domain_labels='not supplied by dataloader; none invented')
  save(path.with_suffix('.json'),record)
  for key in ('train_path','validation_path'):
   p=ROOT/cfg['data'][key];inventory.append(dict(source_id=sid,kind='data',key=key,path=str(p),expected_sha256=sha(p),actual_sha256=sha(p),verified=True))
  print('VERIFIED',sid,flush=True)
 table(OUT/'input_inventory.csv',inventory)
 if not all(r['verified'] for r in inventory):raise RuntimeError('Required inputs missing or mismatched')
 save(OUT/'runtime_sources.json',sources);table(OUT/'initial_manifest.csv',grid());table(OUT/'run_manifest.csv',grid())
 save(OUT/'registration.json',dict(count=75,sparse=60,dense=15,seeds=SEEDS,ages=AGES,sparsities=SP,steps=STEPS,primary_rank=16,ranks=(8,16,32),initial_manifest_sha256=sha(OUT/'initial_manifest.csv'),inventory_sha256=sha(OUT/'input_inventory.csv'),source_state_bytes=sum(t.numel()*t.element_size() for t in ck['model'].values()),storage_strategy='CPU model-state diagnostic snapshots at all seven steps; not resumable. Final native model-only checkpoint retained. No silent continuation.'))
if __name__=='__main__':
 torch.set_num_threads(2);prepare()
