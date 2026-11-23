from gcommon import *
from moe_lth.models import TinyMoELanguageModel
from moe_lth.pruning.magnitude_prune import expert_local_magnitude_masks

def verify():
 torch.set_num_threads(2);sources=read(OUT/'runtime_sources.json');canonical={(int(r['num_experts']),int(r['seed']),int(r['router_age'])):r for r in rows(ROOT/'results/final_dataset_a/inputs/source_checkpoint_manifest.csv')};counts=[];inspection=[]
 for sid,info in sources.items():
  seed=int(info['config']['seed'])
  for age in AGES:
   rec=info['checkpoints'][str(age)];assert rec['sha256']==canonical[32,seed,age]['sha256'];ck=torch.load(rec['path'],map_location='cpu',weights_only=False);assert ck['step']==25*age;assert ck['config']['model']==info['config']['model']
  model=TinyMoELanguageModel(info['config']['model']);model.load_state_dict(ck['model'])
  if seed==7:
   inspection=[dict(name=n,shape=list(v.shape),pruned=('.moe.experts.' in n and n.endswith('.weight')),recovery_initialization='selected source age' if '.router.' in n else 'source R0',trainable=not n.endswith('.temperature')) for n,v in model.state_dict().items()]
  for sp in SP:
   m=info['masks'][str(sp)];learned=torch.load(m['learned_path'],weights_only=True);rand=torch.load(m['random_path'],weights_only=True);generated=expert_local_magnitude_masks(model,sp);assert P.state_dict_hash(generated)==m['learned_state_hash']
   for name,v in learned.items():
    assert int(v.sum())==int(rand[name].sum());counts.append(dict(source_id=sid,sparsity=sp,tensor=name,total=v.numel(),learned_retained=int(v.sum()),random_retained=int(rand[name].sum()),learned_state_hash=m['learned_state_hash'],random_state_hash=m['random_state_hash']))
  print('SOURCE PROTOCOL VERIFIED',seed,flush=True)
 table(OUT/'mask_counts.csv',counts);save(OUT/'model_tensor_inventory.json',inspection);save(OUT/'validation/source_protocol.json',dict(passed=True,finalized_checkpoint_identities=True,actual_checkpoint_steps=True,learned_masks_regenerated=True,random_masks_regenerated_in_preparation=True,per_tensor_counts_verified=True))
if __name__=='__main__':verify()
