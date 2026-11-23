from common import *
from moe_lth.pruning.masks import load_masks
records=[]
for seed in SEEDS:
 root=ROOT/f'results/imp_80_85/trajectories/e32/seed_{seed}';final_meta=read(root/'mask_08.json');parent_meta=read(root/'mask_07.json');final=load_masks(root/'mask_08.pt');parent=load_masks(root/'mask_07.pt')
 fh=p.native.recovery._mask_hash(final);ph=p.native.recovery._mask_hash(parent)
 for analysis,round_idx in [('discovery',7),('dense',0)]:
  stage=read(root/f'round_{round_idx:02d}/stage.json');checkpoint=root/f'round_{round_idx:02d}/checkpoints/final_recovered.pt';payload=torch.load(checkpoint,map_location='cpu',weights_only=False,mmap=True)
  flags=dict(final_mask_hash_matches=fh==final_meta['mask_hash'],parent_mask_hash_matches=ph==parent_meta['mask_hash']==final_meta['parent_mask_hash'],discovery_parent_identity_matches=(stage['mask_hash']==ph) if analysis=='discovery' else stage['sparsity']==0,actual_selecting_checkpoint_matches=(checkpoint.resolve()==Path(final_meta['trained_checkpoint']).resolve()) if analysis=='discovery' else stage['identity']['round']==0,config_seed_matches=payload['config']['seed']==seed,initialization_identity_matches=stage['identity']['initial_sha256']==final_meta['initial_sha256'])
  assert all(flags.values()),(seed,analysis,flags)
  records.append(dict(analysis=analysis,seed=seed,checkpoint_round=round_idx,checkpoint=str(checkpoint),**flags))
write('checkpoint_role_audit.csv',records)
print('ten checkpoint roles and both mask identities per seed verified')
