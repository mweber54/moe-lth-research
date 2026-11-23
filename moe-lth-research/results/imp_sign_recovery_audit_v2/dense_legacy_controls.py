from common import *
from sign_agreement import stats,derive,RAW
from moe_lth.pruning.masks import load_masks
@torch.inference_mode()
def main():
 data=[];manifest=rows(OLD/'variant_manifest.csv')
 for seed in SEEDS:
  r=next(r for r in manifest if int(r['seed'])==seed and r['variant']=='imp_original' and r['order']=='A')
  init=torch.load(r['initial_checkpoint'],map_location='cpu',weights_only=False,mmap=True)
  checkpoint=ROOT/f'results/imp_80_85/trajectories/e32/seed_{seed}/round_00/checkpoints/final_recovered.pt';trained=torch.load(checkpoint,map_location='cpu',weights_only=False,mmap=True)
  for mask_id in (1,2):
   r=next(r for r in manifest if int(r['seed'])==seed and r['variant']==f'random{mask_id}_original' and r['order']=='A');masks=load_masks(Path(r['mask_path']));assert p.native.recovery._mask_hash(masks)==r['mask_sha256']
   for name,m in masks.items():
    w=init['model'][name].cuda().flatten();t=trained['model'][name].cuda().flatten();mask=m.cuda().flatten()
    for group,ids in [('legacy_random_survivors',mask.nonzero().flatten()),('legacy_random_pruned',(~mask).nonzero().flatten())]:
     data.append(dict(analysis='dense',seed=seed,checkpoint_round=0,layer=int(name.split('.')[1]),expert=int(name.split('.')[4]),projection=name.split('.')[-2],tensor=name,group=group,draw=mask_id,mask_path=r['mask_path'],mask_sha256=r['mask_sha256'],**stats(w,t,ids)))
  print(seed,'legacy dense random controls complete',flush=True)
 df=derive(pd.DataFrame(data));df.to_csv(OUT/'dense_legacy_random_matrix_metrics.csv',index=False)
 for suffix,extra in [('by_layer_projection',['layer','projection']),('per_seed',[])]:
  x=df.groupby(['analysis','seed','checkpoint_round','group','draw']+extra)[RAW].sum().reset_index();derive(x).to_csv(OUT/f'dense_legacy_random_{suffix}.csv',index=False)
if __name__=='__main__':main()
