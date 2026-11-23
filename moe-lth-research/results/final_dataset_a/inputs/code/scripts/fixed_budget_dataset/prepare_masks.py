"""Prepare missing canonical R100 masks from already validated sources; no training."""
from common import *
import os,fcntl
import pandas as pd
from moe_lth.experiments import run_router_age_recovery as recovery
from moe_lth.pruning.magnitude_prune import expert_local_magnitude_masks

def main():
 lock=(OUT/'prepare_masks.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
 jobs=pd.read_csv(OUT/'run_manifest.csv');sources=jobs[jobs.job_type=='source']
 assert len(sources)==25 and sources.status.isin(['COMPLETE','REUSED']).all()
 torch.set_num_threads(4);rows=[]
 for source in sources.itertuples():
  e=int(source.architecture);seed=int(source.seed);root=Path(source.artifact_path)
  missing=[sp for sp in SP if not (OUT/f'masks/e{e}/seed_{seed}/s{sp:.2f}.pt').exists()]
  if not missing:continue
  path=root/'checkpoints/step_2500.pt';ck=load(path)
  assert ck['step']==2500 and not protocol_errors(ck['config'],e,seed)
  proof=jread(OUT/'progress_cache'/f'{source.condition_id}.json')
  mature=next(x for x in proof['checkpoints'] if x['router_age']==100)
  assert sha(path)==mature['sha256'] and state_dict_hash(ck['model'])==mature['model_hash']
  model=recovery.load_model_from_checkpoint(ck['config']['model'],str(path),torch.device('cpu'))
  assert state_dict_hash(model.state_dict())==mature['model_hash']
  for sp in missing:
   mask=expert_local_magnitude_masks(model,sp);digest=state_dict_hash(mask)
   total=sum(v.numel() for v in mask.values());keep=sum(int(v.sum()) for v in mask.values())
   expected_keep=0
   experts={}
   for name,value in mask.items():
    assert not name.endswith('bias');key=name.split('.experts.')[0]+'.experts.'+name.split('.experts.')[1].split('.')[0]
    experts.setdefault(key,[]).append(value)
   assert len(experts)==4*e
   for values in experts.values():
    assert len(values)==2
    expected=max(1,round(sum(v.numel() for v in values)*(1-sp)))
    assert sum(int(v.sum()) for v in values)==expected
    expected_keep+=expected
   assert keep==expected_keep
   dest=OUT/f'masks/e{e}/seed_{seed}/s{sp:.2f}.pt';dest.parent.mkdir(parents=True,exist_ok=True)
   temporary=dest.with_name(dest.name+f'.prepare-{os.getpid()}.tmp');torch.save(mask,temporary)
   try:
    # Atomic no-overwrite publication. If another worker created it, verify equality.
    os.link(temporary,dest)
   except FileExistsError:
    assert state_dict_hash(torch.load(dest,map_location='cpu',weights_only=True))==digest
   finally:temporary.unlink()
   rows.append({'num_experts':e,'seed':seed,'sparsity':sp,'mask_hash':digest,'keep_count':keep,'pruned_count':total-keep,'actual_sparsity':1-keep/total,'source_checkpoint_sha256':mature['sha256'],'path':str(dest)})
   print('MASK READY',e,seed,sp,digest,flush=True)
  del model,ck
 writecsv(OUT/'additional_mask_preparation.csv',rows)
 print('MISSING MASK PREPARATION COMPLETE',len(rows),flush=True)
if __name__=='__main__':main()
