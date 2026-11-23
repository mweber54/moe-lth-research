"""Read-only measurement provenance audit; never imports a training entry point for execution."""
from common import *
import argparse
import pandas as pd
REVIEW=OUT/'analysis_audit_v2'
def array_hash(a):
 a=np.ascontiguousarray(a);return hashlib.sha256(str((a.dtype,a.shape)).encode()+a.tobytes()).hexdigest()
def paths_for(j,track):
 if track=='historical':return {0:None,2500:j['boundary'],**{int(k):v for k,v in j['historical_checkpoints'].items()}}
 d=read(OUT/'reruns'/j['id']/'completed.json');return {t:str(Path(d['attempt'])/'snapshots'/f'step_{t:04d}.pt') for t in STEPS}
def audit_seed(seed):
 f=frozen();files={v['path']:v['sha256'] for v in f['files']};jobs=[j for j in f['conditions'] if j['seed']==seed];probe_path=OUT/'probes'/f'seed_{seed}.pt';probe=load(probe_path);probe_id=verify_file(probe_path,read(probe_path.with_suffix('.json'))['sha256']);poshash=array_hash(probe['positions'].numpy());tokenhash=array_hash(probe['tokens'].numpy());records=[];families=[];pair_records=[];initials={};hashcache={};summaries=[]
 def identity(p,expected=None):
  if p not in hashcache:hashcache[p]=verify_file(p,expected)
  if expected:assert hashcache[p]['sha256']==expected
  return hashcache[p]
 for j in jobs:
  init=state(j['initial']);mask=load_masks(j['mask']) if j['mask'] else {};initials[j['id']]=masked(init,mask)
  for track in ('historical','rerun'):
   for step,cp in paths_for(j,track).items():
    folder=OUT/'analysis'/track/j['id']/f'step_{step:04d}';meta=read(folder/'completed.json');assert (meta['condition'],meta['seed'],meta['step'],meta['track'])==(j['id'],seed,step,track)
    if cp is None:s=initials[j['id']];origin={'path':'reconstructed:'+j['initial'],'sha256':identity(j['initial'],files[j['initial']])['sha256']}
    else:
     expected=files.get(cp) if track=='historical' else read(Path(cp).with_suffix('.json'))['snapshot_sha256'];origin=identity(cp,expected);payload=load(cp);assert payload['step']==step;s=payload['model']
    modelhash=state_dict_hash(s);components={g:state_dict_hash({n:t for n,t in s.items() if parameter_group(n)==g}) for g in ('expert','router','shared')}
    if track=='rerun':assert modelhash==read(Path(cp).with_suffix('.json'))['model_hash']
    for n,m in mask.items():assert torch.count_nonzero(s[n][~m]).item()==0
    raw=np.load(folder/'raw.npz');assert np.array_equal(raw['tokens'],probe['tokens'].numpy()) and np.array_equal(raw['positions'],probe['positions'].numpy())
    embed=s['token_embedding.weight'][probe['tokens'].reshape(-1)[probe['positions']]].numpy();assert np.array_equal(embed,raw['representation_token_embedding'])
    valid_outputs=True
    for name,h in meta['outputs'].items():
     actual=sha(folder/name);assert actual==h,(folder,name);families.append(dict(track=track,condition=j['id'],seed=seed,sparsity=j['sparsity'],step=step,family=name,sha256=actual,cache_key=meta['key']))
    for mode in ('fixed_source','native'):
     for l in range(4):
      # Scores recomputed on the exact cached inputs bind router rows to this checkpoint.
      x=raw[f'representation_router_input_L{l}'] if mode=='native' else probe['source_full_inputs'][l].reshape(-1,256)[probe['positions']].numpy();expected_scores=x@s[f'blocks.{l}.moe.router.projection.weight'].numpy().T
      difference=float(np.max(np.abs(expected_scores-raw[f'{mode}_L{l}_scores'])));assert difference<3e-5,(track,j['id'],step,l,difference)
      for selection in ('all_expert','fixed_initial','fixed_source','current_proposed','current_accepted'):
       pair_records.append(dict(track=track,condition=j['id'],seed=seed,sparsity=j['sparsity'],step=step,mode=mode,layer=l,selection=selection,pair_sha256=array_hash(raw[f'{mode}_L{l}_selection_{selection}']),input_sha256=array_hash(x),token_sha256=tokenhash,position_sha256=poshash,raw_path=str(folder/'raw.npz'),cache_key=meta['key']))
    records.append(dict(track=track,condition=j['id'],seed=seed,sparsity=j['sparsity'],step=step,checkpoint_path=origin['path'],checkpoint_sha256=origin['sha256'],model_hash=modelhash,**{g+'_hash':h for g,h in components.items()},probe_path=str(probe_path),probe_sha256=probe_id['sha256'],probe_config_sha256=sha(OUT/'probe_config.json'),token_sha256=tokenhash,position_sha256=poshash,source_reference=j['source_reference'],mask_selection_reference=read(Path(j['mask']).with_suffix('.json'))['trained_checkpoint'] if j['mask'] else j['source_reference'],initial_reference=j['initial'],same_step_dense_condition=j['dense_id'],fixed_dense_final_reference=paths_for(next(x for x in jobs if x['sparsity'] is None),track)[5000],raw_path=str(folder/'raw.npz'),cache_key=meta['key'],measurement_completed_sha256=sha(folder/'completed.json'),recorded_measurement_ce=meta['validation_ce'],valid_output_hashes=valid_outputs,components_verified=True))
    print('AUDIT',seed,track,j['id'],step,flush=True);del s
 table(REVIEW/'provenance'/f'seed_{seed}_states.csv',records);table(REVIEW/'provenance'/f'seed_{seed}_families.csv',families);table(REVIEW/'provenance'/f'seed_{seed}_pairs.csv',pair_records)
 # Link actual historical/rerun tensors and measurement families on exactly the same step.
 d=pd.DataFrame(records);a=d[d.track.eq('historical')].merge(d[d.track.eq('rerun')],on=['condition','seed','step'],suffixes=('_historical','_rerun'),validate='one_to_one')
 for g in ('model','expert','router','shared'):a[g+'_identical']=a[g+'_hash_historical']==a[g+'_hash_rerun']
 a.to_csv(REVIEW/'provenance'/f'seed_{seed}_cross_track.csv',index=False)
 save(REVIEW/'provenance'/f'seed_{seed}_complete.json',dict(seed=seed,states=len(records),common_states=len(a),identical_models=int(a.model_identical.sum()),passed=True,unix=time.time()))
def main():
 p=argparse.ArgumentParser();p.add_argument('--seeds',type=int,nargs='+',required=True);a=p.parse_args();torch.set_num_threads(2)
 for seed in a.seeds:audit_seed(seed)
if __name__=='__main__':main()
