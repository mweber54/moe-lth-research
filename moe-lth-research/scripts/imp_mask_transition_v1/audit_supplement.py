"""Verify native IMP derivation and available state/alias provenance, without outcomes."""
from common import *
def main():
 m=read(OUT/'manifest.json');rows=[];aliases=[];files={};started=time.time()
 for v in m['variants']:
  payload=load(v['initial']);assert payload['step']==0
  rows.append(dict(seed=v['seed'],kind='original_initialization',path=v['initial'],model_only='optimizer' not in payload,available_payload_keys=sorted(payload),exact_optimizer_continuation=False))
  del payload
  for k in range(1,9):
   meta=read(Path(v['mask']).parent/f'mask_{k:02d}.json');stage=read(Path(meta['stage_path'])/'stage.json')
   assert meta['initial_sha256']==sha(v['initial']) and meta['original_initialization_checkpoint']==v['initial']
   assert stage['optimizer_step_attempts']==2500 and stage['original_initialization_checkpoint']==v['initial']
   cp=Path(meta['trained_checkpoint']);h=sha(cp);assert h==stage['checkpoint_sha256']
   files[str(cp)]=h;files[str(Path(meta['stage_path'])/'stage.json')]=sha(Path(meta['stage_path'])/'stage.json')
   rows.append(dict(seed=v['seed'],kind='IMP_parent_recovered_training',round=k-1,path=str(cp),sha256=h,model_only=True,exact_optimizer_continuation=False))
  for condition,ref in v['references'].items():
   for offset in (0,2500):
    p=Path(ref['attempt'])/f'segment_{offset}/interruption.pt'
    # Many historical reruns retained only model checkpoints and separate optimizer snapshots.
    rows.append(dict(seed=v['seed'],condition=condition,kind='interruption_state',path=str(p),exists=p.exists(),exact_optimizer_continuation=False,note='State not loaded or verified for continuation; references are readout-only'))
    for p in Path(ref['attempt']).glob(f'optimizer_{offset}_*.pt'):
     files[str(p)]=sha(p);rows.append(dict(seed=v['seed'],condition=condition,kind='optimizer_scaler_snapshot',path=str(p),sha256=files[str(p)],exact_optimizer_continuation=False,note='Optimizer/scaler alone lack sampler/RNG; no exact continuation claim'))
   c=next(c for c in read(HIST/'cohort.json')['conditions'] if c['seed']==v['seed'] and (c['sparsity'] is None if condition=='dense' else c['sparsity']==.8))
   for step,path in c['historical_checkpoints'].items():
    selected=ref['snapshots'].get(step,ref['checkpoint'] if step=='5000' else None)
    # Historical native payloads include NumPy MT19937 arrays; keep weights-only loading
    # and explicitly allow only NumPy array/dtype reconstruction for this local audit.
    allowed=[(np.core.multiarray._reconstruct,'numpy.core.multiarray._reconstruct'),np.ndarray,np.dtype,type(np.dtype(np.uint32))]
    with torch.serialization.safe_globals(allowed):model=load_model_from_checkpoint(v['config']['model'],path,torch.device('cpu'))
    mh=state_dict_hash(model.state_dict());del model
    expected=read(Path(selected).with_suffix('.json'))['model_hash'] if selected and Path(selected).with_suffix('.json').exists() else None
    aliases.append(dict(seed=v['seed'],condition=condition,step=step,historical_path=path,selected_reference=selected,use='historical source excluded; not an extra replication',historical_sha256=sha(path),historical_model_hash=mh,selected_model_hash=expected,identical_weight_alias=mh==expected if expected else None))
 save(OUT/'supplemental_input_hashes.json',files);table(OUT/'available_states.csv',rows);table(OUT/'historical_aliases.csv',aliases)
 save(OUT/'audit_supplement.json',dict(passed=True,nested_masks_from_retrained_IMP_parents=True,original_step0_verified=True,available_states_documented=True,seconds=time.time()-started,input_hashes_sha256=sha(OUT/'supplemental_input_hashes.json')))
 print('Supplemental IMP ancestry/state audit passed',flush=True)
if __name__=='__main__':main()
