"""Verify and inventory archived artifacts; never modifies v1."""
import hashlib,json,pathlib,sys,time
import torch
HERE=pathlib.Path(__file__).resolve().parent
V1=HERE.parent/'moe_online_reinforcement_pruning_v1'
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(1024*1024),b''): h.update(b)
 return h.hexdigest()
def record(path,expected=None):
 digest=sha(path)
 if expected is not None and digest!=expected: raise ValueError(f'hash mismatch {path}')
 return {'path':str(path),'sha256':digest,'bytes':path.stat().st_size}
manifest={'created_unix':time.time(),'cells':{}}
for seed in (7,17,29):
 for order in ('A','B'):
  name=f'seed{seed}_{order}'; p=V1/'prefixes'/name; m=V1/'matching'/name
  v=json.loads((p/'verification.json').read_text()); c=json.loads((m/'complete.json').read_text())
  assert v['status']=='VERIFIED' and all(v['restore_step251_checks'].values()) and c['passed']
  files={'checkpoint':record(p/'native/interruption.pt',v['checkpoint_sha256']),'scores':record(p/'scores.pt',v['scores_sha256']),'oracle':record(p/'expected_step251.pt'),'initial_checkpoint':record(pathlib.Path(v['initial_checkpoint']),v['initial_sha256']),'selections':record(m/'selections.pt',c['selections_sha256']),'matrix_audit':record(m/'matrix_audit.json',c['matrix_audit_sha256']),'verification':record(p/'verification.json'),'matching_metadata':record(m/'complete.json')}
  cp=torch.load(p/'native/interruption.pt',map_location='cpu',weights_only=False)
  req={'identity','model','optimizer','scaler','step','optimizer_steps_applied','amp_overflow_steps','python_rng','numpy_rng','torch_rng','cuda_rng'}
  assert req<=cp.keys() and cp['step']==250 and cp['optimizer_steps_applied']==249 and cp['amp_overflow_steps']==1
  assert cp['optimizer']['state'] and cp['optimizer']['param_groups'] and len(cp['cuda_rng'])>=1
  manifest['cells'][name]={'files':files,'checkpoint_keys':sorted(cp.keys()),'applied':cp['optimizer_steps_applied'],'skipped':cp['amp_overflow_steps'],'recorded_restore_checks':v['restore_step251_checks']}
  del cp
  print(name,'verified',flush=True)
manifest['v1_protocol']=record(V1/'protocol.md','403caae38a808e39571bcd1526a48cf44e06801e57e239f18800789bb690e2fc')
tmp=HERE/'input_manifest.tmp';tmp.write_text(json.dumps(manifest,indent=2)+'\n');tmp.replace(HERE/'input_manifest.json')
