"""Audit the exact five-seed oracle masks and register all twenty endpoints."""
from common import *
import shutil

def main():
 torch.set_num_threads(2);OUT.mkdir(parents=True,exist_ok=True)
 if (OUT/'manifest.json').exists():raise RuntimeError('Existing study preserved')
 protocol='''# Known IMP80 mask transition protocol

Prospective controlled experiment motivated by previous exploratory findings; not project-level preregistration.

All five source seeds7,17,29,43,61 are retained, regardless of matching status.
Seed17 is the engineering pilot. Four conditions per seed: dense, fixed before
update1, abrupt immediately after attempted1000, gradual immediately after
attempted100,200,...1000. All sparse paths have the identical original final IMP80
mask for updates1001..5000. No new masks are discovered. The existing IMP mask is
privileged oracle information, not an online pruning algorithm. Its historical
discovery cost is excluded from diagnostic training but explicitly retained as a cost.

Gradual event assignments use SHA256('imp-mask-transition-v1:seed:parameter'),
first8 bytes big endian modulo(2^63-1), CPU torch.randperm on exactly final-discarded
coordinates. Event k deletes the prefix floor(k*N_discard/10); survivors never deleted.
One schedule per seed; no future statistics or outcome-dependent ordering.

Native32E model, complete initialization, batch128×128, deterministic batches0..4999,
AdamW lr3e-4 weight_decay.01, native default betas/eps, clip1, FP16 GradScaler,
auxiliary objective weight.1, full-softmax native selected gates and capacity1.25.
All parameters trainable. Masked gradients are zeroed before unscale/clipping;
weights enforced after every attempted step. New deletion zeros coordinate-shaped
Adam moments only; retained state and scalar counters persist. Events count AMP
skips and never reset RNG/scaler/optimizer. After2500 attempts, fresh optimizer,
scaler and RNG seeded with the source seed exactly as the established second call;
weights unchanged, second segment uses batches2500..4999.

Common task-only evaluation: FP32, TF32 off, eval/inference, ALL native validation
batches including final partial, actual-target weighting. Grid0,100..1000 by100,
1250,1500,1750,2000,2250,2500(pre-reset),2525,2600,2750,3000,3500,4000,4500,5000.
At1000 retain pre/post event; main trajectory is post. Reused endpoints must be
re-read under common evaluation. Unobserved reference grid points stay missing.
AUC uses common observed post1000 grid with≥3 points including1000 and5000;
otherwise unavailable. Trapezoidal AUC describes observed-grid coverage only.

Two complete training diagnostic batches: materialized batches0 and1, fixed before
new losses. Before/after every pruning event record task/aux, native proposals,
acceptance/drop rates, selected gates, per-layer expert loads, agreement/set overlap,
parameter norms. No diagnostic backward. Selected training updates on the grid and
first update in each segment record actual E/R/S update norms, native unscaled
gradient norms, skips/scaler and active counts. Tied parameters counted once.
Diagnostics preserve RNG/modes/gradients/model/buffers; do not touch optimizer,
scaler, sampler, masks or event counters. Cache only bounded compact route arrays.

Primary seed-level contrasts at5000: abrupt−fixed and gradual−fixed. Secondary:
gradual−abrupt, each−dense, pre-reset2500 and subsequent trajectory differences.
Report every seed before mean/median/range/sampleSD; n=5 source seeds, no token/
expert/checkpoint/event pseudoreplication, no winning-status filtering, no adaptive
duration or hyperparameter choice. No causal routing or isolated smoothness claim.
Equal attempted-token budgets have unequal cumulative active-coordinate budgets:
abrupt retains dense capacity longer than gradual; both exceed fixed early capacity.
Dense tensor zeros imply no measured FLOP/wall-clock savings.

Engineering before production: seed17 short native equivalence atol1e-6 rtol1e-5,
CE atol1e-6 (existing infrastructure), exact schedule/moment/event/resume/reset/RNG
checks. No tolerance relaxation after outcomes. Document engineering corrections
with versioned seals; no changes to timing/cohort/endpoints. Compatible dense/fixed
references can be reused with absent trajectories/diagnostics marked unavailable.
One detached worker per host, exclusive claims/host locks,≥24GiB and1.25×benchmark
peak admission, atomic status/heartbeats, full resumable states every100 attempts,
event state included, no duplicate resume events. Never overwrite an attempt or kill
unrelated jobs. Technical failures stop dispatch and preserve artifacts. Seed17 new
jobs run first; remaining seeds dispatch only after pilot numerical completion,
regardless of scientific effect direction. Only the declared20 endpoints authorized.
'''
 (OUT/'protocol.md').write_text(protocol)
 save(OUT/'registration.json',dict(label='Prospective controlled experiment motivated by previous exploratory findings; not project-level preregistration.',seeds=list(SEEDS),conditions=list(CONDITIONS),slots=20,grid=list(GRID),pilot=17,unix=time.time(),new_scientific_outcomes_viewed=False,equivalence_atol=1e-6,equivalence_rtol=1e-5,ce_atol=1e-6))
 cohort=read(HIST/'cohort.json');oldhash={x['path']:x['sha256'] for x in cohort['files']}
 artifacts=[];compat=[];variants=[];conditions=[];inputs={};streams=[];schedule_rows=[];groups=[]
 def record(path,kind,seed,expected=None):
  path=Path(path);h=sha(path)
  if expected is not None:assert h==expected,str(path)
  inputs[str(path)]=h;artifacts.append(dict(seed=seed,kind=kind,path=str(path),sha256=h,bytes=path.stat().st_size));return h
 for seed in SEEDS:
  cs=[c for c in cohort['conditions'] if c['seed']==seed and c['sparsity'] in (None,.8)]
  assert len(cs)==2
  sparse=next(c for c in cs if c['sparsity']==.8);final=load_masks(sparse['mask'])
  for path in (sparse['initial'],sparse['mask']):record(path,'initial' if path==sparse['initial'] else 'IMP80_mask',seed,oldhash[path])
  meta=read(Path(sparse['mask']).with_suffix('.json'));assert meta['pruning_round']==8 and meta['target_sparsity']==.8
  assert meta['mask_hash']==R._mask_hash(final) and meta['original_initialization_checkpoint']==sparse['initial']
  # Verify the full nested IMP ancestry; do not manufacture a replacement mask.
  parent=None
  for k in range(1,9):
   path=Path(sparse['mask']).parent/f'mask_{k:02d}.pt';m=load_masks(path);mm=read(path.with_suffix('.json'))
   record(path,'IMP_ancestor_mask',seed);record(path.with_suffix('.json'),'IMP_round_metadata',seed)
   assert R._mask_hash(m)==mm['mask_hash'] and mm['pruning_round']==k
   if parent is not None:
    assert mm['parent_mask_hash']==R._mask_hash(parent)
    assert all(not torch.any(m[n]&~parent[n]) for n in m)
   parent=m
  model=load_model_from_checkpoint(sparse['config']['model'],sparse['initial'],torch.device('cpu'))
  assert set(final)==set(expert_weight_parameters(model)) and all(n.endswith(('.fc1.weight','.fc2.weight')) for n in final)
  assert all(p.requires_grad for p in model.parameters())
  for n,m in final.items():assert m.dtype==torch.bool and m.shape==dict(model.named_parameters())[n].shape
  expertkeys={n.rsplit('.',2)[0] for n in final}
  for key in expertkeys:
   names=[n for n in final if n.startswith(key+'.')]
   assert len(names)==2 and sum(int(final[n].sum()) for n in names)==round(sum(final[n].numel() for n in names)*.2)
  raw=load(sparse['initial']);groups.extend(dict(seed=seed,**r) for r in inventory(model,final))
  component_hashes={g:state_dict_hash({n:t for n,t in model.state_dict().items() if parameter_group(n)==g}) for g in ('expert','router','shared')}
  del raw,model,parent
  assignment=assignments(final,seed);path=OUT/'schedules'/f'seed_{seed}.pt';atomic_torch(path,assignment);record(path,'registered_event_assignments',seed)
  for n,a in assignment.items():
   assert torch.equal(a==0,final[n])
   for k in range(1,11):
    mask=(a==0)|(a>k);N=int((~final[n]).sum());assert int((~mask).sum())==k*N//10
    schedule_rows.append(dict(seed=seed,parameter=n,stable_seed=stable_seed(seed,n),event=k,attempt=100*k,removed_coordinates=int((~mask).sum()),surviving_coordinates=int(mask.sum()),mask_sha256=hashlib.sha256(mask.numpy().tobytes()).hexdigest()))
  v=dict(seed=seed,config=sparse['config'],initial=sparse['initial'],mask=sparse['mask'],schedule=str(path),mask_hash=R._mask_hash(final),initial_component_hashes=component_hashes,canonical_pair=sparse['canonical_pair'],extension_identity=sparse['extension_identity'],references={})
  train,val=batches(v);assert len(val)==31 and val[-1][1].numel()==896
  for i,batch in enumerate(train):streams.append(dict(seed=seed,batch=i,segment=0 if i<2500 else 2500,target_tokens=batch[1].numel(),sha256=R._batch_sequence_hash([batch])))
  for i,batch in enumerate(val):streams.append(dict(seed=seed,batch=i,segment='validation',target_tokens=batch[1].numel(),sha256=R._batch_sequence_hash([batch])))
  v['validation_tokens']=sum(y.numel() for x,y in val);v['diagnostic_hash']=R._batch_sequence_hash(train[:2]);v['training_hash']=R._batch_sequence_hash(train)
  del train,val
  for c in cs:
   cond='dense' if c['sparsity'] is None else 'fixed';done=read(HIST/'reruns'/c['id']/'completed.json');attempt=Path(done['attempt'])
   record(done['checkpoint'],'reference_endpoint',seed,done['sha256']);assert done['reset_weights_unchanged'] and done['cohort_sha256']==sha(HIST/'cohort.json')
   for off in (0,2500):
    p=attempt/f'segment_{off}/record.json';r=read(p);record(p,'reference_training_record',seed)
    assert r['optimizer_step_attempts']==2500 and r['optimizer_state_entries_at_start']==0 and r['precision']=='fp16' and r['router_mode']=='trainable'
    assert r['training_batch_sequence_hash']==(c['canonical_pair']['data_hash'] if off==0 else c['extension_identity']['training_batch_hash'])
    assert r['validation_batch_sequence_hash']==c['canonical_pair']['validation_hash']
   a=read(attempt/'snapshots/step_2500.json');b=read(attempt/'snapshots/step_2500_after_reset.json');assert a['model_hash']==b['model_hash'] and b['optimizer_entries']==0
   expected=load_model_from_checkpoint(c['config']['model'],c['initial'],torch.device('cpu'));apply_masks_(expected,final if cond=='fixed' else {})
   assert read(attempt/'snapshots/step_0000.json')['model_hash']==state_dict_hash(expected.state_dict());del expected
   snapshots={}
   for step in GRID:
    p=attempt/f'snapshots/step_{step:04d}.pt'
    if p.exists():record(p,'compatible_reference_snapshot',seed,read(p.with_suffix('.json'))['snapshot_sha256']);snapshots[str(step)]=str(p)
   v['references'][cond]=dict(checkpoint=done['checkpoint'],attempt=str(attempt),snapshots=snapshots,applied=done['first_applied']+done['second_applied'])
   compat.append(dict(seed=seed,condition=cond,reuse=True,initialization_verified=True,mask_verified=True,stream_verified=True,attempts=5000,reset_verified=True,training_precision='FP16 native TF32 on',historical_readout='FP32 TF32 on; not directly compared',common_readout_required=True,missing_grid_points=';'.join(str(t) for t in GRID if str(t) not in snapshots),optimizer_continuation=False,reason='exact two-call own-init native trajectory; common FP32 TF32-off re-read only'))
  variants.append(v)
  for cond in CONDITIONS:
   j=dict(id=f'seed{seed}_{cond}',seed=seed,condition=cond,reuse=cond in ('dense','fixed'));conditions.append(j);save(OUT/'status'/f"{j['id']}.json",dict(status='PENDING',reuse=j['reuse']))
  print('Audited seed',seed,flush=True)
 # Seal reference implementation provenance, not just endpoint files.
 for p,h in read(HIST/'launch.json')['implementation'].items():
  if p.startswith('src/') or p.endswith('runner.py'):record(ROOT/p,'native_reference_implementation',None,h)
 record(ROOT/'scripts/imp_80_85/run.py','IMP_implementation',None)
 record(ROOT/'scripts/imp_extended_recovery/pipeline.py','extension_implementation',None)
 for p in ('data/multidomain_balanced/multidomain_train.txt','data/multidomain_balanced/multidomain_validation.txt'):record(ROOT/p,'corpus',None)
 table(OUT/'artifact_inventory.csv',artifacts);table(OUT/'reference_compatibility.csv',compat);table(OUT/'condition_manifest.csv',conditions);table(OUT/'event_schedule.csv',schedule_rows);table(OUT/'batch_manifest.csv',streams);table(OUT/'parameter_groups.csv',groups)
 save(OUT/'input_hashes.json',inputs);save(OUT/'manifest.json',dict(variants=variants,jobs=conditions,new_training_jobs=10,reused_references=10,registered_endpoints=20))
 save(OUT/'resource_estimate.json',dict(new_training_jobs=10,reused_references=10,attempts_per_new_job=5000,estimated_storage_bytes=60*1024**3,available_storage_bytes=shutil.disk_usage(OUT).free,benchmark_required_before_launch=True,minimum_free_gpu_bytes=24*1024**3,one_worker_per_host=True))
 (OUT/'audit.md').write_text('''# Input audit

Verified five complete own initializations, actual nested IMP rounds1..8 and final
joint-per-expert fc1+fc2 keep budgets. Per-tensor allocations are preserved; no
bias/router/shared mask entries. All native parameters remain trainable. Original
component hashes and exact data hashes are recorded. Native top1 selected gates
retain full-softmax probabilities; capacity accepts highest gate priorities, dropped
tokens have zero expert contribution; aux uses detached proposed loads and full
probability means. No routing substitution or router-age sweep is used.

Validation comprises31 batches:30 complete128×128 plus7×128 targets in the last
batch,492416 target tokens total. Historical readouts used TF32; all reused states
require common FP32 TF32-off re-evaluation, never direct historical CE comparison.
Compatible references are ten high-sparsity reruns with the exact two-call reset,
complete initialization and stream identities. Their model-only snapshots support
readout, not optimizer continuation. Missing grid points/diagnostics stay missing.
Each final endpoint is counted once; historical aliases are not replications.

Native clipping occurs after gradient hooks/unscale. Dynamic mask hooks require
explicit updates because native hooks capture a tensor at registration. The new
adapter retains the native training function code and overrides its masking and
readout/serialization globals; engineering equivalence is required before launch.
There are exactly10 new5000-attempt training jobs and10 reused controls.
''')
 print('REGISTERED: 20 endpoints,10 new training jobs,10 reused controls',flush=True)
if __name__=='__main__':main()
