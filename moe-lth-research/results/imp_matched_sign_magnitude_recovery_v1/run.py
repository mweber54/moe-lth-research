"""Matched-distance study only. One native segment, no resume/retry or continuation path."""
from pathlib import Path
import sys
sys.dont_write_bytecode=True
ROOT=Path.cwd();OUT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT/'results/imp_sign_recovery_audit_v2'))
from common import p,torch,np,pd,read,rows,sha
import csv,json,hashlib,datetime,time,os,socket,fcntl,shutil,subprocess,traceback,argparse,math
import gpu_policy
N=p.native;OLD=p.OUT;N.OUT=OUT
torch.set_num_threads(2)
SEEDS=(7,17,29);MASKS=('imp','random1','random2');ORDERS=('A','B');DRAWS=(1,2,3);ARMS=('M','Bm');STEPS=(0,250,500,1000,1500,2000,2500)
N.SNAPSHOT_STEPS=STEPS
BUDGET=377257594841
HOST=socket.getfqdn();DEVICE=torch.device('cuda:0')
def utc():return datetime.datetime.now(datetime.timezone.utc).isoformat()
def save(path,value):N._atomic_json(Path(path),value)
def table(path,data):
 data=list(data);path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
 temp=path.with_suffix(path.suffix+'.tmp')
 with temp.open('w') as f:
  w=csv.DictWriter(f,fieldnames=list(data[0]));w.writeheader();w.writerows(data)
 temp.replace(path)
def size(path):return sum(f.stat().st_size for f in Path(path).rglob('*') if f.is_file())
def storage_check(label):
 used=size(OUT);st=os.statvfs(OUT);free=st.f_bavail*st.f_frsize;required=max(0,BUDGET-used)
 record=dict(label=label,host=HOST,checked_utc=utc(),available_bytes=free,matched_study_written_bytes=used,registered_total_with_allowance=BUDGET,required_remaining_bytes=required,fits=free>=required)
 path=OUT/'storage_checks'/f'{time.time_ns()}_{HOST}.json';save(path,record)
 if not record['fits']:raise RuntimeError('Registered matched study outputs no longer fit with allowance; no files deleted')
 return record
def check_authority():
 c=rows(OUT/'amendment_commit.csv')[0];amend=subprocess.check_output(['git','show',f"{c['commit']}:{c['file']}"])
 assert hashlib.sha256(amend).hexdigest()==c['sha256']
 assert amend==(OUT/'amendment_2026-11-14.md').read_bytes()
 orig=subprocess.check_output(['git','show',f"{c['registration_commit']}:{c['registration_file']}"])
 assert hashlib.sha256(orig).hexdigest()==c['registration_sha256']
 assert orig==(OUT/'preregistration_2026-11-12.md').read_bytes()
 old=subprocess.check_output(['git','show','1f60d90:moe-lth-research/results/imp_sign_recovery_audit_v2/preregistration.md'])
 assert old==(ROOT/'results/imp_sign_recovery_audit_v2/preregistration.md').read_bytes()
 previous=subprocess.check_output(['git','show','3dae414:moe-lth-research/results/imp_sign_recovery_stage1_v1/amendment_2026-11-11.md'])
 assert previous==(ROOT/'results/imp_sign_recovery_stage1_v1/amendment_2026-11-11.md').read_bytes()
 assert c['stage1_training_runs_preceding_commit']=='0' and c['stage1_attempts_preceding_commit']=='0'
 return c
@torch.inference_mode()
def evaluate(model,validation):
 import torch.nn.functional as F
 model.eval();total=0.;count=0
 torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
 for x,y in validation:
  logits=model(x.to(DEVICE)).logits.float();y=y.to(DEVICE)
  total+=float(F.cross_entropy(logits.reshape(-1,logits.shape[-1]),y.reshape(-1),reduction='sum').double().cpu());count+=y.numel()
 assert count==492416 and math.isfinite(total)
 return total/count

def perturb_pair(v,seed,mask_id,draw,name):
 seeds={};hashes={};z=v.clone()
 for arm in ARMS:
  key=f'matched-sign-magnitude-recovery-v1|E32|{arm}|{seed}|{mask_id}|{draw}|{name}'
  seeds[arm]=int.from_bytes(hashlib.sha256(key.encode()).digest()[:8],'little')%(2**63)
 gm=torch.Generator(device=v.device).manual_seed(seeds['M'])
 for sign in (1,-1):
  ids=torch.where(v*sign>0)[0];ids=ids[torch.randperm(len(ids),device=v.device,generator=gm)]
  perm=torch.randperm(len(ids),device=v.device,generator=gm);z[ids]=v[ids[perm]]
  hashes['M_'+str(sign)]=hashlib.sha256(ids.cpu().numpy().tobytes()+perm.cpu().numpy().tobytes()).hexdigest()
 target=(z.double()-v.double()).square().sum();gb=torch.Generator(device=v.device).manual_seed(seeds['Bm'])
 pos=torch.where(v>0)[0];neg=torch.where(v<0)[0]
 pos=pos[torch.randperm(len(pos),device=v.device,generator=gb)];neg=neg[torch.randperm(len(neg),device=v.device,generator=gb)]
 n=min(len(pos),len(neg));pos=pos[:n];neg=neg[:n]
 costs=4*(v[pos].double().square()+v[neg].double().square());cumulative=torch.cat((costs.new_zeros(1),costs.cumsum(0)))
 k=int((cumulative-target).abs().argmin());b=v.clone();b[pos[:k]]*=-1;b[neg[:k]]*=-1
 hashes['Bm_selected']=hashlib.sha256(pos[:k].cpu().numpy().tobytes()+neg[:k].cpu().numpy().tobytes()).hexdigest()
 assert torch.equal(z.sign(),v.sign())
 for sign in (1,-1):assert torch.equal(v[v*sign>0].abs().sort().values,z[z*sign>0].abs().sort().values)
 assert torch.equal(b.abs(),v.abs()) and torch.equal(b.sign().sort().values,v.sign().sort().values)
 return {'M':z,'Bm':b},seeds,hashes,k

class PreparationDeferred(Exception):
 pass

def prepare():
 (OUT/'locks').mkdir(exist_ok=True)
 with (OUT/'locks/preparation.lock').open('a') as lock:
  fcntl.flock(lock.fileno(),fcntl.LOCK_EX|fcntl.LOCK_NB)
  if (OUT/'STOP.json').exists():raise RuntimeError('Global stop marker; preparation prohibited')
  if any((OUT/name).exists() for name in ('preparation_started.json','registration.json','job_manifest.csv','variants','jobs')):
   raise RuntimeError('Existing preparation or training artifacts; manual review required, no retry')
  decision=gpu_policy.admit('preparation')
  if not decision['accepted']:raise PreparationDeferred(decision['reason'])
  if (OUT/'STOP.json').exists():raise RuntimeError('Global stop marker; preparation prohibited')
  with (OUT/'preparation_started.json').open('x') as marker:
   json.dump(dict(host=HOST,pid=os.getpid(),started_utc=utc(),no_automatic_retry=True),marker)
   marker.flush();os.fsync(marker.fileno())
  _prepare()

def _prepare():
 assert not (OUT/'registration.json').exists() and not (OUT/'jobs').exists()
 amendment=check_authority();assert torch.__version__.startswith('2.7.0')
 started=time.perf_counter();legacy=rows(OLD/'variant_manifest.csv');logged=rows(OLD/'trajectory_losses.csv');manifest=[];checks=[];rngrows=[];matrixrows=[]
 for file in ('canonical_stream_manifest.json','recovery_order_manifest.json'):
  target=OUT/file
  if target.exists():assert sha(target)==sha(OLD/file),('Existing input differs',str(target))
  else:shutil.copyfile(OLD/file,target)
 if (OUT/'permutations').exists():
  source={p.name for p in (OLD/'permutations').iterdir()}
  assert source=={p.name for p in (OUT/'permutations').iterdir()}
  for file in source:assert sha(OUT/'permutations'/file)==sha(OLD/'permutations'/file)
 else:shutil.copytree(OLD/'permutations',OUT/'permutations')
 for seed in SEEDS:
  if (OUT/'STOP.json').exists():raise RuntimeError('Global stop marker during preparation')
  config,train,val=N._load_training_material(seed);del train
  assert config['model']['num_experts']==32
  for mask_id in MASKS:
   variant=mask_id+'_original';base=next(r for r in legacy if int(r['seed'])==seed and r['order']=='A' and r['variant']==variant)
   state,masks=p.variant_state(base);original=N._read_model_payload(Path(base['initial_checkpoint']))['model'];basehash=N._tensor_state_hash(state)
   with N._preserved_rng():
    model=N.TinyMoELanguageModel(config['model']).to(DEVICE);model.load_state_dict(state);ce=evaluate(model,val)
    identity={n:t.clone() for n,t in original.items()};identity.update({n:state[n].clone() for n in masks});assert N._tensor_state_hash(identity)==basehash
    model.load_state_dict(identity);identity_ce=evaluate(model,val);assert identity_ce==ce
   for order in ORDERS:
    expected=float(next(r for r in logged if int(r['seed'])==seed and r['order']==order and r['variant']==variant and int(r['step'])==0)['cross_entropy'])
    assert abs(ce-expected)<=.001,(seed,mask_id,order,ce,expected)
    checks.append(dict(seed=seed,mask_type=mask_id,order=order,original_CE=ce,logged_CE=expected,difference=ce-expected,identity_CE=identity_ce,identity_delta=identity_ce-ce,identity_state_hash=basehash,passed=True))
   del model,identity;torch.cuda.empty_cache();table(OUT/'engineering_checks.csv',checks)
   for draw in DRAWS:
    if (OUT/'STOP.json').exists():raise RuntimeError('Global stop marker during preparation')
    weights={a:{} for a in ARMS};distances={a:0. for a in ARMS};flips={a:0 for a in ARMS};energy=0.;survivors=0
    for name,mask in masks.items():
     m=mask.to(DEVICE);ref=m.clone();w=state[name].to(DEVICE);coords=m.flatten().nonzero().flatten();v=w.flatten()[coords]
     pair,seeds,hashes,k=perturb_pair(v,seed,mask_id,draw,name);en=float(v.double().square().sum());energy+=en;survivors+=len(v)
     ds={a:float((pair[a].double()-v.double()).square().sum()) for a in ARMS};d={a:math.sqrt(ds[a]/en) for a in ARMS}
     mismatch=abs(d['Bm']-d['M']);assert mismatch<=.015,(seed,mask_id,draw,name,d,mismatch)
     rngrows.append(dict(seed=seed,mask_type=mask_id,draw=draw,tensor=name,M_rng_seed=seeds['M'],Bm_rng_seed=seeds['Bm'],M_positive_permutation_hash=hashes['M_1'],M_negative_permutation_hash=hashes['M_-1'],Bm_selected_indices_hash=hashes['Bm_selected'],Bm_pairs_flipped=k))
     for arm,z in pair.items():
      e1=float(z.double().square().sum());assert math.isclose(en,e1,rel_tol=1e-12,abs_tol=1e-12)
      result=w.clone();result.flatten()[coords]=z
      assert torch.count_nonzero(result[~m])==0 and torch.equal(m,ref) and torch.equal(m.sum(0),ref.sum(0)) and torch.equal(m.sum(1),ref.sum(1))
      nf=int((z.sign()!=v.sign()).sum());weights[arm][name]=result.cpu();distances[arm]+=ds[arm];flips[arm]+=nf
      matrixrows.append(dict(seed=seed,mask_type=mask_id,draw=draw,tensor=name,arm=arm,survivors=len(v),zero_survivors=int((v==0).sum()),target_distance=d['M'],realized_distance=d[arm],absolute_distance_mismatch=mismatch,relative_distance_mismatch=mismatch/d['M'] if d['M'] else 0.,frac_sign_flipped=nf/len(v),energy_original=en,energy_perturbed=e1,mask_unchanged=True,row_counts_unchanged=True,column_counts_unchanged=True,signs_unchanged=bool(torch.equal(z.sign(),v.sign())),magnitudes_unchanged=bool(torch.equal(z.abs(),v.abs())),sign_counts_unchanged=True,magnitude_multiset_per_sign_preserved=(arm=='M'),distance_passed=True,energy_passed=True))
    target=math.sqrt(distances['M']/energy);assert abs(math.sqrt(distances['Bm']/energy)-target)<=.001
    for arm in ARMS:
     perturbed={n:t.clone() for n,t in original.items()};perturbed.update(weights[arm])
     assert set(weights[arm])==set(masks)
     for n in original:
      if n not in masks:assert torch.equal(perturbed[n],original[n])
     state_hash=N._tensor_state_hash(perturbed);path=OUT/'variants'/f'seed{seed}_{mask_id}_draw{draw}_{arm}.pt';path.parent.mkdir(exist_ok=True)
     torch.save(dict(weights=weights[arm],seed=seed,mask_type=mask_id,draw=draw,arm=arm),path);bundle_hash=sha(path)
     for order in ORDERS:
      manifest.append(dict(job_id=f'matched_e32_seed{seed}_{mask_id}_draw{draw}_{arm}_order{order}',seed=seed,order=order,mask_type=mask_id,draw=draw,model_type='imp80',arm=arm,initial_checkpoint=base['initial_checkpoint'],initial_checkpoint_sha256=base['initial_checkpoint_sha256'],variant_bundle=str(path),variant_sha256=bundle_hash,initial_state_hash=state_hash,mask_path=base['mask_path'],mask_hash=base['mask_sha256'],target_distance=target,realized_distance=math.sqrt(distances[arm]/energy),frac_sign_flipped=flips[arm]/survivors,cap_updates=2500,amendment_commit=amendment['commit']))
    table(OUT/'rng_seeds.csv',rngrows);table(OUT/'matrix_invariants.csv',matrixrows);print('prepared',seed,mask_id,draw,flush=True)
   del state,original,weights,perturbed;torch.cuda.empty_cache()
 assert len(manifest)==108 and sum(r['cap_updates'] for r in manifest)==270000
 # Alternate arms within each order/draw, so temporal execution order does not block one arm behind all of the other.
 manifest.sort(key=lambda r:(r['seed'],r['mask_type'],r['draw'],r['order'],r['arm']))
 table(OUT/'job_manifest.csv',manifest);code_inputs={str(Path(__file__).resolve()):sha(Path(__file__))}
 prior=read(ROOT/'results/imp_sign_recovery_stage1_v1/registration.json')['code_inputs']
 for file,h in prior.items():
  if '/src/' in file or '/scripts/' in file:assert sha(file)==h;code_inputs[file]=h
 for control in (OLD/'trajectory_losses.csv', ROOT/'results/imp_sign_recovery_stage1_v1/completion.json', OUT/'amendment_commit.csv', OUT/'policy_checks.csv', OUT/'report_check.csv'):
  code_inputs[str(control.resolve())]=sha(control)
 for file in ('gate.py','report.py','preregistration_2026-11-12.md','amendment_2026-11-14.md','dispatch.py','coordinator.py','monitor.py','supervise.py','gpu_policy.py'):
  path=OUT/file;code_inputs[str(path.resolve())]=sha(path)
 save(OUT/'registration.json',dict(stage='matched_distance',condition='M_vs_Bm',seeds=list(SEEDS),masks=list(MASKS),draws=list(DRAWS),orders=list(ORDERS),arms=list(ARMS),jobs=108,per_job_cap=2500,total_cap=270000,amendment_commit=amendment['commit'],amendment_sha256=amendment['sha256'],job_manifest_sha256=sha(OUT/'job_manifest.csv'),code_inputs=code_inputs,gpu_policy=gpu_policy.policy(),engineering_checks_sha256=sha(OUT/'engineering_checks.csv'),invariants_sha256=sha(OUT/'matrix_invariants.csv'),prepared_utc=utc(),preparation_elapsed_seconds=time.perf_counter()-started,automatic_continuation=False))
 storage_check('preparation_complete');print('sealed 108 jobs, 270000 total attempted-update cap',flush=True)

def verify():
 reg=read(OUT/'registration.json');check_authority()
 assert reg['gpu_policy']==gpu_policy.policy(),'GPU admission policy changed after preparation'
 assert reg['jobs']==108 and reg['per_job_cap']==2500 and reg['total_cap']==270000
 assert sha(OUT/'job_manifest.csv')==reg['job_manifest_sha256']
 for file,h in reg['code_inputs'].items():assert sha(file)==h,('sealed code changed',file)
 assert sha(OUT/'engineering_checks.csv')==reg['engineering_checks_sha256'];assert sha(OUT/'matrix_invariants.csv')==reg['invariants_sha256']
 return reg

def reconstruct(row):
 assert int(row['seed']) in SEEDS and row['mask_type'] in MASKS and int(row['draw']) in DRAWS and row['order'] in ORDERS and row['arm'] in ARMS and int(row['cap_updates'])==2500
 assert sha(row['variant_bundle'])==row['variant_sha256'];assert sha(row['initial_checkpoint'])==row['initial_checkpoint_sha256']
 payload=torch.load(row['initial_checkpoint'],map_location='cpu',weights_only=False);assert payload['step']==0
 state={n:t.clone() for n,t in payload['model'].items()};bundle=torch.load(row['variant_bundle'],map_location='cpu',weights_only=True);masks=N.load_masks(Path(row['mask_path']))
 assert set(bundle['weights'])==set(masks);assert N.recovery._mask_hash(masks)==row['mask_hash'];state.update(bundle['weights'])
 for n in state:
  if n not in masks:assert torch.equal(state[n],payload['model'][n])
 assert N._tensor_state_hash(state)==row['initial_state_hash']
 return state,masks

def run_one(row,cache):
 job=row['job_id'];attempt=OUT/'jobs'/job;attempt.mkdir(parents=True,exist_ok=False)
 save(OUT/'status'/f'{job}.json',dict(status='RUNNING',job_id=job,host=HOST,pid=os.getpid(),started_utc=utc(),attempt=str(attempt),attempts=0))
 start=time.perf_counter();seed=int(row['seed']);config,canonical,val=cache(seed);perm=N._permutations()[row['order']]['segment_0'];batches=N._apply_order(canonical,perm);assert len(batches)==2500
 state,masks=reconstruct(row);snap=attempt/'snapshots/step_0000.pt'
 with N._preserved_rng():
  model=N.assemble_router_age_model(config['model'],state,row['initial_checkpoint'],masks,DEVICE);N._assert_model_mask(model,masks);N._model_snapshot(snap,model,0,config,'matched_study_initial_model_only');del model
 original=N.recovery._run_recovery_condition;count=0;ledger=attempt/'attempt_ledger.jsonl'
 with ledger.open('x') as log:
  def guarded(**kwargs):
   assert kwargs['recovery_steps']==2500 and kwargs['router_step']==0 and kwargs['condition_name']==job and kwargs['output_dir']==attempt/'segment_0'
   cb=kwargs['training_forward_callback']
   def forward(model,ids,step):
    nonlocal count
    if (OUT/'STOP.json').exists():raise RuntimeError('Global matched study stop marker set')
    assert step==count+1 and step<=2500
    log.write(json.dumps(dict(attempt=step,unix=time.time(),job_id=job))+'\n');log.flush();os.fsync(log.fileno());count+=1
    return cb(model,ids,step)
   kwargs['training_forward_callback']=forward
   return original(**kwargs)
  N.recovery._run_recovery_condition=guarded
  try:record=N._run_segment(row,attempt,config,state,masks,Path(row['initial_checkpoint']),batches,val,0,'segment_0',DEVICE)
  finally:N.recovery._run_recovery_condition=original
 assert count==2500
 logs=[json.loads(l) for l in (attempt/'segment_0/gradient_stats/gradient_stats.jsonl').read_text().splitlines()]
 assert len(logs)==2500 and [r['step'] for r in logs]==list(range(1,2501))
 assert not (attempt/'segment_2500').exists()
 snapshots=[(t,attempt/f'snapshots/step_{t:04d}.pt','matched_study_model_only') for t in STEPS];assert all(path.exists() for _,path,_ in snapshots)
 evaluations=N._common_evaluate(snapshots,config,val,seed,row['order'],'imp80',job,N._seed_entry(seed)['validation_batch_sequence_hash'],DEVICE,allowed_steps=STEPS)
 assert [r['step'] for r in evaluations]==list(STEPS) and all(math.isfinite(r['cross_entropy']) for r in evaluations)
 with (attempt/'common_evaluations.jsonl').open('w') as f:
  for r in evaluations:f.write(json.dumps(r)+'\n')
 stats=dict(status='COMPLETE',job_id=job,host=HOST,pid=os.getpid(),attempt=str(attempt),attempted_updates=count,applied_updates=sum(bool(r['optimizer_step_applied']) for r in logs),amp_skipped_updates=sum(r['measurement_status']=='amp_overflow_skipped' for r in logs),slot_seconds=time.perf_counter()-start,native_segment_seconds=record['segment_seconds'],retained_bytes=size(attempt),amendment_commit=row['amendment_commit'],completed_utc=utc(),second_segment_started=False)
 assert stats['applied_updates']+stats['amp_skipped_updates']==2500
 save(attempt/'completion.json',stats);save(OUT/'status'/f'{job}.json',stats)
 print('COMPLETE',job,'attempts',count,'slot_seconds',stats['slot_seconds'],flush=True)
 del state,masks;torch.cuda.empty_cache()

def worker():
 reg=verify();alljobs=rows(OUT/'job_manifest.csv');assert len(alljobs)==108
 (OUT/'locks').mkdir(exist_ok=True);lockpath=OUT/'locks'/f'{HOST}_gpu0.lock'
 with lockpath.open('a') as lock:
  fcntl.flock(lock.fileno(),fcntl.LOCK_EX|fcntl.LOCK_NB)
  cache_data={}
  def cache(seed):
   if seed not in cache_data:cache_data.clear();cache_data[seed]=N._load_training_material(seed)
   return cache_data[seed]
  allocated=None;completed=0;job=None
  try:
   while True:
    if (OUT/'STOP.json').exists():raise RuntimeError('Global stop marker; no further dispatch')
    pending=[r for r in alljobs if not (OUT/'claims'/r['job_id']).exists()]
    if not pending:break
    decision=gpu_policy.admit('before_job')
    if not decision['accepted']:
     save(OUT/'workers'/f'{HOST}.json',dict(status='WAITING_FOR_GPU',host=HOST,gpu=decision,updated_utc=utc(),completed_jobs=completed))
     time.sleep(30);continue
    if (OUT/'STOP.json').exists():raise RuntimeError('Global stop marker; no further dispatch')
    if allocated is None:
     allocated=time.perf_counter();torch.cuda.set_device(0)
    verify();storage_check('immediately_before_job_dispatch')
    if (OUT/'STOP.json').exists():raise RuntimeError('Global stop marker; no further dispatch')
    row=None
    for candidate in pending:
     path=OUT/'claims'/candidate['job_id'];path.parent.mkdir(exist_ok=True)
     try:path.mkdir();row=candidate;break
     except FileExistsError:continue
    if row is None:continue
    job=row['job_id'];save(OUT/'claims'/job/'claim.json',dict(host=HOST,pid=os.getpid(),claimed_utc=utc(),amendment_commit=reg['amendment_commit']))
    if (OUT/'STOP.json').exists():raise RuntimeError('Global stop marker after claim; no training')
    if not (OUT/'training_started.json').exists():
     save(OUT/f'training_started_{HOST}.json',dict(first_job=job,host=HOST,started_utc=utc(),amendment_commit=reg['amendment_commit']))
    run_one(row,cache);completed+=1;job=None
    save(OUT/'workers'/f'{HOST}.json',dict(status='RUNNING',host=HOST,completed_jobs=completed,updated_utc=utc()))
  except BaseException as exc:
   details=dict(reason=repr(exc),traceback=traceback.format_exc(),host=HOST,job_id=job,stopped_utc=utc(),no_automatic_retry=True)
   if not (OUT/'STOP.json').exists():save(OUT/'STOP.json',details)
   if job:save(OUT/'status'/f'{job}.json',dict(status='FAILED',**details))
   raise
  finally:
   save(OUT/'workers'/f'{HOST}_accounting.json',dict(host=HOST,worker_slot_seconds=0 if allocated is None else time.perf_counter()-allocated,completed_jobs=completed,finished_utc=utc(),stage2_started=False))
 if all((OUT/'status'/f"{r['job_id']}.json").exists() and read(OUT/'status'/f"{r['job_id']}.json")['status']=='COMPLETE' for r in alljobs):
  with (OUT/'locks/report.lock').open('a') as lock:
   fcntl.flock(lock.fileno(),fcntl.LOCK_EX)
   subprocess.run([sys.executable,str(OUT/'report.py')],cwd=ROOT,check=True)
 print('Worker stopped after matched study; no continuation path',flush=True)

if __name__=='__main__':
 parser=argparse.ArgumentParser();parser.add_argument('command',choices=['prepare','verify','worker']);a=parser.parse_args()
 if a.command=='prepare':
  try:prepare()
  except PreparationDeferred as exc:print(str(exc),file=sys.stderr);sys.exit(75)
 elif a.command=='verify':print(json.dumps(verify()))
 else:worker()
