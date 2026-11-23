"""Stage 1 only. One native segment, no resume/retry or continuation path."""
from pathlib import Path
import sys
sys.dont_write_bytecode=True
ROOT=Path.cwd();OUT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT/'results/imp_sign_recovery_audit_v2'))
from common import p,torch,np,pd,read,rows,sha
import csv,json,hashlib,datetime,time,os,socket,fcntl,shutil,subprocess,traceback,argparse,math
N=p.native;OLD=p.OUT;N.OUT=OUT
torch.set_num_threads(2)
SEEDS=(7,17,29);MASKS=('imp','random1','random2');ORDERS=('A','B');DRAWS=(1,2,3);STEPS=(0,250,500,1000,1500,2000,2500)
N.SNAPSHOT_STEPS=STEPS
BUDGET=188628797420.25
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
 record=dict(label=label,host=HOST,checked_utc=utc(),available_bytes=free,stage1_written_bytes=used,registered_total_with_allowance=BUDGET,required_remaining_bytes=required,fits=free>=required)
 path=OUT/'storage_checks'/f'{time.time_ns()}_{HOST}.json';save(path,record)
 if not record['fits']:raise RuntimeError('Registered Stage1 outputs no longer fit with allowance; no files deleted')
 return record
def check_authority():
 c=rows(OUT/'amendment_commit.csv')[0];content=subprocess.check_output(['git','show',f"{c['commit']}:{c['file']}"])
 assert hashlib.sha256(content).hexdigest()==c['sha256']
 assert content==(OUT/'amendment_2026-11-11.md').read_bytes()
 old=subprocess.check_output(['git','show','1f60d90:moe-lth-research/results/imp_sign_recovery_audit_v2/preregistration.md'])
 assert old==(ROOT/'results/imp_sign_recovery_audit_v2/preregistration.md').read_bytes()
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

def prepare():
 assert not (OUT/'registration.json').exists() and not (OUT/'jobs').exists()
 amendment=check_authority();assert torch.__version__.startswith('2.7.0')
 idle,details=N._gpu_idle(0);assert idle,('GPU busy; no preparation started',details)
 started=time.perf_counter();legacy=rows(OLD/'variant_manifest.csv');logged=rows(OLD/'trajectory_losses.csv');manifest=[];checks=[];rngrows=[];matrixrows=[]
 for file in ('canonical_stream_manifest.json','recovery_order_manifest.json'):
  shutil.copyfile(OLD/file,OUT/file)
 shutil.copytree(OLD/'permutations',OUT/'permutations')
 for seed in SEEDS:
  config,train,val=N._load_training_material(seed);del train
  for mask_id in MASKS:
   variant='imp_original' if mask_id=='imp' else mask_id+'_original'
   base=next(r for r in legacy if int(r['seed'])==seed and r['order']=='A' and r['variant']==variant)
   state,masks=p.variant_state(base);original=N._read_model_payload(Path(base['initial_checkpoint']))['model'];basehash=N._tensor_state_hash(state)
   with N._preserved_rng():
    model=N.TinyMoELanguageModel(config['model']).to(DEVICE);model.load_state_dict(state)
    ce=evaluate(model,val)
    identity={n:t.clone() for n,t in original.items()};identity.update({n:state[n].clone() for n in masks})
    assert N._tensor_state_hash(identity)==basehash
    model.load_state_dict(identity);identity_ce=evaluate(model,val);assert identity_ce==ce
   for order in ORDERS:
    expected=float(next(r for r in logged if int(r['seed'])==seed and r['order']==order and r['variant']==variant and int(r['step'])==0)['cross_entropy'])
    assert abs(ce-expected)<=.001,(seed,mask_id,order,ce,expected)
    checks.append(dict(seed=seed,mask_type=mask_id,order=order,original_CE=ce,logged_CE=expected,difference=ce-expected,identity_CE=identity_ce,identity_delta=identity_ce-ce,identity_state_hash=basehash,passed=True))
   del model,identity;torch.cuda.empty_cache();table(OUT/'engineering_checks.csv',checks)
   for draw in DRAWS:
    weights={};distance=0.;energy=0.;flips=0;total_survivors=0
    for name,mask in masks.items():
     m=mask.to(DEVICE);reference=m.clone();w=state[name].to(DEVICE);coords=m.flatten().nonzero().flatten();v=w.flatten()[coords];eligible=(v!=0).nonzero().flatten()
     key=f'sign-recovery-phase2-v1|E32|sign|{seed}|{mask_id}|{draw}|{name}';s=int.from_bytes(hashlib.sha256(key.encode()).digest()[:8],'little')%(2**63);g=torch.Generator(device='cuda').manual_seed(s)
     selection=eligible[torch.randperm(len(eligible),device='cuda',generator=g)];perm=torch.randperm(len(selection),device='cuda',generator=g);z=v.clone();z[selection]=v[selection].abs()*v[selection[perm]].sign()
     assert torch.equal(v.abs(),z.abs());assert int((v>0).sum())==int((z>0).sum()) and int((v<0).sum())==int((z<0).sum())
     e0=v.double().square().sum();e1=z.double().square().sum();assert torch.allclose(e0,e1,rtol=1e-12,atol=1e-12)
     result=w.clone();result.flatten()[coords]=z
     assert torch.count_nonzero(result[~m])==0 and torch.equal(m,reference)
     assert torch.equal(m.sum(0),reference.sum(0)) and torch.equal(m.sum(1),reference.sum(1))
     ds=float((z.double()-v.double()).square().sum());en=float(e0);nf=int((z.sign()!=v.sign()).sum());ns=len(v)
     distance+=ds;energy+=en;flips+=nf;total_survivors+=ns;weights[name]=result.cpu()
     rngrows.append(dict(seed=seed,mask_type=mask_id,draw=draw,tensor=name,rng_seed=s,selection_sha256=hashlib.sha256(selection.cpu().numpy().tobytes()).hexdigest(),sign_permutation_sha256=hashlib.sha256(perm.cpu().numpy().tobytes()).hexdigest()))
     matrixrows.append(dict(seed=seed,mask_type=mask_id,draw=draw,tensor=name,survivors=ns,zero_survivors=int((v==0).sum()),realized_distance=math.sqrt(ds/en),frac_sign_flipped=nf/ns,energy_original=en,energy_perturbed=float(e1),mask_unchanged=True,row_counts_unchanged=True,column_counts_unchanged=True,position_magnitudes_unchanged=True,sign_counts_unchanged=True))
    assert set(weights)==set(masks)
    perturbed={n:t.clone() for n,t in original.items()};perturbed.update(weights)
    for n in original:
     if n not in masks:assert torch.equal(perturbed[n],original[n])
    state_hash=N._tensor_state_hash(perturbed);path=OUT/'variants'/f'seed{seed}_{mask_id}_draw{draw}.pt';path.parent.mkdir(exist_ok=True)
    torch.save(dict(weights=weights,seed=seed,mask_type=mask_id,draw=draw),path);bundle_hash=sha(path)
    for order in ORDERS:
     manifest.append(dict(job_id=f's1_e32_seed{seed}_{mask_id}_draw{draw}_order{order}',seed=seed,order=order,mask_type=mask_id,draw=draw,model_type='imp80',arm='sign',initial_checkpoint=base['initial_checkpoint'],initial_checkpoint_sha256=base['initial_checkpoint_sha256'],variant_bundle=str(path),variant_sha256=bundle_hash,initial_state_hash=state_hash,mask_path=base['mask_path'],mask_hash=base['mask_sha256'],realized_distance=math.sqrt(distance/energy),frac_sign_flipped=flips/total_survivors,cap_updates=2500,amendment_commit=amendment['commit']))
    table(OUT/'rng_seeds.csv',rngrows);table(OUT/'matrix_invariants.csv',matrixrows)
    print('prepared',seed,mask_id,draw,flush=True)
   del state,original,weights,perturbed;torch.cuda.empty_cache()
 assert len(manifest)==54 and sum(r['cap_updates'] for r in manifest)==135000
 table(OUT/'job_manifest.csv',manifest)
 code_inputs={str(Path(__file__).resolve()):sha(Path(__file__))}
 for file,h in read(OLD/'registration.json')['sealed_files'].items():
  if '/src/' in file or '/scripts/' in file:assert sha(file)==h;code_inputs[file]=h
 for file in ('gate.py','report.py'):
  path=OUT/file;code_inputs[str(path)]=sha(path)
 seal=dict(stage=1,condition='sign_only',seeds=list(SEEDS),masks=list(MASKS),draws=list(DRAWS),orders=list(ORDERS),jobs=54,per_job_cap=2500,total_cap=135000,amendment_commit=amendment['commit'],amendment_sha256=amendment['sha256'],job_manifest_sha256=sha(OUT/'job_manifest.csv'),code_inputs=code_inputs,engineering_checks_sha256=sha(OUT/'engineering_checks.csv'),invariants_sha256=sha(OUT/'matrix_invariants.csv'),prepared_utc=utc(),preparation_elapsed_seconds=time.perf_counter()-started,stage1_only=True,automatic_continuation=False)
 save(OUT/'registration.json',seal);storage_check('preparation_complete')
 print('sealed 54 jobs, 135000 total attempted-update cap',flush=True)

def verify():
 reg=read(OUT/'registration.json');check_authority()
 assert reg['jobs']==54 and reg['per_job_cap']==2500 and reg['total_cap']==135000
 assert sha(OUT/'job_manifest.csv')==reg['job_manifest_sha256']
 for file,h in reg['code_inputs'].items():assert sha(file)==h,('sealed code changed',file)
 assert sha(OUT/'engineering_checks.csv')==reg['engineering_checks_sha256'];assert sha(OUT/'matrix_invariants.csv')==reg['invariants_sha256']
 return reg

def reconstruct(row):
 assert int(row['seed']) in SEEDS and row['mask_type'] in MASKS and int(row['draw']) in DRAWS and row['order'] in ORDERS and row['arm']=='sign' and int(row['cap_updates'])==2500
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
  model=N.assemble_router_age_model(config['model'],state,row['initial_checkpoint'],masks,DEVICE);N._assert_model_mask(model,masks);N._model_snapshot(snap,model,0,config,'stage1_initial_model_only');del model
 original=N.recovery._run_recovery_condition;count=0;ledger=attempt/'attempt_ledger.jsonl'
 with ledger.open('x') as log:
  def guarded(**kwargs):
   assert kwargs['recovery_steps']==2500 and kwargs['router_step']==0 and kwargs['condition_name']==job and kwargs['output_dir']==attempt/'segment_0'
   cb=kwargs['training_forward_callback']
   def forward(model,ids,step):
    nonlocal count
    if (OUT/'STOP.json').exists():raise RuntimeError('Global Stage1 stop marker set')
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
 snapshots=[(t,attempt/f'snapshots/step_{t:04d}.pt','stage1_model_only') for t in STEPS];assert all(path.exists() for _,path,_ in snapshots)
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
 reg=verify();alljobs=rows(OUT/'job_manifest.csv');assert len(alljobs)==54
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
    if allocated is None:
     idle,gpu=N._gpu_idle(0)
     if not idle:
      save(OUT/'workers'/f'{HOST}.json',dict(status='WAITING_FOR_GPU',host=HOST,gpu=gpu,updated_utc=utc(),completed_jobs=completed))
      time.sleep(30);continue
     allocated=time.perf_counter();torch.cuda.set_device(0)
    verify();storage_check('immediately_before_job_dispatch')
    row=None
    for candidate in pending:
     path=OUT/'claims'/candidate['job_id'];path.parent.mkdir(exist_ok=True)
     try:path.mkdir();row=candidate;break
     except FileExistsError:continue
    if row is None:continue
    job=row['job_id'];save(OUT/'claims'/job/'claim.json',dict(host=HOST,pid=os.getpid(),claimed_utc=utc(),amendment_commit=reg['amendment_commit']))
    if not (OUT/'training_started.json').exists():
     save(OUT/f'training_started_{HOST}.json',dict(first_job=job,host=HOST,started_utc=utc(),amendment_commit=reg['amendment_commit']))
    run_one(row,cache);completed+=1;job=None
    save(OUT/'workers'/f'{HOST}.json',dict(status='RUNNING',host=HOST,completed_jobs=completed,updated_utc=utc()))
  except BaseException as exc:
   details=dict(reason=repr(exc),traceback=traceback.format_exc(),host=HOST,job_id=job,stopped_utc=utc(),no_automatic_retry=True)
   save(OUT/'STOP.json',details)
   if job:save(OUT/'status'/f'{job}.json',dict(status='FAILED',**details))
   raise
  finally:
   save(OUT/'workers'/f'{HOST}_accounting.json',dict(host=HOST,worker_slot_seconds=0 if allocated is None else time.perf_counter()-allocated,completed_jobs=completed,finished_utc=utc(),stage2_started=False))
 if all((OUT/'status'/f"{r['job_id']}.json").exists() and read(OUT/'status'/f"{r['job_id']}.json")['status']=='COMPLETE' for r in alljobs):
  with (OUT/'locks/report.lock').open('a') as lock:
   fcntl.flock(lock.fileno(),fcntl.LOCK_EX)
   subprocess.run([sys.executable,str(OUT/'report.py')],cwd=ROOT,check=True)
 print('Worker stopped after Stage1; no continuation path',flush=True)

if __name__=='__main__':
 parser=argparse.ArgumentParser();parser.add_argument('command',choices=['prepare','verify','worker']);a=parser.parse_args()
 if a.command=='prepare':prepare()
 elif a.command=='verify':print(json.dumps(verify()))
 else:worker()
