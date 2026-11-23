"""One persistent worker per host, exclusive claims, fail-closed bounded phases."""
from v2_common import *
import argparse
import contextlib
import threading
import traceback
from moe_lth.data import build_dataloaders
from moe_lth.models import TinyMoELanguageModel
from moe_lth.pruning.mask_information import assert_control
from moe_lth.pruning.router_age import load_model_from_checkpoint,selected_experts_per_batch
from moe_lth.utils import seed_everything,configure_device

ADMISSION_BYTES=24*1024**3


def batches(cohort):
    cfg=cohort['config'];seed=int(cohort['seed'])
    seed_everything(seed)
    tr,va=build_dataloaders(cfg['data'],128,seed)
    train=recovery._materialize_batches(tr,2500);val=recovery._materialize_validation_batches(va,32)
    if recovery._batch_sequence_hash(train)!=cohort['training_batch_hash']:raise RuntimeError('Training batch mismatch')
    if recovery._batch_sequence_hash(val)!=cohort['validation_batch_hash']:raise RuntimeError('Validation batch mismatch')
    return train,val


def check_host(manifest):
    torch.set_num_threads(4)
    if socket.getfqdn() not in HOSTS:raise RuntimeError('Unregistered host')
    if not torch.__version__.startswith('2.7.0') or not torch.cuda.is_available():raise RuntimeError('Canonical runtime unavailable')
    checks=[]
    for seed in SEEDS:
        c=manifest['cohorts'][str(seed)]
        for path,h in [(c['rewind_source'],c['rewind_sha256']),(c['mask_source'],c['mask_sha256']),(c['recovery_checkpoint'],c['recovery_sha256']),(c['dense_checkpoint'],c['dense_sha256']),(c['diagnostic_reference'],c['diagnostic_reference_sha256'])]:
            if sha(path)!=h:raise RuntimeError('Host source mismatch '+path)
        masks=load_masks(c['mask_source'])
        if recovery._mask_hash(masks)!=c['mask_hash']:raise RuntimeError('Host original mask mismatch')
        train,val=batches(c);checks.append(dict(seed=seed,train_hash=recovery._batch_sequence_hash(train),validation_hash=recovery._batch_sequence_hash(val),verified=True));del train,val
    reused_checks=[]
    for j in manifest['jobs']:
        if 'reused_stage' not in j:continue
        record=result(j);path=Path(record['stage_path']);mask=load_masks(j['mask_path'])
        valid=recovery._mask_hash(mask)==j['mask_hash'] and sha(j['mask_path'])==j['mask_sha256'] and sha(path.parent/'checkpoints/final_recovered.pt')==record['checkpoint_sha256'] and Path(record['curves_path']).is_file()
        if not valid:raise RuntimeError('Host reused-artifact mismatch '+j['id'])
        reused_checks.append(dict(job=j['id'],verified=True))
    free,total=torch.cuda.mem_get_info()
    if free<ADMISSION_BYTES:raise RuntimeError('Insufficient free GPU memory at preflight; no process killed')
    save(OUT/'host_checks'/f'{socket.getfqdn()}.json',dict(host=socket.getfqdn(),torch=str(torch.__version__),gpu=torch.cuda.get_device_name(),free_gpu_bytes=free,total_gpu_bytes=total,admission_bytes=ADMISSION_BYTES,seeds=checks,reused_checks=reused_checks,passed=True,updated=time.time()))


def refresh(manifest):
    from report import analysis
    with lock(OUT/'report.lock'):analysis(manifest)


def transition(manifest):
    try:phase_lock=lock(OUT/'phase.lock',False)
    except BlockingIOError:return
    with phase_lock:
        if (OUT/'phase2_gate.json').exists():return
        phase1=[j for j in manifest['jobs'] if j['phase']==1]
        if not all(job_status(j)['status'] in ('COMPLETE','REUSED') for j in phase1):return
        effects={s:sum(result(j)['final_ce'] for j in phase1 if j['seed']==s and j['condition']=='matrix')/5-next(result(j)['final_ce'] for j in phase1 if j['seed']==s and j['condition']=='IMP') for s in SEEDS}
        save(OUT/'replication_gate_evaluation.json',dict(effects=effects,rule='all five strictly >0.02',passed=matrix_gate(effects),unit='source seed',not_a_significance_test=True,updated=time.time()))
        if not matrix_gate(effects):
            with lock(OUT/'dispatch.lock'):
                for j in manifest['jobs']:
                    if j['phase']==2:save(OUT/'status'/f"{j['id']}.json",dict(status='NOT_ADMITTED',reason='Effect rule failed',updated=time.time()))
                save(OUT/'phase2_gate.json',dict(status='NOT_ADMITTED_EFFECT_RULE',effects=effects,updated=time.time()))
            return
        # This host's persistent heartbeat remains active while mixing on CPU.
        from degree import prepare_degree
        success=prepare_degree(manifest)
        with lock(OUT/'dispatch.lock'):
            for j in manifest['jobs']:
                if j['phase']==2:save(OUT/'status'/f"{j['id']}.json",dict(status='PENDING' if success else 'NOT_ADMITTED',reason='Mixing checks passed' if success else 'Mixing limitation',updated=time.time()))
            save(OUT/'phase2_gate.json',dict(status='ADMITTED' if success else 'NOT_ADMITTED_MIXING',effects=effects,limitation=None if success else 'Prespecified degree randomization diagnostics failed; no degree masks trained.',updated=time.time()))


def claim(manifest,host):
    with lock(OUT/'dispatch.lock'):
        if (OUT/'STOP.json').exists():return None
        if torch.cuda.mem_get_info()[0]<ADMISSION_BYTES:return None
        for j in manifest['jobs']:
            if job_status(j)['status']!='PENDING':continue
            if j['phase']==2 and (not (OUT/'phase2_gate.json').exists() or read(OUT/'phase2_gate.json')['status']!='ADMITTED'):continue
            dep=[x for x in manifest['jobs'] if x['id'] in j['dependencies']]
            if not all(job_status(x)['status'] in ('COMPLETE','REUSED') for x in dep):continue
            # Claims are durable statuses guarded by the same shared filesystem
            # lock on both hosts. A fresh directory is reserved without reuse.
            parent=OUT/'recoveries'/j['id'];parent.mkdir(parents=True,exist_ok=True)
            attempt=parent/'attempt_0001'
            if attempt.exists():raise RuntimeError('Existing attempt requires explicit review '+str(attempt))
            save(OUT/'status'/f"{j['id']}.json",dict(status='RUNNING',host=host,pid=os.getpid(),attempt=str(attempt),claimed=time.time(),updated=time.time()))
            return j
    return None


def run_job(job,manifest):
    c=manifest['cohorts'][str(job['seed'])];cfg=c['config'];seed=job['seed'];device=torch.device('cuda')
    for path,h in [(c['rewind_source'],c['rewind_sha256']),(c['dense_checkpoint'],c['dense_sha256'])]:
        if sha(path)!=h:raise RuntimeError('Source changed before recovery')
    masks=load_masks(job['mask_path']);original=load_masks(c['mask_source'])
    model=TinyMoELanguageModel(cfg['model']);validate_masks(model,masks)
    info=read(OUT/'degree_masks'/f"{job['id']}.json") if job['condition']=='degree' else job
    if recovery._mask_hash(masks)!=info['mask_hash'] or (info.get('mask_sha256') and sha(job['mask_path'])!=info['mask_sha256']):raise RuntimeError('Mask identity changed')
    if job['condition']=='matrix':assert_control(model,original,masks,'matrix')
    if job['condition']=='degree':
        from degree import same_degrees
        if not all(same_degrees(original[n],masks[n]) for n in masks):raise RuntimeError('Degree budget differs')
    payload=torch.load(c['rewind_source'],map_location='cpu',weights_only=True)
    expected=validate_coordinates(payload['model'],masks)
    if expected['shared']!=c['initial_masked_hashes']['shared'] or expected['router']!=c['initial_masked_hashes']['router']:raise RuntimeError('Shared/router initialization differs')
    train,val=batches(c)
    seed_everything(seed);configure_device(device)
    reference_model=load_model_from_checkpoint(cfg['model'],c['diagnostic_reference'],device)
    reference=selected_experts_per_batch(reference_model,recovery._calibration_batches(val),device)
    del reference_model,model,payload
    status=job_status(job);dest=Path(status['attempt'])
    save(OUT/'configs'/f"{job['id']}.json",dict(job=job,config=cfg,cohort=c,expected_initialization=expected))
    identity=dict(experiment='imp-mask-information-v2',job=job['id'],seed=seed,condition=job['condition'],mask_seed=job['mask_seed'],mask_hash=info['mask_hash'],rewind_sha256=c['rewind_sha256'],sparsity=float(c['sparsity']),steps=2500,torch_version=str(torch.__version__),manifest_sha256=sha(OUT/'manifest.json'))
    path=OUT/'logs'/f"{job['id']}.log";path.parent.mkdir(exist_ok=True)
    with path.open('a',buffering=1) as log,contextlib.redirect_stdout(log),contextlib.redirect_stderr(log):
        record=canonical().run_stage(cfg,Path(c['rewind_source']),masks,train,val,reference,device,dest,float(c['dense_comparison_ce']),identity)
    checks=dict(expert_initialization=record['expert_state_hash']==expected['expert'],shared_initialization=record['shared_state_hash']==expected['shared'],router_initialization=record['initial_router_state_hash']==expected['router'],train_hash=record['training_batch_sequence_hash']==c['training_batch_hash'],validation_hash=record['validation_batch_sequence_hash']==c['validation_batch_hash'],budget=record['optimizer_step_attempts']==2500,optimizer=record['optimizer']=='fresh_AdamW' and record['optimizer_state_entries_at_start']==0,scheduler=record['scheduler']=='none',trainability=record['router_mode']=='trainable' and record['integrity_checks_passed'],amp_accounting=record['optimizer_steps_applied']+record['amp_overflow_step_count']==2500,endpoint_finite=__import__('math').isfinite(record['final_validation_loss']))
    if job['condition']=='IMP':checks['reproduction']=abs(record['final_validation_loss']-float(c['original_recovery_ce']))<=.005
    save(dest/'checks.json',checks)
    if not all(checks.values()):raise RuntimeError('Recovery validation failed: '+str(checks))
    output=dict(job=job['id'],seed=seed,condition=job['condition'],replicate=job['replicate'],mask_seed=job['mask_seed'],mask_path=job['mask_path'],mask_hash=info['mask_hash'],reused=False,host=socket.getfqdn(),stage_path=str(dest/'stage.json'),curves_path=str(dest/'metrics.jsonl'),final_ce=record['final_validation_loss'],initial_ce=record['initial_validation_loss'],training_ce=record['final_training_ce'],historical_dense_ce=float(c['dense_comparison_ce']),historical_imp_ce=float(c['original_recovery_ce']),attempted_updates=record['optimizer_step_attempts'],applied_updates=record['optimizer_steps_applied'],amp_skips=record['amp_overflow_step_count'],duration_seconds=record['duration_seconds'],checkpoint_sha256=record['checkpoint_sha256'],initialization_hashes=expected,checks=checks)
    save(dest/'result.json',output)
    save(OUT/'status'/f"{job['id']}.json",dict(status='COMPLETE',host=socket.getfqdn(),pid=os.getpid(),attempt=str(dest),result_path=str(dest/'result.json'),updated=time.time()))
    del train,val,reference,masks,original;torch.cuda.empty_cache()


def worker():
    torch.set_num_threads(4);host=socket.getfqdn()
    with lock(OUT/'workers'/f'{host}.lock',False):
        manifest=frozen();stop=threading.Event();active={'job':None,'state':'preflight'}
        def heartbeat():
            while not stop.is_set():
                save(OUT/'workers'/f'{host}.heartbeat.json',dict(host=host,pid=os.getpid(),active=dict(active),unix=time.time()))
                stop.wait(15)
        thread=threading.Thread(target=heartbeat,daemon=True);thread.start()
        try:
            check_host(manifest);active['state']='dispatch'
            while not (OUT/'STOP.json').exists():
                frozen()
                job=claim(manifest,host)
                if job:
                    active.update(job=job['id'],state='recovery')
                    try:run_job(job,manifest)
                    except BaseException as e:
                        prior=job_status(job);save(OUT/'status'/f"{job['id']}.json",dict(prior,status='FAILED',error=repr(e),traceback=traceback.format_exc(),updated=time.time()))
                        raise
                    finally:active.update(job=None,state='dispatch')
                    refresh(manifest)
                    continue
                active['state']='phase-transition-or-wait'
                transition(manifest);refresh(manifest)
                statuses=[job_status(j)['status'] for j in manifest['jobs']]
                if all(s in ('COMPLETE','REUSED','NOT_ADMITTED') for s in statuses):break
                time.sleep(15)
            refresh(manifest)
            save(OUT/'workers'/f'{host}.finished.json',dict(host=host,pid=os.getpid(),unix=time.time(),stopped=(OUT/'STOP.json').exists()))
        except BaseException as e:
            save(OUT/'STOP.json',dict(host=host,pid=os.getpid(),error=repr(e),traceback=traceback.format_exc(),unix=time.time(),policy='Preserve attempts; no retries; no degree expansion'))
            save(OUT/'workers'/f'{host}.failed.json',read(OUT/'STOP.json'))
            refresh(manifest);raise
        finally:
            stop.set();thread.join(timeout=2)
            save(OUT/'workers'/f'{host}.heartbeat.json',dict(host=host,pid=os.getpid(),active=dict(active),unix=time.time(),finished=True))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--check-host',action='store_true');a=p.parse_args()
    if a.check_host:check_host(frozen())
    else:worker()
