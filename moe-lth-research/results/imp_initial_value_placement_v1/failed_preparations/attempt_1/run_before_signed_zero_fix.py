"""Fixed 32E surviving initial-value placement study.

Native segment training, snapshots, common readout and interruption semantics
are imported from the audited order study. Historical artifacts are read-only.
"""
from __future__ import annotations
import argparse
import csv
import fcntl
import hashlib
import importlib.util
import inspect
import json
import math
import os
from pathlib import Path
import shutil
import socket
import subprocess
import sys
import time
from copy import deepcopy

ROOT = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location('placement_native_order', ROOT/'scripts/imp_recovery_order_stability_v1/run.py')
native = importlib.util.module_from_spec(spec)
spec.loader.exec_module(native)
import numpy as np
import torch
from moe_lth.pruning.imp import validate_masks
from moe_lth.pruning.masks import load_masks, save_masks, expert_weight_parameters

OLD = native.OUT
OUT = ROOT/'results/imp_initial_value_placement_v1'
SEEDS = native.SEEDS
STEPS = native.SNAPSHOT_STEPS
VARIANTS = ('imp_original','imp_shuffle1','imp_shuffle2','random1_original','random1_shuffle','random2_original','random2_shuffle')
PILOTS = ('imp_shuffle1','random1_original','random1_shuffle')
sha = native._sha256_file
read = native._json
save = native._atomic_json
table = native._write_csv

def rows(path):
    with Path(path).open() as f:
        return list(csv.DictReader(f))

def derived_seed(base, source, tensor, kind):
    text = f'initial-value-placement-v1|{kind}|{base}|{source}|{tensor}'
    return int.from_bytes(hashlib.sha256(text.encode()).digest()[:8], 'little') % (2**63)

def transform(w0, mask, source, name, mask_seed=None, shuffle_seed=None):
    """Identity arrays are persisted, canonical coordinates are ascending C order."""
    assert w0.device.type == mask.device.type == 'cpu'
    assert mask.dtype == torch.bool and w0.shape == mask.shape and w0.ndim == 2
    m = mask.clone()
    if mask_seed is not None:
        g = torch.Generator(device='cpu').manual_seed(derived_seed(mask_seed,source,name,'mask'))
        coordinates = torch.randperm(mask.numel(), generator=g)[:int(mask.sum())]
        m.zero_(); m.view(-1)[coordinates] = True
    coordinates = m.flatten().nonzero().flatten()
    permutation = torch.arange(len(coordinates), dtype=torch.int64)
    if shuffle_seed is not None:
        g = torch.Generator(device='cpu').manual_seed(derived_seed(shuffle_seed,source,name,'shuffle'))
        permutation = torch.randperm(len(coordinates), generator=g)
    values = w0.flatten()[coordinates]
    output = torch.zeros_like(w0)
    output.view(-1)[coordinates] = values[permutation]
    return output, m, coordinates, permutation

def validate_value(w0, result, mask, coordinates, permutation):
    original = w0.flatten()[coordinates]
    actual = result.flatten()[coordinates]
    assert torch.equal(coordinates, mask.flatten().nonzero().flatten())
    assert torch.equal(torch.sort(permutation).values, torch.arange(len(coordinates)))
    assert torch.equal(actual, original[permutation])
    assert torch.equal(torch.sort(original).values, torch.sort(actual).values)
    assert torch.count_nonzero(result[~mask]) == 0
    assert (original > 0).sum() == (actual > 0).sum()
    assert (original < 0).sum() == (actual < 0).sum()
    energy0, energy1 = original.double().square().sum().item(), actual.double().square().sum().item()
    # FP64 sum is order dependent; tolerance fixed before any intervention loss.
    assert math.isclose(energy0,energy1,rel_tol=1e-12,abs_tol=1e-12)
    return energy0, energy1

def verify():
    reg = read(OUT/'registration.json')
    for p,h in reg['sealed_files'].items():
        if sha(Path(p)) != h: raise RuntimeError('Sealed input changed: '+p)
    return reg

def configure():
    native.OUT = OUT
    native._verify_registration = verify
    native._estimated_remaining_storage_bytes = storage_estimate
    native._pilot_gate_passed = lambda: (OUT/'pilot_gate.json').exists() and read(OUT/'pilot_gate.json').get('passed',False)

def storage_estimate():
    jobs=rows(OUT/'job_manifest.csv')
    remaining=sum(not (OUT/'status'/f"{r['job_id']}.json").exists() or read(OUT/'status'/f"{r['job_id']}.json")['status']!='COMPLETE' for r in jobs)
    size=max(Path(r['initial_checkpoint']).stat().st_size for r in jobs)
    # Per job: 11 model snapshots, 1 reset snapshot, 2 native final models,
    # two model+Adam-moments interruption states (3 model equivalents each).
    # Six extra equivalents cover atomic-write staging on both hosts.
    return int(size*(20*remaining+6)*1.03+10*1024**3)


def variant_state(row):
    bundle = torch.load(row['variant_path'],map_location='cpu',weights_only=True)
    if sha(Path(row['variant_path'])) != row['variant_sha256']: raise RuntimeError('Variant changed')
    original = native._read_model_payload(Path(row['initial_checkpoint']))
    if original['step'] != 0 or sha(Path(row['initial_checkpoint'])) != row['initial_checkpoint_sha256']:
        raise RuntimeError('Original initialization changed')
    state = {n:t.clone() for n,t in original['model'].items()}
    state.update(bundle['weights'])
    if native._tensor_state_hash(state) != row['initial_model_state_hash']: raise RuntimeError('Starting model changed')
    masks = load_masks(Path(row['mask_path']))
    if native.recovery._mask_hash(masks) != row['mask_sha256']: raise RuntimeError('Mask changed')
    return state,masks

# A narrowly edited copy of _run_one_job lives beside this adapter. It replaces
# only source/mask assembly; native training/reset/evaluation remain imported.

def prepare():
    if (OUT/'registration.json').exists(): raise RuntimeError('Registration exists; preserve it')
    OUT.mkdir(parents=True,exist_ok=True)
    # Check old registration first, including source code and frozen cohort.
    native.OUT=OLD
    original_verify()
    sparse,dense = native._cohort_jobs()
    inventory = rows(OLD/'artifact_inventory.csv')
    for item in inventory:
        if sha(Path(item['path'])) != item['sha256']: raise RuntimeError('Historical input changed: '+item['path'])
    for filename in ('canonical_stream_manifest.json','recovery_order_manifest.json','initialization_and_mask_hashes.json'):
        shutil.copy2(OLD/filename,OUT/filename)
    shutil.copytree(OLD/'permutations',OUT/'permutations',dirs_exist_ok=True)
    checks=[]; manifest=[]; validation=[]; transformations={}; reused=[]
    oldjobs = rows(OLD/'job_manifest.csv')
    stream = read(OUT/'canonical_stream_manifest.json')['seeds']
    for seed in SEEDS:
        base=next(r for r in oldjobs if int(r['seed'])==seed and r['model_type']=='imp80')
        init=native._read_model_payload(Path(base['initial_checkpoint']))
        assert init['step']==0 and init['config']['seed']==seed
        assert init['config']['model']['num_experts']==32
        masks=load_masks(Path(base['mask_path']))
        with native._preserved_rng(): model=native.TinyMoELanguageModel(deepcopy(init['config']['model']))
        model.load_state_dict(init['model'],strict=True);validate_masks(model,masks)
        meta=read(Path(base['mask_path']).with_suffix('.json'))
        assert meta['initial_sha256']==base['initial_checkpoint_sha256']
        assert meta['original_initialization_checkpoint']==base['initial_checkpoint']
        assert meta['pruning_round']==8 and meta['target_sparsity']==.8
        assert meta['mask_hash']==native.recovery._mask_hash(masks)==base['mask_sha256']
        for variant in VARIANTS:
            j=1 if '1' in variant else 2
            family='random' if variant.startswith('random') else 'imp'
            shuffled='shuffle' in variant
            maskbase=92000+j if family=='random' else None
            shufflebase=93000+j if shuffled else None
            weights={}; vmasks={}; identity={}
            for name in sorted(masks):
                w,m,c,p=transform(init['model'][name],masks[name],seed,name,maskbase,shufflebase)
                e0,e1=validate_value(init['model'][name],w,m,c,p)
                assert m.sum()==masks[name].sum()
                weights[name]=w;vmasks[name]=m
                identity[name]={'coordinates':c,'permutation':p}
                validation.append(dict(seed=seed,variant=variant,tensor=name,shape=str(list(m.shape)),survivors=int(m.sum()),binary=True,multiset_exact=True,masked_zero=True,sign_counts_exact=True,energy_original=e0,energy_transformed=e1,energy_tolerance='FP64 rtol=atol=1e-12',mask_hash=native._tensor_state_hash({name:m}),row_counts_hash=native._tensor_state_hash({'rows':m.sum(1)}),column_counts_hash=native._tensor_state_hash({'columns':m.sum(0)}),dead_rows=int((m.sum(1)==0).sum()),dead_columns=int((m.sum(0)==0).sum())))
            d=OUT/'variants'/f'seed_{seed}'/variant;d.mkdir(parents=True,exist_ok=False)
            torch.save(dict(weights=weights,identities=identity),d/'values_and_identities.pt')
            save_masks(vmasks,d/'mask.pt')
            full={n:t.clone() for n,t in init['model'].items()};full.update(weights)
            assert all(torch.equal(t,full[n]) for n,t in init['model'].items() if n not in weights)
            if variant=='imp_original': assert native._tensor_state_hash(full)==base['initial_model_state_hash']
            transformations[f'{seed}/{variant}']=dict(path=str(d/'values_and_identities.pt'),sha256=sha(d/'values_and_identities.pt'),mask_path=str(d/'mask.pt'),mask_file_sha256=sha(d/'mask.pt'),mask_hash=native.recovery._mask_hash(vmasks),mask_generator_seed=maskbase,shuffle_generator_seed=shufflebase,starting_state_hash=native._tensor_state_hash(full))
            for order in ('A','B'):
                control=next(r for r in oldjobs if int(r['seed'])==seed and r['order']==order and r['model_type']=='imp80')
                row=dict(control)
                row.update(job_id=f'e32_seed{seed}_order_{order}_{variant}',variant=variant,family=family,realization=0 if variant=='imp_original' else j,placement='shuffled' if shuffled else 'original',variant_path=str(d/'values_and_identities.pt'),variant_sha256=sha(d/'values_and_identities.pt'),mask_path=str(d/'mask.pt'),mask_sha256=native.recovery._mask_hash(vmasks),initial_model_state_hash=native._tensor_state_hash(full),pilot=seed==17 and order=='A' and variant in PILOTS,reused=variant=='imp_original')
                row['status_path']=f"status/{row['job_id']}.json";row['output_path']=f"jobs/{row['job_id']}"
                if variant=='imp_original':
                    status=read(OLD/'status'/f"{control['job_id']}.json");attempt=Path(status['attempt'])
                    evals=[json.loads(l) for l in (attempt/'common_evaluations.jsonl').read_text().splitlines()]
                    valid=dict(completed=status['status']=='COMPLETE',attempts=status['attempted_updates']==5000,reset=status['step_2500_model_reset_hash_equal'],init=status['initial_masked_state_hash']==row['initial_model_state_hash'],readout_hash=sha(attempt/'common_evaluations.jsonl')==status['common_evaluation_sha256'],steps=[v['step'] for v in evals]==list(STEPS),readout=all(v['precision']=='FP32' and v['tf32'] is False and v['target_tokens']==stream[str(seed)]['validation_target_tokens'] and v['validation_stream_hash']==stream[str(seed)]['validation_batch_sequence_hash'] for v in evals))
                    assert all(valid.values()),valid
                    snap=native._read_model_payload(attempt/'snapshots/step_0000.pt')
                    assert native._tensor_state_hash(snap['model'])==row['initial_model_state_hash']
                    update=native._update_accounting(attempt);assert len(update)==5000 and len({r['step'] for r in update})==5000
                    checks.append(dict(seed=seed,order=order,compatible=True,**valid))
                    row['reused_attempt']=str(attempt);row['reused_status_path']=str(OLD/'status'/f"{control['job_id']}.json")
                    reused.append((row['job_id'],status))
                manifest.append(row)
            del weights,vmasks,identity,full
        del model,init,masks
    assert len(manifest)==70 and len(reused)==10
    table(OUT/'variant_manifest.csv',manifest);table(OUT/'job_manifest.csv',[r for r in manifest if not r['reused']])
    table(OUT/'control_reuse_checks.csv',checks);table(OUT/'mask_and_value_validation.csv',validation)
    save(OUT/'transformation_hashes.json',transformations)
    # Full user protocol is authoritative, with implementation details appended.
    text=(OUT/'user_request.txt').read_text()
    (OUT/'protocol.md').write_text('# Surviving initial-value placement protocol\n\n'+text+'\n\nImplementation: canonical row-major tensor flattening; SHA256 derivation described in registration. Energy comparisons use FP64 rtol=atol=1e-12. All 11 snapshots are preserved and explicitly nonresumable; native optimizer/scaler/RNG interruption checkpoints are resumable. No replacement initialization. Existing control readouts are reused only after exact input and schema checks.\n')
    code=native._code_inventory()+[dict(path=str(p),sha256=sha(p),bytes=p.stat().st_size) for p in sorted((ROOT/'scripts/imp_initial_value_placement_v1').glob('*.py'))]
    sealed={}
    for item in inventory+code:sealed[item['path']]=item['sha256']
    for p in (OUT/'variants').rglob('*.pt'):sealed[str(p)]=sha(p)
    for p in (OUT/'permutations').glob('*.npy'):sealed[str(p)]=sha(p)
    for row in manifest:
        if row['reused']:
            attempt=Path(row['reused_attempt'])
            for p in (attempt/'common_evaluations.jsonl',attempt/'completion.json',Path(row['reused_status_path'])):sealed[str(p)]=sha(p)
    for filename in ('protocol.md','user_request.txt','canonical_stream_manifest.json','recovery_order_manifest.json','initialization_and_mask_hashes.json','variant_manifest.csv','job_manifest.csv','control_reuse_checks.csv','mask_and_value_validation.csv','transformation_hashes.json'):
        sealed[str(OUT/filename)]=sha(OUT/filename)
    table(OUT/'artifact_inventory.csv',[dict(path=p,sha256=h,bytes=Path(p).stat().st_size) for p,h in sealed.items()])
    sealed[str(OUT/'artifact_inventory.csv')]=sha(OUT/'artifact_inventory.csv')
    reg=dict(title='32E IMP80 surviving initial-value placement',sealed_files=sealed,created_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),sources=list(SEEDS),orders=['A','B'],variants=list(VARIANTS),new_jobs=60,reused_controls=10,attempts=5000,readout_steps=list(STEPS),primary_endpoint=5000,secondary_endpoint=2500,primary_estimand='mean(IMP shuffle1, IMP shuffle2) - IMP original - mean(random1 shuffle-original, random2 shuffle-original)',aggregation='equal A/B mean within source; five sources are replication units',hypothesis='IMP shuffling penalty exceeds random-mask shuffling penalty; not an established mechanism',generator='isolated CPU torch.Generator; seed = little-endian first 8 SHA256 bytes of initial-value-placement-v1|kind|base|source|tensor modulo 2**63; randperm',mask_seeds=[92001,92002],shuffle_seeds=[93001,93002],production_ready=False,pilot=dict(source=17,order='A',variants=list(PILOTS),outcome_independent=True),no_grid_expansion=True,numerical_tolerances=read(OLD/'registration.json')['numerical_tolerances'])
    save(OUT/'registration.json',reg)
    shutil.copy2(OUT/'registration.json',OUT/'registration.initial.json')
    configure()
    for job,status in reused:save(OUT/'status'/f'{job}.json',dict(status,job_id=job,reused=True))
    print(json.dumps(dict(status='REGISTERED',new=60,reused=10)))

original_verify=native._verify_registration


def contrast(v):
    di=(v['imp_shuffle1']+v['imp_shuffle2'])/2-v['imp_original']
    dr=((v['random1_shuffle']-v['random1_original'])+(v['random2_shuffle']-v['random2_original']))/2
    a0=(v['random1_original']+v['random2_original'])/2-v['imp_original']
    a1=(v['random1_shuffle']+v['random2_shuffle'])/2-(v['imp_shuffle1']+v['imp_shuffle2'])/2
    assert math.isclose(di-dr,a0-a1,abs_tol=1e-12,rel_tol=1e-12)
    return dict(delta_imp=di,delta_random=dr,interaction=di-dr,A_original=a0,A_shuffled=a1)

def engineering():
    verify();configure()
    # Native two-segment engineering suite exercises interruption before/after
    # reset, exact resumed endpoints, RNG-preserving evaluation, masking, updates.
    # Its manifest-only checks expect its own original dense/IMP cohort.
    real_manifest=native._read_job_manifest
    native._read_job_manifest=lambda:rows(OLD/'job_manifest.csv')
    native.verify_engineering()
    native._read_job_manifest=real_manifest
    report=read(OUT/'engineering_checks.json')
    checks=report['checks']
    w=torch.arange(1,49,dtype=torch.float32).reshape(6,8)-24
    m=(torch.arange(48).reshape(6,8)%3==0)
    before=torch.get_rng_state().clone()
    first=transform(w,m,17,'synthetic.fc1.weight',92001,93001)
    second=transform(w,m,17,'synthetic.fc1.weight',92001,93001)
    checks['transform_global_rng_preserved']=torch.equal(before,torch.get_rng_state())
    checks['deterministic_transform']=all(torch.equal(a,b) for a,b in zip(first,second))
    validate_value(w,*first[:2],*first[2:])
    checks['exact_values_and_zero_contamination']=True
    v=dict(imp_original=1.,imp_shuffle1=2.,imp_shuffle2=4.,random1_original=3.,random1_shuffle=4.,random2_original=5.,random2_shuffle=7.)
    c=contrast(v);checks['contrast_known_answer']=c['interaction']==.5 and c['A_original']==3. and c['A_shuffled']==2.5
    # Real pairs: same mask, degrees/dead units/connectivity, retained multiset.
    allrows=rows(OUT/'variant_manifest.csv')
    for seed in SEEDS:
        for shuffled,original in (('imp_shuffle1','imp_original'),('imp_shuffle2','imp_original'),('random1_shuffle','random1_original'),('random2_shuffle','random2_original')):
            a=next(r for r in allrows if int(r['seed'])==seed and r['variant']==shuffled)
            b=next(r for r in allrows if int(r['seed'])==seed and r['variant']==original)
            assert a['mask_sha256']==b['mask_sha256']
    checks['fixed_binary_connectivity_across_placement_and_orders']=True
    # AdamW with nonzero decay: test a transformed matrix after real backward.
    w0,mask,coords,perm=first
    p=torch.nn.Parameter(w0.clone());p.register_hook(lambda g:g*mask)
    opt=torch.optim.AdamW([p],lr=.01,weight_decay=.1)
    for _ in range(3):
        opt.zero_grad();p.square().sum().backward();opt.step()
        with torch.no_grad():p.mul_(mask)
    checks['transformed_mask_persists_under_nonzero_adamw_decay']=torch.count_nonzero(p[~mask]).item()==0
    report.update(status='PASS' if all(checks.values()) else 'FAIL',checks=checks)
    save(OUT/'engineering_checks.json',report)
    assert report['status']=='PASS'
    print(json.dumps(report))

def benchmark():
    verify();configure()
    result=native.benchmark()
    result['estimated_total_gpu_seconds']*=3
    result['estimated_total_gpu_hours']*=3
    result['estimated_elapsed_hours_at_two_available_gpus']*=3
    result['new_jobs']=60
    # native conservative estimate already reads our 60-job manifest.
    save(OUT/'engineering/throughput_benchmark/benchmark.json',result)
    save(OUT/'compute_accounting.json',dict(projected=result,measured_production_gpu_seconds=0))
    print(json.dumps(result))

def verify_host():
    configure();print(json.dumps(native._verify_artifacts_on_host()))

def seal():
    reg=verify()
    assert read(OUT/'engineering_checks.json')['status']=='PASS'
    profile=read(OUT/'engineering/throughput_benchmark/benchmark.json')
    assert math.isfinite(profile['seconds_per_attempt'])
    for host in native.HOST_SEEDS:
        h=read(OUT/'hosts'/f'{host}.json');assert h['verified'] and h['host']==host
    free=shutil.disk_usage(OUT).free
    need=profile['snapshot_storage_estimate_bytes']
    save(OUT/'resource_gate.json',dict(free_bytes=free,required_bytes=need,passed=free>=need))
    if free<need:raise RuntimeError(f'Storage infeasible: need {need}, free {free}. No production launched.')
    reg.update(production_ready=True,estimated_gpu_hours=profile['estimated_total_gpu_hours'],estimated_storage_bytes=need,host_input_verification={h:sha(OUT/'hosts'/f'{h}.json') for h in native.HOST_SEEDS})
    save(OUT/'registration.json',reg)
    save(OUT/'production_seal.json',dict(registration_sha256=sha(OUT/'registration.json'),engineering_sha256=sha(OUT/'engineering_checks.json'),benchmark_sha256=sha(OUT/'engineering/throughput_benchmark/benchmark.json'),status='SEALED'))

def pilot_gate():
    pilots=[r for r in rows(OUT/'job_manifest.csv') if r['pilot']=='True']
    if not all((OUT/'status'/f"{r['job_id']}.json").exists() and read(OUT/'status'/f"{r['job_id']}.json")['status']=='COMPLETE' for r in pilots):return False
    details=[]
    for row in pilots:
        s=read(OUT/'status'/f"{row['job_id']}.json");a=Path(s['attempt'])
        assert s['attempted_updates']==5000 and s['applied_updates']+s['amp_skipped_updates']==5000
        assert s['step_2500_model_reset_hash_equal'] and s['common_evaluation_steps']==list(STEPS)
        assert s['initial_masked_state_hash']==row['initial_model_state_hash']
        assert sha(a/'common_evaluations.jsonl')==s['common_evaluation_sha256']
        es=[json.loads(l) for l in (a/'common_evaluations.jsonl').read_text().splitlines()]
        assert all(math.isfinite(e['cross_entropy']) and e['precision']=='FP32' and e['tf32'] is False for e in es)
        mask=load_masks(Path(row['mask_path']))
        for step in STEPS:
            state=native._read_model_payload(a/'snapshots'/f'step_{step:04d}.pt')['model']
            assert all(torch.count_nonzero(state[n][~m])==0 for n,m in mask.items())
        details.append(dict(job_id=row['job_id'],attempts=5000,applied=s['applied_updates'],amp_skips=s['amp_skipped_updates'],complete=True,mask_integrity=True,reset_equal=True))
    need=native._estimated_remaining_storage_bytes();free=shutil.disk_usage(OUT).free
    gate=dict(passed=free>=need,provenance=True,numerical=True,complete_endpoints=True,correct_accounting=True,masking=True,resources=free>=need,free_bytes=free,required_bytes=need,outcome_signs_used=False,jobs=details)
    save(OUT/'pilot_gate.json',gate)
    return gate['passed']

def worker(resume=False,device=0):
    configure();reg=verify();assert reg['production_ready']
    seal_record=read(OUT/'production_seal.json')
    assert sha(OUT/'registration.json')==seal_record['registration_sha256']
    host=socket.getfqdn();assert host in reg['host_input_verification']
    native._verify_artifacts_on_host()
    assert torch.__version__.startswith('2.7.0') and torch.cuda.is_available()
    lock=OUT/'locks'/f'worker_{host}_{device}.lock';lock.parent.mkdir(parents=True,exist_ok=True)
    with lock.open('a') as f:
        fcntl.flock(f,fcntl.LOCK_EX|fcntl.LOCK_NB)
        from job import _run_one_job
        assigned=set(native._assigned_seeds(host))
        jobs=rows(OUT/'job_manifest.csv')
        while True:
            candidates=[];allcomplete=True
            gate=native._pilot_gate_passed()
            for row in jobs:
                p=OUT/'status'/f"{row['job_id']}.json";s=read(p) if p.exists() else {}
                if s.get('status')!='COMPLETE':allcomplete=False
                if int(row['seed']) not in assigned or s.get('status')=='COMPLETE':continue
                if s.get('status')=='FAILED' and not resume:continue
                if not gate and row['pilot']!='True':continue
                candidates.append(row)
            save(OUT/'workers'/f'{host}.json',dict(pid=os.getpid(),host=host,heartbeat_unix=time.time(),pilot_passed=gate,status='ACTIVE'))
            if allcomplete:
                analyze();return
            if not candidates:
                if any((OUT/'status'/f"{r['job_id']}.json").exists() and read(OUT/'status'/f"{r['job_id']}.json")['status']=='FAILED' for r in jobs if r['pilot']=='True'):
                    analyze();raise RuntimeError('Engineering pilot failed; cohort withheld')
                time.sleep(30);continue
            candidates.sort(key=lambda r:(r['pilot']!='True',int(r['seed']),r['order'],VARIANTS.index(r['variant'])))
            torch.cuda.empty_cache()
            idle,details=native._gpu_idle(device)
            if not idle:
                save(OUT/'workers'/f'{host}.json',dict(pid=os.getpid(),heartbeat_unix=time.time(),status='WAITING_FOR_EXTERNAL_GPU',gpu=details));time.sleep(30);continue
            try:_run_one_job(candidates[0],resume=resume,device_index=device)
            except BaseException:
                analyze()
                if candidates[0]['pilot']=='True':raise
            torch.cuda.empty_cache()
            if not gate:pilot_gate()
            analyze()


def analyze():
    # Serialize two-host reports; only observed results are populated.
    p=OUT/'locks'/'analysis.lock';p.parent.mkdir(exist_ok=True)
    with p.open('a') as f:
        fcntl.flock(f,fcntl.LOCK_EX)
        _analyze()

def _analyze():
    manifest=rows(OUT/'variant_manifest.csv');observed=[];account=[];pending=[];failed=[]
    for row in manifest:
        status_path=OUT/'status'/f"{row['job_id']}.json"
        s=read(status_path) if status_path.exists() else dict(status='PENDING')
        if s['status']!='COMPLETE':
            pending.append(dict(job_id=row['job_id'],status=s['status'],error=s.get('error')))
            if s['status']=='FAILED':failed.append(row['job_id'])
            continue
        attempt=Path(s['attempt'])
        for line in (attempt/'common_evaluations.jsonl').read_text().splitlines():
            item=json.loads(line);item.update({k:row[k] for k in ('variant','family','realization','placement')});item['job_id']=row['job_id'];item['reused']=row['reused']
            observed.append(item)
        clip=float(native._seed_entry(int(row['seed']))['config']['training']['grad_clip'])
        for segment in ('segment_0','segment_2500'):
            record=read(attempt/segment/'segment_record.json')
            gs=[json.loads(l) for l in (attempt/segment/'gradient_stats/gradient_stats.jsonl').read_text().splitlines()]
            account.append(dict(job_id=row['job_id'],seed=row['seed'],order=row['order'],variant=row['variant'],segment=segment,reused=row['reused'],attempted=record['optimizer_step_attempts'],applied=record['optimizer_steps_applied'],amp_skipped=record['optimizer_step_attempts']-record['optimizer_steps_applied'],seconds=record['segment_seconds'],gradient_log_path=str(attempt/segment/'gradient_stats/gradient_stats.jsonl'),clipping_records=len(gs),clipped_updates=sum(g.get('global_grad_norm_pre_clip') is not None and float(g['global_grad_norm_pre_clip'])>clip for g in gs)))
    lookup={(int(r['seed']),r['order'],r['variant'],int(r['step'])):r for r in observed}
    # Dense controls are references only; never launch dense training.
    for r in observed:
        oldid=f"e32_seed{r['seed']}_order_{r['order']}_unpruned"
        ds=read(OLD/'status'/f'{oldid}.json')
        dense=[json.loads(l) for l in (Path(ds['attempt'])/'common_evaluations.jsonl').read_text().splitlines()]
        d=next(v for v in dense if v['step']==r['step'])
        assert d['precision']=='FP32' and d['tf32'] is False and d['validation_stream_hash']==r['validation_stream_hash']
        r['sparse_minus_unpruned_CE']=r['cross_entropy']-d['cross_entropy']
    contrasts=[];effects=[];source=[]
    for seed in SEEDS:
        for order in ('A','B'):
            for step in STEPS:
                v={name:lookup[(seed,order,name,step)]['cross_entropy'] for name in VARIANTS if (seed,order,name,step) in lookup}
                for j in (1,2):
                    for fam,a,b in (('imp',f'imp_shuffle{j}','imp_original'),('random',f'random{j}_shuffle',f'random{j}_original')):
                        if a in v and b in v:effects.append(dict(seed=seed,order=order,step=step,family=fam,realization=j,original_ce=v[b],shuffled_ce=v[a],penalty=v[a]-v[b],nested=True))
                if len(v)==7:contrasts.append(dict(seed=seed,order=order,step=step,**contrast(v)))
        for step in STEPS:
            cs=[c for c in contrasts if c['seed']==seed and c['step']==step]
            if len(cs)==2:source.append(dict(seed=seed,step=step,**{k:sum(c[k] for c in cs)/2 for k in ('delta_imp','delta_random','interaction','A_original','A_shuffled')}))
    table(OUT/'trajectory_losses.csv',observed,['job_id','seed','order','variant','step','cross_entropy','sparse_minus_unpruned_CE'])
    table(OUT/'endpoint_losses.csv',[r for r in observed if r['step']==5000],['job_id','seed','order','variant','cross_entropy'])
    table(OUT/'realization_level_effects.csv',effects,['seed','order','step','family','realization','penalty'])
    table(OUT/'source_order_contrasts.csv',contrasts,['seed','order','step','interaction','delta_imp','delta_random','A_original','A_shuffled'])
    table(OUT/'source_level_summary.csv',source,['seed','step','interaction'])
    table(OUT/'optimizer_compute_accounting.csv',account,['job_id','segment','attempted','applied','amp_skipped','seconds'])
    measured=sum(float(a['seconds']) for a in account if a['reused']=='False')
    profile=read(OUT/'engineering/throughput_benchmark/benchmark.json') if (OUT/'engineering/throughput_benchmark/benchmark.json').exists() else None
    save(OUT/'compute_accounting.json',dict(projected=profile,measured_production_segment_seconds=measured,measured_production_gpu_hours=measured/3600,reused_compute_excluded=True,storage_bytes=sum(p.stat().st_size for p in OUT.rglob('*') if p.is_file())))
    complete=len(observed)==70*len(STEPS) and not pending
    save(OUT/'completion.json',dict(status='COMPLETE' if complete else 'IN_PROGRESS',registered_endpoints=70,new_jobs=60,reused_controls=10,completed_endpoints=len(observed)//len(STEPS),pending_or_failed=pending,failed_jobs=failed,analysis_complete=complete))
    primary=[s['interaction'] for s in source if s['step']==5000]
    summary='Primary five-source result is pending.'
    if len(primary)==5:
        summary=f'Five-source mean interaction: {np.mean(primary):.8f} CE; sample SD {np.std(primary,ddof=1):.8f}; range [{min(primary):.8f}, {max(primary):.8f}]. Orders averaged equally within each source.'
    (OUT/'findings.md').write_text(f'''# Surviving initial-value placement findings

Status: {'COMPLETE' if complete else 'IN PROGRESS'}. {len(observed)//len(STEPS)} of 70 registered endpoints available; missing and failed outcomes are listed in completion.json.

{summary}

Absolute sparse CE and sparse-minus-unpruned CE are in endpoint_losses.csv. Initial and final penalties are reported separately in realization_level_effects.csv; trajectories retain every realization. Positive interaction means IMP is harmed more by shuffling than sampled random controls. Shared IMP-original controls are used once within each source/order contrast. Random masks and shuffles are nested, not independent source replications.

Initial damage changes the model function; a final deficit alone does not establish slower optimization. No initial-CE adjustment is used. The treatment does not distinguish sign, magnitude, weighted paths or compatibility with biases/router/shared components. IMP and random retained-value distributions and energy are not matched. No novelty or topology-only mechanism is claimed. No equivalence threshold or follow-up grid is introduced.

Projected and measured production costs are separate in compute_accounting.json. Update/AMP/reset records and native gradient logs preserve numerical-accounting differences.
''')
    figures(observed,contrasts,effects)

def figures(observed,contrasts,effects):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    d=OUT/'figures';d.mkdir(exist_ok=True)
    # All plots are refreshed as outcomes arrive; empty panels explicitly pending.
    for number,name in enumerate(('original_vs_shuffled','advantage_before_after','interaction_by_source_order','initial_vs_final_penalty','all_recovery_trajectories'),1):
        fig,axes=plt.subplots(1,5,figsize=(20,4))
        for ax,seed in zip(axes,SEEDS):
            ax.set_title(f'seed {seed}')
            ax.text(.02,.98,'Pending' if not any(int(r['seed'])==seed and r['reused']=='False' for r in observed) else '',transform=ax.transAxes,va='top')
            if number==1:
                for e in effects:
                    if e['seed']==seed and e['step']==5000:ax.plot(['original','shuffled'],[e['original_ce'],e['shuffled_ce']],marker='o',label=f"{e['order']} {e['family']}{e['realization']}")
                ax.set_ylabel('Endpoint CE')
            elif number==2:
                for c in contrasts:
                    if c['seed']==seed and c['step']==5000:ax.plot(['original','shuffled'],[c['A_original'],c['A_shuffled']],marker='o',label=c['order'])
                ax.set_ylabel('Random minus IMP CE')
            elif number==3:
                for c in contrasts:
                    if c['seed']==seed and c['step']==5000:ax.scatter(c['order'],c['interaction'])
                ax.axhline(0,color='black',lw=.5);ax.set_ylabel('Interaction CE')
            elif number==4:
                for e in effects:
                    if e['seed']==seed and e['step']==0:
                        end=next((a for a in effects if a['seed']==seed and a['order']==e['order'] and a['family']==e['family'] and a['realization']==e['realization'] and a['step']==5000),None)
                        if end:ax.scatter(e['penalty'],end['penalty'],label=f"{e['order']} {e['family']}{e['realization']}")
                ax.set_xlabel('Initial CE penalty');ax.set_ylabel('Final CE penalty')
            else:
                for order in ('A','B'):
                    for variant in VARIANTS:
                        rs=sorted([r for r in observed if int(r['seed'])==seed and r['order']==order and r['variant']==variant],key=lambda r:r['step'])
                        if rs:ax.plot([r['step'] for r in rs],[r['cross_entropy'] for r in rs],label=f'{order} {variant}',lw=.8)
                ax.set_xlabel('Attempted updates');ax.set_ylabel('CE')
        handles,labels=axes[-1].get_legend_handles_labels()
        if handles:fig.legend(handles,labels,loc='lower center',ncol=7,fontsize=6)
        fig.tight_layout(rect=(0,.13,1,1));fig.savefig(d/f'{number}_{name}.png',dpi=140);plt.close(fig)


def main():
    parser=argparse.ArgumentParser();parser.add_argument('command',choices=['prepare','engineering','benchmark','verify-host','seal','worker','analyze','status']);parser.add_argument('--resume',action='store_true');parser.add_argument('--device-index',type=int,default=0)
    args=parser.parse_args();torch.set_num_threads(2)
    if args.command=='prepare':prepare();return
    configure()
    if args.command=='worker':worker(args.resume,args.device_index)
    elif args.command=='status':
        print(json.dumps(read(OUT/'completion.json') if (OUT/'completion.json').exists() else {'status':'REGISTERED'},indent=2))
    else:globals()[args.command.replace('-','_')]()

if __name__=='__main__':
    # job.py imports this module by canonical name to avoid a second native copy.
    sys.modules['run']=sys.modules[__name__]
    main()
