"""Bounded one-checkpoint pilot. No discovery or automatic expansion."""
import argparse
import csv
import fcntl
import importlib.util
import json
import math
import os
import statistics
import sys
import time
from pathlib import Path

# Existing canonical loaders explicitly request unrestricted pickle; force the
# supported torch safe loader globally, including calls inside the old trainer.
os.environ['TORCH_FORCE_WEIGHTS_ONLY_LOAD'] = '1'
ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT/'src'),str(ROOT/'scripts/imp_mask_information')]
import torch
from audit import OUT, rows, sha, write_csv
from moe_lth.models import TinyMoELanguageModel
from moe_lth.pruning.mask_information import randomize, assert_control
from moe_lth.pruning.masks import save_masks
from moe_lth.data import build_dataloaders
from moe_lth.pruning.router_age import load_model_from_checkpoint, selected_experts_per_batch
from moe_lth.experiments import run_router_age_recovery as recovery
from moe_lth.utils import configure_device, seed_everything


def canonical():
    spec=importlib.util.spec_from_file_location('canonical_imp_runner', ROOT/'scripts/imp_80_85/run.py')
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    return module


def jwrite(path,obj):
    tmp=path.with_suffix('.tmp');tmp.write_text(json.dumps(obj,indent=2)+'\n');tmp.replace(path)


def prepare():
    torch.set_num_threads(4)
    if not (OUT/'protocol.md').is_file() or not (OUT/'audit_review.json').is_file(): raise RuntimeError('Completed audit review and protocol required first')
    inventory=rows(OUT/'checkpoint_inventory.csv')
    choices=[r for r in inventory if r['eligible']=='True' and r['configuration_id'].endswith('final_s0.80')]
    if not choices: raise RuntimeError('No verified 80% IMP recoveries')
    # Strongest relative endpoint, deterministic tie break. Selection uses IMP
    # only, prior to seeing any randomized-mask outcome; generalization is limited.
    choice=min(choices,key=lambda r:(float(r['original_recovery_ce'])/float(r['dense_comparison_ce']),int(r['num_experts']),int(r['seed'])))
    config=torch.load(choice['rewind_source'],map_location='cpu',weights_only=True)['config']
    original=torch.load(choice['mask_source'],map_location='cpu',weights_only=True)
    model=TinyMoELanguageModel(config['model'])
    dest=OUT/'masks';dest.mkdir(exist_ok=True)
    plan=[]
    seen=set()
    for ci,condition in enumerate(['global','expert','matrix']):
        for replicate in range(5):
            seed=810000+ci*100+replicate
            path=dest/f'{condition}_{seed}.pt'
            masks=randomize(model,original,condition,seed)
            if path.exists():
                saved=torch.load(path,map_location='cpu',weights_only=True)
                if not all(torch.equal(saved[n],masks[n]) for n in masks): raise RuntimeError('Saved mask identity differs')
            else: save_masks(masks,path)
            assert_control(model,original,masks,condition)
            digest=recovery._mask_hash(masks)
            if digest in seen: raise RuntimeError('Duplicate pilot mask realization')
            seen.add(digest)
            plan.append(dict(condition=condition,replicate=replicate,mask_seed=seed,mask_path=str(path),mask_hash=recovery._mask_hash(masks),mask_sha256=sha(str(path))))
    durations=[float(r['duration_seconds']) for r in inventory if r['family']=='IMP' and r['num_experts']==choice['num_experts'] and 'final_s' in r['configuration_id']]
    estimate=dict(number_full_recoveries=16,steps_each=2500,median_observed_recovery_seconds=statistics.median(durations),observed_recovery_seconds_range=[min(durations),max(durations)],estimated_gpu_hours=16*statistics.median(durations)/3600,range_scenario_gpu_hours=[16*min(durations)/3600,16*max(durations)/3600],limitations='Observed same-architecture full recovery times; shared GPU contention may change throughput. Occupancy estimate, not billed cost. Includes an IMP reproduction. No discovery.')
    source_files=['scripts/imp_80_85/run.py','src/moe_lth/experiments/run_router_age_recovery.py','src/moe_lth/pruning/mask_information.py','src/moe_lth/training/checkpoint.py','src/moe_lth/pruning/router_age.py','src/moe_lth/data.py']
    source_hashes={p:sha(str(ROOT/p)) for p in source_files}
    result=dict(source_hashes=source_hashes,configuration=choice,config=config,controls=plan,reproduction_tolerance_ce=.005,equivalence_margin_ce=.02,estimate=estimate)
    if (OUT/'pilot_plan.json').exists():
        old=json.loads((OUT/'pilot_plan.json').read_text())
        if old!=result: raise RuntimeError('Plan changed; preserve existing pilot and review explicitly')
    jwrite(OUT/'pilot_plan.json',result)
    jwrite(OUT/'resource_estimate.json',estimate)
    jwrite(OUT/'pilot_config.json',config)
    print(json.dumps(dict(selected=choice['configuration_id'],estimate=estimate)),flush=True)
    return result


def execute(plan):
    if not torch.__version__.startswith('2.7.0'): raise RuntimeError('Canonical torch 2.7.0 required')
    if not torch.cuda.is_available(): raise RuntimeError('CUDA unavailable; no CPU substitution')
    jwrite(OUT/'pilot_status.json',dict(status='running',completed_recoveries=0,full_expansion=False))
    c=plan['configuration'];cfg=plan['config'];seed=int(c['seed']);device=torch.device('cuda')
    for key,path in [('rewind_sha256',c['rewind_source']),('recovery_sha256',c['recovery_checkpoint']),('dense_sha256',c['dense_checkpoint'])]:
        sha.cache_clear()
        if sha(path)!=c[key]: raise RuntimeError(f'Artifact changed: {path}')
    original=torch.load(c['mask_source'],weights_only=True,map_location='cpu')
    if recovery._mask_hash(original)!=c['mask_hash']: raise RuntimeError('Original mask identity changed')
    trainer=canonical()
    if trainer.common.protocol_errors(cfg,int(c['num_experts']),seed): raise RuntimeError('Canonical config mismatch')
    seed_everything(seed);configure_device(device)
    tr,va=build_dataloaders(cfg['data'],int(cfg['training']['batch_size']),seed)
    train=recovery._materialize_batches(tr,2500);val=recovery._materialize_validation_batches(va,32)
    if recovery._batch_sequence_hash(train)!=c['training_batch_hash'] or recovery._batch_sequence_hash(val)!=c['validation_batch_hash']: raise RuntimeError('Data mismatch; stop')
    # Diagnostic reference is recorded explicitly in source IMP runner, never
    # used as a rewind. Require it; never substitute another run.
    reference_path=Path(c['rewind_source']).parent/'step_2500.pt'
    if not reference_path.is_file(): raise FileNotFoundError(reference_path)
    reference_model=load_model_from_checkpoint(cfg['model'],str(reference_path),device)
    reference=selected_experts_per_batch(reference_model,recovery._calibration_batches(val),device)
    del reference_model
    jwrite(OUT/'runtime_provenance.json',dict(torch=torch.__version__,gpu=torch.cuda.get_device_name(),diagnostic_reference=str(reference_path),diagnostic_reference_sha256=sha(str(reference_path)),source_config=cfg))
    model=TinyMoELanguageModel(cfg['model'])
    conditions=[dict(condition='IMP',replicate=0,mask_seed=None,mask_path=c['mask_source'],mask_hash=c['mask_hash'])]+plan['controls']
    observations=[]
    try:
        for condition in conditions:
            name=condition['condition'];mask=torch.load(condition['mask_path'],map_location='cpu',weights_only=True)
            if recovery._mask_hash(mask)!=condition['mask_hash']: raise RuntimeError('Mask changed')
            if name!='IMP': assert_control(model,original,mask,name)
            identity=dict(experiment='imp-mask-information-v1',configuration=c['configuration_id'],condition=name,mask_seed=condition['mask_seed'],mask_hash=condition['mask_hash'],sparsity=float(c['sparsity']),rewind_sha256=c['rewind_sha256'],steps=2500,torch_version=torch.__version__)
            dest=OUT/'recoveries'/f"{name}_{condition['replicate']}"
            seed_everything(seed)
            print(f"Starting {name} replicate {condition['replicate']}",flush=True)
            jwrite(OUT/'pilot_status.json',dict(status='running',active_condition=name,active_replicate=condition['replicate'],completed_recoveries=len(observations),full_expansion=False))
            record=trainer.run_stage(cfg,Path(c['rewind_source']),mask,train,val,reference,device,dest,float(c['dense_comparison_ce']),identity)
            row=dict(configuration=c['configuration_id'],num_experts=c['num_experts'],training_seed=seed,sparsity=c['sparsity'],condition=name,replicate=condition['replicate'],mask_seed=condition['mask_seed'],mask_path=condition['mask_path'],mask_hash=condition['mask_hash'],rewind_checkpoint=c['rewind_source'],source_imp_ce=c['original_recovery_ce'],initial_ce=record['initial_validation_loss'],final_ce=record['final_validation_loss'],train_ce=record['final_training_ce'],attempted_steps=record['optimizer_step_attempts'],applied_steps=record['optimizer_steps_applied'],duration_seconds=record['duration_seconds'],gpu_hours=record['duration_seconds']/3600,curves_path=str(dest/'metrics.jsonl'),stage_metadata=str(dest/'stage.json'))
            observations.append(row);write_csv(OUT/'pilot_results.csv',observations)
            if name=='IMP':
                gap=abs(record['final_validation_loss']-float(c['original_recovery_ce']))
                jwrite(OUT/'reproduction_check.json',dict(absolute_ce_discrepancy=gap,tolerance=.005,passed=math.isfinite(gap) and gap<=.005,metadata=str(dest/'stage.json')))
                if not math.isfinite(gap) or gap>.005: raise RuntimeError('IMP reproduction failed; null comparisons prohibited')
            from analyze import analyze
            analyze()
        jwrite(OUT/'pilot_status.json',dict(status='complete',recoveries=len(observations),full_expansion=False))
    except Exception as error:
        jwrite(OUT/'pilot_status.json',dict(status='stopped',error=str(error),completed_recoveries=len(observations)))
        from analyze import analyze
        analyze()
        raise


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--execute',action='store_true');args=parser.parse_args()
    OUT.mkdir(exist_ok=True)
    with (OUT/'pilot.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        plan=prepare()
        if args.execute: execute(plan)

if __name__=='__main__':main()
