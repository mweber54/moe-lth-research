#!/usr/bin/env python3
"""Detached gate replay on the established 18-pair grid, using existing donors."""
from pathlib import Path
import gc,json,shutil,tempfile,traceback
from datetime import datetime,timezone
import torch
from moe_lth.config import load_config
from moe_lth.data import build_dataloaders
from moe_lth.experiments import run_router_age_recovery as r
from moe_lth.experiments import optimization_geometry_diagnostics as d
from moe_lth.experiments.fixed_schedule_intervention import Schedule,file_hash
from moe_lth.experiments.gate_amplitude_replay import GateSchedule
from moe_lth.experiments.amplitude_pattern_replay import PatternSchedule
from moe_lth.pruning.magnitude_prune import expert_local_magnitude_masks
from moe_lth.pruning.router_age import state_dict_hash,selected_experts_per_batch
from moe_lth.utils import configure_device,resolve_data_seed,seed_everything

ROOT=Path('results/gate_amplitude_pattern_decomposition')
LEVEL2=Path('results/gate_amplitude_replay_intervention')
LEVEL1=Path('results/cross_arch_optimization_geometry/fixed_schedule_intervention')
PRIOR=Path('results/cross_arch_optimization_geometry/diagnostic_rerun')

def group(s,k):return {n:v for n,v in s.items() if r.parameter_group(n)==k}
def write_manifest(m):(ROOT/'decomposition_run_manifest.json').write_text(json.dumps(m,indent=2)+'\n')

def main():
    ROOT.mkdir(parents=True,exist_ok=True)
    if (ROOT/'decomposition_run_manifest.json').exists():raise FileExistsError('Existing decomposition; refusing overwrite')
    capture=json.loads((ROOT/'capture_run_manifest.json').read_text());assert capture['status']=='all_sources_validated' and capture['validated_sources']==18
    transforms=json.loads((ROOT/'amplitude_transform_manifest.json').read_text());assert transforms['status']=='all_transforms_validated' and transforms['sources']==18
    checks=json.loads((ROOT/'compiled_forward_integrity.json').read_text());assert checks['status']=='passed' and checks['sources']==18 and checks['conditions']==7
    if shutil.disk_usage(ROOT).free<750*1024**3:raise RuntimeError('Need 750 GiB stable storage for the full 252-recovery retained grid')
    rows=capture['conditions'];bulk=ROOT.resolve();conditions=transforms['conditions'];assert len(conditions)==7
    manifest={'status':'preflight','created_utc':datetime.now(timezone.utc).isoformat(),'conditions':rows,'amplitude_conditions':conditions,'required_sources':18,'validated_sources':18,'completed_recipient_recoveries':0,'completed_pairs':0,'required_recipient_recoveries':252,'required_pairs':126,'source':'canonical age/seed dense amplitude source tensor, identical transformed schedule for both model kinds','primary_readout':'primary_r100_common_readout','secondary_readouts':['secondary_condition_matched_readout','secondary_native_readout'],'retention_tolerance':1e-5,'execution':'sequential','code_hashes':{}}
    write_manifest(manifest);(ROOT/'decomposition_provenance').mkdir()
    for name in [__file__,r.__file__,'src/moe_lth/experiments/amplitude_pattern_replay.py','scripts/build_amplitude_pattern_schedules.py','scripts/analyze_amplitude_pattern_decomposition.py']:
        manifest['code_hashes'][name]=file_hash(name);shutil.copyfile(name,ROOT/'decomposition_provenance'/Path(name).name)
    write_manifest(manifest)
    device=torch.device('cuda');configure_device(device)
    data={};data_checks=[]
    for row in rows:
        cfg=load_config(row['config']);s=row['seed'];sm=json.loads((Path(row['sparse'])/'metadata.json').read_text())
        if s not in data:
            seed_everything(s);a,b=build_dataloaders(cfg['data'],int(cfg['training']['batch_size']),resolve_data_seed(cfg))
            train=r._materialize_batches(a,2500);val=r._materialize_validation_batches(b,32)
            data[s]=(train,val,r._batch_sequence_hash(train),r._batch_sequence_hash(val))
        train,val,th,vh=data[s]
        assert th==sm['training_batch_sequence_hash'] and vh==sm['validation_batch_sequence_hash']
        data_checks.append(dict(experts=row['experts'],seed=s,age=row['age'],training_hash=th,validation_hash=vh))
    (ROOT/'data_identity_audit.json').write_text(json.dumps(data_checks,indent=2)+'\n')
    for amplitude_condition in conditions:
        for e in [8,12]:
            for seed in [7,17,29]:
                if shutil.disk_usage(bulk).free<5*1024**3:raise RuntimeError('Bulk storage nearly full; preserving completed results')
                selected=[x for x in rows if x['experts']==e and x['seed']==seed];donor_row=next(x for x in selected if x['age']==100)
                cfg=load_config(donor_row['config']);train,val,th,vh=data[seed];cal=r._calibration_batches(val)
                dm=json.loads((Path(donor_row['dense'])/'metadata.json').read_text());cp=Path(dm['router_checkpoint']);ref=cp.parent
                init=r.load_model_from_checkpoint(cfg['model'],str(ref/'step_0.pt'),torch.device('cpu'));initial={k:v.detach().cpu().clone() for k,v in init.state_dict().items()};del init
                final=r.load_model_from_checkpoint(cfg['model'],str(ref/'step_2500.pt'),device);masks=expert_local_magnitude_masks(final,.85)
                ticket={k:v*masks[k].cpu() if k in masks else v for k,v in initial.items()};reference=selected_experts_per_batch(final,cal,device);del final
                for row in selected:
                    for kind,base in [('dense',initial),('sparse',ticket)]:
                        meta=json.loads((Path(row[kind])/'metadata.json').read_text())
                        assert state_dict_hash(group(base,'expert'))==meta['expert_state_hash'] and state_dict_hash(group(base,'shared'))==meta['shared_state_hash']
                        if kind=='sparse':assert r._mask_hash(masks)==meta['mask_hash']
                        router=r.load_model_from_checkpoint(cfg['model'],meta['router_checkpoint'],torch.device('cpu'))
                        assert state_dict_hash(group(router.state_dict(),'router'))==meta['initial_router_state_hash'];del router
                gc.collect();torch.cuda.empty_cache()
                schedule_path=LEVEL2/f'schedules/e{e}/seed{seed}'
                def run_one(row,kind,schedule,donor=False):
                    age=row['age'];meta=json.loads((Path(row[kind])/'metadata.json').read_text());base=initial if kind=='dense' else ticket;maskset={} if kind=='dense' else masks
                    phase=amplitude_condition;prefix=bulk/f'snapshots/{amplitude_condition}/e{e}/seed{seed}/R{age}/{kind}';prefix.mkdir(parents=True)
                    output=bulk/(f'donors/e{e}/seed{seed}' if donor else f'recovery/{amplitude_condition}/e{e}/seed{seed}/R{age}/{kind}')
                    manifest.update(status='running',current_condition={'amplitude_condition':amplitude_condition,'experts':e,'seed':seed,'age':age,'kind':kind,'phase':phase});write_manifest(manifest)
                    def snapshot(model,opt,scaler,step):
                        if step not in d.STEPS:return
                        rng=d.rng_state();mode=[q.training for q in model.modules()];mh=d.digest(model.state_dict());oh=d.digest(opt.state_dict());sh=d.digest(scaler.state_dict())
                        p=prefix/f'step_{step:04d}.pt';assert not p.exists()
                        torch.save({'model':model.state_dict(),'optimizer':opt.state_dict(),'grad_scaler':scaler.state_dict(),'scheduler':None,'lr':[g['lr'] for g in opt.param_groups],'step':step,'rng':rng,'config':cfg,'seed':seed,'mask':masks,'mask_applied':kind=='sparse','historical_metadata':meta,'data_sequence_identity':{'training':th,'validation':vh,'next_batch_index':step},'route_schedule':str(schedule_path),'schedule_mode':schedule.mode,'parameter_hash':mh,'optimizer_hash':oh},p)
                        d.append(ROOT/'snapshot_manifest.csv',[dict(amplitude_condition=amplitude_condition,experts=e,seed=seed,age=age,kind=kind,phase=phase,step=step,path=str(p),sha256=file_hash(p),model_hash=mh,optimizer_hash=oh)])
                        if not donor:
                            native=r._evaluate_condition_loss(model,val,device,False,1.)
                            d.append(ROOT/'native_evaluation_curves.csv',[dict(amplitude_condition=amplitude_condition,experts=e,seed=seed,age=age,kind=kind,step=step,loss=native)])
                        for q,training in zip(model.modules(),mode):q.training=training
                        d.restore_rng(rng)
                        assert d.digest(model.state_dict())==mh and d.digest(opt.state_dict())==oh and d.digest(scaler.state_dict())==sh and d.digest(d.rng_state())==d.digest(rng)
                        d.append(ROOT/'mutation_audit.csv',[dict(amplitude_condition=amplitude_condition,experts=e,seed=seed,age=age,kind=kind,phase=phase,step=step,model_unchanged=True,optimizer_unchanged=True,scaler_unchanged=True,rng_restored=True)])
                        print(f'[snapshot] {phase} e{e} seed{seed} R{age} {kind} step{step}: mutation audit passed',flush=True)
                    dense_loss=json.loads((Path(row['dense'])/'summary.json').read_text())['final_validation_loss']
                    record=r._run_recovery_condition(config=cfg,condition_name=f'pattern_{amplitude_condition}_e{e}_seed{seed}_R{age}_{kind}',pruned_base_state=base,router_checkpoint=meta['router_checkpoint'],router_age_percent=age,router_step=meta['router_step'],masks=maskset,expert_hash=meta['expert_state_hash'],shared_hash=meta['shared_state_hash'],mask_hash=meta['mask_hash'],reference_selected=reference,calibration_batches=cal,train_batches=train,validation_batches=val,train_batch_hash=th,validation_batch_hash=vh,device=device,recovery_steps=2500,dense_loss=dense_loss,output_dir=output,confidence_control=False,target_confidence=None,seed=seed,sparsity=.85 if kind=='sparse' else 0.,router_mode='trainable',state_snapshot_callback=snapshot,training_forward_callback=schedule.train,validation_forward_callback=primary.validation)
                    # Historical helper's recovery-fraction denominator is not an intervention endpoint.
                    # Only explicitly matched outcomes below are used for causal comparisons.
                    return output,record
                for row in sorted(selected,key=lambda x:(x['age']!=100,x['age'])):
                    age=row['age'];endpoints={}
                    for kind in ['dense','sparse']:
                        transform_path=ROOT/f'transformed/e{e}/seed{seed}/R{age}'
                        replay=PatternSchedule(schedule_path,transform_path,amplitude_condition);primary=GateSchedule(schedule_path)
                        output,record=run_one(row,kind,replay);audit=replay.verify_training()
                        assert primary.counts['validation']==primary.manifest['counts']['validation']
                        assert primary.rolling['validation'].hexdigest()==primary.manifest['sequence_hashes']['validation']
                        d.append(ROOT/'replay_audit.csv',[dict(amplitude_condition=amplitude_condition,experts=e,seed=seed,age=age,kind=kind,**audit,primary_validation_occurrences=primary.counts['validation'],primary_validation_hash=primary.rolling['validation'].hexdigest(),optimizer_steps_applied=record['optimizer_steps_applied'],amp_overflow_steps=record['amp_overflow_step_count'])])
                        model=r.load_model_from_checkpoint(cfg['model'],str(output/'checkpoints/final_recovered.pt'),device);before=d.digest(model.state_dict())
                        endpoints[kind]={'primary_r100_common_readout':record['final_validation_loss']}
                        endpoints[kind]['secondary_native_readout']=r._evaluate_condition_loss(model,val,device,False,1.)
                        secondary=PatternSchedule(schedule_path,transform_path,amplitude_condition)
                        endpoints[kind]['secondary_condition_matched_readout']=r._evaluate_condition_loss(model,val,device,False,1.,secondary.validation,2500)
                        assert before==d.digest(model.state_dict());del model
                        manifest['completed_recipient_recoveries']+=1;write_manifest(manifest);gc.collect();torch.cuda.empty_cache()
                    for readout in ['primary_r100_common_readout','secondary_condition_matched_readout','secondary_native_readout']:
                        d.append(ROOT/'outcomes.csv',[dict(experts=e,seed=seed,age=age,amplitude_condition=amplitude_condition,sparse_final_loss=endpoints['sparse'][readout],dense_final_loss=endpoints['dense'][readout],ticket_gap=endpoints['sparse'][readout]-endpoints['dense'][readout],primary_readout=readout)])
                    manifest['completed_pairs']+=1;write_manifest(manifest)
                    print(f"[pattern pair complete] {manifest['completed_pairs']}/126 {amplitude_condition} e{e} seed{seed} R{age}",flush=True)
    manifest.update(status='recoveries_complete',current_condition=None);write_manifest(manifest)
    print('All 252 decomposition recipients completed. Analysis follows.',flush=True)
    import importlib.util
    spec=importlib.util.spec_from_file_location('pattern_analysis','scripts/analyze_amplitude_pattern_decomposition.py');module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module);module.main()

if __name__=='__main__':
    try:main()
    except Exception:
        if ROOT.exists():
            (ROOT/'failure.txt').write_text(traceback.format_exc())
            if (ROOT/'decomposition_run_manifest.json').exists():
                m=json.loads((ROOT/'decomposition_run_manifest.json').read_text());m['status']='failed';write_manifest(m)
        raise
