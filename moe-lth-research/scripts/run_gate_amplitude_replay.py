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
from moe_lth.pruning.magnitude_prune import expert_local_magnitude_masks
from moe_lth.pruning.router_age import state_dict_hash,selected_experts_per_batch
from moe_lth.utils import configure_device,resolve_data_seed,seed_everything

ROOT=Path('results/gate_amplitude_replay_intervention')
LEVEL1=Path('results/cross_arch_optimization_geometry/fixed_schedule_intervention')
PRIOR=Path('results/cross_arch_optimization_geometry/diagnostic_rerun')

def group(s,k):return {n:v for n,v in s.items() if r.parameter_group(n)==k}
def write_manifest(m):(ROOT/'gate_replay_manifest.json').write_text(json.dumps(m,indent=2)+'\n')

def main():
    ROOT.mkdir(parents=True,exist_ok=True)
    if (ROOT/'gate_replay_manifest.json').exists():raise FileExistsError('Existing gate experiment; refusing overwrite')
    if shutil.disk_usage(ROOT).free<130*1024**3:raise RuntimeError('Need 130 GiB stable storage')
    prior=json.loads((PRIOR/'run_manifest.json').read_text());assert prior['status']=='complete';rows=prior['conditions'];assert len(rows)==18
    import pandas as pd
    audit=pd.read_csv(ROOT/'donor_amplitude_manifest.csv');assert len(audit)==6 and audit.all_files_verified.all()
    controls=pd.read_csv(ROOT/'donor_same_state_forward_checks.csv');assert len(controls)==6 and controls[['logits_exact','assignments_exact','accepted_tokens_exact','amplitudes_exact']].all().all()
    controls.to_csv(ROOT/'donor_self_replay.csv',index=False)
    bulk=ROOT.resolve()
    manifest={'status':'preparing_stable_storage','created_utc':datetime.now(timezone.utc).isoformat(),'conditions':rows,'completed_pairs':0,'completed_recipient_recoveries':0,'recipient_recoveries':36,'new_donor_recoveries':0,'reused_donors':6,'sparsity':.85,'steps':2500,'snapshot_steps':d.STEPS,'validation_steps':list(range(0,2501,50)),'bulk_storage':str(bulk),'self_replay_criterion':'User approved exact same-state forward/amplitude/acceptance and gradient checks, not donor final-model identity under detached gates. Final trajectory divergence is reported separately.','gradient_semantics':'Detached gates remove task gradient through router amplitudes; historical native-probability auxiliary loss remains active into router and upstream states. No surrogate gradients.','primary_readout':'Same donor assignments AND detached donor gates for all three training levels','primary_interaction':'target/control contrast level2 minus level1 at common assignment+gate readout','total_interaction':'target/control contrast level2 minus native at common assignment+gate readout','attenuation_epsilon':1e-12,'classification_rule':'Descriptive: substantial attenuation means >=50 percent; consistent means all three seeds per contrast. Strong support requires collapse or reversal of favorable level1 contrasts in both architectures; consistent lesser attenuation is partial; mixed seeds are seed-conditioned; otherwise robust ordering is insufficient. Evaluate all contrasts, including unfavorable or near-zero baseline cases.','trajectory_policy':'Level2 common gate readout at all 51 validations. Existing level0/1 curves retain their original readout and are explicitly labeled. Common gate reevaluation of saved level0/1 states is available at nine snapshots only; no interpolation is called observed evidence.','execution':'sequential','code_hashes':{}}
    write_manifest(manifest)
    # Copy only existing schedules to durable storage; never recapture donors.
    shutil.copytree(LEVEL1/'schedules',ROOT/'schedules')
    (ROOT/'provenance').mkdir()
    for name in [__file__,r.__file__,'src/moe_lth/experiments/gate_amplitude_replay.py','src/moe_lth/experiments/fixed_schedule_intervention.py','scripts/analyze_gate_amplitude_replay.py','scripts/evaluate_gate_replay_snapshots.py','tests/test_gate_replay_gradients.py']:
        manifest['code_hashes'][name]=file_hash(name);shutil.copyfile(name,ROOT/'provenance'/Path(name).name)
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
            schedule_path=bulk/f'schedules/e{e}/seed{seed}'
            def run_one(row,kind,schedule,donor=False):
                age=row['age'];meta=json.loads((Path(row[kind])/'metadata.json').read_text());base=initial if kind=='dense' else ticket;maskset={} if kind=='dense' else masks
                phase='gate';prefix=bulk/f'snapshots/{phase}/e{e}/seed{seed}/R{age}/{kind}';prefix.mkdir(parents=True)
                output=bulk/(f'donors/e{e}/seed{seed}' if donor else f'recovery/e{e}/seed{seed}/R{age}/{kind}')
                manifest.update(status='running',current_condition={'experts':e,'seed':seed,'age':age,'kind':kind,'phase':phase});write_manifest(manifest)
                def snapshot(model,opt,scaler,step):
                    if step not in d.STEPS:return
                    rng=d.rng_state();mode=[q.training for q in model.modules()];mh=d.digest(model.state_dict());oh=d.digest(opt.state_dict());sh=d.digest(scaler.state_dict())
                    p=prefix/f'step_{step:04d}.pt';assert not p.exists()
                    torch.save({'model':model.state_dict(),'optimizer':opt.state_dict(),'grad_scaler':scaler.state_dict(),'scheduler':None,'lr':[g['lr'] for g in opt.param_groups],'step':step,'rng':rng,'config':cfg,'seed':seed,'mask':masks,'mask_applied':kind=='sparse','historical_metadata':meta,'data_sequence_identity':{'training':th,'validation':vh,'next_batch_index':step},'route_schedule':str(schedule_path),'schedule_mode':schedule.mode,'parameter_hash':mh,'optimizer_hash':oh},p)
                    d.append(ROOT/'snapshot_manifest.csv',[dict(experts=e,seed=seed,age=age,kind=kind,phase=phase,step=step,path=str(p),sha256=file_hash(p),model_hash=mh,optimizer_hash=oh)])
                    if not donor:
                        native=r._evaluate_condition_loss(model,val,device,False,1.)
                        d.append(ROOT/'native_evaluation_curves.csv',[dict(experts=e,seed=seed,age=age,kind=kind,step=step,loss=native)])
                    for q,training in zip(model.modules(),mode):q.training=training
                    d.restore_rng(rng)
                    assert d.digest(model.state_dict())==mh and d.digest(opt.state_dict())==oh and d.digest(scaler.state_dict())==sh and d.digest(d.rng_state())==d.digest(rng)
                    d.append(ROOT/'mutation_audit.csv',[dict(experts=e,seed=seed,age=age,kind=kind,phase=phase,step=step,model_unchanged=True,optimizer_unchanged=True,scaler_unchanged=True,rng_restored=True)])
                    print(f'[snapshot] {phase} e{e} seed{seed} R{age} {kind} step{step}: mutation audit passed',flush=True)
                dense_loss=json.loads((Path(row['dense'])/'summary.json').read_text())['final_validation_loss']
                record=r._run_recovery_condition(config=cfg,condition_name=f'gate_replay_{phase}_e{e}_seed{seed}_R{age}_{kind}',pruned_base_state=base,router_checkpoint=meta['router_checkpoint'],router_age_percent=age,router_step=meta['router_step'],masks=maskset,expert_hash=meta['expert_state_hash'],shared_hash=meta['shared_state_hash'],mask_hash=meta['mask_hash'],reference_selected=reference,calibration_batches=cal,train_batches=train,validation_batches=val,train_batch_hash=th,validation_batch_hash=vh,device=device,recovery_steps=2500,dense_loss=dense_loss,output_dir=output,confidence_control=False,target_confidence=None,seed=seed,sparsity=.85 if kind=='sparse' else 0.,router_mode='trainable',state_snapshot_callback=snapshot,training_forward_callback=schedule.train,validation_forward_callback=schedule.validation)
                # Historical helper's recovery-fraction denominator is not an intervention endpoint.
                # Only explicitly matched outcomes below are used for causal comparisons.
                return output,record
            # Full archive integrity already passed; stable copy is checked again on every replay.
            donor_model=torch.load(LEVEL1/f'donors/e{e}/seed{seed}/checkpoints/final_recovered.pt',map_location='cpu',weights_only=False)['model'];donor_hash=d.digest(donor_model);del donor_model
            for row in sorted(selected,key=lambda x:(x['age']!=100,x['age'])):
                age=row['age'];endpoints={}
                for kind in ['dense','sparse']:
                    replay=GateSchedule(schedule_path,ROOT/'gate_amplitude_statistics.csv',dict(experts=e,seed=seed,age=age,kind=kind));output,record=run_one(row,kind,replay)
                    audit=replay.verify_complete_replay()
                    d.append(ROOT/'replay_audit.csv',[dict(experts=e,seed=seed,age=age,kind=kind,training_forwards=audit['counts']['train'],validation_forwards=audit['counts']['validation'],training_schedule_hash=audit['sequence_hashes']['train'],validation_schedule_hash=audit['sequence_hashes']['validation'],all_assignments_and_accepted_tokens_identical=True,amplitude_schedule_hash=audit['amplitude_schedule_hash'],maximum_absolute_amplitude_error=audit['maximum_absolute_amplitude_error'],detached_amplitudes=True,optimizer_steps_applied=record['optimizer_steps_applied'],amp_overflow_steps=record['amp_overflow_step_count'])])
                    if age==100 and kind=='dense':
                        own=torch.load(output/'checkpoints/final_recovered.pt',map_location='cpu',weights_only=False)['model']
                        d.append(ROOT/'donor_training_divergence.csv',[dict(experts=e,seed=seed,final_model_identical_to_native_donor=d.digest(own)==donor_hash,final_gate_readout_loss=record['final_validation_loss'],interpretation='Expected possible divergence: detached gates change backward pass')]);del own
                    endpoints[kind]={}
                    for level,checkpoint in [('gate',output/'checkpoints/final_recovered.pt'),('assign',LEVEL1/f'recovery/e{e}/seed{seed}/R{age}/{kind}/checkpoints/final_recovered.pt'),('native',Path(row[kind])/'checkpoints/final_recovered.pt')]:
                        model=r.load_model_from_checkpoint(cfg['model'],str(checkpoint),device);before=d.digest(model.state_dict())
                        for readout in ['fixedgate','native','assign']:
                            reader=GateSchedule(schedule_path) if readout=='fixedgate' else Schedule(schedule_path,'replay') if readout=='assign' else None
                            value=r._evaluate_condition_loss(model,val,device,False,1.,reader.validation if reader else None,2500)
                            endpoints[kind][level+'_'+readout]=value
                        assert before==d.digest(model.state_dict());del model
                    manifest['completed_recipient_recoveries']+=1;write_manifest(manifest);gc.collect();torch.cuda.empty_cache()
                outcome={'experts':e,'seed':seed,'age':age}
                for policy in [level+'_'+readout for level in ['native','assign','gate'] for readout in ['fixedgate','native','assign']]:
                    for kind in ['sparse','dense']:outcome[f'{policy}_{kind}_loss']=endpoints[kind][policy]
                    outcome[f'{policy}_ticket_gap']=endpoints['sparse'][policy]-endpoints['dense'][policy]
                d.append(ROOT/'outcomes.csv',[outcome]);manifest['completed_pairs']+=1;write_manifest(manifest)
                print(f"[pair complete] {manifest['completed_pairs']}/18 e{e} seed{seed} R{age}",flush=True)
    manifest.update(status='recoveries_complete',current_condition=None);write_manifest(manifest)
    print('All 36 gate recipients completed using six existing donors. Analysis follows.',flush=True)
    import importlib.util
    spec=importlib.util.spec_from_file_location('gate_analysis','scripts/analyze_gate_amplitude_replay.py');module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module);module.main()

if __name__=='__main__':
    try:main()
    except Exception:
        if ROOT.exists():
            (ROOT/'failure.txt').write_text(traceback.format_exc())
            if (ROOT/'gate_replay_manifest.json').exists():
                m=json.loads((ROOT/'gate_replay_manifest.json').read_text());m['status']='failed';write_manifest(m)
        raise
