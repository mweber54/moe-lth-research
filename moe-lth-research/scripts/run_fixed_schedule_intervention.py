#!/usr/bin/env python3
"""One R100-dense trajectory replay intervention on the established 18-pair grid."""
from pathlib import Path
import gc,json,shutil,tempfile,traceback
from datetime import datetime,timezone
import torch
from moe_lth.config import load_config
from moe_lth.data import build_dataloaders
from moe_lth.experiments import run_router_age_recovery as r
from moe_lth.experiments import optimization_geometry_diagnostics as d
from moe_lth.experiments.fixed_schedule_intervention import Schedule,file_hash
from moe_lth.pruning.magnitude_prune import expert_local_magnitude_masks
from moe_lth.pruning.router_age import state_dict_hash,selected_experts_per_batch
from moe_lth.utils import configure_device,resolve_data_seed,seed_everything

ROOT=Path('results/cross_arch_optimization_geometry/fixed_schedule_intervention')
PRIOR=Path('results/cross_arch_optimization_geometry/diagnostic_rerun')

def group(s,k):return {n:v for n,v in s.items() if r.parameter_group(n)==k}
def write_manifest(m):(ROOT/'run_manifest.json').write_text(json.dumps(m,indent=2)+'\n')

def main():
    if ROOT.exists():raise FileExistsError(f'Refusing to overwrite {ROOT}')
    if shutil.disk_usage('/tmp').free<120*1024**3:raise RuntimeError('Need at least 120 GiB free for complete retained state and schedules')
    ROOT.mkdir(parents=True);bulk=Path(tempfile.mkdtemp(prefix='mweber54_fixed_schedule_',dir='/tmp'))
    for name in ['schedules','donors','recovery','snapshots']:
        (bulk/name).mkdir();(ROOT/name).symlink_to(bulk/name,target_is_directory=True)
    prior=json.loads((PRIOR/'run_manifest.json').read_text());assert prior['status']=='complete';rows=prior['conditions'];assert len(rows)==18
    manifest={'status':'preflight','created_utc':datetime.now(timezone.utc).isoformat(),'conditions':rows,'sparsity':.85,'recovery_steps':2500,'runtime_seeds':[7,17,29],'donor':'Matched R100 dense native recovery trajectory, one per architecture/seed','donor_recoveries':6,'recipient_pairs':18,'recipient_recoveries':36,'total_new_recoveries':42,'bulk_storage':str(bulk),'bulk_storage_note':'Local /tmp storage, required because historical-results filesystem had only 3.5 GB free. No automatic deletion; preserve this directory with the report.','primary_estimand':'Change in within-seed router-age ticket-gap contrasts between native-trained and schedule-clamped-trained models, with BOTH arms evaluated under the SAME donor validation schedule at step 2500. Ticket gap is sparse minus matched dense validation loss.','primary_contrasts':{'8':'R20 minus R0 and R100','12':'R60 minus R20 and R100'},'primary_interaction_formula':'(gap_clamped(target)-gap_clamped(control))-(gap_native_trained(target)-gap_native_trained(control)), all evaluated with common replayed validation routes','secondary_endpoints':['Same training-arm comparison under native evaluation','Evaluation-only ordering changes in unchanged native-trained checkpoints','Seed-level winners and full rankings','Recovery curves','AMP applied/skipped optimizer-step counts'],'interpretation':'Replay fixes endogenous selection to one common evolving donor sequence; it does not make assignments constant over time. IDs and capacity decisions fixed, own differentiable gate amplitudes and trainable routers retained. Donor-specific controlled intervention, not a universal mediation estimate.','statistical_unit':'Seed, n=3 per architecture; means, sample SD and all paired seed contrasts; no expert/token pseudoreplication. No significance claim.','preregistered_descriptive_rule':'Report every contrast and rank. Consistent pairwise ordering change requires a target/control ordering flip in all three seeds under common fixed evaluation. Interaction signs and magnitudes are reported separately. Evaluation-only changes must not be attributed to training. Mixed signs are seed-conditioned; no change in rankings is reported as preserved ordering.','snapshot_steps':d.STEPS,'validation_steps':list(range(0,2501,50)),'matched_baseline':'Existing exactly reproduced native recoveries; reevaluate checkpoints, never retrain or modify baselines.','native_evaluation_precision':'Historical fp32 validation; training remains fp16 with original GradScaler.','completed_donors':0,'completed_pairs':0,'completed_recipient_recoveries':0,'execution':'sequential','unit_tests':'Native/capture/self-replay bit identity; capacity and assignment checks; live router gradients; mismatched batch rejection; prior instrumentation regression passed','code_hashes':{str(p):file_hash(p) for p in [Path(__file__),Path(r.__file__),Path('src/moe_lth/experiments/fixed_schedule_intervention.py'),Path('scripts/analyze_fixed_schedule_intervention.py')]}}
    write_manifest(manifest);(ROOT/'provenance').mkdir()
    for name in [__file__,r.__file__,'src/moe_lth/experiments/fixed_schedule_intervention.py','tests/test_fixed_schedule_intervention.py','scripts/analyze_fixed_schedule_intervention.py']:
        shutil.copyfile(name,ROOT/'provenance'/Path(name).name)
    (ROOT/'protocol_audit.md').write_text('# Fixed recovery-trajectory schedule intervention\n\nSix native R100 dense recoveries capture every actual training forward and every historical validation forward. All recipient router ages, and both sparse/dense models, replay the corresponding architecture/seed schedule. Replayed IDs, capacity priorities, and accepted masks are checked against the donor on every forward; complete chronological archive-hash sequences must match. Recipient gate amplitudes remain differentiable and router parameters trainable.\n\nPrimary endpoint: both existing native-trained and new clamped-trained models are evaluated using the identical donor validation routes. Native evaluation of both arms is secondary, separating evaluation-policy changes from training effects. No extra native baseline recoveries are run beyond the six necessary schedule donors. Donor capture must reproduce history, and R100 dense self-replay must reproduce the donor exactly before other ages run.\n\nThe previous grid, E0/shared/router hashes, ET-derived 85% masks, batch sequences, seeds, fresh AdamW hyperparameters, no LR schedule, fp16 training, clipping, GradScaler, masking, and 2500-step budget are retained. Nine recoverable snapshots are saved. Native routing-stat probes are non-mutating measurements of router proposals; they are not used for parameter updates. Actual update routing is defined by the replay audit.\n\nBulk data are stored through symlinks to the dedicated local /tmp directory recorded in run_manifest.json because the historical-results filesystem is nearly full. Historical artifacts are never changed.\n')
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
                phase='donor' if donor else 'clamped';prefix=bulk/f'snapshots/{phase}/e{e}/seed{seed}/R{age}/{kind}';prefix.mkdir(parents=True)
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
                    d.append(ROOT/'snapshot_mutation_audit.csv',[dict(experts=e,seed=seed,age=age,kind=kind,phase=phase,step=step,model_unchanged=True,optimizer_unchanged=True,scaler_unchanged=True,rng_restored=True)])
                    print(f'[snapshot] {phase} e{e} seed{seed} R{age} {kind} step{step}: mutation audit passed',flush=True)
                dense_loss=json.loads((Path(row['dense'])/'summary.json').read_text())['final_validation_loss']
                record=r._run_recovery_condition(config=cfg,condition_name=f'fixed_schedule_{phase}_e{e}_seed{seed}_R{age}_{kind}',pruned_base_state=base,router_checkpoint=meta['router_checkpoint'],router_age_percent=age,router_step=meta['router_step'],masks=maskset,expert_hash=meta['expert_state_hash'],shared_hash=meta['shared_state_hash'],mask_hash=meta['mask_hash'],reference_selected=reference,calibration_batches=cal,train_batches=train,validation_batches=val,train_batch_hash=th,validation_batch_hash=vh,device=device,recovery_steps=2500,dense_loss=dense_loss,output_dir=output,confidence_control=False,target_confidence=None,seed=seed,sparsity=.85 if kind=='sparse' else 0.,router_mode='trainable',state_snapshot_callback=snapshot,training_forward_callback=schedule.train,validation_forward_callback=schedule.validation)
                # Historical helper's recovery-fraction denominator is not an intervention endpoint.
                # Only explicitly matched outcomes below are used for causal comparisons.
                return output,record
            capture=Schedule(schedule_path,'capture');donor_out,donor_record=run_one(donor_row,'dense',capture,True)
            assert capture.counts=={'train':2500,'validation':51*len(val)}
            cap=capture.finish_capture(dict(experts=e,seed=seed,donor_age=100,donor_kind='dense',training_hash=th,validation_hash=vh,training_precision=cfg['training']['precision'],validation_precision='fp32'))
            old=torch.load(Path(donor_row['dense'])/'checkpoints/final_recovered.pt',map_location='cpu',weights_only=False)['model'];new=torch.load(donor_out/'checkpoints/final_recovered.pt',map_location='cpu',weights_only=False)['model'];donor_hash=d.digest(new)
            assert d.digest(old)==donor_hash,'Donor capture failed exact historical reproduction';del old,new
            d.append(ROOT/'donor_reproduction.csv',[dict(experts=e,seed=seed,bit_identical_to_history=True,donor_model_hash=donor_hash,training_schedule_hash=cap['sequence_hashes']['train'],validation_schedule_hash=cap['sequence_hashes']['validation'])]);manifest['completed_donors']+=1;write_manifest(manifest)
            # R100 dense is first: it is an exact self-replay positive control.
            for row in sorted(selected,key=lambda x:(x['age']!=100,x['age'])):
                age=row['age'];endpoints={}
                for kind in ['dense','sparse']:
                    replay=Schedule(schedule_path,'replay');output,record=run_one(row,kind,replay)
                    audit=replay.verify_complete_replay()
                    d.append(ROOT/'replay_audit.csv',[dict(experts=e,seed=seed,age=age,kind=kind,training_forwards=audit['counts']['train'],validation_forwards=audit['counts']['validation'],training_schedule_hash=audit['sequence_hashes']['train'],validation_schedule_hash=audit['sequence_hashes']['validation'],all_assignments_and_accepted_tokens_identical=True,optimizer_steps_applied=record['optimizer_steps_applied'],amp_overflow_steps=record['amp_overflow_step_count'])])
                    if age==100 and kind=='dense':
                        own=torch.load(output/'checkpoints/final_recovered.pt',map_location='cpu',weights_only=False)['model'];assert d.digest(own)==donor_hash,'R100 dense self-replay changed training';del own
                        d.append(ROOT/'self_replay_control.csv',[dict(experts=e,seed=seed,bit_identical_to_donor=True)])
                    checkpoint=output/'checkpoints/final_recovered.pt';model=r.load_model_from_checkpoint(cfg['model'],str(checkpoint),device)
                    fixed=record['final_validation_loss'];native=r._evaluate_condition_loss(model,val,device,False,1.);del model
                    old_model=r.load_model_from_checkpoint(cfg['model'],str(Path(row[kind])/'checkpoints/final_recovered.pt'),device)
                    old_hash=d.digest(old_model.state_dict());baseline_native=r._evaluate_condition_loss(old_model,val,device,False,1.)
                    expected=json.loads((Path(row[kind])/'summary.json').read_text())['final_validation_loss'];assert abs(baseline_native-expected)<1e-8,'Baseline evaluation mismatch'
                    eval_reader=Schedule(schedule_path,'replay');baseline_fixed=r._evaluate_condition_loss(old_model,val,device,False,1.,eval_reader.validation,2500)
                    assert d.digest(old_model.state_dict())==old_hash;del old_model
                    endpoints[kind]=dict(clamped_fixed=fixed,clamped_native=native,baseline_fixed=baseline_fixed,baseline_native=baseline_native)
                    manifest['completed_recipient_recoveries']+=1;write_manifest(manifest);gc.collect();torch.cuda.empty_cache()
                outcome={'experts':e,'seed':seed,'age':age}
                for policy in ['baseline_native','baseline_fixed','clamped_native','clamped_fixed']:
                    for kind in ['sparse','dense']:outcome[f'{policy}_{kind}_loss']=endpoints[kind][policy]
                    outcome[f'{policy}_ticket_gap']=endpoints['sparse'][policy]-endpoints['dense'][policy]
                d.append(ROOT/'outcomes.csv',[outcome]);manifest['completed_pairs']+=1;write_manifest(manifest)
                print(f"[pair complete] {manifest['completed_pairs']}/18 e{e} seed{seed} R{age}",flush=True)
    manifest.update(status='recoveries_complete',current_condition=None);write_manifest(manifest)
    print('All six donors and 18 intervention pairs completed. Analysis follows.',flush=True)
    import importlib.util
    spec=importlib.util.spec_from_file_location('fixed_schedule_analysis','scripts/analyze_fixed_schedule_intervention.py');module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module);module.main()

if __name__=='__main__':
    try:main()
    except Exception:
        if ROOT.exists():
            (ROOT/'failure.txt').write_text(traceback.format_exc())
            if (ROOT/'run_manifest.json').exists():
                m=json.loads((ROOT/'run_manifest.json').read_text());m['status']='failed';write_manifest(m)
        raise
