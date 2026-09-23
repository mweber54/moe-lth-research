#!/usr/bin/env python3
"""Capture the 18 canonical dense amplitude sources; no decomposition recovery is permitted here."""
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
from moe_lth.experiments.canonical_dense_amplitude import DenseAmplitudeCapture
import pandas as pd
from moe_lth.pruning.magnitude_prune import expert_local_magnitude_masks
from moe_lth.pruning.router_age import state_dict_hash,selected_experts_per_batch
from moe_lth.utils import configure_device,resolve_data_seed,seed_everything

ROOT=Path('results/gate_amplitude_pattern_decomposition')
LEVEL2=Path('results/gate_amplitude_replay_intervention')
LEVEL1=Path('results/cross_arch_optimization_geometry/fixed_schedule_intervention')
PRIOR=Path('results/cross_arch_optimization_geometry/diagnostic_rerun')

def group(s,k):return {n:v for n,v in s.items() if r.parameter_group(n)==k}
def write_manifest(m):(ROOT/'capture_run_manifest.json').write_text(json.dumps(m,indent=2)+'\n')

def main():
    ROOT.mkdir(parents=True,exist_ok=True)
    if (ROOT/'capture_run_manifest.json').exists():raise FileExistsError('Capture already exists; refusing overwrite')
    if shutil.disk_usage(ROOT).free<60*1024**3:raise RuntimeError('Need 60 GiB stable storage for source artifacts')
    prior=json.loads((LEVEL2/'gate_replay_manifest.json').read_text());rows=prior['conditions'];assert len(rows)==18 and prior['completed_recipient_recoveries']==36
    bulk=ROOT.resolve();manifest={'status':'preflight','created_utc':datetime.now(timezone.utc).isoformat(),'conditions':rows,'required_sources':18,'validated_sources':0,'decomposition_recoveries_launched':0,'canonical_source':'each age/seed dense Level-2 trajectory, shared by both matched model kinds','primary_readout':'primary_r100_common_readout','secondary_readouts':['secondary_condition_matched_readout','secondary_native_readout'],'capture_validation':'all nine model/optimizer/scaler states and full validation curve exactly equal Level 2; indexed native proposal means match existing observations; complete route replay verified','decomposition_gate':'ALL 18 source manifests validated before any transformed recovery','execution':'sequential','code_hashes':{}}
    write_manifest(manifest);(ROOT/'capture_provenance').mkdir()
    for name in [__file__,r.__file__,'src/moe_lth/experiments/canonical_dense_amplitude.py','src/moe_lth/experiments/gate_amplitude_replay.py']:
        manifest['code_hashes'][name]=file_hash(name);shutil.copyfile(name,ROOT/'capture_provenance'/Path(name).name)
    write_manifest(manifest)
    old_snapshots=pd.read_csv(LEVEL2/'snapshot_manifest.csv');old_statistics=pd.read_csv(LEVEL2/'gate_amplitude_statistics.csv')
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
            schedule_path=LEVEL2/f'schedules/e{e}/seed{seed}'
            def run_one(row,schedule):
                age=row['age'];meta=json.loads((Path(row['dense'])/'metadata.json').read_text());output=bulk/f'capture_recovery/e{e}/seed{seed}/R{age}/dense'
                manifest.update(status='capturing',current_condition={'experts':e,'seed':seed,'age':age});write_manifest(manifest)
                def snapshot(model,opt,scaler,step):
                    if step not in d.STEPS:return
                    oldrow=old_snapshots[(old_snapshots.experts==e)&(old_snapshots.seed==seed)&(old_snapshots.age==age)&(old_snapshots.kind=='dense')&(old_snapshots.step==step)].iloc[0]
                    old=torch.load(oldrow.path,map_location='cpu',weights_only=False)
                    mh=d.digest(model.state_dict());oh=d.digest(opt.state_dict());sh=d.digest(scaler.state_dict())
                    assert mh==oldrow.model_hash and oh==oldrow.optimizer_hash and sh==d.digest(old['grad_scaler']),f'Capture trajectory mismatch e{e} s{seed} R{age} step{step}'
                    d.append(ROOT/'capture_state_integrity.csv',[dict(experts=e,seed=seed,age=age,step=step,model_exact=True,optimizer_exact=True,scaler_exact=True,model_hash=mh,optimizer_hash=oh)])
                    print(f'[source state exact] e{e} seed{seed} R{age} step{step}',flush=True)
                dense_loss=json.loads((Path(row['dense'])/'summary.json').read_text())['final_validation_loss']
                record=r._run_recovery_condition(config=cfg,condition_name=f'canonical_dense_capture_e{e}_seed{seed}_R{age}',pruned_base_state=initial,router_checkpoint=meta['router_checkpoint'],router_age_percent=age,router_step=meta['router_step'],masks={},expert_hash=meta['expert_state_hash'],shared_hash=meta['shared_state_hash'],mask_hash=meta['mask_hash'],reference_selected=reference,calibration_batches=cal,train_batches=train,validation_batches=val,train_batch_hash=th,validation_batch_hash=vh,device=device,recovery_steps=2500,dense_loss=dense_loss,output_dir=output,confidence_control=False,target_confidence=None,seed=seed,sparsity=0.,router_mode='trainable',state_snapshot_callback=snapshot,training_forward_callback=schedule.train,validation_forward_callback=schedule.validation)
                # Historical helper's recovery-fraction denominator is not an intervention endpoint.
                # Only explicitly matched outcomes below are used for causal comparisons.
                return output,record
            for row in sorted(selected,key=lambda x:x['age']):
                age=row['age'];identity=dict(experts=e,seed=seed,age=age,kind='dense')
                expected=old_statistics[(old_statistics.experts==e)&(old_statistics.seed==seed)&(old_statistics.age==age)&(old_statistics.kind=='dense')].set_index(['step','layer','expert'])
                source_path=bulk/f'amplitude_sources/e{e}/seed{seed}/R{age}'
                capture=DenseAmplitudeCapture(schedule_path,source_path,identity,expected)
                output,record=run_one(row,capture);source=capture.finish()
                old_model=torch.load(LEVEL2/f'recovery/e{e}/seed{seed}/R{age}/dense/checkpoints/final_recovered.pt',map_location='cpu',weights_only=False)['model']
                new_model=torch.load(output/'checkpoints/final_recovered.pt',map_location='cpu',weights_only=False)['model'];assert d.digest(old_model)==d.digest(new_model);del old_model,new_model
                old_curve=pd.read_json(LEVEL2/f'recovery/e{e}/seed{seed}/R{age}/dense/metrics.jsonl',lines=True);new_curve=pd.read_json(output/'metrics.jsonl',lines=True);assert old_curve.equals(new_curve),'Capture validation curve differs'
                assert capture.checked_summary_rows==len(expected),'Incomplete native-proposal summary verification'
                source.update(status='validated',final_model_exact=True,all_snapshot_states_exact=True,validation_curve_exact=True)
                (source_path/'manifest.json').write_text(json.dumps(source,indent=2)+'\n')
                d.append(ROOT/'validated_amplitude_sources.csv',[dict(**identity,path=str(source_path),training_occurrences=source['counts']['train'],validation_occurrences=source['counts']['validation'],final_model_exact=True,all_snapshot_states_exact=True,validation_curve_exact=True,summary_rows_checked=capture.checked_summary_rows,manifest_sha256=file_hash(source_path/'manifest.json'))])
                manifest['validated_sources']+=1;write_manifest(manifest);gc.collect();torch.cuda.empty_cache()
                print(f"[source validated] {manifest['validated_sources']}/18 e{e} seed{seed} R{age}",flush=True)
    assert manifest['validated_sources']==18
    manifest.update(status='all_sources_validated',current_condition=None);write_manifest(manifest)
    print('ALL 18 CANONICAL DENSE AMPLITUDE SOURCES VALIDATED. Decomposition may now be constructed and checked.',flush=True)

if __name__=='__main__':
    try:main()
    except Exception:
        if ROOT.exists():
            (ROOT/'failure.txt').write_text(traceback.format_exc())
            if (ROOT/'capture_run_manifest.json').exists():
                m=json.loads((ROOT/'capture_run_manifest.json').read_text());m['status']='failed';write_manifest(m)
        raise
