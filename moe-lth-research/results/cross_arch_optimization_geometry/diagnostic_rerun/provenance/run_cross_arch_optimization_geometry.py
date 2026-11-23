#!/usr/bin/env python3
"""Exactly the preregistered 18 matched diagnostic recoveries; historical inputs read-only."""
import argparse, gc, hashlib, json, shutil, sys, traceback
from pathlib import Path
import torch
from moe_lth.config import load_config
from moe_lth.data import build_dataloaders
from moe_lth.experiments import run_router_age_recovery as r
from moe_lth.experiments import optimization_geometry_diagnostics as d
from moe_lth.models import TinyMoELanguageModel
from moe_lth.pruning.magnitude_prune import expert_local_magnitude_masks
from moe_lth.pruning.router_age import state_dict_hash,component_state_dict,selected_experts_per_batch
from moe_lth.utils import configure_device,resolve_data_seed,seed_everything

OUT=Path('results/cross_arch_optimization_geometry/diagnostic_rerun')

def sha(path):
    h=hashlib.sha256()
    with open(path,'rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
    return h.hexdigest()

def group(s,g):return {k:v for k,v in s.items() if r.parameter_group(k)==g}

def main():
    if OUT.exists():raise FileExistsError(f'Refusing to overwrite {OUT}')
    OUT.mkdir(parents=True)
    rows=json.loads(Path('/tmp/geometry_preflight.json').read_text())
    assert len(rows)==18 and all(all(x['checks'].values()) for x in rows)
    manifest={'status':'preflight','conditions':rows,'sparsity':.85,'recovery_steps':2500,'snapshot_steps':d.STEPS,'primary_metrics':['forced_sparse_route_expert_gradient_cosine','masked_expert_update_cosine','common_input_expert_output_cosine'],'secondary_metrics':['native_gradient_cosine','routing_overlap','parameter_distance','two_microbatch_gradient_coherence'],'aggregation':'Equal expert means within layer, equal layer means within seed; seed is independent replicate. Equal snapshot weights for Qg including step 0; equal interval weights for Qu. Report step 0 separately. Exclude undefined cosines with explicit valid counts.','reproduction_gate':{'max_absolute_individual_loss_delta':.03,'max_absolute_ticket_gap_delta':.03,'ordering':'report all seed rankings and architecture means; investigate changes above .01 gap separation'},'temporal_rule':'First measured step at which all three paired seed differences favor target against each control; report individual seeds and persistence; descriptive only.','diagnostic_precision':'fp32 autocast disabled, unit loss scale, fp32 vectors, fp64 reductions; training fp16 unchanged','forced_routes':'Sparse native IDs and capacity priority scores shared across models. Native gate amplitudes retained and may differ; this controls token selection, not all routing-dependent effects.','functional_inputs':'Identical sparse-native accepted pre-router hidden vectors for each corresponding expert, no cross-architecture expert pairing.','optional_dense_forcing':'skipped to keep rerun minimal','optional_covariance_cka':'not run','code_hashes':{str(p):sha(p) for p in [Path(__file__),Path(r.__file__),Path(d.__file__)]},'torch_version':torch.__version__}
    (OUT/'run_manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    device=torch.device('cuda');configure_device(device)
    # Preflight every data identity before ANY recovery, not merely pair metadata.
    sequences={};probes={};data_checks=[]
    for row in rows:
        c=load_config(row['config']);key=row['seed'];sm=json.loads((Path(row['sparse'])/'metadata.json').read_text())
        if key not in sequences:
            seed_everything(key);a,b=build_dataloaders(c['data'],int(c['training']['batch_size']),resolve_data_seed(c))
            train=r._materialize_batches(a,2500);val=r._materialize_validation_batches(b,32)
            sequences[key]=(train,val,r._batch_sequence_hash(train),r._batch_sequence_hash(val))
            probes[key]=(val[0][0][:8].clone(),val[0][1][:8].clone())
        train,val,th,vh=sequences[key]
        assert th==sm['training_batch_sequence_hash'] and vh==sm['validation_batch_sequence_hash'],'Data sequence incompatible'
        data_checks.append({'experts':row['experts'],'seed':key,'age':row['age'],'train_sequence_hash':th,'validation_sequence_hash':vh,'train_file_sha256':sha(c['data']['train_path']),'validation_file_sha256':sha(c['data']['validation_path'])})
    assert all(d.digest(p)==d.digest(probes[7]) for p in probes.values())
    probe=probes[7];torch.save(probe,OUT/'diagnostic_probe.pt')
    c=load_config(rows[0]['config']);pmanifest={'batch_hash':d.digest(probe),'batch_file_sha256':sha(OUT/'diagnostic_probe.pt'),'dataset_hash':sha(c['data']['validation_path']),'tokenizer':'UTF-8 ByteTokenizer, byte IDs 0..255','tokenizer_hash':d.digest({'type':'byte','mapping':list(range(256))}),'preprocessing':'Path.read_text(encoding=utf-8), then UTF-8 encoding; contiguous TokenBlockDataset blocks','sample_ids':list(range(8)),'token_positions':[list(range(i*128,(i+1)*128)) for i in range(8)],'token_ids':probe[0].tolist(),'target_ids':probe[1].tolist(),'sequence_length':128,'batch_order':list(range(8)),'shared_literally_across_architectures_and_seeds':True}
    (OUT/'diagnostic_probe_manifest.json').write_text(json.dumps(pmanifest,indent=2)+'\n')
    (OUT/'protocol_audit.md').write_text('# Protocol audit\n\nAll 18 reconstructed E0/shared/router/ticket/ET-mask component hashes match historical metadata. All materialized training and validation sequence hashes match before training. Within each architecture/seed the three router ages use identical masks, E0, and shared state.\n\nTraining calls the existing recovery function unchanged except for an optional callback after step 0 and completed training steps. Historical evaluation every 50 steps and routing diagnostics are retained. The callback saves complete state at nine requested steps; dense checkpoints are then loaded into a separate model for sparse diagnostics. Model parameters/buffers, optimizer, GradScaler and RNG are audited. Diagnostic gradients are cleared before the next training step. No training iterator is passed to diagnostics.\n\nSparse/dense vectors use identical sorted named parameter order and select surviving mask indices (unmasked expert biases survive in full). Norms and invalid counts are retained. Forced routing shares sparse assignment IDs and capacity priorities, with exact accepted-token equality asserted; native gate values remain model-specific. Functional comparisons evaluate both expert functions on identical sparse-native accepted inputs.\n\nFP32 diagnostic backward uses unit loss scale, so gradients are already unscaled; it never calls the training GradScaler. Training remains fp16. Undefined zero-norm or nonfinite cosines are unavailable, never zero-filled.\n')
    (OUT/'data_identity_audit.json').write_text(json.dumps(data_checks,indent=2)+'\n')
    manifest['status']='running';(OUT/'run_manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    completed=0
    for row in rows:
        e,seed,age=row['experts'],row['seed'],row['age'];cfg=load_config(row['config'])
        train,val,th,vh=sequences[seed];cal=r._calibration_batches(val)
        sp,dp=Path(row['sparse']),Path(row['dense']);sm=json.loads((sp/'metadata.json').read_text());dm=json.loads((dp/'metadata.json').read_text())
        checkpoint=Path(sm['router_checkpoint']);ref=checkpoint.parent
        final=r.load_model_from_checkpoint(cfg['model'],str(ref/'step_2500.pt'),device)
        init=r.load_model_from_checkpoint(cfg['model'],str(ref/'step_0.pt'),torch.device('cpu'))
        initial={k:v.detach().cpu().clone() for k,v in init.state_dict().items()};del init
        masks=expert_local_magnitude_masks(final,.85)
        assert r._mask_hash(masks)==sm['mask_hash']
        ticket={k:(v*masks[k].cpu() if k in masks else v) for k,v in initial.items()}
        assert state_dict_hash(group(initial,'expert'))==dm['expert_state_hash']
        assert state_dict_hash(group(ticket,'expert'))==sm['expert_state_hash']
        for k,m in masks.items():assert torch.equal(ticket[k][m.cpu().bool()],initial[k][m.cpu().bool()])
        reference_selected=selected_experts_per_batch(final,cal,device)
        dense_reference_loss=json.loads((dp/'summary.json').read_text())['dense_reference_loss']
        del final;gc.collect();torch.cuda.empty_cache()
        previous=None;records={}
        for kind,historical,base,maskset in [('dense',dm,initial,{}),('sparse',sm,ticket,masks)]:
            prefix=OUT/f'snapshots/e{e}/seed{seed}/R{age}/{kind}';prefix.mkdir(parents=True)
            def snapshot(model,optimizer,scaler,step):
                nonlocal previous
                if step not in d.STEPS:return
                rng=d.rng_state();before=d.digest(model.state_dict());opt=d.digest(optimizer.state_dict());scale=d.digest(scaler.state_dict());modes=[m.training for m in model.modules()]
                payload={'model':model.state_dict(),'optimizer':optimizer.state_dict(),'grad_scaler':scaler.state_dict(),'scheduler':None,'lr':[g['lr'] for g in optimizer.param_groups],'step':step,'rng':rng,'config':cfg,'seed':seed,'mask':{k:v.cpu() for k,v in masks.items()},'mask_applied':kind=='sparse','historical_metadata':historical,'data_sequence_identity':{'training':th,'validation':vh,'next_batch_index':step},'diagnostic_probe_hash':pmanifest['batch_hash'],'parameter_hash':before,'optimizer_hash':opt}
                path=prefix/f'step_{step:04d}.pt';assert not path.exists();torch.save(payload,path)
                d.append(OUT/'snapshot_manifest.csv',[{'experts':e,'seed':seed,'age':age,'kind':kind,'step':step,'path':str(path),'sha256':sha(path),'model_hash':before,'optimizer_hash':opt}]);del payload
                try:
                    if kind=='sparse':
                        dense=TinyMoELanguageModel(cfg['model']).to(device)
                        dense_path=OUT/f'snapshots/e{e}/seed{seed}/R{age}/dense/step_{step:04d}.pt'
                        saved=torch.load(dense_path,map_location='cpu',weights_only=False);dense.load_state_dict(saved['model']);del saved
                        previous=d.measure(model,dense,masks,probe,float(cfg['routing']['aux_loss_weight']),OUT,{'experts':e,'seed':seed,'age':age,'step':step},previous)
                        del dense
                finally:
                    model.zero_grad(set_to_none=True)
                    for module,mode in zip(model.modules(),modes):module.training=mode
                    d.restore_rng(rng)
                assert d.digest(model.state_dict())==before,'Model mutated'
                assert d.digest(optimizer.state_dict())==opt,'Optimizer mutated'
                assert d.digest(scaler.state_dict())==scale,'Scaler mutated'
                assert d.digest(d.rng_state())==d.digest(rng),'RNG mutated'
                d.append(OUT/'mutation_audit.csv',[{'experts':e,'seed':seed,'age':age,'kind':kind,'step':step,'model_unchanged':True,'optimizer_unchanged':True,'scaler_unchanged':True,'rng_restored':True,'training_iterator_not_accessed':True}])
                print(f'[snapshot] e{e} seed{seed} R{age} {kind} step{step} mutation audit passed',flush=True)
            records[kind]=r._run_recovery_condition(config=cfg,condition_name=historical['condition'],pruned_base_state=base,router_checkpoint=str(checkpoint),router_age_percent=age,router_step=historical['router_step'],masks=maskset,expert_hash=historical['expert_state_hash'],shared_hash=historical['shared_state_hash'],mask_hash=historical['mask_hash'],reference_selected=reference_selected,calibration_batches=cal,train_batches=train,validation_batches=val,train_batch_hash=th,validation_batch_hash=vh,device=device,recovery_steps=2500,dense_loss=records['dense']['final_validation_loss'] if kind=='sparse' else dense_reference_loss,output_dir=OUT/f'recovery/e{e}/seed{seed}/R{age}/{kind}',confidence_control=False,target_confidence=None,seed=seed,sparsity=.85 if kind=='sparse' else 0.,router_mode='trainable',state_snapshot_callback=snapshot)
        hs=json.loads((sp/'summary.json').read_text())['final_validation_loss'];hd=json.loads((dp/'summary.json').read_text())['final_validation_loss'];rs=records['sparse']['final_validation_loss'];rd=records['dense']['final_validation_loss']
        delta=(rs-rd)-(hs-hd);passed=max(abs(rs-hs),abs(rd-hd),abs(delta))<=.03
        d.append(OUT/'reproduction_check.csv',[{'experts':e,'seed':seed,'age':age,'historical_sparse_loss':hs,'rerun_sparse_loss':rs,'historical_dense_loss':hd,'rerun_dense_loss':rd,'historical_ticket_gap':hs-hd,'rerun_ticket_gap':rs-rd,'delta':delta,'within_preregistered_tolerance':passed}])
        completed+=1;manifest.update(completed_pairs=completed);(OUT/'run_manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
        if not passed:raise RuntimeError(f'Reproduction gate failed for e{e}/seed{seed}/R{age}; investigate before continuing')
    manifest['status']='recoveries_complete';(OUT/'run_manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    print(f'Completed {completed}/18 pairs. Recovery and diagnostics complete; analysis pending.',flush=True)

if __name__=='__main__':
    try:main()
    except Exception:
        if OUT.exists():(OUT/'failure.txt').write_text(traceback.format_exc())
        raise
