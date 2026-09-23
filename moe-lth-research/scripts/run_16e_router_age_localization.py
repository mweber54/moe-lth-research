#!/usr/bin/env python3
"""Staged native 16E localization; no replay or mechanism interventions."""
import argparse
import json
import math
import shutil
from pathlib import Path
from collections import defaultdict
import numpy as np
import pandas as pd
import torch
from moe_lth.config import load_config, save_config
from moe_lth.data import build_dataloaders
from moe_lth.models.transformer import TinyMoELanguageModel
from moe_lth.experiments import run_router_age_recovery as r
from moe_lth.experiments.run_router_development_12expert import sha256_file, group
from moe_lth.pruning.magnitude_prune import expert_local_magnitude_masks
from moe_lth.pruning.masks import save_masks
from moe_lth.pruning.router_age import state_dict_hash, component_state_dict, selected_experts_per_batch
from moe_lth.utils import configure_device, resolve_device, resolve_data_seed, seed_everything

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'results/16e_router_age_localization'
SEEDS=(7,17,29)
COARSE=[0,20,40,60,80,100]

def write_json(path,obj):
    path.parent.mkdir(parents=True,exist_ok=True)
    temporary=path.with_suffix(path.suffix+'.tmp'); temporary.write_text(json.dumps(obj,indent=2)+'\n'); temporary.replace(path)

def append_csv(path,row):
    old=pd.read_csv(path).to_dict('records') if path.exists() else []
    pd.DataFrame(old+[row]).to_csv(path,index=False)

def config_for(seed):
    c=load_config(ROOT/'configs/expert_count_replication/e12_seed7.yaml')
    c['seed']=seed; c['model']['num_experts']=16
    c['training']['checkpoint_steps']=[0,500,2500]
    c['training']['checkpoint_only_steps']=list(range(125,2500,125))
    c['output_dir']=str(OUT/f'references/seed_{seed}')
    return c

def select_refinement(frame):
    """Predeclared: adjacent coarse mean differences <= one paired-seed SD are ambiguous."""
    pivot=frame.pivot(index='seed',columns='router_age',values='ticket_gap')
    means=pivot.mean(); best=int(means.idxmin()); tied=[best]; comparisons=[]
    if best in (0,100): return [10 if best==0 else 90],{'best':best,'basin_coarse':[best],'comparisons':[]}
    for neighbor in (best-20,best+20):
        if neighbor not in means: continue
        diff=pivot[neighbor]-pivot[best]; ambiguous=abs(float(diff.mean()))<=float(diff.std(ddof=1))
        comparisons.append({'age':neighbor,'mean_paired_difference':float(diff.mean()),'paired_seed_sd':float(diff.std(ddof=1)),'ambiguous':bool(ambiguous)})
        if ambiguous: tied.append(neighbor)
    candidates=sorted({x for age in tied for x in (age-10,age+10) if 0<x<100 and x not in means})
    return candidates,{'best':best,'basin_coarse':sorted(tied),'comparisons':comparisons}

def preflight():
    OUT.mkdir(parents=True,exist_ok=True)
    c=config_for(7); seed_everything(7); model=TinyMoELanguageModel(c['model'])
    counts=defaultdict(int)
    for name,p in model.named_parameters(): counts[r.parameter_group(name)]+=p.numel()
    assert len(model.blocks)==4 and all(b.moe.num_experts==16 and b.moe.top_k==1 for b in model.blocks)
    assert c['routing']['mode']=='learned' and c['routing']['aux_loss_weight']==.1
    assert c['training']['steps']==2500 and c['training']['precision']=='fp16'
    manifest={'experts':16,'model':c['model'],'routing':c['routing'],'training':c['training'],'total_parameter_count':sum(counts.values()),'parameter_groups':dict(counts),'active_parameters_per_token_nominal':counts['shared']+counts['router']+counts['expert']//16,'active_count_definition':'Shared and router parameters plus one expert per layer; nominal before capacity drops, counts unique tied parameters once.','architectural_differences':['num_experts: 12 -> 16; router output dimension and expert bank grow accordingly'],'recovery_scheduler':'none','recovery_warmup':0,'recovery_eval_interval':r.RECOVERY_EVAL_INTERVAL,'source_config_sha256':sha256_file(ROOT/'configs/multidomain_gpu.yaml')}
    write_json(OUT/'architecture_manifest.json',manifest)
    decision=OUT/'localization_decision.json'
    if not decision.exists(): write_json(decision,{'status':'preflight','stage_a_ages':COARSE,'stage_b_ages':None,'stage_c_ages':[],'ambiguity_rule':'Before outcomes: an adjacent coarse checkpoint is ambiguous if absolute mean paired gap difference <= sample SD of the three seed-paired differences. Refine one midpoint on each side of best and ambiguous adjacent coarse checkpoints; boundaries refine inward once. This is a localization heuristic, not a significance test.','stage_c_policy':'Optional Stage C omitted; report residual 10%-grid ambiguity as a basin.','primary':'native final sparse minus matched same-age dense loss at step 2500','seeds':list(SEEDS),'sparsity':.85})
    (OUT/'preflight_audit.md').write_text('# 16E native localization preflight\n\nExactly seeds 7,17,29; 85% expert-weight sparsity using historical per-expert rounded keep counts; E_T magnitude masks and E_0 surviving expert/shared values, age-specific native trainable router. Fresh AdamW, 2500 attempts, fp16/GradScaler, lr 0.0003, wd 0.01, clip 1, auxiliary 0.1, no scheduler or warmup; historical recovery validation every 50 steps. No forward overrides, replay callbacks, hidden-state interventions, or extra recoveries. Existing helper diagnostic logging is retained without diagnostic interventions.\n\nThree reference trajectories retain original validation cadence and add checkpoint-only saves every 125 steps (5% age). Saving weights does not evaluate or change RNG. These are artifacts, not a fine recovery sweep. Stage A finishes for all seeds before the preregistered local Stage B decision. Optional Stage C is omitted; unresolved local uncertainty is reported directly. Primary winner uses matched ticket gap exclusively. Secondary pruning shock compares matched E_0/R_age sparse and dense initial models. Developmental probes use the historical reference-model definitions; cumulative exposure is the historical sum of logged assigned-token counts, not complete accepted-token updates.\n')
    for seed in SEEDS: save_config(config_for(seed),OUT/f'configs/e16_seed{seed}.yaml')
    if shutil.disk_usage(OUT).free<40*1024**3: raise RuntimeError('Less than 40 GiB free for reference and recovery artifacts')
    print('[16E] preflight passed; architecture parameters',manifest['total_parameter_count'],flush=True)

def run_seed(seed,ages,stage):
    c=config_for(seed); path=OUT/f'configs/e16_seed{seed}.yaml'
    reference=r._ensure_reference(str(path)); device=resolve_device(c['device']); configure_device(device); seed_everything(seed)
    initial_path=reference/'checkpoints/step_0.pt'; final_path=reference/'checkpoints/step_2500.pt'
    final=r.load_model_from_checkpoint(c['model'],str(final_path),device)
    initial=r.load_model_from_checkpoint(c['model'],str(initial_path),torch.device('cpu'))
    state={k:v.detach().cpu().clone() for k,v in initial.state_dict().items()}; del initial
    masks=expert_local_magnitude_masks(final,.85)
    groups=defaultdict(list)
    for name,mask in masks.items():
        assert r.parameter_group(name)=='expert'
        groups[name.split('.moe.experts.')[0]+'.'+name.split('.moe.experts.')[1].split('.')[0]].append(mask)
    assert len(groups)==64
    for items in groups.values(): assert sum(int(x.sum()) for x in items)==round(sum(x.numel() for x in items)*.15)
    mask_path=OUT/f'masks/seed_{seed}/pruning_mask.pt'
    if mask_path.exists():
        from moe_lth.pruning.masks import load_masks
        assert r._mask_hash(load_masks(mask_path))==r._mask_hash(masks)
    else: save_masks(masks,mask_path)
    ticket=r.build_fixed_pruned_base(c['model'],str(initial_path),masks,device)
    loader,val_loader=build_dataloaders(c['data'],c['training']['batch_size'],resolve_data_seed(c))
    train=r._materialize_batches(loader,2500); val=r._materialize_validation_batches(val_loader,c['data']['validation_blocks']); probe=r._calibration_batches(val)
    train_hash=r._batch_sequence_hash(train); val_hash=r._batch_sequence_hash(val)
    selected=selected_experts_per_batch(final,probe,device)
    ref_loss=r.evaluate_language_model(final,val,device,max_batches=len(val))['loss']; del final
    for age in ages:
        checkpoint,step=r._checkpoint_for_percent(reference,2500,age)
        assert step==age*25 and '16e_router_age_localization' in str(checkpoint)
        payload=torch.load(checkpoint,map_location='cpu',weights_only=False)
        assert payload['step']==step and payload['config']['model']['num_experts']==16
        checkpoint_row={'experts':16,'seed':seed,'router_age':age,'source_step':step,'checkpoint_path':str(checkpoint),'checkpoint_hash':sha256_file(checkpoint),'model_hash':state_dict_hash(payload['model']),**{f'{g}_hash':state_dict_hash(group(payload['model'],g)) for g in ('router','expert','shared')}}
        existing=pd.read_csv(OUT/'checkpoint_manifest.csv') if (OUT/'checkpoint_manifest.csv').exists() else pd.DataFrame()
        if existing.empty or not ((existing.seed==seed)&(existing.router_age==age)).any(): append_csv(OUT/'checkpoint_manifest.csv',checkpoint_row)
        records={}
        for kind,base,mask in [('dense',state,{}),('sparse',ticket,masks)]:
            dest=OUT/f'recovery/seed_{seed}/R{age}/{kind}'
            result_path=dest/'validated_result.json'
            if result_path.exists():
                record=json.loads(result_path.read_text())
                assert record['router_checkpoint_hash']==checkpoint_row['checkpoint_hash'] and record['training_batch_sequence_hash']==train_hash and record['validation_batch_sequence_hash']==val_hash
                assert sha256_file(dest/'checkpoints/final_recovered.pt')==record['final_checkpoint_sha256']
            else:
                hashes={}
                def initial_audit(model,optimizer,scaler,at_step):
                    if at_step: return
                    assert not optimizer.state
                    assert all(b.moe.num_experts==16 for b in model.blocks)
                    assert state_dict_hash(component_state_dict(model,'router'))==checkpoint_row['router_hash']
                    hashes['initial_model_hash']=state_dict_hash(model.state_dict())
                record=r._run_recovery_condition(config=c,condition_name=f'e16_seed{seed}_R{age}_{kind}',pruned_base_state=base,router_checkpoint=str(checkpoint),router_age_percent=age,router_step=step,masks=mask,expert_hash=state_dict_hash(group(base,'expert')),shared_hash=state_dict_hash(group(base,'shared')),mask_hash=r._mask_hash(mask) if mask else 'dense_no_mask',reference_selected=selected,calibration_batches=probe,train_batches=train,validation_batches=val,train_batch_hash=train_hash,validation_batch_hash=val_hash,device=device,recovery_steps=2500,dense_loss=records['dense']['final_validation_loss'] if kind=='sparse' else ref_loss,output_dir=dest,confidence_control=False,target_confidence=None,seed=seed,sparsity=.85 if mask else 0.,router_mode='trainable',state_snapshot_callback=initial_audit)
                final_payload=torch.load(dest/'checkpoints/final_recovered.pt',map_location='cpu',weights_only=False)
                assert final_payload['step']==2500 and record['optimizer_state_entries_at_start']==0 and record['integrity_checks_passed']
                record.update(hashes); record.update({'experts':16,'stage':stage,'kind':kind,'router_age':age,'router_checkpoint_hash':checkpoint_row['checkpoint_hash'],'final_model_hash':state_dict_hash(final_payload['model']),'final_checkpoint_sha256':sha256_file(dest/'checkpoints/final_recovered.pt'),'native_routing':True,'forward_callbacks':None,'condition_dir':str(dest)})
                write_json(result_path,record)
            records[kind]=record
        d,s=records['dense'],records['sparse']
        assert d['initial_router_state_hash']==s['initial_router_state_hash'] and d['shared_state_hash']==s['shared_state_hash']
        refresh_tables()
        print(f'[16E] {stage} seed={seed} R{age}: gap={s["final_validation_loss"]-d["final_validation_loss"]:.8f}',flush=True)
    del ticket,state,masks; torch.cuda.empty_cache()

def refresh_tables():
    records=[json.loads(p.read_text()) for p in OUT.glob('recovery/seed_*/R*/*/validated_result.json')]
    if not records:return
    pd.DataFrame(records).to_csv(OUT/'run_manifest.csv',index=False)
    rows=[]
    for seed in SEEDS:
        for age in sorted({x['router_age'] for x in records if x['seed']==seed}):
            pair={x['kind']:x for x in records if x['seed']==seed and x['router_age']==age}
            if set(pair)!={'dense','sparse'}:continue
            d,s=pair['dense'],pair['sparse']
            rows.append({'experts':16,'seed':seed,'router_age':age,'dense_initial_loss':d['initial_validation_loss'],'sparse_initial_loss':s['initial_validation_loss'],'immediate_pruning_shock':s['initial_validation_loss']-d['initial_validation_loss'],'dense_early_auc':d['early_auc'],'sparse_early_auc':s['early_auc'],'dense_final_loss':d['final_validation_loss'],'sparse_final_loss':s['final_validation_loss'],'ticket_gap':s['final_validation_loss']-d['final_validation_loss'],'dense_recovery_improvement':d['initial_validation_loss']-d['final_validation_loss'],'sparse_recovery_improvement':s['initial_validation_loss']-s['final_validation_loss']})
    pd.DataFrame(rows).to_csv(OUT/'outcomes.csv',index=False)

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--preflight-only',action='store_true');args=parser.parse_args()
    preflight()
    if args.preflight_only:return
    decision=json.loads((OUT/'localization_decision.json').read_text())
    decision['status']='stage_a_running';write_json(OUT/'localization_decision.json',decision)
    for seed in SEEDS:run_seed(seed,COARSE,'A')
    outcomes=pd.read_csv(OUT/'outcomes.csv');coarse=outcomes[outcomes.router_age.isin(COARSE)]
    assert len(coarse)==18 and coarse.groupby('router_age').seed.nunique().eq(3).all()
    refinement,evidence=select_refinement(coarse)
    decision.update({'status':'stage_b_running','stage_b_ages':refinement,'stage_a_decision':evidence});write_json(OUT/'localization_decision.json',decision)
    print('[16E] Stage A complete; Stage B ages',refinement,flush=True)
    for seed in SEEDS:run_seed(seed,refinement,'B')
    decision['status']='recoveries_complete_analysis_pending';write_json(OUT/'localization_decision.json',decision)
    import subprocess,sys
    subprocess.run([sys.executable,'-u',str(ROOT/'scripts/analyze_16e_router_age_localization.py')],check=True)

if __name__=='__main__':main()
