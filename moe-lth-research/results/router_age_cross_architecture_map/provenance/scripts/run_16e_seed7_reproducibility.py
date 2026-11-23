#!/usr/bin/env python3
"""One repeat of six existing seed-7 matched native recoveries, never a fourth seed."""
import json,os,platform
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import run_16e_router_age_localization as base
from moe_lth.config import load_config,save_config
from moe_lth.pruning.masks import load_masks
from moe_lth.utils import configure_device,resolve_device,resolve_data_seed
ROOT=base.ROOT; ORIGINAL=ROOT/'results/16e_router_age_localization';OUT=ROOT/'results/16e_seed7_reproducibility';AGES=[50,60,70,80,90,100]

def audit():
    OUT.mkdir(parents=True,exist_ok=True)
    assert json.loads((ORIGINAL/'localization_decision.json').read_text())['status']=='complete'
    cfg=load_config(ORIGINAL/'configs/e16_seed7.yaml');assert cfg['model']['num_experts']==16 and cfg['seed']==7
    configure_device(resolve_device(cfg['device']))
    code=json.loads((ORIGINAL/'implementation_validation.json').read_text())['code_sha256']
    for name in ['src/moe_lth/experiments/run_router_age_recovery.py','src/moe_lth/pruning/magnitude_prune.py','src/moe_lth/pruning/router_age.py']:
        assert base.sha256_file(ROOT/name)==code[name],f'Historical code changed: {name}'
    train_loader,val_loader=base.build_dataloaders(cfg['data'],cfg['training']['batch_size'],resolve_data_seed(cfg))
    train=base.r._materialize_batches(train_loader,2500);val=base.r._materialize_validation_batches(val_loader,32)
    train_hash=base.r._batch_sequence_hash(train);val_hash=base.r._batch_sequence_hash(val)
    mask=load_masks(ORIGINAL/'masks/seed_7/pruning_mask.pt');mask_hash=base.r._mask_hash(mask)
    final=base.r.load_model_from_checkpoint(cfg['model'],str(Path(cfg['output_dir'])/'checkpoints/step_2500.pt'),torch.device('cpu'))
    assert base.r._mask_hash(base.expert_local_magnitude_masks(final,.85))==mask_hash;del final
    initial=Path(cfg['output_dir'])/'checkpoints/step_0.pt'
    ticket=base.r.build_fixed_pruned_base(cfg['model'],str(initial),mask,torch.device('cpu'))
    initstate=torch.load(initial,map_location='cpu',weights_only=False)['model']
    rows=[]
    for age in AGES:
        checkpoint=Path(cfg['output_dir'])/f'checkpoints/step_{age*25}.pt'
        payload=torch.load(checkpoint,map_location='cpu',weights_only=False)
        assert payload['step']==age*25
        for kind,state in [('dense',initstate),('sparse',ticket)]:
            dest=ORIGINAL/f'recovery/seed_7/R{age}/{kind}';record=json.loads((dest/'validated_result.json').read_text())
            assert base.sha256_file(checkpoint)==record['router_checkpoint_hash']
            assert base.sha256_file(dest/'checkpoints/final_recovered.pt')==record['final_checkpoint_sha256']
            assert record['training_batch_sequence_hash']==train_hash and record['validation_batch_sequence_hash']==val_hash
            assert record['expert_state_hash']==base.state_dict_hash(base.group(state,'expert'))
            assert record['shared_state_hash']==base.state_dict_hash(base.group(state,'shared'))
            assert record['mask_hash']==(mask_hash if kind=='sparse' else 'dense_no_mask')
            assembled=base.r.assemble_router_age_model(cfg['model'],state,str(checkpoint),mask if kind=='sparse' else {},torch.device('cpu'))
            assert base.state_dict_hash(assembled.state_dict())==record['initial_model_hash'];del assembled
            rows.append({'age':age,'kind':kind,'checkpoint_sha256':record['router_checkpoint_hash'],'initial_model_hash':record['initial_model_hash'],'expert_hash':record['expert_state_hash'],'shared_hash':record['shared_state_hash'],'router_hash':record['initial_router_state_hash'],'mask_hash':record['mask_hash'],'training_sequence_hash':train_hash,'validation_sequence_hash':val_hash,'probe_sequence_hash':record['probe_sequence_hash'],'passed':True})
    pd.DataFrame(rows).to_csv(OUT/'source_integrity.csv',index=False)
    settings={'torch':torch.__version__,'cuda':torch.version.cuda,'gpu':torch.cuda.get_device_name(),'python':platform.python_version(),'deterministic_algorithms':torch.are_deterministic_algorithms_enabled(),'cudnn_deterministic':torch.backends.cudnn.deterministic,'cudnn_benchmark':torch.backends.cudnn.benchmark,'matmul_tf32':torch.backends.cuda.matmul.allow_tf32,'cudnn_tf32':torch.backends.cudnn.allow_tf32,'cublas_workspace_config':os.environ.get('CUBLAS_WORKSPACE_CONFIG'),'omp_num_threads':os.environ.get('OMP_NUM_THREADS'),'seed':7,'independent_seeds':3,'additional_statistical_replicates':0,'exact_loss_absolute_tolerance':1e-10,'relative_error_denominator_tolerance':1e-8,'dataset_hashes':{k:base.sha256_file(ROOT/cfg['data'][k]) for k in ('train_path','validation_path')},'reference_initial_sha256':base.sha256_file(initial),'reference_final_sha256':base.sha256_file(Path(cfg['output_dir'])/'checkpoints/step_2500.pt')}
    base.write_json(OUT/'preflight_settings.json',settings)
    (OUT/'preflight_audit.md').write_text('# 16E seed-7 reproducibility preflight\n\nAll 12 original cells, source checkpoints, initial model/component hashes, final artifacts, regenerated E_T masks, minibatch sequences and historical code hashes validated before rerunning. Original references and masks are reused; no reference training. Same seed, native trainable routing, fresh AdamW, fp16/GradScaler, 2500 attempts, clip=1, aux=0.1, lr=0.0003, wd=0.01, no scheduler/warmup, validation every 50, historical final-only full recovery checkpoint cadence.\n\nDeterministic algorithms are not forced by the historical protocol. CUDA reduction/kernel behavior may vary with hardware/software; TF32 is retained. This audit measures repeatability, not a guaranteed bitwise contract. Predeclared exact loss tolerance 1e-10; model hashes and all validation observations are compared separately. Qualitative reproduction requires R100 winner, rank correlation >=0.8 and maximum gap error <25% of original winner-to-runner-up separation. Errors at least as large as that separation are material nondeterminism if invariants pass. Rerun is never averaged into the original n=3 observations.\n')
    save_config(cfg,OUT/'configs/e16_seed7.yaml')
    return cfg

def summarize():
    original=pd.read_csv(ORIGINAL/'outcomes.csv');original=original[(original.seed==7)&original.router_age.isin(AGES)].set_index('router_age')
    rerun=pd.read_csv(OUT/'outcomes.csv').set_index('router_age');assert set(rerun.index)==set(AGES)
    rows=[];curve_rows=[];all_models=True
    for age in AGES:
        o=original.loc[age];n=rerun.loc[age];error=n.ticket_gap-o.ticket_gap
        rows.append({'router_age':age,'original_dense_loss':o.dense_final_loss,'rerun_dense_loss':n.dense_final_loss,'original_sparse_loss':o.sparse_final_loss,'rerun_sparse_loss':n.sparse_final_loss,'original_ticket_gap':o.ticket_gap,'rerun_ticket_gap':n.ticket_gap,'gap_error':error,'absolute_gap_difference':abs(error),'relative_gap_error':error/abs(o.ticket_gap) if abs(o.ticket_gap)>1e-8 else None})
        for kind in ('dense','sparse'):
            old=ORIGINAL/f'recovery/seed_7/R{age}/{kind}';new=OUT/f'recovery/seed_7/R{age}/{kind}'
            om=json.loads((old/'validated_result.json').read_text());nm=json.loads((new/'validated_result.json').read_text())
            for key in ('initial_model_hash','mask_hash','expert_state_hash','shared_state_hash','initial_router_state_hash','training_batch_sequence_hash','validation_batch_sequence_hash','probe_sequence_hash','optimizer','precision','recovery_eval_interval'):
                assert om[key]==nm[key],f'{age} {kind} {key} differs'
            all_models &= om['final_model_hash']==nm['final_model_hash']
            a=pd.read_json(old/'metrics.jsonl',lines=True);b=pd.read_json(new/'metrics.jsonl',lines=True);assert a.step.tolist()==b.step.tolist()
            for x,y in zip(a.itertuples(),b.itertuples()):curve_rows.append({'router_age':age,'kind':kind,'step':x.step,'original_loss':x.loss,'rerun_loss':y.loss,'error':y.loss-x.loss})
            # Complete historical per-step scalar records, including AMP skips, not just final endpoints.
            a=pd.read_json(old/'gradient_stats/gradient_stats.jsonl',lines=True);b=pd.read_json(new/'gradient_stats/gradient_stats.jsonl',lines=True);assert a.step.tolist()==b.step.tolist()
            pd.DataFrame({'step':a.step,**{key+'_equal':a[key].fillna(-999).eq(b[key].fillna(-999)) for key in ('train_loss','optimizer_step_applied','loss_scale_before','loss_scale_after')}}).to_csv(OUT/f'step_repeatability_R{age}_{kind}.csv',index=False)
    frame=pd.DataFrame(rows);frame['original_rank']=frame.original_ticket_gap.rank(method='min');frame['rerun_rank']=frame.rerun_ticket_gap.rank(method='min');frame.to_csv(OUT/'original_vs_rerun.csv',index=False)
    curves=pd.DataFrame(curve_rows);curves.to_csv(OUT/'validation_repeatability.csv',index=False)
    ow=int(frame.loc[frame.original_ticket_gap.idxmin(),'router_age']);nw=int(frame.loc[frame.rerun_ticket_gap.idxmin(),'router_age']);separation=float(frame.original_ticket_gap.nsmallest(2).diff().iloc[-1]);error=float(frame.absolute_gap_difference.max());rho=float(frame.original_rank.corr(frame.rerun_rank));maxcurve=float(curves.error.abs().max())
    if maxcurve<=1e-10 and error<=1e-10:verdict='16E SEED7 EXACTLY REPRODUCED'
    elif error>=separation:verdict='16E SEED7 NONDETERMINISM MATERIAL'
    elif nw==ow==100 and rho>=.8 and error<.25*separation:verdict='16E SEED7 QUALITATIVELY REPRODUCED'
    else:verdict='16E SEED7 ORDERING NOT REPRODUCED'
    summary={'status':'complete','original_winner':ow,'rerun_winner':nw,'original_second_best':int(frame.nsmallest(2,'original_ticket_gap').iloc[-1].router_age),'rerun_second_best':int(frame.nsmallest(2,'rerun_ticket_gap').iloc[-1].router_age),'verdict':verdict,'max_absolute_gap_error':error,'max_validation_error':maxcurve,'rank_correlation':rho,'original_winner_separation':separation,'all_final_model_hashes_equal':bool(all_models),'original_r100_minus_r60':float(original.loc[100].ticket_gap-original.loc[60].ticket_gap),'rerun_r100_minus_r60':float(rerun.loc[100].ticket_gap-rerun.loc[60].ticket_gap),'independent_seed_count':3}
    base.write_json(OUT/'reproducibility_summary.json',summary)
    (OUT/'final_report.md').write_text('# 16E seed-7 repeatability\n\n'+verdict+'\n\n```json\n'+json.dumps(summary,indent=2)+'\n```\n\n1. R100 remains the winner: '+str(nw==100)+'.\n2. R100−R60 contrasts are recorded above.\n3. R70/R80/R90 original and rerun ranks are retained in original_vs_rerun.csv.\n4. Maximum numerical error versus original winner separation is recorded above.\n5. Seed 7 differs from seeds 17/29 descriptively if its boundary preference repeats; this repeat adds no independent seed.\n6. '+('No material run-level variation observed in this one repeat.' if error<.25*separation else 'Run-level variation constrains interpretation; one repeat cannot estimate a stable variance distribution.')+'\n\nOriginal 16E observations remain untouched and primary.\n')
    base.write_json(OUT/'artifact_integrity.json',{'status':'validated','cells':12,'invalid_runs':0,'files':{str(p.relative_to(OUT)):base.sha256_file(p) for p in OUT.glob('*.csv')}})
    print(verdict,flush=True)

def main():
    if (OUT/'reproducibility_summary.json').exists():
        assert json.loads((OUT/'reproducibility_summary.json').read_text())['status']=='complete';return
    cfg=audit();base.OUT=OUT;base.SEEDS=(7,);base.config_for=lambda seed:cfg
    base.run_seed(7,AGES,'reproducibility_audit');summarize()
if __name__=='__main__':main()
