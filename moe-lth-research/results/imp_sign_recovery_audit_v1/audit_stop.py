from pathlib import Path
import csv,json,sys
ROOT=Path.cwd();OUT=ROOT/'results/imp_sign_recovery_audit_v1';OLD=ROOT/'results/imp_initial_value_placement_v1'
sys.path.insert(0,str(ROOT/'src'))
import torch
torch.set_num_threads(4)
def read(p):return json.loads(Path(p).read_text())
def rows(p):
    with Path(p).open() as f:return list(csv.DictReader(f))
def write(name,data):
    with (OUT/name).open('w') as f:
        w=csv.DictWriter(f,fieldnames=list(data[0]));w.writeheader();w.writerows(data)
manifest=rows(OLD/'variant_manifest.csv');cohort=[];provenance=[];initial=[];draws=[]
for seed in (7,17,29,43,61):
    selected=[r for r in manifest if int(r['seed'])==seed]
    r=next(r for r in selected if r['variant']=='imp_original')
    init=torch.load(r['initial_checkpoint'],map_location='cpu',weights_only=False,mmap=True)
    initial.append(dict(seed=seed,checkpoint=r['initial_checkpoint'],checkpoint_step=init['step'],num_experts=init['config']['model']['num_experts'],dropout=init['config']['model']['dropout'],config_seed=init['config']['seed'],all_weights_source='complete original step-zero checkpoint; expert survivor values replaced only for perturbations'))
    meta_path=ROOT/f'results/imp_80_85/trajectories/e32/seed_{seed}/mask_08.json';meta=read(meta_path)
    stage_path=Path(meta['stage_path'])/'stage.json';stage=read(stage_path)
    provenance.append(dict(seed=seed,num_experts=meta['num_experts'],mask_round=meta['pruning_round'],target_sparsity=meta['target_sparsity'],actual_sparsity=meta['actual_sparsity'],trained_checkpoint=meta['trained_checkpoint'],trained_checkpoint_exists=Path(meta['trained_checkpoint']).is_file(),discovery_input_sparsity=stage['sparsity'],discovery_is_dense=stage['sparsity']==0,mask_metadata=str(meta_path),stage_metadata=str(stage_path),source_file='scripts/imp_80_85/run.py',source_lines='147-153',source_quote="record = run_stage(... masks ...); trained = load_model_from_checkpoint(... dest / 'checkpoints/final_recovered.pt' ...); new = nested_expert_local_masks(trained, masks, target)"))
    for order in ('A','B'):
        group=[x for x in selected if x['order']==order]
        draws.append(dict(seed=seed,order=order,imp_shuffle_draws=len([x for x in group if x['family']=='imp' and x['placement']=='shuffled']),random_mask_realizations=len({x['realization'] for x in group if x['family']=='random'}),shuffle_draws_per_random_mask=1,requested_minimum_draws=3,full_shuffle_draw_design_matches=False))
    for r in selected:
        if r['placement']!='shuffled':continue
        status=read(OLD/r['status_path']);attempt=Path(status['attempt']);snap=attempt/'snapshots/step_0000.pt'
        payload=torch.load(snap,map_location='cpu',weights_only=False,mmap=True)
        config=payload['config'];seg0=read(attempt/'segment_0/metadata.json');seg1=read(attempt/'segment_2500/metadata.json')
        keep=meta['keep_count'];eligible=meta['eligible_count']
        cohort.append(dict(job_id=r['job_id'],seed=seed,order=r['order'],variant=r['variant'],family=r['family'],realization=r['realization'],snapshot=str(snap),snapshot_step=payload['step'],config_num_experts=config['model']['num_experts'],manifest_num_experts=int(r['num_experts']),segment0_sparsity=seg0['sparsity'],segment1_sparsity=seg1['sparsity'],imp_mask_target_sparsity=meta['target_sparsity'],imp_mask_actual_sparsity=meta['actual_sparsity'],router_checkpoint_matches_init=seg0['router_checkpoint']==r['initial_checkpoint'],router_step=seg0['router_step'],router_mode=seg0['router_mode'],optimizer=seg0['optimizer'],scheduler=seg0['scheduler'],learning_rate=config['training']['learning_rate'],weight_decay=config['training']['weight_decay'],precision=seg0['precision'],validation_hash=seg0['validation_batch_sequence_hash'],segment0_metadata=str(attempt/'segment_0/metadata.json'),segment1_metadata=str(attempt/'segment_2500/metadata.json'),verified_E32_80=config['model']['num_experts']==32 and seg0['sparsity']==seg1['sparsity']==meta['target_sparsity']==.8))
        print(r['job_id'],cohort[-1]['verified_E32_80'],flush=True)
write('initial_checkpoint_audit.csv',initial);write('full_shuffle_cohort_audit.csv',cohort);write('mask_discovery_provenance.csv',provenance);write('draw_design_audit.csv',draws)
evidence=[]
for question,answer,file,start,end in [
 ('step_zero','true original random initialization, saved before the step-1 training loop','src/moe_lth/training/train.py',142,142),
 ('checkpoint_save_point','step zero saved before model.train() and the training loop','src/moe_lth/training/train.py',166,177),
 ('router_attention_embedding_bias_source','complete original initialization state copied; bundle replaces expert matrix weights','scripts/imp_initial_value_placement_v1/run.py',108,119),
 ('router_source','router swapped from the same original initialization checkpoint','src/moe_lth/pruning/router_age.py',86,101),
 ('masked_coordinates','zero via multiplication of original weights by the binary mask, preserving signed zero','scripts/imp_initial_value_placement_v1/run.py',62,71),
 ('expert_initialization','native nn.Linear initialization; no custom expert scaling in forward','src/moe_lth/models/experts.py',7,16),
 ('shared_initialization','native embeddings/layernorm; LM head tied to token embedding','src/moe_lth/models/transformer.py',78,88)]:
 lines=(ROOT/file).read_text().splitlines()
 evidence.append(dict(question=question,answer=answer,file=file,start_line=start,end_line=end,quote='\n'.join(f'{i}: {lines[i-1]}' for i in range(start,min(end,len(lines))+1))))
write('source_evidence.csv',evidence)
write('phase_status.csv',[dict(phase='0.1',status='audited',reason='E32/80% full-shuffle snapshot configs and segment metadata inspected'),dict(phase='0.2',status='not_run_this_task',reason='Stopped on provenance/design conflicts before further audits'),dict(phase='0.3',status='partial_stop',reason='Existing full-shuffle design has 2 IMP draws and 1 draw per random mask; requested minimum is 3'),dict(phase='0.4',status='not_run',reason='Stop rule'),dict(phase='1',status='not_run',reason='Actual mask selection source is a sparse discovery round, not a dense model'),dict(phase='2',status='not_launched',reason='User approval required; no cost estimate or preregistration finalized')])
