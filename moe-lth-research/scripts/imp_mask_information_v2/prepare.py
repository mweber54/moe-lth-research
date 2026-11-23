"""Verify the five fixed seeds and reuse checks; generate only matrix controls."""
from v2_common import *
import statistics
from moe_lth.models import TinyMoELanguageModel
from moe_lth.pruning.mask_information import randomize,assert_control
from moe_lth.pruning.masks import save_masks


def compatible_config(a,b):
    for group,keys in [('model',list(a['model'])),('data',list(a['data'])),('routing',['mode','aux_loss_weight']),('training',['batch_size','learning_rate','weight_decay','grad_clip','precision','data_seed'])]:
        for k in keys:
            if a[group].get(k)!=b[group].get(k):raise ValueError(f'Protocol mismatch {group}.{k}')
    if a['seed']!=b['seed']:raise ValueError('Checkpoint seed mismatch')


def main():
    torch.set_num_threads(4)
    if (OUT/'manifest.json').exists():raise RuntimeError('Manifest exists; preserve it')
    original_plan=read(OLD/'pilot_plan.json')
    for name,h in original_plan['source_hashes'].items():
        if sha(ROOT/name)!=h:raise RuntimeError('Canonical dependency differs from v1: '+name)
    if sha(OUT/'protocol.md')!=read(OUT/'registration.json')['protocol_sha256']:raise RuntimeError('Registration mismatch')
    inventory=rows(OLD/'checkpoint_inventory.csv')
    cohorts={};jobs=[];reuse=[];validation=[]
    pilot=rows(OLD/'pilot_results.csv')
    for seed in SEEDS:
        rr=[r for r in inventory if r['family']=='IMP' and r['num_experts']=='16' and r['seed']==str(seed) and r['configuration_id'].endswith('final_s0.80')]
        if len(rr)!=1 or rr[0]['eligible']!='True':raise RuntimeError(f'Ineligible seed {seed}; no exclusion/substitution')
        r=rr[0]
        for path,h in [(r['rewind_source'],r['rewind_sha256']),(r['recovery_checkpoint'],r['recovery_sha256']),(r['dense_checkpoint'],r['dense_sha256'])]:
            if sha(path)!=h:raise RuntimeError('Source hash mismatch '+path)
        init=torch.load(r['rewind_source'],map_location='cpu',weights_only=True)
        dense=torch.load(r['dense_checkpoint'],map_location='cpu',weights_only=True)
        cfg=init['config'];compatible_config(cfg,dense['config'])
        if init['step']!=0 or dense['step']!=2500:raise RuntimeError('Wrong initialization/dense budget')
        if abs(float(dense['validation_loss'])-float(r['dense_comparison_ce']))>1e-8:raise RuntimeError('Dense endpoint mismatch')
        if canonical().common.protocol_errors(cfg,16,seed):raise RuntimeError('Canonical configuration mismatch')
        masks=load_masks(r['mask_source']);model=TinyMoELanguageModel(cfg['model']);validate_masks(model,masks)
        if recovery._mask_hash(masks)!=r['mask_hash']:raise RuntimeError('Original mask mismatch')
        reference=Path(r['rewind_source']).parent/'step_2500.pt'
        if not reference.is_file():raise FileNotFoundError(reference)
        cohort=dict(r,config=cfg,diagnostic_reference=str(reference),diagnostic_reference_sha256=sha(reference),mask_sha256=sha(r['mask_source']),initial_masked_hashes=validate_coordinates(init['model'],masks))
        cohorts[str(seed)]=cohort
        print(f'Verified seed {seed}',flush=True)
        for condition,replicate in [('IMP',0)]+[('matrix',i) for i in range(5)]+[('degree',i) for i in range(5)]:
            jobid=f'e16_seed{seed}_{condition}_{replicate}'
            mask_seed=None if condition=='IMP' else (810200+replicate if seed==17 and condition=='matrix' else (9200000 if condition=='matrix' else 9300000)+seed*100+replicate)
            mask_path=r['mask_source'] if condition=='IMP' else str(OUT/'masks'/f'{jobid}.pt')
            job=dict(id=jobid,phase=2 if condition=='degree' else 1,seed=seed,condition=condition,replicate=replicate,mask_seed=mask_seed,mask_path=mask_path,mask_hash=r['mask_hash'] if condition=='IMP' else None,dependencies=[] if condition=='IMP' else [f'e16_seed{seed}_IMP_0'])
            if condition=='matrix' and seed!=17:
                control=randomize(model,masks,'matrix',mask_seed);assert_control(model,masks,control,'matrix')
                save_masks(control,mask_path);saved=load_masks(mask_path)
                if not all(torch.equal(control[n],saved[n]) for n in control):raise RuntimeError('Mask persistence failed')
                job['mask_hash']=recovery._mask_hash(control);job['mask_sha256']=sha(mask_path)
                for n in sorted(masks):validation.append(dict(job=jobid,matrix=n,original_keep=int(masks[n].sum()),control_keep=int(control[n].sum()),changed_bits=int(torch.count_nonzero(masks[n]!=control[n])),passed=True))
            if seed==17 and condition!='degree':
                pr=[x for x in pilot if x['condition']==condition and int(x['replicate'])==replicate]
                if len(pr)!=1:raise RuntimeError('Missing pilot reuse')
                pr=pr[0];stage_path=Path(pr['stage_metadata']);stage=read(stage_path)
                mask_path=pr['mask_path'];reused_masks=load_masks(mask_path);validate_masks(model,reused_masks)
                if condition=='matrix':assert_control(model,masks,reused_masks,'matrix')
                expected_mask=recovery._mask_hash(reused_masks)
                checks=dict(mask=expected_mask==pr['mask_hash']==stage['mask_hash'],rewind=stage['original_initialization_checkpoint']==r['rewind_source'],identity=stage['identity']['rewind_sha256']==r['rewind_sha256'],attempts=stage['optimizer_step_attempts']==2500,optimizer=stage['optimizer']=='fresh_AdamW' and stage['optimizer_state_entries_at_start']==0,scheduler=stage['scheduler']=='none',precision=stage['precision']=='fp16',train_data=stage['training_batch_sequence_hash']==r['training_batch_hash'],validation_data=stage['validation_batch_sequence_hash']==r['validation_batch_hash'],trainability=stage['router_mode']=='trainable' and stage['integrity_checks_passed'],endpoint=abs(float(pr['final_ce'])-stage['final_validation_loss'])<1e-10,checkpoint=sha(stage_path.parent/'checkpoints/final_recovered.pt')==stage['checkpoint_sha256'],curve=(stage_path.parent/'metrics.jsonl').is_file(),initialization=validate_coordinates(init['model'],reused_masks)=={g:stage[g+'_state_hash' if g!='router' else 'initial_router_state_hash'] for g in ('expert','shared','router')})
                if not all(checks.values()):raise RuntimeError(f'Incompatible reuse {jobid}: {checks}')
                job.update(mask_path=mask_path,mask_hash=expected_mask,mask_sha256=sha(mask_path),reused_stage=str(stage_path))
                output=dict(job=jobid,seed=seed,condition=condition,replicate=replicate,mask_seed=mask_seed,mask_path=mask_path,mask_hash=expected_mask,reused=True,stage_path=str(stage_path),curves_path=str(stage_path.parent/'metrics.jsonl'),final_ce=stage['final_validation_loss'],initial_ce=stage['initial_validation_loss'],training_ce=stage['final_training_ce'],historical_dense_ce=float(r['dense_comparison_ce']),historical_imp_ce=float(r['original_recovery_ce']),attempted_updates=stage['optimizer_step_attempts'],applied_updates=stage['optimizer_steps_applied'],amp_skips=stage['amp_overflow_step_count'],duration_seconds=stage['duration_seconds'],checkpoint_sha256=stage['checkpoint_sha256'],initialization_hashes=validate_coordinates(init['model'],reused_masks))
                save(OUT/'reused'/f'{jobid}.json',output)
                save(OUT/'status'/f'{jobid}.json',dict(status='REUSED',result_path=str(OUT/'reused'/f'{jobid}.json'),source=str(stage_path),updated=time.time()))
                reuse.append(dict(job=jobid,checks=checks))
                if condition=='matrix':
                    for n in sorted(masks):validation.append(dict(job=jobid,matrix=n,original_keep=int(masks[n].sum()),control_keep=int(reused_masks[n].sum()),changed_bits=int(torch.count_nonzero(masks[n]!=reused_masks[n])),passed=True))
            else:save(OUT/'status'/f'{jobid}.json',dict(status='CONDITIONAL' if condition=='degree' else 'PENDING',updated=time.time()))
            jobs.append(job)
        del init,dense,model,masks
    table(OUT/'mask_validation.csv',validation)
    save(OUT/'reuse_checks.json',reuse)
    save(OUT/'manifest.json',dict(protocol_sha256=sha(OUT/'protocol.md'),cohorts=cohorts,jobs=jobs,phase2_gate='all five per-seed mean(matrix)-IMP > 0.02',scope='E16/80% only',phase1_new=24,phase1_reused=6,phase2_conditional=25))
    # Seal implementation only after tests and all workflow files are present.
    durations=[float(r['duration_seconds']) for r in inventory if r['family']=='IMP' and r['num_experts']=='16' and r['configuration_id'].endswith('final_s0.80')]
    save(OUT/'resource_estimate.json',dict(phase1_new_recoveries=24,phase1_gpu_hours_median=24*statistics.median(durations)/3600,phase2_if_admitted_recoveries=25,phase2_gpu_hours_median=25*statistics.median(durations)/3600,observed_seconds=durations,limitations='Throughput depends on shared GPU contention; degree mixing adds CPU work. Wall/GPU occupancy is not billed cost.'))
    print('Manifest frozen: 6 reused, 24 new phase1, 25 conditional phase2',flush=True)

if __name__=='__main__':main()
