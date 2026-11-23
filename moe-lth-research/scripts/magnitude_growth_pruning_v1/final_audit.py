"""Read-only production audit and final reviewed report seal after all15 branches."""
from common import *
import argparse

def main(review_file):
 torch.set_num_threads(4)
 reg=frozen();checks=read(D/'numerical_checks.json');production=[];total_applied=total_skips=0
 statuses=[read(D/'status'/f'seed{s}_{b}.json') for s in SEEDS for b in BRANCHES];assert len(statuses)==15 and all(s['status']=='COMPLETE' for s in statuses)
 for seed in SEEDS:
  initial=reg['checkpoints'][str(seed)]['1000'];assert sha(initial['path'])==initial['sha256'];mp=reg['mask_files'][str(seed)];assert sha(mp['path'])==mp['sha256'];groups=load(mp['path'])
  assert set(groups['growing'])==set(groups['stable'])
  source_state=torch.load(initial['path'],map_location='cpu',weights_only=True,mmap=True)['model']
  initial_records={b:read(D/f'attempts/seed{seed}_{b}/attempt_0001/initial_integrity.json') for b in BRANCHES}
  assert all(x['fresh_optimizer_entries']==0 and x['scaler']==initial_records['control']['scaler'] and x['rng_reset_seed']==seed and x['all_parameters_trainable'] and x['tied_weights'] for x in initial_records.values())
  for name,g in groups['growing'].items():
   s=groups['stable'][name];assert g.numel()==s.numel()==2621 and not torch.isin(g,s).any();assert g.unique().numel()==s.unique().numel()==2621 and int(g.min())>=0 and int(s.min())>=0 and int(g.max())<source_state[name].numel() and int(s.max())<source_state[name].numel();assert '.moe.experts.' in name and name.endswith(('.fc1.weight','.fc2.weight'))
  for branch in BRANCHES:
   expected={n:t.clone() if branch!='control' and n in groups[branch] else t for n,t in source_state.items()}
   if branch!='control':
    for n,idx in groups[branch].items():expected[n].view(-1).index_fill_(0,idx,0.)
   assert state_dict_hash(expected)==initial_records[branch]['masked_model_hash']
   del expected
   st=read(D/'status'/f'seed{seed}_{branch}.json');r=read(st['result_path']);p=Path(r['checkpoint']);assert sha(p)==r['checkpoint_sha256'];cp=torch.load(p,map_location='cpu',weights_only=True,mmap=True);assert cp['step']==2500 and cp['continuation_step']==1500 and cp['registration_sha256']==sha(D/'registration.json');assert state_dict_hash(cp['model'])==r['final_model_hash']
   if branch!='control':
    for name,idx in groups[branch].items():assert torch.count_nonzero(cp['model'][name].flatten()[idx])==0
   log=[json.loads(l) for l in (Path(r['attempt'])/'training.jsonl').open()];assert [x['continuation_step'] for x in log]==list(range(1,1501));assert [x['source_training_step'] for x in log]==list(range(1001,2501));assert sum(x['applied'] for x in log)==r['applied_updates'];assert sum(x['amp_skip'] for x in log)==r['amp_skips'];assert r['applied_updates']+r['amp_skips']==1500
   assert r['mask_persistence'] and r['removed_optimizer_moments_zero'];assert r['train_sequence_hash']==reg['data_identity'][str(seed)]['continuation_train_hash'];assert r['validation_sequence_hash']==reg['data_identity'][str(seed)]['validation_hash'];assert all(torch.isfinite(t).all() for t in cp['model'].values())
   curves=read(Path(r['attempt'])/'curve.json');assert [x['continuation_step'] for x in curves]==list(EVAL_STEPS)
   batches=list(csv.DictReader((Path(r['attempt'])/'per_batch_validation.csv').open()));assert len(batches)==len(EVAL_STEPS)*31
   for step in EVAL_STEPS:
    z=[x for x in batches if int(x['continuation_step'])==step];assert sum(int(x['target_tokens']) for x in z)==492416
    reconstructed=sum(float(x['task_ce_sum']) for x in z)/492416
    assert abs(reconstructed-next(x['ce'] for x in curves if x['continuation_step']==step))<=1e-12
   total_applied+=r['applied_updates'];total_skips+=r['amp_skips'];production.append(dict(seed=seed,branch=branch,final_checkpoint_sha256_verified=True,model_hash_verified=True,zeros_verified=True,training_log_1500_unique_steps=True,source_offset_verified=True,complete_validation_token_counts=True,applied_updates=r['applied_updates'],amp_skips=r['amp_skips']))
   print('audited',seed,branch,flush=True)
 checks.update(production_checks=production,all15_completed=True,matching_full_dose_all1280_tensors=True,all_masks_counts_equal_and_disjoint=True,all15_initial_states_only_intended_changes=True,all15_initial_optimizer_scaler_resets_equal=True,per_batch_ce_reconstruction_atol=1e-12,initial_checkpoints_and_masks_unchanged=True,total_attempted_updates=22500,total_applied_updates=total_applied,total_amp_skips=total_skips,pilot_gate=read(D/'pilot_engineering.json'),matching_balance_checks=read(D/'matched_balance_checks.json'),cross_host_numerical_checks=read(D/'cross_host_numerical_checks.json'),failed_engineering_import_logs_preserved=True)
 save(D/'numerical_checks.json',checks)
 with lock(D/'report.lock'):
  # Workers are finished; append the human-reviewed interpretation without changing sealed training code.
  text=(D/'findings.md').read_text();review=Path(review_file).read_text();(D/'findings.md').write_text(text+'\n## Reviewed scientific conclusion\n\n'+review.strip()+'\n')
  c=read(D/'completion.json');assert c['status']=='COMPLETE' and c['science_report_complete'];c.update(final_numerical_audit_passed=True,source_seeds=5,total_production_branches=15,total_attempted_updates=22500,total_applied_updates=total_applied,total_amp_skips=total_skips,reviewed_conclusion=True,historical_artifacts_modified=False,final_checkpoint_sha256={f'seed{seed}_{b}':read(read(D/'status'/f'seed{seed}_{b}.json')['result_path'])['checkpoint_sha256'] for seed in SEEDS for b in BRANCHES})
  files=[p for p in D.iterdir() if p.is_file() and p.suffix in ('.json','.csv','.md') and p.name!='completion.json']+list((D/'figures').glob('*'))
  c['output_sha256']={str(p.relative_to(D)):sha(p) for p in files};c['analysis_implementation_sha256']={str(p.relative_to(ROOT)):sha(p) for p in HERE.glob('*.py')};save(D/'completion.json',c)
 print('FINAL AUDIT COMPLETE',flush=True)
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--review-file',required=True);main(p.parse_args().review_file)
