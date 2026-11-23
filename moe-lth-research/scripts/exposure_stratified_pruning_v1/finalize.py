"""Seal completed feasibility report; never changes thresholds or candidates."""
from common import *
import csv

def main():
 prep=read(D/'preparation_complete.json');assert prep['status']=='STOPPED_BALANCE_GATE';checks=read(D/'numerical_checks.json');assert checks['array_and_provenance_checks_passed'] and checks['stop_policy_respected'] and checks['model_forward_backward_or_optimizer_calls']==0
 assert sha(D/'design.json')==read(D/'design_seal.json')['design_sha256'];assert sha(HERE/'prepare.py')==read(D/'design_seal.json')['prepare_sha256']
 assert read(D/'completion.json')['completed_conditions']==0 and read(D/'completion.json')['production_attempts']==0
 rows=list(csv.DictReader((D/'paired_seed_effects.csv').open()));assert len(rows)==35 and all(r['L_minus_H']==r['L_minus_C']==r['H_minus_C']=='' for r in rows)
 rows=list(csv.DictReader((D/'learning_curves.csv').open()));assert len(rows)==105 and all(r['ce']=='' for r in rows)
 assert not read(D/'coordinate_seal.json')['approved_for_intervention'];assert not (D/'launch.json').exists()
 for p,h in read(D/'prior_artifact_hashes.json').items():assert sha(ROOT/p)==h
 for seed in SEEDS:
  p=D/'masks'/f'seed{seed}.pt';assert not (p.stat().st_mode & 0o222);assert sha(p)==registry()['mask_files'][str(seed)]['sha256']
 assert len(list((D/'figures').glob('seed*_historical_exposure.png')))==5 and len(list((D/'figures').glob('seed*_loss.png')))==5
 required=['protocol.md','exposure_inventory.csv','selection_and_balance.csv','coordinate_seal.json','numerical_checks.json','per_condition_results.csv','learning_curves.csv','paired_seed_effects.csv','findings.md','compute_accounting.json']
 assert all((D/p).exists() for p in required)
 files={str(p.relative_to(ROOT)):sha(p) for p in D.rglob('*') if p.is_file() and p.name!='final_artifact_audit.json' and '__pycache__' not in p.parts};implementation={str(p.relative_to(ROOT)):sha(p) for p in HERE.glob('*.py')}
 save(D/'final_artifact_audit.json',dict(passed=True,status='STOPPED_BALANCE_GATE',intended_pruning_question_answered=False,all_five_seeds_evaluated=True,all15_conditions_reported_unavailable=True,loss_values_empty_not_zero=True,original_artifacts_unchanged=True,coordinate_files_readonly_and_unapproved=True,production_training_attempts=0,intervention_loss_measurements=0,model_operations=0,gradient_scores_recomputed=0,used_score_files_verified=checks['verified_used_score_files'],required_deliverables=required,artifact_hashes=files,implementation_hashes=implementation))
 print('FINAL ARTIFACT AUDIT PASSED',flush=True)
if __name__=='__main__':main()
