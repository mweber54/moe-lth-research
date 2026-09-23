"""Dataset logic tests; no training or recovery runs."""
import sys,tempfile,json
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts/fixed_budget_dataset'))
import common,finalize

def test_protocol_rejects_budget_and_pairing_drift():
 c=common.config(32,43);assert not common.protocol_errors(c,32,43)
 c['training']['steps']=10000;assert 'training.steps=10000' in common.protocol_errors(c,32,43)
 c=common.config(32,43);c['seed']=61;assert 'seed' in common.protocol_errors(c,32,43)
 c=common.config(32,43);c['training']['learning_rate']=.001;assert common.protocol_errors(c,32,43)

def test_exposure_distinguishes_tokens_attempts_and_applied_updates():
 with tempfile.TemporaryDirectory() as tmp:
  old=finalize.OUT;finalize.OUT=Path(tmp)
  try:
   path=Path(tmp)/'counts.jsonl';pro={f'l{l}e{e}':2048 for l in range(4) for e in range(8)};acc={k:1024 for k in pro}
   with path.open('w') as f:
    for step in range(1,2501):f.write(json.dumps({'step':step,'proposed':pro,'accepted':acc,'optimizer_step_applied':step!=1})+'\n')
   result=finalize.exposures(path,True,8)
   assert result['proposed_assignments']==163840000 and result['accepted_assignments']==81920000
   assert result['mean_active_attempt_steps_per_expert']==2500
   assert result['mean_active_applied_steps_per_expert']==2499
   assert result['acceptance_rate']==.5
   legacy=Path(tmp)/'legacy.jsonl'
   with legacy.open('w') as f:
    for step in range(1,2501):f.write(json.dumps({'step':step,'proposed':pro,'accepted':acc})+'\n')
   oldresult=finalize.exposures(legacy,True,8)
   assert oldresult['mean_active_applied_steps_per_expert'] is None and oldresult['applied_steps'] is None
   assert oldresult['mean_active_attempt_steps_per_expert']==2500
  finally:finalize.OUT=old


def test_tied_winners_keep_all_ages_and_pairings():
 import pandas as pd
 import numpy as np
 with tempfile.TemporaryDirectory() as tmp:
  old=finalize.OUT;finalize.OUT=Path(tmp)
  try:
   rows=[]
   for i,seed in enumerate(finalize.SEEDS):
    for age in finalize.AGES:
     for sp in finalize.SP:
      gap=i+sp+(0 if age in (0,10) else age/100)
      rows.append({'num_experts':8,'seed':seed,'router_age':age,'sparsity':sp,'ticket_gap':gap,'sparse_final_validation_loss':3+gap,'dense_final_validation_loss':3.,'pruning_shock':abs(age-10)/100,'matched_early_auc_gap':gap*100})
   frame=pd.DataFrame(rows);finalize.summarize_and_plot(frame)
   best=pd.read_csv(finalize.OUT/'best_age_by_architecture_sparsity.csv')
   assert best.best_age.isna().all() and (best.winning_ages=='R0 | R10').all()
   assert (best.tied_best_ages=='[0, 10]').all() and (best.gap_to_second_ranked_age==0).all()
   assert (best.number_seeds_selecting_mean_winner==5).all() and (best.number_seeds_with_identical_winner_set==5).all()
   for name in ['architecture_sparsity_winner_matrix.csv']+[f'architecture_sparsity_winner_matrix_seed{seed}.csv' for seed in finalize.SEEDS]:
    matrix=pd.read_csv(finalize.OUT/name);assert (matrix.iloc[:,1:]=='R0 | R10').all().all()
   decomposition=pd.read_csv(finalize.OUT/'dense_sparse_boundary_decomposition.csv')
   assert len(decomposition)==120 and set(decomposition.candidate_age)=={0,10}
   assert np.allclose(decomposition.ticket_gap_difference,decomposition.sparse_loss_difference-decomposition.dense_loss_difference)
   endpoints=pd.read_csv(finalize.OUT/'endpoint_winner_comparison.csv')
   assert endpoints.dense_final_validation_loss_winner.isna().all() and not endpoints.pruning_shock_winner_disagrees.any()
   marginal=pd.read_csv(finalize.OUT/'summary_by_architecture_age.csv')
   assert np.allclose(marginal.ticket_gap_sd,np.sqrt(2.5)) and (marginal.n_seeds==5).all()
   assert 'R0 | R10' in finalize.descriptive_report(frame,best,endpoints)
  finally:finalize.OUT=old

if __name__=='__main__':
 test_protocol_rejects_budget_and_pairing_drift();test_exposure_distinguishes_tokens_attempts_and_applied_updates();test_tied_winners_keep_all_ages_and_pairings();print('Dataset logic checks PASS; no training executed')
