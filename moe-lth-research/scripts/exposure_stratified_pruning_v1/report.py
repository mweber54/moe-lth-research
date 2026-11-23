"""Report every registered condition, with unavailable effects left empty."""
from common import *
import csv

def analyze():
 import matplotlib
 matplotlib.use('Agg')
 import matplotlib.pyplot as plt
 prep=read(D/'preparation_complete.json');states=[read(D/'status'/f'seed{s}_{b}.json') for s in SEEDS for b in BRANCHES];results=[];curves={};effects=[]
 for st in states:
  if st['status']=='COMPLETE':
   r=read(st['result_path']);results.append(r);curves[(r['seed'],r['branch'])]=read(Path(st['result_path']).parent/'curve.json')
  else:results.append(dict(seed=st['seed'],branch=st['branch'],status=st['status'],final_ce=None))
 table(D/'per_condition_results.csv',results)
 for seed in SEEDS:
  complete=all((seed,b) in curves for b in BRANCHES)
  for i,step in enumerate(EVAL_STEPS):
   row=dict(seed=seed,continuation_step=step,L_minus_C=None,H_minus_C=None,L_minus_H=None,status='COMPLETE' if complete else 'NOT_AVAILABLE')
   if complete:
    c,l,h=[curves[(seed,b)][i]['ce'] for b in BRANCHES];row.update(L_minus_C=l-c,H_minus_C=h-c,L_minus_H=l-h)
   effects.append(row)
 table(D/'paired_seed_effects.csv',effects)
 trajectory=[]
 for seed in SEEDS:
  for b in BRANCHES:
   measured={r['continuation_step']:r['ce'] for r in curves.get((seed,b),[])}
   for step in EVAL_STEPS:trajectory.append(dict(seed=seed,branch=b,continuation_step=step,source_training_step=1000+step,ce=measured.get(step),status='MEASURED' if step in measured else 'NOT_RUN_BALANCE_GATE' if prep['status']!='COMPLETE' else 'PENDING'))
 table(D/'learning_curves.csv',trajectory)
 if prep['status']!='COMPLETE':
  for filename in ('post_pruning_exposure_trajectories.csv','parameter_update_norms.csv','baseline_expert_function.csv'):
   table(D/filename,[dict(seed=s,branch=b,status='NOT_RUN_BALANCE_GATE',value=None) for s in SEEDS for b in BRANCHES])
 seps=list(csv.DictReader((D/'exposure_separation.csv').open()));inventory=list(csv.DictReader((D/'exposure_inventory.csv').open()));balances=list(csv.DictReader((D/'selection_and_balance.csv').open()));fig=D/'figures';fig.mkdir(exist_ok=True)
 for seed in SEEDS:
  f,axes=plt.subplots(2,2,figsize=(10,6))
  for layer,ax in enumerate(axes.flat):
   z=[r for r in inventory if int(r['seed'])==seed and int(r['layer'])==layer];ax.bar([int(r['expert']) for r in z],[int(r['cumulative_1_1000_accepted']) for r in z],color=[{'L':'tab:blue','H':'tab:orange','middle':'gray'}[r['group']] for r in z]);ax.set(title=f'Layer {layer}',xlabel='Expert',ylabel='Accepted tokens through1000')
  f.suptitle(f'Seed {seed}: historical quantity, not input diversity');f.tight_layout();f.savefig(fig/f'seed{seed}_historical_exposure.png',dpi=140);plt.close(f)
  f,ax=plt.subplots(figsize=(7,4))
  if all((seed,b) in curves for b in BRANCHES):
   for b in BRANCHES:ax.plot(EVAL_STEPS,[r['ce'] for r in curves[(seed,b)]],marker='o',label=b)
   ax.legend()
  else:ax.text(.5,.5,'No pruning losses measured: selection gate not passed' if prep['status']!='COMPLETE' else 'Continuation pending',ha='center',va='center',transform=ax.transAxes)
  ax.set(title=f'Seed {seed}',xlabel='Continuation attempted updates',ylabel='Full-validation task CE');f.tight_layout();f.savefig(fig/f'seed{seed}_loss.png',dpi=140);plt.close(f)
 allcomplete=all(r['status']=='COMPLETE' for r in states);stopped=prep['status']!='COMPLETE';final=[r for r in effects if r['continuation_step']==1500 and r['L_minus_H'] is not None]
 auc=[]
 if allcomplete:
  for seed in SEEDS:
   z=[r for r in effects if r['seed']==seed];r=dict(seed=seed)
   for key in ('L_minus_C','H_minus_C','L_minus_H'):
    a=float(np.trapz([x[key] for x in z],EVAL_STEPS));r[key+'_auc']=a;r[key+'_mean_excess_ce']=a/1500
   auc.append(r)
 table(D/'recovery_auc.csv',auc or [dict(seed=s,status='NOT_AVAILABLE',L_minus_C_auc=None,H_minus_C_auc=None,L_minus_H_auc=None) for s in SEEDS])
 lines=['# Exposure-stratified pruning experiment v1','',f'Status: {"STOPPED_BALANCE_GATE" if stopped else "COMPLETE" if allcomplete else "RUNNING"}.','', 'Historical exposure is cumulative accepted tokens through1000, within layer. It measures quantity, not relevant input coverage. Gate-weighted exposure and historical diversity are unavailable. Calibration sensitivity arrays are reused from the verified native training-only task-gradient measurement; no gradient scoring was repeated.','', '| Seed | Layer0 H/L exposure | Layer1 | Layer2 | Layer3 | Matching pairs passed /64 |','|---|---:|---:|---:|---:|---:|']
 for seed in SEEDS:
  ratios=[float(next(r['ratio'] for r in seps if int(r['seed'])==seed and int(r['layer'])==l)) for l in range(4)];passed=sum(r['passed']=='True' for r in balances if int(r['seed'])==seed);lines.append('| '+str(seed)+' | '+' | '.join(f'{x:.3f}' for x in ratios)+f' | {passed}/64 |')
 if stopped:
  failed=[r for r in balances if r['passed']!='True'];lines+=['','The frozen approximate matching design did not establish adequate five-seed coverage. It was stopped before any pruning loss, mask application to a model, engineering update or production continuation. Candidate coordinate files are retained for audit and are not approved intervention masks. No tolerance or dose was relaxed.','',f'{len(failed)}/320 paired tensor selections failed one or more fixed gates. Full counts, magnitude/score distributions, energy balance and residual history/structure are in the CSVs.','', 'This failure applies to the fixed exposure-rank expert pairing and single-pass nearest16 generator with its joint calipers; it is not a proof that all cross-expert approximate designs lack overlap. The intended pruning comparison remains unanswered. Exposure variation exists, but that alone does not validate exposure-aware protection or show that waiting makes a weight useful. Baseline expert function and post-pruning trajectories were not measured because the pre-loss selection gate failed. No method development is justified by this feasibility result.']
  if (D/'failed_balance_gates.csv').exists():
   failed_rows=list(csv.DictReader((D/'failed_balance_gates.csv').open()));counts={}
   for row in failed_rows:
    for gate in row.get('failed_gates','').split(';'):
     if gate:counts[gate]=counts.get(gate,0)+1
   lines+=['','Failed-gate counts (a pair can fail multiple): '+', '.join(f'{k}={v}' for k,v in sorted(counts.items()))+'.']
  if (D/'pair_support_diagnostics.csv').exists():
   support=list(csv.DictReader((D/'pair_support_diagnostics.csv').open()));n=sum(r['joint_magnitude_calipers_incompatible']=='True' for r in support);lines+=['',f'{n}/320 fixed expert pairs have RMS differences incompatible with the simultaneous raw/normalized magnitude calipers: feasibility requires absolute log RMS ratio <=0.105. This rules out pairs only for those fixed expert pairings and calipers. Other incomplete yields are single-pass heuristic results, not global infeasibility certificates. These constraints were not relaxed. Normalized-magnitude calipers and balance were additional registered requirements, so this failure does not establish that raw-magnitude/sensitivity matching alone is infeasible. For the singleton candidate, SMD=1e99 is a zero-variance unequal-mean failure sentinel, not an interpretable standardized effect; that pair already fails dose.']
 else:
  lines+=['','| Seed | Immediate L-H | Final L-C | Final H-C | Final L-H (primary) |','|---|---:|---:|---:|---:|']
  for seed in SEEDS:
   z=[r for r in effects if r['seed']==seed];i=z[0];r=z[-1];lines.append(f"| {seed} | "+' | '.join('NA' if v is None else f'{v:.8f}' for v in (i['L_minus_H'],r['L_minus_C'],r['H_minus_C'],r['L_minus_H']))+' |')
  if allcomplete:
   x=np.array([r['L_minus_H'] for r in final]);lines+=['',f'Final L-H: mean {x.mean():.8f}, median {np.median(x):.8f}, range {x.min():.8f} to {x.max():.8f}, sample SD {x.std(ddof=1):.8f}, SE {x.std(ddof=1)/np.sqrt(5):.8f}; positive in {sum(x>0)}/5 seeds. Immediate and integrated effects remain separate in paired_seed_effects.csv and recovery_auc.csv.','', 'The intervention establishes only the consequences of removing these compound selected groups under the shared fresh reset and fixed1500-attempt budget. Historical exposure is not randomized. Residual expert function, structure, growth and input coverage may contribute. Post-pruning routing changes are responses and do not independently establish compensation mechanisms. No coordinate/batch pseudoreplication or equivalence inference is used. Waiting for exposure and compute savings remain untested; no pruning algorithm is implemented.']
 (D/'findings.md').write_text('\n'.join(lines)+'\n')
 attempted=sum(r.get('attempted_updates',0) for r in results);save(D/'compute_accounting.json',dict(planned_production_attempts=22500,completed_production_attempts=attempted,completed_production_branches=sum(st['status']=='COMPLETE' for st in states),gradient_scores_recomputed=0,reused_calibration_batches=32*5,selection_cpu_wall_seconds=prep['wall_seconds'],production_wall_seconds_sum=sum(r.get('wall_seconds',0) for r in results),diagnostic_cost='per-attempt diagnostics_timing.json; engineering validation separate' if not stopped else '0 model operations; array-only selection',selection_verification_cpu_wall_seconds=read(D/'numerical_checks.json')['wall_seconds'] if (D/'numerical_checks.json').exists() else None,GPU_model_operations=0 if stopped else None,failed_array_verification_attempts=1 if (D/'verification_attempt1.log').exists() else 0,failed_array_verification_wall_seconds=None,failed_array_verification_cost_note='First verifier failed on a singleton SMD convention; wall/CPU timing was not captured. Preserved log, no model operations.',engineering_training_attempts=0 if stopped else 'per-host numerical_checks validation count',intervention_losses=0 if stopped else 'full validation per completed branch at7 snapshots plus baseline'))
 if stopped or allcomplete:save(D/'completion.json',dict(status='STOPPED_BALANCE_GATE' if stopped else 'COMPLETE',intended_intervention_complete=allcomplete,planned_conditions=15,completed_conditions=sum(st['status']=='COMPLETE' for st in states),production_attempts=attempted,all_five_seeds_required=True,coordinate_selections_approved=not stopped))
if __name__=='__main__':analyze()
