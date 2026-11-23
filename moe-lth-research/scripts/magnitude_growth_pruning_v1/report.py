"""All registered branches retained; source seeds are the replication unit."""
from common import *
import statistics

def analyze():
 rows=[];curves=[];effects=[]
 for seed in SEEDS:
  results={};series={}
  for branch in BRANCHES:
   job=f'seed{seed}_{branch}';st=read(D/'status'/f'{job}.json');r=dict(seed=seed,branch=branch,status=st['status'],error=st.get('error',''))
   if st.get('result_path') and Path(st['result_path']).exists():
    result=read(st['result_path']);results[branch]=result;r.update(result)
   if st.get('attempt') and (Path(st['attempt'])/'curve.json').exists():
    seq=read(Path(st['attempt'])/'curve.json');series[branch]={x['continuation_step']:x['ce'] for x in seq}
    for x in seq:curves.append(dict(seed=seed,branch=branch,**x))
   rows.append(r)
  if len(results)==3:
   initial=[x['baseline_ce'] for x in results.values()];assert max(initial)-min(initial)<=1e-7,'Cross-branch unmasked baseline mismatch'
   e=dict(seed=seed,D_final_growing_minus_stable=results['growing']['final_ce']-results['stable']['final_ce'],D_immediate_growing_minus_stable=results['growing']['immediate_ce']-results['stable']['immediate_ce'],growing_final_gap_control=results['growing']['final_ce']-results['control']['final_ce'],stable_final_gap_control=results['stable']['final_ce']-results['control']['final_ce'],growing_immediate_damage=results['growing']['immediate_ce']-results['control']['immediate_ce'],stable_immediate_damage=results['stable']['immediate_ce']-results['control']['immediate_ce'])
   for b in ('growing','stable'):
    xs=list(EVAL_STEPS);ys=[series[b][x]-series['control'][x] for x in xs];auc=sum((xs[i+1]-xs[i])*(ys[i+1]+ys[i])/2 for i in range(len(xs)-1));e[b+'_excess_loss_auc']=auc;e[b+'_mean_excess_ce']=auc/1500
   effects.append(e)
 table(D/'per_branch_results.csv',rows)
 if curves:
  for r in curves:
   ref=next((x['ce'] for x in curves if (x['seed'],x['branch'],x['continuation_step'])==(r['seed'],'control',r['continuation_step'])),None);r['excess_ce_control']=r['ce']-ref if ref is not None else None
  table(D/'learning_curves.csv',curves)
 if effects:table(D/'paired_seed_effects.csv',effects)
 complete=len(effects)==5
 save(D/'completion.json',dict(status='COMPLETE' if complete else 'RUNNING',registered_branches=15,complete_branches=sum(r['status']=='COMPLETE' for r in rows),failed_branches=sum(r['status']=='FAILED' for r in rows),complete_seeds=len(effects),registration_sha256=sha(D/'registration.json'),updated=time.time(),science_report_complete=False))
 if complete:render(rows,curves,effects)
 else:
  (D/'findings.md').write_text('# Magnitude growth intervention\n\nExecution is incomplete. No final scientific verdict is available. All fifteen registered branches, including failures, are retained in per_branch_results.csv. Engineering gates do not use effect signs.\n')


def render(rows,curves,effects):
 import matplotlib
 matplotlib.use('Agg')
 import matplotlib.pyplot as plt
 F=D/'figures';F.mkdir(exist_ok=True)
 balances=list(csv.DictReader((D/'magnitude_balance.csv').open()));groups=list(csv.DictReader((D/'matched_group_summary.csv').open()));struct=list(csv.DictReader((D/'structural_balance.csv').open()))
 def figsave(name):plt.tight_layout();plt.savefig(F/(name+'.png'),dpi=160);plt.savefig(F/(name+'.pdf'));plt.close()
 fig,axes=plt.subplots(1,2,figsize=(10,4))
 for seed in SEEDS:
  z=[r for r in balances if int(r['seed'])==seed];axes[0].scatter([float(r['stable_mean_magnitude']) for r in z],[float(r['growing_mean_magnitude']) for r in z],s=9,label=str(seed),alpha=.6);axes[1].hist([float(r['worst_mismatch_rms']) for r in z],bins=np.linspace(0,.005,30),alpha=.4,label=str(seed))
 lim=max(float(r['growing_mean_magnitude']) for r in balances);axes[0].plot([0,lim],[0,lim],color='black',lw=1);axes[0].set(xlabel='Stable removed mean |w| per tensor',ylabel='Growing removed mean |w| per tensor');axes[1].set(xlabel='Worst paired mismatch / tensor RMS',ylabel='Tensors');axes[0].legend(title='Source seed',fontsize=7);figsave('current_magnitude_balance')
 fig,axes=plt.subplots(1,5,figsize=(15,3.4),sharey=True)
 for ax,seed in zip(axes,SEEDS):
  for branch in ('growing','stable'):
   z=[r for r in groups if int(r['seed'])==seed and r['branch']==branch];ax.plot([250,500,1000],[statistics.mean(float(r[f'mean_abs{st}']) for r in z) for st in (250,500,1000)],marker='o',label=branch)
  ax.set(title=f'Seed {seed}',xlabel='Source training step');ax.legend(fontsize=7)
 axes[0].set_ylabel('Mean magnitude of removed weights');figsave('historical_magnitude_trajectories')
 fig,axes=plt.subplots(2,5,figsize=(16,6),sharex=True)
 for col,seed in enumerate(SEEDS):
  for branch in BRANCHES:
   z=[r for r in curves if r['seed']==seed and r['branch']==branch];axes[0,col].plot([r['continuation_step'] for r in z],[r['ce'] for r in z],label=branch);axes[1,col].plot([r['continuation_step'] for r in z],[r['excess_ce_control'] for r in z],label=branch)
  axes[0,col].set_title(f'Seed {seed}');axes[1,col].set_xlabel('Attempted continuation step');axes[1,col].axhline(0,color='black',lw=.5)
 axes[0,0].set_ylabel('Task validation CE');axes[1,0].set_ylabel('Excess CE over new control');axes[0,0].legend(fontsize=7);figsave('validation_and_excess_loss')
 fig,axes=plt.subplots(1,2,figsize=(10,4))
 for ax,key,title in zip(axes,['D_immediate_growing_minus_stable','D_final_growing_minus_stable'],['Immediate paired effect','Final paired effect']):
  ax.bar([str(e['seed']) for e in effects],[e[key] for e in effects]);ax.axhline(0,color='black',lw=1);ax.set(title=title,xlabel='Source seed',ylabel='CE growing minus stable')
 figsave('immediate_and_final_paired_effects')
 vals=[e['D_final_growing_minus_stable'] for e in effects];immediate=[e['D_immediate_growing_minus_stable'] for e in effects];mean=statistics.mean(vals);sd=statistics.stdev(vals);reg=registry();maxdiff=max(float(r['worst_mismatch_rms']) for r in balances);maxsmd=max(abs(float(r['standardized_mean_difference'])) for r in balances);energy=[float(r['squared_magnitude_relative_difference']) for r in balances]
 lines=['# Magnitude growth pruning intervention v1','','Prospective intervention motivated by retrospective results; the overall project was not preregistered. All five original unpruned 32E source seeds were branched at step 1000. History 250/500/1000 alone determined matching. All masks were sealed before any loss measurement. All three branches used the same fresh AdamW/scaler/RNG reset because source optimizer/scaler/RNG states were unavailable. Historical uninterrupted endpoints are not controls.','', '| Seed | Immediate CE G−S | Final CE G−S (primary) | Final G−control | Final S−control |','|---|---:|---:|---:|---:|']
 for e in effects:lines.append(f"| {e['seed']} | {e['D_immediate_growing_minus_stable']:.8f} | {e['D_final_growing_minus_stable']:.8f} | {e['growing_final_gap_control']:.8f} | {e['stable_final_gap_control']:.8f} |")
 lines += ['',f'Final paired effects: mean {mean:.8f}, median {statistics.median(vals):.8f}, range {min(vals):.8f}–{max(vals):.8f}, sample SD {sd:.8f}, SE {sd/math.sqrt(5):.8f}. Positive sign in {sum(x>0 for x in vals)}/5 seeds. Source seed is the independent replication unit (n=5); coordinates, experts, layers, batches, and tokens are nested observations. No nonsignificance/equivalence inference is made.','', '## Required interpretation','',f'1. Magnitude matching achieved {min(reg["mask_files"][str(s)]["overall_dose"] for s in SEEDS):.4%} of all eligible expert-weight coordinates per branch in every seed (2,621 per tensor, 670,976 per seed). Each growing/stable pair is within its seed/layer/expert/tensor and current-magnitude stratum. Worst absolute pair mismatch is {maxdiff:.8g} tensor RMS; worst tensor standardized mean difference is {maxsmd:.8g}. Squared removed-magnitude relative differences across tensors range {min(energy):.4%}–{max(energy):.4%}. This is a local small-dose intervention, not full 80/85% pruning.',f'2. Immediate damage: growing minus stable is positive in {sum(x>0 for x in immediate)}/5 seeds; mean {statistics.mean(immediate):.8f}. These immediate effects are separate from the final endpoint.',f'3. After 1,500 attempted updates with the same subsequent original batches, final G−S is positive in {sum(x>0 for x in vals)}/5 seeds; mean {mean:.8f}. Applied updates and AMP skips are reported for every branch; duration was never adapted to the result.',f'4. Consistency is assessed from the five individual effects above, rather than from coordinates or validation batches. Immediate and final effects need not have the same sign. Excess-loss AUC uses trapezoidal integration at registered steps 0,25,100,250,500,1000,1500 relative to the newly run control, without interpolation beyond linear segments. AUC and normalized mean excess CE are in paired_seed_effects.csv.','5. Magnitude matching does not establish equal connectivity. Row/column occupancy, concentration, and neuron-level residual differences are retained in structural_balance.csv. Residual magnitude, squared-energy, sign, and structural differences could contribute to any effect. Growing/stable is a compound selection rule; a positive result would support its predictive value under this intervention, not show that growth itself causes importance or establish a routing mechanism.','6. A full pruning schedule is not implemented or authorized here. Review whether the final effects are consistent, larger than residual balance differences plausibly explain, and aligned with immediate/AUC behavior before deciding whether growth protection merits a later test. Small-dose null or reversed effects do not establish equivalence at high sparsity.','', '## Protocol and limitations','', 'Currently small means the bottom half within each fc1/fc2 tensor. Growing candidates strictly increase in absolute magnitude at both observed intervals and have high tensor-RMS-normalized OLS slopes; stable candidates have low total absolute variation, rather than near-zero net slope alone. Sustained growth refers only to the observed checkpoints. Signed trajectories and sign changes are diagnostics in matched_group_summary.csv and coordinate_examples.csv. No individual-magnitude denominator, future checkpoint, gradient, validation result, or IMP membership enters selection.','', 'Everything else remains trainable, including native routers, shared layers, biases and surviving expert weights. Gradient masking and post-update zeroing maintain exact zeros. Canonical LR, decay, clipping, FP16/autoscaling, auxiliary loss and native capacity settings are preserved. Validation is task-only FP32 with TF32 disabled on identical complete batches and actual-token weighting. Unpruned shared-reset controls were newly run.','', 'The seed-17 engineering pilot remained in the cohort and only numerical/resource checks controlled dispatch of the remaining seeds; effect direction did not enter the gate. Unique attempts, locks, heartbeats, immutable masks, serialized resume checks and checkpoint hashes are preserved. Every registered branch is reported, including any failures; no historical winners were selected.','']
 (D/'findings.md').write_text('\n'.join(lines)+'\n')
 c=read(D/'completion.json');c['science_report_complete']=True;c['final_effect_summary']=dict(mean=mean,median=statistics.median(vals),minimum=min(vals),maximum=max(vals),sd=sd,se=sd/math.sqrt(5));c['output_sha256']={str(p.relative_to(D)):sha(p) for p in [D/n for n in ('protocol.md','registration.json','checkpoint_inventory.csv','matched_group_summary.csv','magnitude_balance.csv','structural_balance.csv','numerical_checks.json','per_branch_results.csv','learning_curves.csv','paired_seed_effects.csv','findings.md')]};save(D/'completion.json',c)

if __name__=='__main__':
 with lock(D/'report.lock'):analyze()
