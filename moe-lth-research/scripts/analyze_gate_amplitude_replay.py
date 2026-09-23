"""Paired three-seed analysis, separating common readouts from legacy curves."""
from pathlib import Path
import json,importlib.util
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=Path('results/gate_amplitude_replay_intervention');L0=Path('results/cross_arch_optimization_geometry/diagnostic_rerun');L1=Path('results/cross_arch_optimization_geometry/fixed_schedule_intervention')

def table(df):
 return '\n'.join(['| '+' | '.join(df.columns)+' |','| '+' | '.join(['---']*len(df.columns))+' |']+['| '+' | '.join(f'{v:.6g}' if isinstance(v,float) else str(v) for v in row)+' |' for row in df.itertuples(index=False,name=None)])
def main():
 m=json.loads((ROOT/'gate_replay_manifest.json').read_text());assert m['completed_recipient_recoveries']==36 and m['completed_pairs']==18
 spec=importlib.util.spec_from_file_location('snapshot_eval','scripts/evaluate_gate_replay_snapshots.py');mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod);mod.main()
 out=pd.read_csv(ROOT/'outcomes.csv');audit=pd.read_csv(ROOT/'replay_audit.csv');assert len(out)==18 and len(audit)==36
 assert audit.maximum_absolute_amplitude_error.eq(0).all() and audit.detached_amplitudes.all() and audit.training_forwards.eq(2500).all()
 assert audit.groupby(['experts','seed'])[['training_schedule_hash','validation_schedule_hash','amplitude_schedule_hash']].nunique().eq(1).all().all()
 assert np.isfinite(out.select_dtypes('number')).all().all()
 rows=[];ranks=[];long=[]
 for row in out.itertuples():
  for level in ['native','assign','gate']:
   long.append(dict(experts=row.experts,seed=row.seed,age=row.age,level=level,ticket_gap=getattr(row,level+'_fixedgate_ticket_gap')))
 long=pd.DataFrame(long);long['rank']=long.groupby(['experts','seed','level']).ticket_gap.rank(method='min');long.to_csv(ROOT/'ranking_comparison.csv',index=False)
 for (e,s),g in out.groupby(['experts','seed']):
  w=g.set_index('age');target=20 if e==8 else 60
  for control in ([0,100] if e==8 else [20,100]):
   c={level:w.loc[target,level+'_fixedgate_ticket_gap']-w.loc[control,level+'_fixedgate_ticket_gap'] for level in ['native','assign','gate']}
   secondary={level:w.loc[target,level+'_native_ticket_gap']-w.loc[control,level+'_native_ticket_gap'] for level in ['native','assign','gate']}
   rows.append(dict(experts=e,seed=s,target_age=target,control_age=control,native_contrast=c['native'],assignment_only_contrast=c['assign'],assignment_gate_contrast=c['gate'],gate_increment_interaction=c['gate']-c['assign'],total_interaction=c['gate']-c['native'],attenuation_fraction=1-abs(c['gate'])/(abs(c['assign'])+1e-12),assignment_to_gate_order_flip=c['gate']*c['assign']<0,level1_favors_target=c['assign']<0,secondary_native_gate_contrast=secondary['gate'],secondary_native_assignment_contrast=secondary['assign'],secondary_native_gate_interaction=secondary['gate']-secondary['assign'],secondary_native_order_flip=secondary['gate']*secondary['assign']<0,near_zero_level1=abs(c['assign'])<1e-5))
 contrasts=pd.DataFrame(rows);contrasts.to_csv(ROOT/'paired_contrasts.csv',index=False)
 summary=contrasts.groupby(['experts','target_age','control_age']).agg(mean_gate_increment_interaction=('gate_increment_interaction','mean'),seed_sd=('gate_increment_interaction','std'),seeds_toward_target=('gate_increment_interaction',lambda s:int((s<0).sum())),seeds_away_from_target=('gate_increment_interaction',lambda s:int((s>0).sum())),ordering_flips=('assignment_to_gate_order_flip','sum'),mean_attenuation=('attenuation_fraction','mean'),attenuation_sd=('attenuation_fraction','std'),seeds_attenuated=('attenuation_fraction',lambda s:int((s>0).sum())),seeds_at_least_half_attenuated=('attenuation_fraction',lambda s:int((s>=.5).sum()))).reset_index();summary['full_three_seed_flip']=summary.ordering_flips.eq(3);summary.to_csv(ROOT/'gate_interaction_summary.csv',index=False)
 arch=out.groupby(['experts','age']).agg(historical_native_gap=('native_fixedgate_ticket_gap','mean'),assignment_replay_gap=('assign_fixedgate_ticket_gap','mean'),assignment_gate_replay_gap=('gate_fixedgate_ticket_gap','mean'),native_readout_after_gate_replay=('gate_native_ticket_gap','mean')).reset_index();arch.to_csv(ROOT/'architecture_summary.csv',index=False)
 long.groupby(['experts','age','level']).ticket_gap.agg(['mean','std','count']).to_csv(ROOT/'architecture_seed_sd.csv')
 common=pd.read_csv(ROOT/'common_readout_snapshot_losses.csv');assert len(common)==972
 curves=common.pivot(index=['level','experts','seed','age','step'],columns='kind',values='loss').reset_index();curves['ticket_gap']=curves['sparse']-curves['dense'];curves['readout']='fixedgate'
 legacy=[];grad=[]
 for row in out.itertuples():
  for level,base,readout in [('native',L0,'native'),('assign',L1,'assign'),('gate',ROOT,'fixedgate')]:
   p=base/f'recovery/e{row.experts}/seed{row.seed}/R{row.age}'
   a=pd.read_json(p/'sparse/metrics.jsonl',lines=True);b=pd.read_json(p/'dense/metrics.jsonl',lines=True);g=a.merge(b,on='step',suffixes=('_sparse','_dense'))
   for x in g.itertuples():legacy.append(dict(level=level,experts=row.experts,seed=row.seed,age=row.age,step=x.step,ticket_gap=x.loss_sparse-x.loss_dense,readout=readout))
   for kind in ['sparse','dense']:
    x=pd.read_json(p/f'{kind}/gradient_stats/gradient_stats.jsonl',lines=True);x=x[x.step>0];x['level']=level;x['experts']=row.experts;x['seed']=row.seed;x['age']=row.age;x['kind']=kind;grad.append(x)
 curves=pd.concat([curves,pd.DataFrame(legacy)],ignore_index=True).drop_duplicates(['level','experts','seed','age','step','readout']);curves.to_csv(ROOT/'ticket_gap_trajectories.csv',index=False)
 grad=pd.concat(grad,ignore_index=True);grad['clipped']=grad.global_grad_norm_pre_clip.gt(1.);grad['nonfinite_step']=~grad.gradient_valid.astype(bool)
 gs=grad.groupby(['level','experts','seed','age','kind']).agg(expert_gradient_mean=('expert_grad_norm','mean'),router_gradient_mean=('router_grad_norm','mean'),shared_gradient_mean=('shared_grad_norm','mean'),clipping_frequency=('clipped','mean'),nonfinite_step_fraction=('nonfinite_step','mean'),scaler_min=('loss_scale_after','min'),scaler_max=('loss_scale_after','max')).reset_index();gs.to_csv(ROOT/'gradient_scale_summary.csv',index=False)
 grad[grad.step.mod(50).eq(0)|grad.step.eq(1)].to_csv(ROOT/'gradient_checkpoint_statistics.csv',index=False)
 gate=pd.read_csv(ROOT/'gate_amplitude_statistics.csv');gate.groupby(['experts','seed','age','kind','layer','expert']).agg(donor_amplitude_mean=('donor_amplitude_mean','mean'),native_recipient_proposal_amplitude_mean=('native_recipient_proposal_amplitude_mean','mean'),difference=('difference','mean'),amplitude_SD=('amplitude_sd','mean'),entropy=('entropy','mean'),margin=('margin','mean')).reset_index().to_csv(ROOT/'gate_statistics_summary.csv',index=False)
 pairedamp=gate.pivot(index=['experts','seed','age','step','layer','expert'],columns='kind',values='native_recipient_proposal_amplitude_mean');pairedamp['sparse_minus_dense']=pairedamp['sparse']-pairedamp['dense'];pairedamp.to_csv(ROOT/'sparse_dense_amplitude_difference.csv')
 # Common-readout target/control trajectories: observed nine baseline snapshots only.
 cr=[]
 for (e,s,level,step),g in curves[curves.readout=='fixedgate'].groupby(['experts','seed','level','step']):
  w=g.set_index('age');target=20 if e==8 else 60
  for control in ([0,100] if e==8 else [20,100]):
   cr.append(dict(experts=e,seed=s,level=level,step=step,control_age=control,contrast=w.loc[target,'ticket_gap']-w.loc[control,'ticket_gap']))
 cr=pd.DataFrame(cr);cr.to_csv(ROOT/'common_readout_contrast_trajectories.csv',index=False)
 figs=ROOT/'figures';figs.mkdir(exist_ok=True)
 colors={'native':'gray','assign':'tab:blue','gate':'tab:orange'}
 def finish(name):plt.tight_layout();plt.savefig(figs/f'{name}.png',dpi=180);plt.savefig(figs/f'{name}.pdf');plt.close()
 for e in [8,12]:
  fig,axs=plt.subplots(1,3,figsize=(14,4))
  for ax,(seed,g) in zip(axs,curves[(curves.experts==e)&(curves.readout=='fixedgate')].groupby('seed')):
   for (level,age),v in g.groupby(['level','age']):ax.plot(v.step,v.ticket_gap,label=f'{level} R{age}',color=colors[level],ls={0:':',20:'--',60:':',100:'-'}[age],marker='.' if level!='gate' else None)
   ax.set(title=f'{e}E seed {seed}',xlabel='step',ylabel='Ticket gap; common gates');ax.legend(fontsize=6,ncol=3)
  finish(f'e{e}_ticket_gap_trajectories')
  fig,axs=plt.subplots(1,2,figsize=(12,4))
  for ax,(control,g) in zip(axs,cr[cr.experts==e].groupby('control_age')):
   for (level,seed),v in g.groupby(['level','seed']):ax.plot(v.step,v.contrast,label=f'{level} s{seed}',color=colors[level],ls={7:'-',17:'--',29:':'}[seed],marker='.' if level!='gate' else None)
   ax.axhline(0,color='black',lw=.5);ax.set(title=f'{e}E target minus R{control}',xlabel='step',ylabel='Common-readout contrast');ax.legend(fontsize=6,ncol=3)
  finish(f'e{e}_target_control_trajectories')
 for field,name in [('gate_increment_interaction','gate_interaction_by_seed'),('attenuation_fraction','contrast_attenuation')]:
  fig,axs=plt.subplots(1,2,figsize=(10,4))
  for ax,e in zip(axs,[8,12]):
   for seed,g in contrasts[contrasts.experts==e].groupby('seed'):ax.plot(g.control_age,g[field],'o-',label=f'seed {seed}')
   ax.axhline(0,color='black',lw=.5);ax.set(title=f'{e}E',xlabel='Control age',ylabel=field);ax.legend()
  finish(name)
 fig,axs=plt.subplots(1,2,figsize=(11,4))
 for ax,e in zip(axs,[8,12]):
  g=gate[(gate.experts==e)&(gate.expert==-1)].groupby(['seed','age','step']).native_recipient_proposal_amplitude_mean.mean().reset_index()
  for (s,age),v in g.groupby(['seed','age']):ax.plot(v.step,v.native_recipient_proposal_amplitude_mean,label=f'R{age}/s{s}')
  ax.set(title=f'{e}E native probabilities on donor edges',xlabel='step',ylabel='Gate amplitude');ax.legend(fontsize=6,ncol=3)
 finish('native_gate_amplitude_trajectories')
 fig,axs=plt.subplots(1,2,figsize=(10,4))
 for ax,e in zip(axs,[8,12]):
  g=gate[(gate.experts==e)&(gate.expert==-1)].groupby(['seed','age']).difference.mean().reset_index()
  for s,v in g.groupby('seed'):ax.plot(v.age,v.difference,'o-',label=f'seed{s}')
  ax.axhline(0,color='black',lw=.5);ax.set(title=f'{e}E',xlabel='age',ylabel='Native minus donor amplitude');ax.legend()
 finish('donor_native_amplitude_difference')
 fig,axs=plt.subplots(1,2,figsize=(11,4))
 for ax,e in zip(axs,[8,12]):
  g=grad[(grad.experts==e)&grad.step.mod(50).eq(0)].groupby(['level','seed','step']).expert_grad_norm.mean().reset_index()
  for (level,s),v in g.groupby(['level','seed']):ax.plot(v.step,v.expert_grad_norm,label=f'{level}/s{s}',color=colors[level],ls={7:'-',17:'--',29:':'}[s])
  ax.set(title=f'{e}E actual training gradients',xlabel='step',ylabel='Expert gradient norm');ax.legend(fontsize=6,ncol=3)
 finish('expert_gradient_norms')
 fig,axs=plt.subplots(2,3,figsize=(12,7))
 for ax,((e,s),g) in zip(axs.flat,long.groupby(['experts','seed'])):
  for level,v in g.groupby('level'):ax.plot(v.age,v['rank'],'o-',label=level,color=colors[level])
  ax.set(title=f'{e}E seed{s}',xlabel='age',ylabel='Rank (1 best)',yticks=[1,2,3]);ax.invert_yaxis();ax.legend(fontsize=7)
 finish('router_age_rankings')
 # Classification cannot promote attenuation of an already unfavorable contrast to support.
 favorable=contrasts[contrasts.level1_favors_target & ~contrasts.near_zero_level1]
 support=favorable.groupby(['experts','control_age']).agg(n=('seed','count'),half=('attenuation_fraction',lambda x:int((x>=.5).sum())),positive=('attenuation_fraction',lambda x:int((x>0).sum())))
 strong=support[(support.n==3)&(support.half==3)]
 if len(strong) and set(strong.index.get_level_values('experts'))=={8,12}:verdict='GATE-AMPLITUDE MECHANISM SUPPORTED'
 elif ((support.n==3)&(support.positive==3)).any():verdict='PARTIAL GATE-AMPLITUDE CONTRIBUTION'
 elif summary.ordering_flips.gt(0).any() or ((summary.seeds_attenuated>0)&(summary.seeds_attenuated<3)).any():verdict='SEED-CONDITIONED GATE EFFECT'
 else:verdict='ROUTING IDENTITY + GATE AMPLITUDE INSUFFICIENT'
 checks=pd.read_csv(ROOT/'mutation_audit.csv');assert len(checks)==324 and checks[['model_unchanged','optimizer_unchanged','scaler_unchanged','rng_restored']].all().all()
 integrity=dict(donor_archives_verified=6,same_state_donor_controls_passed=6,recipients=36,matched_pairs=18,exact_full_replays=36,maximum_amplitude_error=0,snapshots=324,common_readout_snapshot_evaluations=972,figure_pairs=10)
 (ROOT/'artifact_integrity.json').write_text(json.dumps(integrity,indent=2)+'\n')
 # Descriptive answers use seed-level aggregates and observed common-readout times.
 timing=cr.pivot(index=['experts','seed','control_age','step'],columns='level',values='contrast').dropna().reset_index()
 timing['interaction']=timing['gate']-timing['assign'];timing['attenuation']=1-timing['gate'].abs()/(timing['assign'].abs()+1e-12)
 timing.to_csv(ROOT/'observed_gate_effect_timing.csv',index=False)
 onset=[]
 for (e,seed,control),g in timing.groupby(['experts','seed','control_age']):
  changed=g[g.interaction.abs()>1e-5];half=g[(g['assign'] < -1e-5)&(g.attenuation>=.5)]
  onset.append(dict(experts=e,seed=seed,control_age=control,first_observed_change_step=int(changed.step.min()) if len(changed) else None,first_observed_half_attenuation_step=int(half.step.min()) if len(half) else None))
 onset=pd.DataFrame(onset);onset.to_csv(ROOT/'effect_onset.csv',index=False)
 gr=gs.pivot(index=['experts','seed','age','kind'],columns='level',values='expert_gradient_mean').reset_index();gr['gate_to_assign_ratio']=gr['gate']/gr['assign'];gr.to_csv(ROOT/'expert_gradient_scale_ratios.csv',index=False)
 layer=gate[gate.expert==-1].groupby(['experts','seed','layer']).difference.mean().reset_index();layer.to_csv(ROOT/'layer_amplitude_differences.csv',index=False)
 expert=gate[gate.expert>=0].groupby(['experts','seed','layer','expert']).difference.mean().reset_index();expert.to_csv(ROOT/'expert_amplitude_differences.csv',index=False)
 amp=gate[gate.expert==-1].groupby(['experts','seed','age']).difference.mean()
 associations=[]
 for row in contrasts.itertuples():
  associations.append(dict(experts=row.experts,seed=row.seed,control_age=row.control_age,amplitude_target_minus_control=amp.loc[row.experts,row.seed,row.target_age]-amp.loc[row.experts,row.seed,row.control_age],gate_increment_interaction=row.gate_increment_interaction))
 associations=pd.DataFrame(associations);associations.to_csv(ROOT/'seed_amplitude_effect_associations.csv',index=False)
 correlations=[]
 for (e,c),g in associations.groupby(['experts','control_age']):
  corr=g.amplitude_target_minus_control.corr(g.gate_increment_interaction) if g.amplitude_target_minus_control.std()>0 and g.gate_increment_interaction.std()>0 else float('nan')
  correlations.append(dict(experts=e,control_age=c,seed_count=len(g),descriptive_pearson_r=corr))
 correlations=pd.DataFrame(correlations);correlations.to_csv(ROOT/'seed_amplitude_effect_correlations.csv',index=False)
 native_disagree=int(((contrasts.assignment_gate_contrast*contrasts.secondary_native_gate_contrast)<0).sum())
 consistency=summary[['experts','control_age','seeds_toward_target','seeds_away_from_target','ordering_flips','seeds_attenuated','seeds_at_least_half_attenuated']]
 mechanism_answers=['## Answers to the remaining scientific questions','',
 f'5–8. Incremental and total changes, literal flips, attenuation, and three-seed sign consistency are tabulated above. There are {int(contrasts.assignment_to_gate_order_flip.sum())}/12 primary seed-level flips; {int(((contrasts.attenuation_fraction>=.5)&~contrasts.assignment_to_gate_order_flip).sum())}/12 contrasts attenuate by at least half without a flip. Unfavorable and near-zero Level-1 contrasts must not be counted as removal of a favorable advantage.','',table(consistency),'',
 '9. First observed common-readout interaction exceeding 1e-5 loss units, and first observed halving of a favorable contrast, are below. This is a descriptive numerical threshold, not significance. Only nine baseline states exist; onset between them cannot be localized. Initialization contrasts under common fixed gates can be zero by construction, so a zero step-0 contrast is not evidence that training removed an initialization effect.','',table(onset),'',
 f'10. Actual training expert-gradient gate/assignment ratios across seed-age-kind runs range from {gr.gate_to_assign_ratio.min():.4f} to {gr.gate_to_assign_ratio.max():.4f}, with mean {gr.gate_to_assign_ratio.mean():.4f}. See gradient_scale_summary.csv for clipping frequency, nonfinite-step fraction and scaler ranges, and common_readout_gradient_diagnostics.csv for elementwise nonfinite fractions in matched FP32 diagnostic gradients. These scales are a sanity check, not a causal endpoint.','',
 f'11–12. Native-minus-donor amplitude differences vary across layer/seed means from {layer.difference.min():.6f} to {layer.difference.max():.6f}, and across expert/layer/seed means from {expert.difference.min():.6f} to {expert.difference.max():.6f}. Detailed means, SD, quantiles, entropy, margin and sparse/dense differences are retained. These show amplitude heterogeneity; the experiment does not identify separate layer-specific or expert-specific causal effects.','',
 '13. The following correlations use three seed-level target-minus-control amplitude differences per contrast. They describe proposals on gate-intervened hidden states, not an independently measured native-training mediator. They are observational, highly unstable at n=3, and cannot establish mediation.','',table(correlations),'',
 f'14. Native readout disagrees with the primary Level-2 target/control sign in {native_disagree}/12 cases. All secondary native interactions are in paired_contrasts.csv; the assignment-only bridge and common-gate outcomes are in outcomes.csv. Native-readout flips are not the primary causal result.','',
 '15–16. A paired common-readout change supports a contribution of this specific detached gate substitution beyond assignment replay; consistent substantial attenuation of favorable contrasts supports a major contribution within the tested donor schedule. It cannot establish that amplitude is the unique or universal explanation: detachment changes gradient paths as well as forward coefficients, the donor is fixed, and only three seeds were tested. The descriptive verdict below applies only to these runs.','',
 '17. If favorable ordering survives, native auxiliary-loss gradients and endogenous hidden/expert-state evolution remain available pathways. This experiment cannot rule out router-dependent auxiliary dynamics or attribute the remaining effect solely to expert inputs. No further experiment was launched.']
 questions=[]
 for (e,control),g in contrasts.groupby(['experts','control_age']):questions.append(f'{e}E target versus R{control}: target is better after full replay in {int(g.assignment_gate_contrast.lt(0).sum())}/3 seeds; incremental contrast change {g.gate_increment_interaction.mean():.6f} ± {g.gate_increment_interaction.std():.6f} seed SD; attenuation {g.attenuation_fraction.mean():.1%}; flips {int(g.assignment_to_gate_order_flip.sum())}/3.')
 report=['# Gate-amplitude replay intervention','',*questions,'','## Paired contrasts at the common assignment-plus-gate readout','',table(contrasts),'',table(summary),'','## Architecture means (all three training levels at common readout)','',table(arch),'','## Interpretation and limits','',
 'All amplitudes are detached exact stored coefficients. Router task gradients through gates are removed; historical auxiliary-loss gradients remain active into router and upstream states. Thus any remaining age dependence can travel through auxiliary-loss dynamics; surviving effects cannot be attributed exclusively to hidden/expert state independent of routing. Donor final-model identity was explicitly waived by the user because the backward pass changes. Exact same-state forward and gradient checks replace that invalid requirement.','',
 'The primary incremental contrast compares Level 2 and Level 1 under the SAME assignment-plus-gate evaluation. Native and assignment-only bridge readouts are separate columns in outcomes.csv. Absolute loss changes and evaluation-only differences are not substituted for paired sparse-minus-dense contrasts. Attenuation uses epsilon 1e-12; near-zero Level-1 contrasts are flagged. Three seeds are the independent units; there are no significance claims or token/expert pseudoreplicates.','',
 'Trajectory timing is resolved only at nine saved common-readout baseline snapshots. Level 2 has all 51 validation observations. Original 51-point baseline curves are retained with their original readout labels, not treated as common-readout evidence. Gate observations describe native probabilities on recipient intervention hidden states and donor-accepted edges. Layer/expert variation is descriptive and is not a factorial intervention on mean, variance, pattern, or curriculum. Actual training-gradient logs and matched FP32 diagnostic gradients are reported separately; nonfinite-step frequency is not an elementwise gradient fraction.','',
 *mechanism_answers,'',
 '## Artifacts and validation','',table(pd.DataFrame([integrity])),'',f'All new bulk artifacts are in stable storage at {ROOT.resolve()}. No donor or Level-1 recovery was rerun.','',verdict,'',
 '**Most important implication for the paper:** The incremental paired effect measures the contribution of replacing native amplitudes with this specific donor schedule, beyond assignment replay. It does not establish formal or complete mediation; auxiliary-loss and representation dynamics remain endogenous.','',
 '**Most important next experiment:** Replay identical expert input representations in addition to assignments and gates, while preserving the current auxiliary-loss protocol, to separate input-state evolution from the already clamped routing output. This is a proposal only; it has not been launched.']
 (ROOT/'final_report.md').write_text('\n'.join(report)+'\n');m.update(status='analysis_ready_for_review',provisional_verdict=verdict);(ROOT/'gate_replay_manifest.json').write_text(json.dumps(m,indent=2)+'\n')
 print(table(summary));print('Donors reused 6/6; same-state controls 6/6; recipients 36/36; replay errors zero.');print(verdict);print(ROOT/'final_report.md')
if __name__=='__main__':main()
