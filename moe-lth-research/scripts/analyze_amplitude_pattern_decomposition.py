"""Common-readout paired analysis of the prescribed dense-source information hierarchy."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=Path('results/gate_amplitude_pattern_decomposition');LEVEL2=Path('results/gate_amplitude_replay_intervention');LEVEL1=Path('results/cross_arch_optimization_geometry/fixed_schedule_intervention')
C=['global_mean','layer_mean','expert_layer_mean','token_conditioned','temporal','temporal_shuffle','token_shuffle'];ORDER=['level1_native_amplitude']+C+['level2_r100_amplitude'];TOL=1e-5

def table(x):
 return '\n'.join(['| '+' | '.join(x.columns)+' |','| '+' | '.join(['---']*len(x.columns))+' |']+['| '+' | '.join(f'{v:.6g}' if isinstance(v,float) else str(v) for v in row)+' |' for row in x.itertuples(index=False,name=None)])

def main():
 m=json.loads((ROOT/'decomposition_run_manifest.json').read_text());assert m['completed_recipient_recoveries']==252 and m['completed_pairs']==126
 audit=pd.read_csv(ROOT/'replay_audit.csv');assert len(audit)==252 and audit.max_amplitude_error.eq(0).all() and audit.training_occurrences.eq(2500).all() and audit[['detached','assignments_exact','acceptances_exact']].all().all()
 assert audit.groupby(['experts','seed','age','amplitude_condition']).transformed_amplitude_sequence_hash.nunique().eq(1).all()
 assert audit.groupby(['experts','seed']).primary_validation_hash.nunique().eq(1).all()
 mutations=pd.read_csv(ROOT/'mutation_audit.csv');assert len(mutations)==2268 and mutations[['model_unchanged','optimizer_unchanged','scaler_unchanged','rng_restored']].all().all()
 out=pd.read_csv(ROOT/'outcomes.csv');assert len(out)==378
 primary=out[out.primary_readout=='primary_r100_common_readout'].copy();assert len(primary)==126
 bridges=[];old=pd.read_csv(LEVEL2/'outcomes.csv')
 for row in old.itertuples():
  for label,level in [('level1_native_amplitude','assign'),('level2_r100_amplitude','gate')]:
   bridges.append(dict(experts=row.experts,seed=row.seed,age=row.age,amplitude_condition=label,sparse_final_loss=getattr(row,level+'_fixedgate_sparse_loss'),dense_final_loss=getattr(row,level+'_fixedgate_dense_loss'),ticket_gap=getattr(row,level+'_fixedgate_ticket_gap'),primary_readout='primary_r100_common_readout'))
 primary=pd.concat([primary,pd.DataFrame(bridges)],ignore_index=True);assert np.isfinite(primary[['sparse_final_loss','dense_final_loss','ticket_gap']]).all().all();primary.to_csv(ROOT/'final_ticket_gaps.csv',index=False)
 rows=[]
 for (e,seed),g in primary.groupby(['experts','seed']):
  w=g.pivot(index='age',columns='amplitude_condition',values='ticket_gap');target=20 if e==8 else 60
  for control in ([0,100] if e==8 else [20,100]):
   c1=w.loc[target,'level1_native_amplitude']-w.loc[control,'level1_native_amplitude'];c2=w.loc[target,'level2_r100_amplitude']-w.loc[control,'level2_r100_amplitude'];den=c1-c2
   for condition in ORDER:
    value=w.loc[target,condition]-w.loc[control,condition]
    rows.append(dict(experts=e,seed=seed,target_age=target,control_age=control,amplitude_condition=condition,target_gap=w.loc[target,condition],control_gap=w.loc[control,condition],contrast=value,favors_target=value<0,contrast_change_vs_level1=value-c1,contrast_change_vs_level2=value-c2,retention_score=(value-c2)/den if abs(den)>TOL else np.nan,near_zero_denominator=abs(den)<=TOL,sign_flip_vs_level1=value*c1<0,level1_contrast=c1,level2_contrast=c2))
 paired=pd.DataFrame(rows);paired.to_csv(ROOT/'paired_contrasts.csv',index=False);paired[paired.amplitude_condition.isin(C)].to_csv(ROOT/'retention_scores.csv',index=False)
 summary=paired.groupby(['experts','target_age','control_age','amplitude_condition']).agg(mean_contrast=('contrast','mean'),seed_sd=('contrast','std'),seeds_favoring_target=('contrast',lambda s:int((s<0).sum())),seeds_favoring_control=('contrast',lambda s:int((s>0).sum())),mean_retention_score=('retention_score','mean'),retention_seed_sd=('retention_score','std'),valid_retention_seeds=('retention_score','count'),ordering_flips_vs_level1=('sign_flip_vs_level1','sum')).reset_index();summary.to_csv(ROOT/'architecture_summary.csv',index=False)
 hierarchy=summary.pivot(index=['experts','target_age','control_age'],columns='amplitude_condition',values='mean_contrast').reindex(columns=ORDER).reset_index();hierarchy.to_csv(ROOT/'amplitude_information_hierarchy.csv',index=False)
 primary['rank']=primary.groupby(['experts','seed','amplitude_condition']).ticket_gap.rank(method='min');primary.to_csv(ROOT/'ranking_comparison.csv',index=False)
 trajectories=[];gradient_rows=[]
 for row in primary[primary.amplitude_condition.isin(C)].itertuples():
  p=ROOT/f'recovery/{row.amplitude_condition}/e{row.experts}/seed{row.seed}/R{row.age}'
  a=pd.read_json(p/'sparse/metrics.jsonl',lines=True);b=pd.read_json(p/'dense/metrics.jsonl',lines=True);g=a.merge(b,on='step',suffixes=('_sparse','_dense'))
  for q in g.itertuples():trajectories.append(dict(experts=row.experts,seed=row.seed,age=row.age,amplitude_condition=row.amplitude_condition,step=q.step,ticket_gap=q.loss_sparse-q.loss_dense,readout='primary_r100_common_readout'))
  for kind in ['sparse','dense']:
   v=pd.read_json(p/f'{kind}/gradient_stats/gradient_stats.jsonl',lines=True);v=v[v.step>0]
   gradient_rows.append(dict(experts=row.experts,seed=row.seed,age=row.age,kind=kind,amplitude_condition=row.amplitude_condition,expert_gradient_norm=v.expert_grad_norm.mean(),shared_gradient_norm=v.shared_grad_norm.mean(),router_auxiliary_gradient_norm=v.router_grad_norm.mean(),router_task_gradient_availability='none under detached coefficients',clipping_frequency=v.global_grad_norm_pre_clip.gt(1.).mean(),scaler_min=v.loss_scale_after.min(),scaler_max=v.loss_scale_after.max(),nonfinite_step_fraction=(~v.gradient_valid.astype(bool)).mean()))
 trajectories=pd.DataFrame(trajectories);trajectories.to_csv(ROOT/'ticket_gap_trajectories.csv',index=False)
 gs=pd.DataFrame(gradient_rows)
 baseline_grad=pd.read_csv(LEVEL2/'gradient_scale_summary.csv');baseline_grad=baseline_grad[baseline_grad.level=='assign'][['experts','seed','age','kind','expert_gradient_mean']]
 gs=gs.merge(baseline_grad,on=['experts','seed','age','kind']);gs['expert_gradient_ratio_vs_level1']=gs.expert_gradient_norm/gs.expert_gradient_mean;gs.to_csv(ROOT/'gradient_scale_summary.csv',index=False)
 ct=[]
 for (e,seed,condition,step),g in trajectories.groupby(['experts','seed','amplitude_condition','step']):
  w=g.set_index('age');target=20 if e==8 else 60
  for control in ([0,100] if e==8 else [20,100]):ct.append(dict(experts=e,seed=seed,amplitude_condition=condition,step=step,control_age=control,contrast=w.loc[target,'ticket_gap']-w.loc[control,'ticket_gap']))
 ct=pd.DataFrame(ct);prior_ct=pd.read_csv(LEVEL2/'common_readout_contrast_trajectories.csv');prior_ct=prior_ct[prior_ct.level.isin(['assign','gate'])].copy();prior_ct['amplitude_condition']=prior_ct.level.map({'assign':'level1_native_amplitude','gate':'level2_r100_amplitude'})
 ct=pd.concat([ct,prior_ct[ct.columns]],ignore_index=True);ct.to_csv(ROOT/'contrast_trajectories.csv',index=False)
 onset=[]
 for (e,seed,control),g in ct.groupby(['experts','seed','control_age']):
  w=g.pivot(index='step',columns='amplitude_condition',values='contrast')
  for condition in C:
   nonzero=w.index[w[condition].abs()>TOL];different=w.index[(w[condition]-w.level2_r100_amplitude).abs()>TOL];den=w.level1_native_amplitude-w.level2_r100_amplitude;rr=(w[condition]-w.level2_r100_amplitude)/den;half=w.index[(den.abs()>TOL)&rr.ge(.5)]
   onset.append(dict(experts=e,seed=seed,control_age=control,amplitude_condition=condition,first_nontrivial_step=int(nonzero.min()) if len(nonzero) else None,first_divergence_from_level2_step=int(different.min()) if len(different) else None,first_observed_half_retention_step=int(half.min()) if len(half) else None))
 onset=pd.DataFrame(onset);onset.to_csv(ROOT/'effect_onset.csv',index=False)
 # Every bridge comparison uses the same original R100 readout. Secondary policies stay separate.
 secondary=[]
 for (e,seed,condition,readout),g in out.groupby(['experts','seed','amplitude_condition','primary_readout']):
  target=20 if e==8 else 60;w=g.set_index('age')
  for control in ([0,100] if e==8 else [20,100]):secondary.append(dict(experts=e,seed=seed,control_age=control,amplitude_condition=condition,readout=readout,contrast=w.loc[target,'ticket_gap']-w.loc[control,'ticket_gap']))
 pd.DataFrame(secondary).to_csv(ROOT/'readout_comparison.csv',index=False)
 shuffle=pd.read_csv(ROOT/'shuffle_effective_perturbation.csv');assert len(shuffle)==36
 shuffle_summary=shuffle.groupby(['experts','phase']).agg(temporal_moved_edge_min=('temporal_step_moved_edges_fraction','min'),temporal_moved_edge_max=('temporal_step_moved_edges_fraction','max'),temporal_changed_value_min=('temporal_values_changed_fraction','min'),temporal_changed_value_max=('temporal_values_changed_fraction','max'),token_changed_value_min=('token_values_changed_fraction','min'),token_changed_value_max=('token_values_changed_fraction','max')).reset_index()
 amps=pd.read_csv(ROOT/'amplitude_statistics.csv');var=pd.read_csv(ROOT/'amplitude_variance_decomposition.csv');el=pd.read_csv(ROOT/'amplitude_layer_expert_means.csv');evolution=pd.read_csv(ROOT/'amplitude_step_means.csv')
 el['weighted_sum']=el['mean']*el['count'];layers=el.groupby(['experts','seed','age','layer']).agg(total=('weighted_sum','sum'),count=('count','sum')).reset_index();layers['mean']=layers.total/layers['count'];layers.to_csv(ROOT/'amplitude_layer_means.csv',index=False)
 figs=ROOT/'figures';figs.mkdir(exist_ok=True);colors={c:plt.get_cmap('tab10')(i) for i,c in enumerate(ORDER)};labels={c:c.replace('_',' ') for c in ORDER}
 def finish(name):plt.tight_layout();plt.savefig(figs/f'{name}.png',dpi=180);plt.savefig(figs/f'{name}.pdf');plt.close()
 for e in [8,12]:
  fig,axes=plt.subplots(1,2,figsize=(15,5))
  for ax,(control,g) in zip(axes,paired[paired.experts==e].groupby('control_age')):
   for seed,v in g.groupby('seed'):
    v=v.set_index('amplitude_condition').reindex(ORDER);ax.plot(range(len(ORDER)),v.contrast,'o-',label=f'seed {seed}')
   ax.axhline(0,color='black',lw=.6);ax.set(title=f'{e}E target minus R{control}',xticks=range(len(ORDER)),xticklabels=[labels[c] for c in ORDER],ylabel='Primary contrast');ax.tick_params(axis='x',rotation=55,labelsize=7);ax.legend()
  finish(f'e{e}_final_contrasts')
 fig,axes=plt.subplots(2,2,figsize=(14,9))
 for ax,((e,control),g) in zip(axes.flat,paired[paired.amplitude_condition.isin(C)].groupby(['experts','control_age'])):
  for seed,v in g.groupby('seed'):ax.plot(range(7),v.set_index('amplitude_condition').reindex(C).retention_score,'o-',label=f's{seed}')
  ax.axhline(0,color='black',lw=.5);ax.axhline(1,color='gray',ls='--');ax.set(title=f'{e}E target vs R{control}',xticks=range(7),xticklabels=[labels[c] for c in C],ylabel='Retention');ax.tick_params(axis='x',rotation=45,labelsize=7);ax.legend()
 finish('contrast_retention')
 for e in [8,12]:
  fig,axes=plt.subplots(1,3,figsize=(15,4))
  for ax,(seed,g) in zip(axes,ct[(ct.experts==e)&(ct.control_age==100)].groupby('seed')):
   for condition,v in g.groupby('amplitude_condition'):ax.plot(v.step,v.contrast,label=labels[condition],color=colors[condition],marker='.' if condition=='level1_native_amplitude' else None)
   ax.set(title=f'{e}E target vs R100 / seed{seed}',xlabel='step',ylabel='Primary ticket-gap contrast');ax.legend(fontsize=5,ncol=2)
  finish(f'e{e}_target_vs_r100_trajectory')
 for condition in ['token_shuffle','temporal_shuffle']:
  fig,axes=plt.subplots(2,3,figsize=(14,7))
  for ax,((e,seed),g) in zip(axes.flat,ct[(ct.control_age==100)&ct.amplitude_condition.isin([condition,'level1_native_amplitude','level2_r100_amplitude'])].groupby(['experts','seed'])):
   for c,v in g.groupby('amplitude_condition'):ax.plot(v.step,v.contrast,label=labels[c],color=colors[c],marker='.' if c=='level1_native_amplitude' else None)
   ax.set(title=f'{e}E seed{seed}',xlabel='step',ylabel='Target minus R100');ax.legend(fontsize=6)
  finish(condition+'_vs_native_amplitude')
 fig,axes=plt.subplots(2,3,figsize=(13,7))
 for ax,((e,seed),g) in zip(axes.flat,layers.groupby(['experts','seed'])):
  for age,v in g.groupby('age'):ax.plot(v.layer,v['mean'],'o-',label=f'R{age}')
  ax.set(title=f'{e}E seed{seed}',xlabel='layer',ylabel='Edge-weighted mean amplitude');ax.legend()
 finish('layer_mean_amplitudes')
 fig,axes=plt.subplots(6,3,figsize=(13,16))
 for ax,((e,seed,age),g) in zip(axes.flat,el.groupby(['experts','seed','age'])):
  a=g.pivot(index='layer',columns='expert',values='mean');im=ax.imshow(a,aspect='auto',vmin=0,vmax=1);ax.set(title=f'{e}E seed{seed} R{age}',xlabel='expert',ylabel='layer');fig.colorbar(im,ax=ax)
 finish('expert_layer_heatmaps')
 fig,axes=plt.subplots(2,3,figsize=(13,7))
 for ax,((e,seed),g) in zip(axes.flat,evolution.groupby(['experts','seed'])):
  for age,v in g.groupby('age'):ax.plot(v.step,v['mean'],label=f'R{age}')
  ax.set(title=f'{e}E seed{seed}',xlabel='step',ylabel='Canonical dense source mean');ax.legend()
 finish('amplitude_evolution')
 fig,axes=plt.subplots(2,3,figsize=(13,7))
 for ax,((e,seed),g) in zip(axes.flat,var.groupby(['experts','seed'])):
  a=g.pivot(index='age',columns='component',values='fraction');a.plot.bar(stacked=True,ax=ax);ax.set(title=f'{e}E seed{seed}',ylabel='Descriptive variance fraction');ax.legend(fontsize=5)
 finish('amplitude_variance_decomposition')
 fig,axes=plt.subplots(1,2,figsize=(13,5))
 for ax,e in zip(axes,[8,12]):
  g=gs[gs.experts==e].groupby(['seed','amplitude_condition']).expert_gradient_ratio_vs_level1.mean().reset_index()
  for seed,v in g.groupby('seed'):ax.plot(range(7),v.set_index('amplitude_condition').reindex(C).expert_gradient_ratio_vs_level1,'o-',label=f'seed{seed}')
  ax.axhline(1,color='gray',ls='--');ax.set(title=f'{e}E',xticks=range(7),xticklabels=[labels[c] for c in C],ylabel='Expert gradient norm / Level 1');ax.tick_params(axis='x',rotation=45,labelsize=7);ax.legend()
 finish('expert_gradient_norm_by_condition')
 reduced=paired[paired.amplitude_condition.isin(C)].copy();reduced['recovers']=reduced.retention_score.between(.8,1.2)&~reduced.sign_flip_vs_level1&~reduced.near_zero_denominator;reduced['distance_from_full']=(reduced.retention_score-1).abs()
 scores=reduced.groupby(['experts','control_age','amplitude_condition']).agg(recovering_seeds=('recovers','sum'),valid_seeds=('retention_score','count'),distance_from_full=('distance_from_full','mean')).reset_index();scores.to_csv(ROOT/'retention_consistency.csv',index=False)
 all_recover={c:bool((scores[scores.amplitude_condition==c].recovering_seeds==3).all()) for c in C}
 all_fail={c:bool((reduced[reduced.amplitude_condition==c].retention_score.abs()<.5).all()) for c in C}
 if all_recover['global_mean']:verdict='GLOBAL GATE SCALE SUFFICIENT'
 elif all_fail['global_mean'] and all_recover['layer_mean']:verdict='LAYER-SPECIFIC GATE STRUCTURE REQUIRED'
 elif all_fail['layer_mean'] and all_recover['expert_layer_mean']:verdict='EXPERT-SPECIFIC GATE STRUCTURE REQUIRED'
 elif scores.recovering_seeds.between(1,2).any():verdict='GATE-PATTERN RESULT SEED-CONDITIONED'
 else:verdict='NO CLEAR GATE-PATTERN COMPONENT IDENTIFIED'
 # Token/time/high-order necessity cannot be inferred without a full detached canonical-source control.
 best=scores.sort_values('distance_from_full').groupby(['experts','control_age']).first().reset_index();best.to_csv(ROOT/'best_retention_condition.csv',index=False)
 answers=[]
 for c,q in [('global_mean','1. Global scale'),('layer_mean','2. Added layer structure'),('expert_layer_mean','3. Added expert/layer structure'),('token_conditioned','4. Token-conditioned information'),('temporal','5. Temporal information'),('temporal_shuffle','6. Temporal shuffling'),('token_shuffle','7. Token shuffling')]:
  g=summary[summary.amplitude_condition==c];answers.extend([q+': common-readout results and three-seed consistency are below. A retention near one reproduces the reference contrast; near zero resembles Level 2.','',table(g),''])
 answers.extend(['8. Closest reduced representation to full reference retention, by architecture/control:','',table(best),'',
 f'9–10. Conditions satisfying 0.8–1.2 retention with preserved signs in all three seeds for all four contrasts: {", ".join(c for c,v in all_recover.items() if v) or "none"}. Per-contrast seed consistency is in retention_consistency.csv. This criterion is descriptive, not statistical significance.','',
 '11. First observed nontrivial contrast, divergence from Level 2, and half retention are listed in effect_onset.csv. The Level-1 common readout exists only at nine retained snapshots, so half-retention timing is limited to those observations. No interpolation is treated as evidence.','',table(onset),'',
 f'12. Expert-gradient ratios to Level 1 range from {gs.expert_gradient_ratio_vs_level1.min():.4f} to {gs.expert_gradient_ratio_vs_level1.max():.4f} across matched runs. Clipping, scale and nonfinite-step fractions are in gradient_scale_summary.csv. Router task gradients are absent; recorded router gradients are from the preserved weighted auxiliary loss. Gradient scaling alone does not establish a mechanism.','',
 f'13–14. Canonical dense-source layer means range from {layers["mean"].min():.5f} to {layers["mean"].max():.5f}; expert/layer means range from {el["mean"].min():.5f} to {el["mean"].max():.5f}. Heatmaps and variance tables show where amplitude variation is concentrated descriptively. They do not identify independent layer/expert causal effects.','',
 '15–16. The contrast hierarchy and retention consistency identify which tested coarse or fine representations reproduce the age-conditioned effect. The verdict applies only to the canonical dense-source transformations tested here. Failure of reduced conditions is not proof that a full high-order canonical pattern is necessary: no full detached canonical-source recovery control was added.','',
 '17. Level 1 used online native amplitudes with task gradients; the canonical sources here are native proposals along dense Level-2 trajectories. Detachment and source-trajectory differences, preserved auxiliary gradients, and endogenous hidden/expert dynamics remain unresolved pathways. Token and temporal shuffles cannot by themselves establish necessity against that unmatched full-reference training procedure.'])
 integrity=dict(validated_dense_sources=18,new_source_statistical_replicates=0,conditions_run=C,recipient_recoveries=252,matched_pairs=126,exact_training_replays=252,paired_amplitude_mismatches=0,max_amplitude_error=0,snapshots=2268,figure_pairs=12,invalid_runs=0)
 (ROOT/'artifact_integrity.json').write_text(json.dumps(integrity,indent=2)+'\n')
 report=['# Gate-amplitude pattern decomposition','',
 'All seven specified conditions completed on the 85% sparsity grid. Eighteen canonical age/seed dense amplitude sources were reconstructed solely to generate artifacts and validated against Level 2. Each transformed amplitude schedule was replayed identically into its matched sparse and dense recipients.','',
 'The primary readout is the same original R100 assignments, acceptances, capacity priorities and gate amplitudes for every condition and both reference levels. Condition-matched and native readouts are secondary and are never mixed into primary contrasts or retention.','',
 '## Source identity and actual shuffle perturbation','',table(amps[['experts','seed','age','recurring_edge_fraction','token_conditioned_definition']]),'',table(shuffle_summary),'',
 '## Final contrast hierarchy','',table(hierarchy),'','## Paired contrasts','',table(paired),'','## Scientific questions','',*answers,'',
 '## Protocol and interpretation','',
 'All coefficients are detached; historical optimizer, clipping, GradScaler, masks, data order and auxiliary losses are preserved. Sources are canonical dense amplitude trajectories associated with router age, not sparse models own native proposals. Temporal shuffling is restricted to equal-length expert/layer blocks and leaves singleton strata fixed; it can also change token mapping across batches. Token means use exact recurring identities where at least 80% of accepted-edge observations recur; any tokenizer-ID substitution is explicitly recorded in each transform manifest.','',
 'Retention uses a preregistered denominator tolerance of 1e-5. Near-zero cases are flagged. Means and sample SD use three independent seeds; tokens, layers, experts and steps are mechanism observations. Some reference target/control contrasts already favor the control; preserving such a contrast is preservation of an age effect, not recovery of a favorable target advantage.','',
 'Raw secondary outcomes and readout_comparison.csv expose evaluation-policy differences. Descriptive nested variance components follow layer, expert within layer, time within expert/layer, then token residual; they are not independent causal variance contributions.','',
 f'All new artifacts are in stable storage at {ROOT.resolve()}. No additional experiment was launched.','',
 verdict,'',
 '**Most important implication for the paper:** The measured retention shows which information in the canonical age-conditioned dense amplitude trajectory is sufficient under detached replay and a common readout. It does not establish exclusive forward-value mediation in ordinary training or necessity of an untested full detached source pattern.','',
 '**Most important next experiment:** A full, untransformed detached replay of the same canonical dense amplitude sources would provide the missing matched reference for interpreting token/time/high-order necessity. This is proposed only and has not been launched.']
 (ROOT/'final_report.md').write_text('\n'.join(report)+'\n');m.update(status='analysis_ready_for_review',provisional_verdict=verdict);(ROOT/'decomposition_run_manifest.json').write_text(json.dumps(m,indent=2)+'\n')
 terminal=['GATE AMPLITUDE PATTERN DECOMPOSITION COMPLETE','']
 for row in hierarchy.itertuples():
  terminal.append(f'{row.experts}E R{row.target_age} vs R{row.control_age}:')
  for c in ORDER:terminal.append(f'- {c}: {getattr(row,c):.8f}')
  b=best[(best.experts==row.experts)&(best.control_age==row.control_age)].iloc[0];terminal.append(f'- best retention condition: {b.amplitude_condition}')
 component='global_mean' if verdict=='GLOBAL GATE SCALE SUFFICIENT' else 'layer structure' if verdict=='LAYER-SPECIFIC GATE STRUCTURE REQUIRED' else 'expert/layer structure' if verdict=='EXPERT-SPECIFIC GATE STRUCTURE REQUIRED' else 'not uniquely identified; closest retention: '+', '.join(best.amplitude_condition.unique())
 terminal += ['Most informative amplitude component: '+component,'Cross-architecture consistency: '+str(all_recover),'Seed consistency: see retention_consistency.csv','Number of invalid runs: 0','Final verdict: '+verdict,'Path to final_report.md: '+str((ROOT/'final_report.md').resolve())]
 (ROOT/'terminal_summary.txt').write_text('\n'.join(terminal)+'\n');print('\n'.join(terminal))
if __name__=='__main__':main()
