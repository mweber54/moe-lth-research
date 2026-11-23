"""Review-only summaries and figures; never executes training or diagnostic backward."""
from pathlib import Path
import json
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=Path('results/cross_arch_optimization_geometry/diagnostic_rerun')

def table(frame, digits=4):
    def cell(v):
        if pd.isna(v):return '—'
        if isinstance(v,float):return f'{v:.{digits}f}'
        return str(v)
    rows=['| '+' | '.join(map(str,frame.columns))+' |','| '+' | '.join(['---']*len(frame.columns))+' |']
    rows+=['| '+' | '.join(cell(x) for x in row)+' |' for row in frame.itertuples(index=False,name=None)]
    return '\n'.join(rows)

def main():
    repro=pd.read_csv(ROOT/'reproduction_check.csv');cross=pd.read_csv(ROOT/'cross_architecture_summary.csv');paired=pd.read_csv(ROOT/'paired_seed_differences.csv');time=pd.read_csv(ROOT/'metric_timecourse_seed_level.csv');seed=pd.read_csv(ROOT/'seed_level_summary.csv');curves=pd.read_csv(ROOT/'validation_ticket_gap_timecourse.csv');hetero=pd.read_csv(ROOT/'expert_layer_paired_effects.csv');coverage=pd.read_csv(ROOT/'diagnostic_coverage_summary.csv')
    paired_summary=paired.groupby(['experts','metric','control_age']).agg(mean_difference=('paired_difference','mean'),seed_sd=('paired_difference','std'),seeds_favoring_target=('favors_target','sum')).reset_index()
    paired_summary.to_csv(ROOT/'paired_seed_effect_summary.csv',index=False)
    timing=[]
    def onset(series,persistent=False):
        s=series.sort_index().dropna()
        for step in s.index:
            if s.loc[step]>0 and (not persistent or (s.loc[step:]>0).all()):return int(step)
        return None
    for e in [8,12]:
        target=20 if e==8 else 60;controls=[0,100] if e==8 else [20,100]
        fig,axes=plt.subplots(3,2,figsize=(12,9),sharex=True)
        for col,control in enumerate(controls):
            for seed_id,color in zip([7,17,29],['tab:blue','tab:orange','tab:green']):
                c=curves[(curves.experts==e)&(curves.seed==seed_id)&curves.age.isin([target,control])]
                gap=c.pivot(index='step',columns='age',values='ticket_gap');loss=c.pivot(index='step',columns='age',values='loss_sparse')
                gd=gap[control]-gap[target];ld=loss[control]-loss[target]
                for metric in ['forced_gradient','update_alignment','functional_similarity']:
                    p=time[(time.experts==e)&(time.seed==seed_id)&(time.metric==metric)&time.age.isin([target,control])].pivot(index='step',columns='age',values='value');delta=p[target]-p[control]
                    timing.append(dict(experts=e,seed=seed_id,target_age=target,control_age=control,metric=metric,first_positive_metric_step=onset(delta),first_persistent_metric_step=onset(delta,True),first_positive_ticket_gap_step=onset(gd),first_persistent_ticket_gap_step=onset(gd,True),first_positive_sparse_loss_step=onset(ld),first_persistent_sparse_loss_step=onset(ld,True),persistent_definition='Positive at this and every later measured step; descriptive supplement, not a replacement primary metric'))
                    if metric!='functional_similarity':axes[0 if metric=='forced_gradient' else 1,col].plot(delta.index,delta.values,label=f'seed {seed_id}',color=color)
                axes[2,col].plot(gd.index,gd.values,label=f'seed {seed_id}',color=color)
            for row,label in enumerate(['Forced-gradient cosine: target − control','Update cosine: target − control','Ticket gap: control − target']):
                ax=axes[row,col];ax.axhline(0,color='black',ls='--',lw=.7);ax.set(ylabel=label,title=f'{e}E R{target} versus R{control}');ax.legend(fontsize=8)
            axes[2,col].set_xlabel('Recovery step (update plotted at interval end)')
        fig.suptitle('Paired contrasts: positive values favor the target router',fontsize=12)
        fig.tight_layout();fig.savefig(ROOT/f'figures/{e}E_paired_temporal_contrasts.png',dpi=180);fig.savefig(ROOT/f'figures/{e}E_paired_temporal_contrasts.pdf');plt.close(fig)
    pd.DataFrame(timing).to_csv(ROOT/'temporal_seed_specific_review.csv',index=False)
    # Compact seed ranking table, always within the specified router grid.
    rankings=[]
    for (e,s),g in seed.groupby(['experts','seed']):
        rr={'experts':e,'seed':s}
        rg=repro[(repro.experts==e)&(repro.seed==s)].sort_values('rerun_ticket_gap')
        rr['ticket_gap_ranking']=' > '.join('R'+str(a) for a in rg.age)
        for metric,b in g.groupby('metric'):rr[metric]=' > '.join('R'+str(a) for a in b.sort_values('metric_rank').age)
        rankings.append(rr)
    rank=pd.DataFrame(rankings);rank.to_csv(ROOT/'seed_rankings_review.csv',index=False)
    means=cross.copy();means['mean ± seed SD']=means.apply(lambda x:f'{x["mean"]:.4f} ± {x.seed_sd:.4f}',axis=1)
    primary=means[means.metric.isin(['forced_gradient','update_alignment','functional_similarity'])].pivot(index=['experts','age'],columns='metric',values='mean ± seed SD').reset_index()
    gaps=repro.groupby(['experts','age']).rerun_ticket_gap.agg(['mean','std']).reset_index();gaps['ticket gap ± SD']=gaps.apply(lambda x:f'{x["mean"]:.4f} ± {x["std"]:.4f}',axis=1)
    primary=primary.merge(gaps[['experts','age','ticket gap ± SD']],on=['experts','age'])
    zero=time[time.step.eq(0)].groupby(['experts','age','metric']).value.agg(['mean','std']).reset_index();zero['mean ± seed SD']=zero.apply(lambda x:f'{x["mean"]:.5f} ± {x["std"]:.5f}',axis=1)
    ztable=zero[zero.metric.isin(['forced_gradient','native_gradient','functional_similarity'])].pivot(index=['experts','age'],columns='metric',values='mean ± seed SD').reset_index()
    cov=coverage.groupby('metric').agg(total=('size','sum'),valid=('count','sum'),minimum_condition_step_coverage=('fraction_available','min')).reset_index()
    hs=hetero.groupby(['experts','metric','control_age']).fraction_experts_favoring_target.mean().reset_index().rename(columns={'fraction_experts_favoring_target':'fraction_experts_favoring_target_mean_over_seeds_layers'})
    layers=hetero[hetero.metric.eq('forced_gradient')].groupby(['experts','control_age','layer']).paired_expert_mean_difference.agg(['mean','std']).reset_index();layers.to_csv(ROOT/'forced_gradient_layer_contrasts.csv',index=False)
    report=['# Cross-architecture optimization geometry: completed diagnostic rerun','',
    'All **18 matched pairs / 36 recoveries** completed the historical 2500-step protocol. All 36 final validation losses and all 18 ticket gaps reproduce history exactly at the stored numeric precision. All 36 final model states are also bit-identical to their historical checkpoints. All **324 complete model/optimizer snapshots** passed their mutation audits. The requested mechanism is not supported: favorable mean ticket gaps do not coincide with a common advantage in forced gradient, optimizer-update, and expert-function alignment.','',
    '## Protocol and measurement integrity','',
    'Only 85% sparsity was used: 8E R0/R20/R100 and 12E R20/R60/R100, each at seeds 7/17/29. Historical artifacts and the availability audit were preserved. E0, ET-derived masks, shared state, routers, and materialized training/validation sequences passed the preflight hash checks. Training retained fresh AdamW, its historical hyperparameters, no scheduler, fp16, trainable routers, mask enforcement, 2500 steps, and validation every 50 steps.','',
    'Snapshots at 0/100/250/500/750/1000/1500/2000/2500 retain the model, optimizer, GradScaler, RNGs, LR, step, mask, architecture/configuration, seed, historical router metadata, and next-batch identity. Diagnostics use a single persisted eight-sequence validation probe (1024 input-token occurrences), shared literally across architectures and seeds. They use fp32 backward with unit loss scale: gradients are already unscaled and the training scaler is untouched. Vectors are fp32 with fp64 reductions.','',
    'Forced routing shares the sparse model’s expert IDs **and capacity priority scores**, with exact accepted-token equality asserted. Dense gradients/updates are selected on identical surviving coordinates in sorted named-parameter order; unpruned expert biases survive in full. Native gate amplitudes and upstream representations may still differ. Functional comparisons separately hold expert inputs literally equal. Optional dense-route forcing, covariance similarity, and CKA were not run. Coherence uses two fixed diagnostic microbatches.','',
    'The run remained sequential. An attempted parallel scheduling change was rejected by automatic approval review and was not executed. This did not affect the scientific protocol or results.','',
    '## Primary results','',
    'Entries are means ± sample SD over three seed-level summaries. Gradient/function means weight the nine saved steps equally; update means weight the eight intervals equally. Experts are averaged within layer, then layers within seed. Seed is the independent replication unit; no token/expert pseudoreplication or significance claim is made. “Favorable” refers to the **matched ticket gap**, not necessarily the lowest absolute sparse loss.','',table(primary),'',
    'The age ordering for mean forced alignment is R100 > R20 > R0 in 8E and R100 > R60 > R20 in 12E. Both native and forced gradient rankings select R100 for every seed in both architectures. Thus forcing tokens does not reveal a favorable-checkpoint advantage that explains the intermediate-age ticket-gap optimum.','',
    '### Paired effects against each control','',table(paired_summary),'',
    'Differences are target minus control; positive favors the target except for trajectory distance, where negative is favorable. All three seeds favor R100 over the architecture-specific target on average forced gradient alignment. R20 loses to R0 on update alignment in all three 8E seeds. The small 12E mean update advantage is inconsistent across seeds.','',
    '## Answers to the scientific questions','',
    '1. **Does 8E:R20 improve forced alignment against both controls?** No. Its mean is 0.1554 versus 0.1405 (R0) and 0.1791 (R100). It beats R0 in only two seeds and loses to R100 in all three.','',
    '2. **Does 12E:R60 improve forced alignment against both controls?** No. Its mean is 0.1699 versus 0.1515 (R20) and 0.1919 (R100). It beats R20 in all three seeds but loses to R100 in all three.','',
    '3. **Does native alignment show the same pattern?** Yes for the failure of the proposed optimum: native averages also rank R100 first in every architecture/seed group. Architecture means are 0.1297/0.1383/0.1545 for 8E R0/R20/R100 and 0.1254/0.1345/0.1453 for 12E R20/R60/R100.','',
    '4. **Does controlling routing preserve or eliminate a favorable alignment advantage?** There is no consistent favorable-target alignment advantage to preserve. Forced and native rankings both prefer R100, so neither a common token-selection explanation nor a common expert-state explanation of the ticket-gap optimum is established.','',
    '5. **Do optimizer updates support the gradient mechanism?** No. 8E update means favor R0 (0.0649) over R20 (0.0423) and R100 (0.0385), whereas forced gradients favor R100. The 12E means (0.0377/0.0394/0.0391) have small, seed-dependent differences.','',
    '6. **Are sparse trajectories closer under favorable routers?** Not commonly. R20 has greater distance than both 8E controls in every seed (mean 0.4346 versus 0.3897/0.4166). R60 lies between the 12E controls (0.4232 versus 0.4307/0.4142) in every seed.','',
    '7. **Is dense expert function preserved more closely?** Not commonly. 8E R20’s mean function cosine is lower than both controls; 12E R60 is intermediate. Functional winners match the ticket-gap winner in only two of six architecture/seed groups.','',
    '8. **Is the mechanism present at step 0?** No shared effect. 8E R20 has greater forced alignment than R0 in all seeds, but not than R100 in seed 29. In 12E, R60 starts below R20 in forced alignment in all seeds. All parameter trajectory distances are exactly zero at initialization, as required.','',
    '9. **Does a common effect emerge later?** No. R100 becomes the consistently strongest forced-alignment router despite not being the mean ticket-gap optimum. Some target-versus-earlier-control improvements develop, but the required advantage against both controls does not emerge consistently.','',
    '10. **Does alignment separate before or alongside the loss advantage?** No common favorable-target sequence is supported. The preregistered first-all-seed concurrence finds no forced advantage over R100 at any saved step for either architecture. Against the earlier control, 8E’s R20 gradient difference can exist at step 0, while 12E R60 first wins across all seeds at step 250, after a ticket-gap difference is already observed. These first occurrences can be transient: initial gaps are negative and early curves cross. The supplemental seed-specific persistent-onset table explicitly distinguishes first-positive from positive-at-every-later-measurement timing, and separates absolute sparse loss from ticket gap. No causal timing claim is based on a transient step-0 crossing.','',
    '11. **Do metrics track seed-specific recovery winners?** Poorly. Forced/native gradient winners, update winners, and function winners each agree with the actual winner in only 2/6 groups; trajectory distance agrees in 1/6. Seed 17 favors R100 in both architectures in this exact grid. Seed 7 favors R20 in 8E and R60 in 12E; the historical note about an earlier 12E optimum must not override these measured outcomes within the restricted grid.','',
    '12. **Is the effect consistent across experts?** No common benefit across nearly all experts. Against R100, only about 34% of 8E and 24% of 12E experts favor the target in time-averaged forced alignment, averaged over seeds/layers. Full expert rows and distributions retain the exceptions.','',
    '13. **Is it consistent across MoE layers?** The failure against R100 holds in the seed-mean forced-gradient contrast of all four MoE layers in both architectures. Relative to the earlier control, all four layer means improve, but seed/expert effects vary. Thus the layer profiles also fail to identify the intermediate router as best against both controls. The table below retains each layer, and the full expert/layer table retains seed-specific signs; expert counts are not extra independent replicates.','',
    '14. **Is the shared property token selection, expert optimization geometry, both, or neither?** Neither is established as the common explanation of the favorable ticket-gap routers by this probe. This is not proof that routing or geometry is irrelevant to recovery.','',
    '15. **What is the strongest supported mechanistic statement?** Under exactly reproduced recoveries, higher matched dense/sparse gradient, update, or function alignment on this fixed probe does not reliably identify the router checkpoint with the smallest ticket gap. Common accepted-token forcing fails to recover the proposed intermediate-age advantage.','',
    '16. **Which tempting claims remain unsupported?** Universal geometry, architecture independence, causal mediation, an established token-selection mechanism, and a shared emergent co-adaptation explanation remain unsupported. Correlations with absolute sparse loss must not be substituted post hoc for the preregistered matched ticket-gap test.','',
    '## Initialization, seeds, and heterogeneity','',table(ztable),'',
    'Rankings below run from best to worst, with lower distance and ticket gap treated as better.','',table(rank),'',
    'Fraction of experts favoring the target, averaged over seeds and layers (time averaged within each expert):','',table(hs),'',
    'Layer-specific forced-gradient contrasts (target minus control; SD over seeds):','',table(layers),'',
    '## Coverage and limitations','',table(cov),'',
    'Undefined cosines come from empty or zero-norm comparisons and are excluded with counts retained. The lowest single native condition/step coverage is 40.6%; forced/function coverage is at least 78.1%. Conditional means can depend on which experts are active, so routing-dependent coverage limits interpretation. This is one small fixed probe, three seeds per architecture, fp32 diagnostics alongside historical fp16 training, and two-microbatch coherence. The negative conclusion is limited to the tested common alignment explanation; it does not exclude mechanisms outside this probe or these metrics.','',
    '## Artifacts and verification','',
    '- `reproduction_check.csv`: all 18 sparse/dense comparisons, every delta zero.\n- `snapshot_manifest.csv`, `mutation_audit.csv`, and `artifact_integrity.json`: snapshot identities, non-mutation evidence, and completed artifact checks.\n- `run_manifest.json`, `protocol_audit.md`, `provenance/`, and probe/data manifests: fixed protocol, primary metrics, implementation and data identities.\n- `paired_seed_effect_summary.csv`, `seed_rankings_review.csv`, `expert_layer_paired_effects.csv`: paired effects and heterogeneity.\n- `temporal_mechanism_summary.csv`: preregistered first-concurrence timing. `temporal_seed_specific_review.csv`: explicitly supplemental persistent/seed-specific timing.\n- `figures/`: all required figure families in PNG and PDF, plus paired temporal contrasts. Thin lines in original time courses identify seeds 7 (solid), 17 (dashed), and 29 (dotted); thick lines are architecture means.','',
    '## Primary verdict','',
    'NO COMMON OPTIMIZATION GEOMETRY FOUND','',
    '**Most important implication for the paper:**',
    'The historical intermediate-router ticket-gap optima reproduce exactly, but they should not be explained by a shared tendency for sparse experts to follow dense gradients, updates, or functions more closely. R100 has the strongest average forced alignment in every seed while often losing on ticket gap, and the other primary metrics do not resolve this mismatch. Present this as a bounded negative mechanism result, not proof that optimization geometry is universally irrelevant.','',
    '**Most important next experiment:**',
    'Run one matched recovery intervention that fixes the complete routed-token schedule across router initializations, to test whether evolving token selection causally changes the ticket-gap ordering. Do not launch it automatically.']
    (ROOT/'final_report.md').write_text('\n'.join(report)+'\n')
    print('Reviewed report and supplemental temporal figures written. No training or diagnostic backward executed.')

if __name__=='__main__':main()
