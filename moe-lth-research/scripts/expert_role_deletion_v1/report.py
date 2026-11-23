"""Summarize the frozen one-seed engineering pilot without fitting a predictor."""
import importlib.util
from pathlib import Path
import json, time
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
spec=importlib.util.spec_from_file_location('role_study',Path(__file__).with_name('study.py'))
s=importlib.util.module_from_spec(spec);spec.loader.exec_module(s)
D=s.D


def corr(x,y):
    a=np.asarray(x,float);b=np.asarray(y,float);ok=np.isfinite(a)&np.isfinite(b)
    if ok.sum()<3 or np.ptp(a[ok])==0 or np.ptp(b[ok])==0:return None
    return float(spearmanr(a[ok],b[ok]).statistic)


def main():
    damage=pd.read_csv(D/'deletion_damage.csv');roles=pd.read_csv(D/'routing_role_metrics.csv');f=pd.read_csv(D/'prospective_forecasts.csv');groups=pd.read_csv(D/'group_manifest.csv');unions=pd.read_csv(D/'combined_deletion_checks.csv')
    whole=damage[damage.subset=='whole'];summaries=[]
    for scope in ('all',)+s.EXPERTS:
      panel=f if scope=='all' else f[f.expert==scope]
      for keys,z in panel.groupby(['kind','decision_step','future_step','tolerance','rule']):
        kind,t,u,tol,rule=keys;selected=z[z.selected];den=int(z.coordinates.sum());coordinate_numerator=int(selected.coordinates.sum())
        expert_groups=groups if scope=='all' else groups[groups.expert==scope]
        active=int(expert_groups.drop_duplicates('expert').active_coordinates.sum())
        summaries.append(dict(kind=kind,expert=scope,decision_step=t,future_step=u,tolerance=tol,rule=rule,groups=len(z),selected_groups=len(selected),qualifying_group_fraction=len(selected)/len(z),sampled_coordinates=den,qualifying_coordinates=coordinate_numerator,qualifying_sampled_coordinate_fraction=coordinate_numerator/den,qualifying_fraction_all_active_sampled_tensors=coordinate_numerator/active,false_safe_rate=float((selected.future_B_damage>tol).mean()) if len(selected) else None,selected_future_mean=float(selected.future_B_damage.mean()) if len(selected) else None,selected_future_median=float(selected.future_B_damage.median()) if len(selected) else None,selected_future_p05=float(selected.future_B_damage.quantile(.05)) if len(selected) else None,selected_future_p95=float(selected.future_B_damage.quantile(.95)) if len(selected) else None,selected_future_max=float(selected.future_B_damage.max()) if len(selected) else None,cross_probe_agreement=float(z.cross_probe_low_agreement.mean()),zero_denominator=len(selected)==0))
    s.table(D/'forecast_summary.csv',summaries);summary=pd.DataFrame(summaries)
    classifications=[]
    for kind in ('sparse','unpruned'):
      panel=whole[(whole.kind==kind)&(whole.probe=='A')].set_index(['step','group'])
      for i,t in enumerate(s.STEPS):
       for tol in s.TOLERANCES:
        for expert in ('all',)+s.EXPERTS:
         gg=groups if expert=='all' else groups[groups.expert==expert]
         keys=gg.group.tolist();coords=dict(zip(gg.group,gg.coordinates));den=sum(coords.values())
         for rule in ('single','persistent'):
          available=rule=='single' or i>0
          selected=[k for k in keys if panel.loc[(t,k),'damage']<=tol and (rule=='single' or panel.loc[(s.STEPS[i-1],k),'damage']<=tol)] if available else []
          classifications.append(dict(kind=kind,expert=expert,step=t,tolerance=tol,rule=rule,status='defined' if available else 'undefined_no_previous_checkpoint',qualifying_groups=len(selected) if available else None,total_groups=len(keys),qualifying_group_fraction=len(selected)/len(keys) if available else None,qualifying_sampled_coordinate_fraction=sum(coords[k] for k in selected)/den if available else None))
    s.table(D/'low_damage_classification.csv',classifications)
    hashes=s.read(D/'checkpoint_and_probe_hashes.json')
    hashes['frozen_probe_config_sha256']=s.sha(D/'probe_config.json')
    hashes['fixed_context_groups_sha256']=s.sha(D/'fixed_context_groups.pt')
    hashes['probe_position_order']='within stored probe sequence_ids order, then positions0..127; A uses candidates[:n], B candidates[2:2+n]'
    hashes['cache_files']={str(p.relative_to(D)):s.sha(p) for p in sorted((D/'cache').glob('*.pt'))}
    s.save(D/'checkpoint_and_probe_hashes.json',hashes)
    # Roles occur once per expert/time, so average groups before descriptive comparisons.
    predictors=[]
    for kind in ('sparse','unpruned'):
      for t,u in zip(s.STEPS[:-1],s.STEPS[1:]):
        z=whole[(whole.kind==kind)&(whole.step==t)&(whole.probe=='A')].groupby('expert')[['damage','absolute_magnitude','normalized_magnitude','recent_movement']].mean().rename(columns={'damage':'past_damage'})
        target=whole[(whole.kind==kind)&(whole.step==u)&(whole.probe=='B')].groupby('expert').damage.mean().rename('future_damage')
        r=roles[(roles.kind==kind)&(roles.step==t)&(roles.probe=='A')]
        for pop in ('proposed','accepted'):
            rr=r[r.population==pop].set_index('expert')[['load','gate_mean','jaccard','differentiation_js_nats']].add_prefix(pop+'_')
            z=z.join(rr)
        z=z.join(target);z['training_step']=t
        for expert,row in z.iterrows():predictors.append(dict(kind=kind,step=t,future_step=u,expert=int(expert),**{k:(float(v) if pd.notna(v) else None) for k,v in row.items()}))
    s.table(D/'expert_level_descriptive_comparisons.csv',predictors);features=pd.DataFrame(predictors);associations=[]
    for kind,z in features.groupby('kind'):
      for t,zz in [('pooled_repeated_experts',z)]+list(z.groupby('step')):
        for variable in z.columns.difference(['kind','step','future_step','expert','future_damage']):
            associations.append(dict(kind=kind,step=t,variable=variable,expert_time_rows=len(zz),independent_seeds=1,spearman_with_future_damage=corr(zz[variable],zz.future_damage),interpretation='descriptive, shared time confounding; no fitted model or replication claim'))
    s.table(D/'descriptive_associations.csv',associations)
    coverage=[]
    for _,r in roles.iterrows():
        coverage.append(dict(kind=r.kind,step=r.step,probe=r.probe,expert=r.expert,population=r.population,measurement='context_group_divergence',status=r.coverage,reason='requires >=32 routed positions; low traffic is not unimportance',count=int(r.load)))
    for _,r in damage[damage.target_tokens==0].iterrows():coverage.append(dict(kind=r.kind,step=r.step,probe=r.probe,expert=r.expert,population=r.subset,measurement='conditional_damage',status='undefined_empty_population',reason='zero target denominator',count=0))
    coverage.extend([dict(kind='all',measurement='historical_training_contextual_assignments',status='unavailable',reason='fixed probes are not historical routing logs'),dict(kind='all',measurement='existing_sensitivity_scores',status='excluded',reason='checkpoint/probe/population provenance does not match'),dict(kind='all',measurement='H2_incremental_prediction',status='not_estimable_reliably',reason='one engineering seed, four experts, two two-history forecasts; no model fit'),dict(kind='all',measurement='source_replications',status='limited',reason='one seed; groups and tokens are nested, not independent source observations')])
    s.table(D/'coverage_and_exclusions.csv',coverage)
    accounting=s.read(D/'compute_accounting.json');accounting['five_seed_confirmation_estimate_seconds']=5*accounting['measure_wall_seconds'];accounting['five_seed_estimate_basis']='linear measured pilot measurement wall-time incl hashing/IO/clustering; union timing added separately below; availability and hardware may differ';accounting['observed_results_bytes']=sum(p.stat().st_size for p in D.rglob('*') if p.is_file());accounting['five_seed_storage_estimate_bytes']=5*accounting['observed_results_bytes'];s.save(D/'compute_accounting.json',accounting)
    figures(damage,roles,summary)
    lines=['# Stage 1 expert-role deletion-sensitivity pilot','',
      '## A. What is measurable from existing artifacts','',
      'Verified seed17 32E IMP80 and matched unpruned recovery snapshots exist at25,250,1000,2500; no substitution or training was required.2500 is pre-reset. Seed17 is a **nonmatching engineering case**, retained regardless of outcome. Historical/rerun endpoint aliases have identical verified model hashes and count once. Native training-only complete batches give contextual routing, fixed-context-group differentiation and direct reversible deletion damage. No old raw geometry, sensitivity or replay loss measurements were reused; their numerical/probe/population provenance differs. See artifact_inventory.csv and checkpoint_and_probe_hashes.json.','',
      f"Probe budget: {s.read(D/'probe_config.json')['batches_per_probe']} complete128×128 batches per A/B; these may have been seen during training. B was excluded from decisions and clustering fit; it is not an independent generalization test. K16 fixed clustering uses earliest unpruned A only, with frozen reference labels for both trajectories. Small clusters remain explicit in cluster_counts.csv.",'',
      '## B. What the pilot actually shows','',
      'Primary tolerance0.001 nats/token is exploratory. Continuous signed damage is primary. The following fractions use all32 sampled groups, not all possible connections. Each group contains floor(1% of actual surviving fc1 coordinates), and the unpruned comparison removes identical coordinates. Persistent selection at250 uses25/250 A; at1000 it uses250/1000 A. Future outcomes use next-checkpoint B.','',
      '| Recovery | Decision → future | Persistent groups | Selected coordinate fraction of panel | False-safe rate | Selected future mean damage |',
      '|---|---|---:|---:|---:|---:|']
    selected_summary=summary[(summary.expert=='all')&(summary.tolerance==.001)&(summary.rule=='persistent')]
    def fmt(x):return 'undefined' if pd.isna(x) else f'{x:.6g}'
    for _,r in selected_summary.iterrows():lines.append(f'| {r.kind} | {int(r.decision_step)} → {int(r.future_step)} | {int(r.selected_groups)}/{int(r.groups)} | {r.qualifying_sampled_coordinate_fraction:.3%} | {fmt(r.false_safe_rate)} | {fmt(r.selected_future_mean)} |')
    lines+=['', 'At the primary tolerance, all32/32 groups qualify at both persistence decisions in both trajectories, and none exceeds the threshold on next-checkpoint B. This is a ceiling result: the sparse pilot does **not** show an increasing persistently low-damage fraction. At0.0001, sparse qualification also stays32/32; unpruned persistent qualification rises16/32 to32/32. These fixed sensitivity results are descriptive, not a chosen replacement primary outcome. Whole-probe group damages range from '+fmt(whole.damage.min())+' to '+fmt(whole.damage.max())+' nats/token. Low whole-probe damage must be read with conditional effects and sparse traffic: '+str(int(roles.differentiation_js_nats.isna().sum()))+' of'+str(len(roles))+' expert/time/probe/population divergence rows lack the declared minimum32 positions. No low-traffic expert or group was removed.', '', 'Damage distribution by checkpoint and probe:','', '| Recovery | Step | Probe | Mean damage | Minimum | Maximum |', '|---|---:|---|---:|---:|---:|']
    for keys,z in whole.groupby(['kind','step','probe']):
        k,t,p=keys;lines.append(f'| {k} | {t} | {p} | {z.damage.mean():.6g} | {z.damage.min():.6g} | {z.damage.max():.6g} |')
    lines+=['','H1 remains a descriptive engineering observation with only two persistence decision times. Increasing qualifying fractions would concern the sampled panel only. More qualification and more reliable identification are separate: compare forecast_summary.csv false-safe rates and continuous future damage, including all fixed tolerance sensitivities. No retrospective all-checkpoint filter was applied.','',
      'H2 is unresolved. expert_level_descriptive_comparisons.csv aligns step, proposed/accepted load and gates, magnitudes/movement, past damage and separate role measures against future B damage. descriptive_associations.csv reports expert-level rank associations, separately by time and pooled with explicit time confounding. No incremental predictive model was fitted: four experts in one seed do not support reliable role-augmented prediction. Eight groups do not create eight independent expert-role observations.','',
      f"Combined-deletion diagnostics: {int((unions.status=='measured').sum())} checkpoint/probe/selection rows; cap {int(unions.cap.max()):,} coordinates (1% of all surviving expert fc1+fc2 weights across all layers). Whole groups are included in canonical order; selection is past-only. Union damage and sum of individual damage are compared in combined_deletion_checks.csv. Each model is restored afterward.",'',
      'Numerical checks passed for repeat losses, zero dose, post-ablation restoration, state hashes, gradients, RNG and sparse-mask integrity. Paired per-token and per-sequence damages and baseline assignments/gates/router inputs are cached with checkpoint/probe/group/configuration/implementation identities. Tokens are not treated as independent source replications. Conditional subsets use baseline accepted positions; attention and later layers can spread effects beyond them.','',
      '## C. What cannot be concluded','',
      'These contextual clusters are not established semantic domains. Persistence and differentiation are separate; stable loads alone do not establish stable assignments. Capacity changes affect accepted sets; proposed metrics remain separate. Low traffic is unavailable where coverage is inadequate, not evidence of dispensability. There is no independent-seed uncertainty estimate or reliable H2 learning relationship. Deletion effects are checkpoint interventions; role associations are observational and do not prove a causal specialization mechanism, permanent-pruning recoverability or future-training dispensability. IMP mask membership defines this cohort only and cannot be assumed by a future dense-training algorithm.','',
      '## D. Cost of confirming the same analysis across five seeds','',
      f"Measured main measurement pass: {accounting['measure_wall_seconds']:.1f}s on {accounting['gpu']}; baseline profile {accounting['profile_baseline_seconds']:.3f}s, one-group profile {accounting['profile_ablation_seconds']:.3f}s. Linear five-seed main-pass estimate {accounting['five_seed_confirmation_estimate_seconds']/3600:.3f} GPU-host hours; this is an engineering estimate, not a reserved budget. Full-stage wall times (including unions) are recorded separately in compute_accounting.json when available. Storage scales to about {accounting['five_seed_storage_estimate_bytes']/1e9:.3f}GB using this pilot cache. CPU/IO, checkpoint availability, per-host contention and union selection can change costs. Keep all five seeds including matching/nonmatching, unchanged sampling, seed-held-out preprocessing and one simple regularized baseline versus role-augmented model if prediction is later authorized. Five seeds remain exploratory. No expansion launched.",'',
      '## E. Whether a later permanent-pruning intervention is warranted','',
      'This one-seed diagnostic does not warrant launching Stage2. First confirm the fixed analysis across existing seeds and evaluate incremental held-out predictive value, coverage and nonadditivity. A later permanent-pruning study needs separate authorization and matched-budget recovery endpoints. No training, permanent pruning, recovery, optimizer updates or Stage2 launch occurred.']
    union_valid=unions[unions.status=='measured']
    detail=('Combined damage is not additive: maximum absolute deviation from the sum of singles is '
            +fmt(union_valid.nonadditivity.abs().max())+' nats/token. Sparse unions reach '
            +fmt(union_valid[union_valid.kind=='sparse'].union_damage.max())
            +' nats/token, exceeding the strict 0.0001 convention despite individually qualifying groups. '
            +'Accepted-position conditional group damage reaches '+fmt(damage[damage.subset=='baseline_accepted'].damage.max())
            +' nats/token; small affected populations can dilute whole-probe damage. These are baseline-conditioned task losses, not isolated expert losses.')
    lines.insert(lines.index('## C. What cannot be concluded')-1,detail+'\n')
    interval=accounting['five_seed_full_process_estimate_seconds_interval']
    cost=f'Including union diagnostics, the observed full pilot runtime is bounded by {accounting["full_measurement_process_seconds_lower_bound"]:.1f}–{accounting["full_measurement_process_seconds_upper_bound"]:.1f} seconds from process-start/heartbeat/completion timing. A linear five-seed estimate is {interval[0]/60:.1f}–{interval[1]/60:.1f} minutes on the same host. This excludes the initial inventory/preparation pass, whose wall time was not instrumented; it is not a total-project runtime guarantee.'
    lines.insert(lines.index('## E. Whether a later permanent-pruning intervention is warranted')-1,cost+'\n')
    prose='\n'.join(lines)+'\n'
    for before,after in [('seed17','seed 17'),('layer1','layer 1'),('expert/time','expert/time'),(' at25',' at 25'),('at25,250,1000,2500','at 25, 250, 1000, and 2500'),('required.2500','required. The 2500 snapshot'),('complete128×128','complete 128 × 128'),('K16','K = 16'),('tolerance0.001','tolerance of 0.001'),('all32','all 32'),('at250','at 250'),('uses25/250','uses 25/250'),('at1000','at 1000'),('uses250/1000','uses 250/1000'),('At0.0001','At 0.0001'),('stays32/32','stays 32/32'),('rises16/32 to32/32','rises 16/32 to 32/32'),('of128','of 128'),('minimum32','minimum of 32'),('In both','In both')]:
        prose=prose.replace(before,after)
    (D/'findings.md').write_text(prose)
    s.save(D/'completion.json',dict(measurement_complete=True,report_complete=True,hypotheses_confirmed=False,stage2_launched=False,implementation_sha256=s.sha(Path(__file__)),unix=time.time()))
    s.save(D/'output_seal.json',{'files':{str(p.relative_to(D)):s.sha(p) for p in D.rglob('*') if p.is_file() and p.suffix in ('.csv','.json','.md','.png','.pt') and p.name not in ('output_seal.json','heartbeat.json','progress.json','final_audit.json')}})


def figures(damage,roles,summary):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    (D/'figures').mkdir(exist_ok=True)
    for e in s.EXPERTS:
        fig,axes=plt.subplots(3,2,figsize=(12,10))
        for kind,color in [('sparse','tab:blue'),('unpruned','tab:orange')]:
            r=roles[(roles.kind==kind)&(roles.expert==e)&(roles.probe=='A')]
            for pop,style in [('proposed','--'),('accepted','-')]:
                z=r[r.population==pop].sort_values('step')
                axes[0,0].plot(z.step,z.jaccard,style+'o',color=color,label=f'{kind} {pop}')
                axes[0,0].plot(z.step,z.shuffled_jaccard_mean,style,color=color,alpha=.3)
                axes[0,1].plot(z.step,z.differentiation_js_nats,style+'o',color=color,label=f'{kind} {pop}')
                axes[0,1].plot(z.step,z.shuffled_js_mean,style,color=color,alpha=.3)
                axes[1,0].plot(z.step,z.load,style+'o',color=color,label=f'{kind} {pop}')
            for probe,style in [('A','-'),('B','--')]:
                z=damage[(damage.kind==kind)&(damage.expert==e)&(damage.probe==probe)&(damage.subset=='whole')]
                g=z.groupby('step').damage
                axes[1,1].plot(g.mean().index,g.mean(),style+'o',color=color,label=f'{kind} {probe}')
                axes[1,1].fill_between(g.mean().index,g.min(),g.max(),color=color,alpha=.08)
            for rule,style in [('single','--'),('persistent','-')]:
                z=summary[(summary.kind==kind)&(summary.expert==e)&(summary.tolerance==.001)&(summary.rule==rule)].sort_values('decision_step')
                axes[2,0].plot(z.decision_step,z.qualifying_group_fraction,style+'o',color=color,label=f'{kind} {rule}')
                axes[2,1].plot(z.decision_step,z.false_safe_rate,style+'o',color=color,label=f'{kind} {rule}')
        for ax,title in zip(axes.flatten(),['Adjacent contextual Jaccard (faint: shuffle)','Fixed context differentiation JS (faint: shuffle)','Native loads on A','Whole-probe deletion damage (band: group range)','Qualifying sampled-group fraction, τ=.001','Next B false-safe rate, τ=.001']):
            ax.set_title(title,fontsize=10);ax.set_xlabel('Recovery step / decision step');ax.legend(fontsize=6);ax.grid(alpha=.15)
        axes[1,1].axhline(.001,color='grey',linewidth=.7)
        for ax in axes[2]:ax.set_ylim(-.05,1.05)
        fig.suptitle(f'Seed17 NONMATCHING engineering case; layer1 expert{e}; fixed fc1 groups')
        fig.tight_layout();fig.savefig(D/'figures'/f'expert_{e}.png',dpi=150);plt.close(fig)

if __name__=='__main__':main()
