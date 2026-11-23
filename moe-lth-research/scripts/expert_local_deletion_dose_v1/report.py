"""Continuous equal-dose A/B reliability; no cross-dose ranking or seed inference."""
from dose_common import *
import pandas as pd
from scipy.stats import spearmanr


def agreement(a,b,floor=1e-12):
    a=np.asarray(a,dtype=float);b=np.asarray(b,dtype=float);valid=np.isfinite(a)&np.isfinite(b);a=a[valid];b=b[valid];n=len(a)
    if not n:return dict(pairs=0,spearman=None,mae=None,rmse=None,bias_B_minus_A=None,pairwise_concordance=None,comparable_pairs=0,both_tied_pairs=0,one_probe_tied_pairs=0,sign_agreement=None)
    error=b-a;concordant=comparable=both_tied=one_tied=0
    for i in range(n):
        for j in range(i+1,n):
            da=a[i]-a[j];db=b[i]-b[j];ta=abs(da)<=floor;tb=abs(db)<=floor
            if ta and tb:both_tied+=1
            elif ta or tb:one_tied+=1
            else:comparable+=1;concordant+=int(da*db>0)
    scale=max(float(np.ptp(a)),float(np.ptp(b)))
    rho=float(spearmanr(a,b).statistic) if n>=3 and np.ptp(a)>floor and np.ptp(b)>floor else None
    sign=lambda x:np.where(np.abs(x)<=floor,0,np.sign(x))
    return dict(pairs=n,spearman=rho,mae=float(np.abs(error).mean()),rmse=float(np.sqrt(np.mean(error**2))),bias_B_minus_A=float(error.mean()),range_A=float(np.ptp(a)),range_B=float(np.ptp(b)),normalized_mae=float(np.abs(error).mean())/scale if scale>floor else None,pairwise_concordance=concordant/comparable if comparable else None,comparable_pairs=comparable,both_tied_pairs=both_tied,one_probe_tied_pairs=one_tied,sign_agreement=float((sign(a)==sign(b)).mean()),nonzero_A=int((np.abs(a)>floor).sum()),nonzero_B=int((np.abs(b)>floor).sum()),rank_status='defined' if rho is not None else 'undefined_small_or_constant_panel')


def reliability(df,floor):
    rows=[]
    for coverage in ('all_retained','adequate_in_both_probes'):
      for subset,ss in df.groupby('subset'):
       for metric in ('damage','output_rms_difference','relative_output_rms_difference','gate_weighted_rms_difference'):
        for (kind,t,typ,dose),z in ss.groupby(['trajectory','step','intervention_kind','dose_groups'],dropna=False):
            a=z[z.probe=='A'].set_index('intervention');b=z[z.probe=='B'].set_index('intervention');paired=a[[metric,'accepted_positions','expert']].join(b[[metric,'accepted_positions']],lsuffix='_A',rsuffix='_B',how='inner')
            if coverage=='adequate_in_both_probes':paired=paired[(paired.accepted_positions_A>=MIN_ACCEPTED)&(paired.accepted_positions_B>=MIN_ACCEPTED)]
            panels=[('across_experts_equal_dose',None,paired)]+[('within_expert_equal_dose',e,paired[paired.expert==e]) for e in s.EXPERTS]
            for scope,e,p in panels:
                rows.append(dict(trajectory=kind,step=int(t),intervention_kind=typ,dose_groups=int(dose) if pd.notna(dose) else None,subset=subset,metric=metric,scope=scope,expert=e,coverage=coverage,independent_seeds=1,deduplicated=True,repeat_floor=floor,**agreement(p[metric+'_A'],p[metric+'_B'],floor)))
    return rows


def fmt(x):return 'undefined' if pd.isna(x) else f'{x:.5g}'


def main():
    started=time.perf_counter();verify_registration()
    df=pd.read_csv(D/'measurements.csv');checks=s.read(D/'numerical_checks.json')
    maxrepeat=max(r['repeat_loss_max_abs'] for r in checks['checks'] if r['repeat_tested']);floor=max(1e-12,maxrepeat)
    rel=reliability(df,floor);table(D/'ab_reliability.csv',rel);rr=pd.DataFrame(rel)
    # Functional metrics repeat over the three CE subsets: reliability consumes only whole rows for summary.
    nonadd=[]
    weights=df[df.intervention_kind=='weight_union']
    for keys,z in weights.groupby(['trajectory','step','probe','dose_groups','subset']):
        k,t,p,dose,subset=keys;v=z.nonadditivity.dropna();nonadd.append(dict(trajectory=k,step=int(t),probe=p,dose_groups=int(dose),subset=subset,unique_unions=len(z),defined=len(v),mean_nonadditivity=float(v.mean()) if len(v) else None,min_nonadditivity=float(v.min()) if len(v) else None,max_nonadditivity=float(v.max()) if len(v) else None,max_abs_nonadditivity=float(v.abs().max()) if len(v) else None,mean_union_damage=float(z.damage.mean()) if z.damage.notna().any() else None,mean_sum_individual_damage=float(z.sum_individual_damage.mean()) if z.sum_individual_damage.notna().any() else None))
    table(D/'nonadditivity_summary.csv',nonadd)
    dose_summary=[]
    for keys,z in df.groupby(['trajectory','step','probe','intervention_kind','dose_groups','subset'],dropna=False):
        kind,t,p,typ,dose,subset=keys
        dose_summary.append(dict(trajectory=kind,step=int(t),probe=p,intervention_kind=typ,dose_groups=int(dose) if pd.notna(dose) else None,subset=subset,unique_interventions=len(z),defined_damage=int(z.damage.notna().sum()),mean_damage=float(z.damage.mean()) if z.damage.notna().any() else None,median_damage=float(z.damage.median()) if z.damage.notna().any() else None,min_damage=float(z.damage.min()) if z.damage.notna().any() else None,max_damage=float(z.damage.max()) if z.damage.notna().any() else None,low_traffic_or_empty=int((z.accepted_positions<MIN_ACCEPTED).sum())))
    table(D/'dose_summary.csv',dose_summary)
    coverage=df[df.subset=='whole'][['trajectory','step','probe','expert','intervention','intervention_kind','dose_groups','accepted_positions','output_coverage','relative_rms_status','coordinates','active_matrix_fraction']]
    coverage.to_csv(D/'coverage.csv',index=False)
    # Explicit matched eight-group temporal contrasts, without pooling doses or claiming monotonicity.
    temporal=[]
    for keys,z in weights[(weights.dose_groups==8)].groupby(['trajectory','expert','probe','subset']):
        kind,e,p,subset=keys;z=z.set_index('step')
        for start,end in zip(s.STEPS[:-1],s.STEPS[1:]):
            a=z.loc[start,'damage'];b=z.loc[end,'damage']
            temporal.append(dict(trajectory=kind,expert=int(e),probe=p,subset=subset,start_step=start,end_step=end,start_damage=float(a) if pd.notna(a) else None,end_damage=float(b) if pd.notna(b) else None,damage_change=float(b-a) if pd.notna(a) and pd.notna(b) else None,accepted_start=int(z.loc[start,'accepted_positions']),accepted_end=int(z.loc[end,'accepted_positions'])))
    table(D/'temporal_dependence.csv',temporal)
    figures(df,rr)
    accounting=s.read(D/'compute_accounting.json');accounting['report_wall_seconds_before_final_seal']=time.perf_counter()-started;accounting['storage_bytes_before_final_seal']=sum(p.stat().st_size for p in D.rglob('*') if p.is_file());accounting['new_cache_bytes']=sum(p.stat().st_size for p in (D/'cache').glob('*.pt'));save(D/'compute_accounting.json',accounting)
    whole=weights[(weights.subset=='whole')&(weights.dose_groups>0)]
    rank=rr[(rr.subset=='whole')&(rr.metric=='damage')&(rr.scope=='within_expert_equal_dose')&(rr.coverage=='adequate_in_both_probes')&(rr.dose_groups.isin([1,2,4]))]
    defined=rank.spearman.dropna()
    lines=['# Expert-local deletion-dose and reliability check','',
      'Seed 17 is the same **nonmatching 32E IMP80 engineering case**. Both trajectories use steps 25, 250, 1000 and pre-reset 2500, layer 1, experts 0/8/16/24, original fc1 groups and complete A/B batches. Both A and B have already been examined. No training, optimizer update, permanent pruning, predictor, clustering refit, extra seed or Stage 2 ran.','',
      'Four damage-independent group permutations generate nested 0/1/2/4/8-group doses. Identical physical unions are evaluated once; ordering aliases remain in ordering_aliases.csv. The prior global union cap is absent. Manifest fractions distinguish sparse surviving-matrix dose from the smaller coordinate-matched unpruned fraction. Output-zero controls are separate interventions, not extra weight-deletion doses.','',
      '## Verdict','',
      f'**Numerical measurement is reproducible.** Repeated same-input interventions have maximum token-loss disagreement {maxrepeat:.6g} nats; zero dose, Stage 1 cached baselines/single-group checks, restoration, full hashes, native layer-1 assignments/gates, RNG, gradients and sparse masks pass. Across positive weight doses, whole-probe damage spans {fmt(whole.damage.min())} to {fmt(whole.damage.max())} nats/token. Output RMS changes and positive controls establish whether a small whole-probe effect accompanies a real function change; a zero-traffic expert has undefined output RMS rather than fabricated zero dependence.','',
      f'**Equal-dose A/B rankings are a separate question.** Among adequate-coverage within-expert panels at 1/2/4 groups, {len(defined)}/{len(rank)} CE rank correlations are defined; their median is {fmt(defined.median())}, range {fmt(defined.min())} to {fmt(defined.max())}. These combine neither doses nor duplicate orderings and are descriptive nested observations, not independent source replications. Stratified values below and ab_reliability.csv take precedence over a pooled impression. Exact-repeat reliability does not establish cross-population ranking reliability.','',
      '**Whole-probe averaging can conceal conditional effects.** Baseline-accepted and complement effects, accepted-token counts and per-sequence paired deltas are explicit. The conditional subset stays fixed before deletion; it is still a task-loss subset, not an isolated expert loss. Read low-coverage effects with their token and sequence counts.','',
      '**Dependence changes over recovery must be read expert by expert.** temporal_dependence.csv holds adjacent-time contrasts for the same eight-group coordinate set, together with traffic. The tables and per-expert plots below show signed changes rather than assuming monotonic dose response or a common recovery trend. Early high loss or absent traffic can dominate a pooled comparison. No specialization interpretation follows.','',
      '**Next step:** this check can justify cross-seed confirmation of the *measurement method*, but does not justify a specialization predictor or Stage 2. Improve or explicitly retain unresolved low-traffic coverage and verify equal-dose A/B rankings across seeds before making dispensability forecasts. No additional seed is launched here.','',
      '## Equal-dose CE agreement','',
      'Across experts, unique physical unions only; adequate coverage in both probes. Correlations across different doses are deliberately absent. Empty/constant or fewer-than-three-point rank panels are undefined. Within-expert versions, functional-output agreement, sign agreement, pairwise concordance and error scales are in ab_reliability.csv.','',
      '| Trajectory | Step | Groups | A/B pairs | Spearman | MAE (nats/token) | Pair order concordance |','|---|---:|---:|---:|---:|---:|---:|']
    view=rr[(rr.subset=='whole')&(rr.metric=='damage')&(rr.scope=='across_experts_equal_dose')&(rr.coverage=='adequate_in_both_probes')&(rr.intervention_kind=='weight_union')&(rr.dose_groups>0)]
    for r in view.itertuples():lines.append(f'| {r.trajectory} | {r.step} | {int(r.dose_groups)} | {r.pairs} | {fmt(r.spearman)} | {fmt(r.mae)} | {fmt(r.pairwise_concordance)} |')
    lines+=['','## Fixed eight-group dose and output-zero control','',
      'Mean and range across the four sampled experts, not a population estimate; conditional means omit undefined empty populations but the count of defined cases remains explicit in dose_summary.csv. Low traffic is retained.','',
      '| Trajectory | Step | Probe | Intervention | Whole CE mean | Accepted CE mean | Accepted CE range |','|---|---:|---|---|---:|---:|---:|']
    for keys,z in df[(df.dose_groups==8)|(df.intervention_kind=='expert_output_zero')].groupby(['trajectory','step','probe','intervention_kind']):
        kind,t,p,typ=keys;w=z[z.subset=='whole'];a=z[z.subset=='baseline_accepted']
        lines.append(f'| {kind} | {t} | {p} | {"8 groups" if typ=="weight_union" else "output zero (separate control)"} | {fmt(w.damage.mean())} | {fmt(a.damage.mean())} | {fmt(a.damage.min())} … {fmt(a.damage.max())} |')
    union=df[(df.intervention_kind=='weight_union')&(df.dose_groups>=2)]
    lines+=['','## Nonadditivity, resolution and limitations','',
      f'Maximum absolute union-minus-sum difference is {fmt(union[union.subset=="whole"].nonadditivity.abs().max())} nats/token whole-probe and {fmt(union[union.subset=="baseline_accepted"].nonadditivity.abs().max())} on accepted positions. This is the directly measured difference, not assumed additivity. Nonadditivity is undefined for output-zero controls. No threshold is used to choose groups or create a trend; 0.001 and 0.0001 are descriptive plot reference lines only.','',
      'Raw expert-output RMS is measured on baseline-accepted native-dispatch positions and all output channels. Relative RMS uses the corresponding baseline expert RMS; values at or below 1e-12 are undefined. Gate-weighted RMS is the RMS of baseline gate times output difference. Baseline expert RMS and baseline gate-weighted RMS are also stored. The expert-only replay for cached CE doses never reroutes isolated tokens. Positive-control routing/gates remain bitwise unchanged at the affected layer; later layers may change.','',
      'A/B disagreement reflects distinct, previously examined diagnostic contexts as well as traffic. Exact repeatability measures numerical stability, not sampling uncertainty or independent generalization. Eight groups, four orderings, token positions, sequences and checkpoints are not independent seed replications. An unpruned comparison deletes the same coordinates at a different active-matrix fraction. No causal specialization or permanent-pruning recoverability claim is supported.','',
      '## Runtime and storage','',
      f'Preparation {accounting["prepare_wall_seconds"]:.2f}s; complete measurement and table serialization {accounting["measure_wall_seconds"]:.2f}s on {accounting["gpu"]}; report/figures before sealing {accounting["report_wall_seconds_before_final_seal"]:.2f}s. {accounting["complete_batch_forwards"]} native complete-batch forwards; {accounting["cached_single_group_measurements_reused"]} single-group CE measurements reused. CUDA event evaluation intervals total {accounting["cuda_event_evaluation_seconds"]:.2f}s (includes orchestration/transfers between device events, not a kernel-only bill). New cache {accounting["new_cache_bytes"]/1e6:.2f} MB; total outputs before final seal {accounting["storage_bytes_before_final_seal"]/1e6:.2f} MB. Source Stage 1 caches and model snapshots are referenced, not copied. Actual byte counts and stage timing remain in compute_accounting.json.']
    (D/'findings.md').write_text('\n'.join(lines)+'\n')
    save(D/'completion.json',dict(measurement_complete=True,report_complete=True,training_updates=0,stage2_launched=False,additional_seeds_launched=False,unix=time.time()))
    save(D/'output_seal.json',dict(files={str(p.relative_to(D)):sha(p) for p in D.rglob('*') if p.is_file() and p.suffix in ('.csv','.json','.md','.png','.pt') and p.name not in ('output_seal.json','final_audit.json','heartbeat.json','progress.json') and not p.name.startswith('failure_')}))
    print('Equal-dose reliability report complete',flush=True)


def figures(df,rr):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    (D/'figures').mkdir(exist_ok=True)
    weights=df[df.intervention_kind=='weight_union'];colors={'sparse':'tab:blue','unpruned':'tab:orange'}
    for e in s.EXPERTS:
        fig,axes=plt.subplots(4,3,figsize=(14,14))
        for row,t in enumerate(s.STEPS):
            for kind in ('sparse','unpruned'):
              for probe,style in [('A','-'),('B','--')]:
                z=weights[(weights.expert==e)&(weights.step==t)&(weights.trajectory==kind)&(weights.probe==probe)]
                for col,subset in enumerate(('whole','baseline_accepted')):
                    zz=z[z.subset==subset]
                    for dose,q in zz.groupby('dose_groups'):axes[row,col].scatter([dose]*len(q),q.damage,color=colors[kind],alpha=.45,s=12)
                    g=zz.groupby('dose_groups').damage.mean();axes[row,col].plot(g.index,g,style+'o',color=colors[kind],label=f'{kind} {probe}')
                zz=z[z.subset=='whole'];g=zz.groupby('dose_groups').relative_output_rms_difference.mean();axes[row,2].plot(g.index,g,style+'o',color=colors[kind],label=f'{kind} {probe}')
            for col in (0,1):
                axes[row,col].axhline(.001,color='grey',ls=':',lw=.6,label='.001 reference');axes[row,col].axhline(.0001,color='grey',ls='-.',lw=.6,label='.0001 reference')
                axes[row,col].axhline(0,color='black',lw=.4)
            for col,title in enumerate(('Whole-probe signed CE','Baseline-accepted signed CE','Relative expert-output RMS')):
                axes[row,col].set_title(f'Step {t}: {title}',fontsize=10);axes[row,col].set_xticks(DOSES);axes[row,col].set_xlabel('Original groups in this expert matrix');axes[row,col].grid(alpha=.15)
                if row==0:axes[row,col].legend(fontsize=6)
        fig.suptitle(f'Seed 17 NONMATCHING engineering case · layer 1 expert {e}\nUnique unions; dots = unions, lines = mean; output-zero control separate')
        fig.tight_layout();fig.savefig(D/'figures'/f'expert_{e}_dose_damage.png',dpi=150);plt.close(fig)
    # A/B equality within each fixed dose; do not mix doses in one fitted rank.
    fig,axes=plt.subplots(2,4,figsize=(14,7))
    for row,kind in enumerate(('sparse','unpruned')):
      for col,dose in enumerate((1,2,4,8)):
        ax=axes[row,col]
        for t in s.STEPS:
            z=weights[(weights.trajectory==kind)&(weights.dose_groups==dose)&(weights.subset=='whole')&(weights.step==t)]
            p=z.pivot(index='intervention',columns='probe',values='damage');ax.scatter(p.A,p.B,label=f'step {t}',s=15)
        lo,hi=ax.get_xlim();bl,bh=ax.get_ylim();lo=min(lo,bl);hi=max(hi,bh);ax.plot([lo,hi],[lo,hi],color='grey',ls=':',lw=.7)
        ax.set_title(f'{kind}: {dose} groups');ax.set_xlabel('A CE damage');ax.set_ylabel('B CE damage');ax.ticklabel_format(style='sci',axis='both',scilimits=(0,0));ax.legend(fontsize=6)
    fig.suptitle('Equal-dose A/B agreement · checkpoints colored separately · no pooled rank correlation');fig.tight_layout();fig.savefig(D/'figures/ab_equal_dose.png',dpi=150);plt.close(fig)
    control=df[(df.intervention_kind=='expert_output_zero')&(df.subset=='whole')]
    fig,axes=plt.subplots(1,2,figsize=(12,4))
    for ax,kind in zip(axes,('sparse','unpruned')):
        for e in s.EXPERTS:
            for p,style in [('A','-'),('B','--')]:
                z=control[(control.trajectory==kind)&(control.expert==e)&(control.probe==p)].sort_values('step');ax.plot(z.step,z.damage,style+'o',label=f'e{e} {p}')
        ax.set_title(f'{kind}: separate output-zero control');ax.set_xlabel('Recovery step');ax.set_ylabel('Signed whole-probe CE damage');ax.legend(fontsize=6);ax.grid(alpha=.15)
    fig.tight_layout();fig.savefig(D/'figures/output_zero_control.png',dpi=150);plt.close(fig)

if __name__=='__main__':
    with worker():main()
