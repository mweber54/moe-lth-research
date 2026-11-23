"""Figures and exhaustive, descriptive temporal interpretation; no model execution."""
import json
import time
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from prepare import ROOT,D,SOURCE,sha,save

COLORS={'current':'#2469aa','history_mean':'#de8722','history_max':'#3c9766'}

def table(name): return pd.read_csv(D/f'{name}.csv')

def format_ci(row,prefix=''):
    return f"{row[prefix+'ci95_low']:.3e} to {row[prefix+'ci95_high']:.3e}"

def figures():
    out=D/'figures';out.mkdir(exist_ok=True)
    s=table('temporal_scores_and_outcomes');e=table('selected_high_low_future_contrasts')
    p=table('low_damage_selection_comparison');st=table('selection_stability');th=table('false_safe_summary');cov=table('temporal_coverage')
    for (trajectory,transition,subset),z in s[s.intervention_kind=='weight_union'].groupby(['trajectory','transition','subset']):
        fig,axes=plt.subplots(4,5,figsize=(17,12),squeeze=False)
        for i,expert in enumerate([0,8,16,24]):
            for j,dose in enumerate([0,1,2,4,8]):
                ax=axes[i,j];a=z[(z.expert==expert)&(z.dose_groups==dose)]
                for rule,b in a.groupby('rule'):
                    ax.scatter(b.prediction_score,b.future_confirmation_damage,color=COLORS[rule],label=rule,s=22,alpha=.8)
                if len(a) and not a.coverage_qualified.all():
                    ax.set_facecolor('#fff0d9');ax.text(.02,.97,'coverage flag',transform=ax.transAxes,va='top',fontsize=7)
                ax.axhline(0,color='.8',lw=.6);ax.axvline(0,color='.8',lw=.6)
                ax.ticklabel_format(style='sci',axis='both',scilimits=(0,0));ax.set_title(f'expert{expert}, dose{dose} (n={a.intervention.nunique()})',fontsize=9)
                if j==0: ax.set_ylabel('future signed CE damage')
                if i==3: ax.set_xlabel('whole discovery score')
        axes[0,0].legend(fontsize=6)
        fig.suptitle(f'{trajectory}: {transition}; future {subset}; separate equal-dose panels')
        fig.tight_layout();fig.savefig(out/f'scores_{trajectory}_{transition}_{subset}.png',dpi=140);plt.close(fig)
    for subset in ['whole','baseline_accepted']:
      for trajectory in ['sparse','unpruned']:
        fig,axes=plt.subplots(4,3,figsize=(14,12),squeeze=False)
        for i,expert in enumerate([0,8,16,24]):
          for j,tr in enumerate(['25_to_250','250_to_1000','1000_to_2500']):
            ax=axes[i,j];z=e[(e.trajectory==trajectory)&(e.expert==expert)&(e.transition==tr)&(e.subset==subset)]
            for off,(rule,a) in enumerate(z.groupby('rule')):
                x=a.dose_groups.to_numpy()+(.13*(off-1));y=a.future_high_minus_low.to_numpy()
                lo=a.ci95_low.to_numpy();hi=a.ci95_high.to_numpy()
                ax.vlines(x,lo,hi,color=COLORS[rule]);ax.scatter(x,y,color=COLORS[rule],s=18,label=rule)
            if len(z) and not z.coverage_qualified.all():
                ax.set_facecolor('#fff0d9');ax.text(.02,.97,'some rules fail coverage',transform=ax.transAxes,va='top',fontsize=7)
            if len(z) and z.extreme_selection_unidentified.any():
                ax.text(.02,.87,'tied choices unidentified',transform=ax.transAxes,va='top',fontsize=7)
            ax.axhline(0,color='.4',lw=.7);ax.ticklabel_format(style='sci',axis='y',scilimits=(0,0))
            ax.set_title(f'expert{expert}, {tr}',fontsize=9);ax.set_xticks([1,2,4]);ax.set_xlabel('dose (groups)')
        axes[0,1].legend(fontsize=7)
        fig.suptitle(f'{trajectory}: {subset}; future high−low, fixed-selection paired batch95% intervals')
        fig.tight_layout();fig.savefig(out/f'future_contrasts_{trajectory}_{subset}.png',dpi=140);plt.close(fig)
    for metric,prefix,label in [('selection_advantage','advantage_','panel-uniform minus selected'),('current_minus_rule_future_damage','current_minus_rule_','current-selected minus rule-selected')]:
      for trajectory in ['sparse','unpruned']:
        fig,axes=plt.subplots(4,3,figsize=(14,12),squeeze=False)
        for i,expert in enumerate([0,8,16,24]):
          for j,tr in enumerate(['25_to_250','250_to_1000','1000_to_2500']):
            ax=axes[i,j];z=p[(p.trajectory==trajectory)&(p.expert==expert)&(p.transition==tr)&(p.subset=='whole')]
            for off,(rule,a) in enumerate(z.groupby('rule')):
                x=a.dose_groups.to_numpy()+.13*(off-1);y=a[metric].to_numpy()
                ax.vlines(x,a[prefix+'ci95_low'],a[prefix+'ci95_high'],color=COLORS[rule]);ax.scatter(x,y,color=COLORS[rule],s=18,label=rule)
            if len(z) and not z.coverage_qualified.all():
                ax.set_facecolor('#fff0d9');ax.text(.02,.97,'some rules fail coverage',transform=ax.transAxes,va='top',fontsize=7)
            if len(z) and z.low_selection_unidentified.any():
                ax.text(.02,.87,'low choice unidentified',transform=ax.transAxes,va='top',fontsize=7)
            ax.axhline(0,color='.4',lw=.7);ax.ticklabel_format(style='sci',axis='y',scilimits=(0,0));ax.set_xticks([1,2,4])
            ax.set_title(f'expert{expert}, {tr}',fontsize=9)
        axes[0,1].legend(fontsize=7);fig.suptitle(f'{trajectory}: whole future {label}; paired95% intervals')
        fig.tight_layout();fig.savefig(out/f'{metric}_{trajectory}.png',dpi=140);plt.close(fig)
    for trajectory in ['sparse','unpruned']:
        fig,axes=plt.subplots(4,3,figsize=(14,10),squeeze=False)
        for i,expert in enumerate([0,8,16,24]):
          for j,tr in enumerate(['25_to_250','250_to_1000','1000_to_2500']):
            ax=axes[i,j];z=st[(st.trajectory==trajectory)&(st.expert==expert)&(st.transition==tr)&(st.subset=='whole')&st.original_low]
            for rule,a in z.groupby('rule'):ax.plot(a.dose_groups,a.low_selection_frequency,'o-',label=rule,color=COLORS[rule],lw=.7)
            ax.set_ylim(-.05,1.05);ax.set_xticks([1,2,4]);ax.set_title(f'expert{expert}, {tr}',fontsize=9)
            if len(z) and not z.coverage_qualified.all():ax.set_facecolor('#fff0d9')
        axes[0,1].legend(fontsize=7);fig.suptitle(f'{trajectory}: frequency of original low choice; amber = coverage failure')
        fig.tight_layout();fig.savefig(out/f'selection_stability_{trajectory}.png',dpi=140);plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(14,5))
    for ax,(trajectory,z) in zip(axes,cov.groupby('trajectory')):
        z=z.drop_duplicates(['expert','checkpoint','partition']).copy();z['column']=z.checkpoint.astype(str)+' '+z.partition
        v=z.pivot(index='expert',columns='column',values='minimum_block_accepted').reindex([0,8,16,24])
        ax.imshow(np.log10(v.to_numpy()+1),aspect='auto',cmap='Blues')
        for i in range(v.shape[0]):
            for j in range(v.shape[1]):ax.text(j,i,str(int(v.iloc[i,j])),ha='center',va='center',fontsize=8,color='red' if v.iloc[i,j]<32 else 'black')
        ax.set_xticks(range(len(v.columns)),v.columns,rotation=65,ha='right');ax.set_yticks(range(4),v.index);ax.set_title(trajectory)
    fig.suptitle('Minimum native accepted positions per four-batch block (red <32)');fig.tight_layout();fig.savefig(out/'temporal_coverage.png',dpi=140);plt.close(fig)
    for tol,z in th[(th.subset=='whole')&th.dose_groups.isin([1,2,4])].groupby('tolerance'):
        rows=[];labels=[]
        for key,a in z.groupby(['trajectory','transition','expert','rule']):
            a=a.set_index('dose_groups').reindex([1,2,4]);rows.append(a.selected_count/a.candidate_count);labels.append('/'.join(map(str,key)))
        vals=np.array(rows);fig,ax=plt.subplots(figsize=(9,max(9,len(rows)*.21)))
        ax.imshow(vals,aspect='auto',vmin=0,vmax=1,cmap='Blues')
        for i,(_,a) in enumerate(z.groupby(['trajectory','transition','expert','rule'])):
            for j,(_,b) in enumerate(a.sort_values('dose_groups').iterrows()):
                fs='undef' if pd.isna(b.false_safe_fraction) else f'{b.false_safe_fraction:.0%}'
                ax.text(j,i,f'{int(b.selected_count)}/{int(b.candidate_count)}; FS={fs}'+(' CEILING' if b.status=='ceiling_no_discrimination' else ''),ha='center',va='center',fontsize=7)
        ax.set_yticks(range(len(labels)),labels,fontsize=6);ax.set_xticks(range(3),['dose1','dose2','dose4']);ax.set_title(f'Whole future: tolerance{tol:.0e}; qualifying unions and point false-safe(FS)')
        fig.tight_layout();fig.savefig(out/f'threshold_{tol:.0e}.png',dpi=130);plt.close(fig)

def main():
    started=time.perf_counter();figures()
    e=table('selected_high_low_future_contrasts');p=table('low_damage_selection_comparison');r=table('temporal_rank_agreement')
    th=table('false_safe_summary');cov=table('temporal_coverage');s=table('temporal_scores_and_outcomes')
    e['positive']=(e.ci95_low>0)&~e.extreme_selection_unidentified
    e['negative']=(e.ci95_high<0)&~e.extreme_selection_unidentified
    e['stable_positive']=e.future_block_contrasts.map(lambda v:all(x is not None and x>0 for x in json.loads(v)))&e.positive
    lines=['# Exploratory temporal deletion damage findings','',
        'All measurements and both context populations were previously inspected. This is a post hoc checkpoint-only analysis of the seed17 NONMATCHING engineering case. No model execution, training, new probes or new model seeds occurred. Primary transitions are250→1000 and1000→2500;25→250 is secondary. Step2500 is pre-reset. Identical endpoint aliases count once.','',
        'Prediction scores always use whole-probe discovery damage. The same selected IDs are evaluated on whole and future-native baseline-accepted positions. Future accepted populations change over time; they are not the same routed population. Sparse and unpruned trajectories, experts, doses and transitions remain separate. Exact per-union magnitude is unavailable; expert load is constant across unions within a panel and is context only.','',
        'Tables contain every registered panel, including singleton trajectories and low-traffic cases. Dose1 has3 distinct unions, doses2/4 have4, doses0/8 and output-zero have1. Output-zero is a separate positive control. Singleton panels have no ranks, selected contrasts or policy comparisons. Overlapping unions are dependent. All95% intervals use2000 paired complete-batch resamples; they describe sampled contexts within one source seed, with no multiplicity correction.','',
        '## Current-damage temporal ordering','',
        'Counts below audit separate panels; they are not pooled correlations, independent successes or model-level replications. A positive fixed-selection contrast supports ordering; consistent signs across all four future blocks are a stricter descriptive check. Tied extrema are unidentified.','',
        '|Trajectory|Transition|Outcome|Covered panels|Positive fixed CI|Negative fixed CI|Positive CI and all4 blocks positive|','|---|---|---|---:|---:|---:|---:|']
    cur=e[e.rule=='current']
    for key,z in cur.groupby(['trajectory','transition','subset']):
        a=z[z.coverage_qualified]
        lines.append('|'+ '|'.join(map(str,[*key,len(a),int(a.positive.sum()),int(a.negative.sum()),int(a.stable_positive.sum())]))+'|')
    named=cur[(cur.subset=='whole')&cur.coverage_qualified&cur.stable_positive]
    lines += ['','Every covered whole-outcome current-rule condition with a positive fixed-selection interval and all four future block contrasts positive is listed below. These are exploratory named candidates, not preselected confirmatory discoveries.','',
        '|Trajectory|Transition|Expert|Dose|Spearman|Future high−low (nats/token)|95% interval|Reselected95% interval|','|---|---|---:|---:|---:|---:|---|---|']
    for _,row in named.iterrows():
        rr=r[(r.trajectory==row.trajectory)&(r.transition==row.transition)&(r.expert==row.expert)&(r.dose_groups==row.dose_groups)&(r.subset=='whole')&(r.rule=='current')].iloc[0]
        lines.append(f'|{row.trajectory}|{row.transition}|{row.expert}|{row.dose_groups:g}|{rr.spearman:.2f}|{row.future_high_minus_low:.3e}|{format_ci(row)}|{format_ci(row,"selection_resampled_")}|')
    lines += ['','All five primary named contrasts have discovery-reselection intervals crossing zero. Their fixed-ID future effects are measurable, but selection instability prevents calling these robust online temporal ranking rules at this budget. The secondary unpruned25→250 expert0 dose4 example survives reselection; it is explicitly outside the primary transitions.','',
        '## History and a concrete low-damage decision','',
        'Selection advantage is future panel mean minus selected future damage. The panel mean is uniform selection among these3/4 fixed overlapping unions, not random pruning throughout the model. Current-minus-rule damage is positive when history improves on current. Identical choices have exactly zero policy difference and are not independent corroboration. Future regret is an optimistic hindsight diagnostic.','',
        '|Trajectory|Transition|Rule|Covered panels|Same as current|Advantage CI>0|History improvement CI>0|History worse CI<0|','|---|---|---|---:|---:|---:|---:|---:|']
    for key,z in p[(p.subset=='whole')&p.coverage_qualified].groupby(['trajectory','transition','rule']):
        identified=z[~z.low_selection_unidentified]
        lines.append('|'+ '|'.join(map(str,[*key,len(z),int(z.identical_selection.sum()),int((identified.advantage_ci95_low>0).sum()),int((identified.current_minus_rule_ci95_low>0).sum()),int((identified.current_minus_rule_ci95_high<0).sum())]))+'|')
    hist=p[(p.subset=='whole')&p.coverage_qualified&(p.rule!='current')&~p.low_selection_unidentified]
    lines += ['','All covered, identified history/current differences with fixed-selection intervals excluding zero (including losses) follow:','',
        '|Trajectory|Transition|Expert|Dose|Rule|Current−history future damage|Fixed95% interval|Reselected95% interval|','|---|---|---:|---:|---|---:|---|---|']
    for _,row in hist[(hist.current_minus_rule_ci95_low>0)|(hist.current_minus_rule_ci95_high<0)].iterrows():
        lines.append(f'|{row.trajectory}|{row.transition}|{row.expert}|{row.dose_groups:g}|{row.rule}|{row.current_minus_rule_future_damage:.3e}|{format_ci(row,"current_minus_rule_")}|{format_ci(row,"resampled_current_minus_rule_")}|')
    lines += ['','No primary current-rule panel has a positive selection-advantage interval after discovery reselection. The complete surviving set is secondary unpruned25→250 experts0 and8, both dose4. All four fixed-selection history improvements above touch or cross zero after discovery reselection. Thus history has not shown stable added value over current damage.','',
        '## Coverage, conditional effects and thresholds','',
        'Coverage qualification uses only the frozen≥32 accepted positions in each four-batch block, separately for early discovery and future confirmation, and previous discovery for history rules. All whole-probe outcomes remain in the tables despite failure. Native damage can evolve through weights, routing, gates, context populations and downstream computation; these data do not isolate a cause or specialization.','',
        '|Trajectory|Checkpoint|Partition|Expert|Minimum accepted/block|Contributing batches|Max batch fraction|HHI|','|---|---:|---|---:|---:|---:|---:|---:|']
    cc=cov.drop_duplicates(['trajectory','expert','checkpoint','partition'])
    for _,row in cc[~cc.coverage_qualified].iterrows():
        lines.append(f'|{row.trajectory}|{row.checkpoint}|{row.partition}|{row.expert}|{row.minimum_block_accepted}|{row.contributing_batches}|{row.max_batch_accepted_fraction:.3g}|{row.accepted_batch_HHI:.3g}|')
    lines += ['','|Outcome|Tolerance|Threshold panels|Ceilings|Empty|Panels with point false-safe>0|','|---|---:|---:|---:|---:|---:|']
    for key,z in th.groupby(['subset','tolerance']):
        lines.append('|'+ '|'.join(map(str,[*key,len(z),int((z.status=='ceiling_no_discrimination').sum()),int((z.status=='empty_undefined').sum()),int((z.false_safe_fraction>0).sum())]))+'|')
    lines += ['','Thresholds1e-4 and1e-3 are descriptive secondary conventions, not optimized cutoffs or safety criteria. Ceiling means lack of discrimination. Empty false-safe is undefined. Zero observed false-safe events do not prove safety. Coordinate fractions use actual set unions, with panel and active-matrix denominators; overlapping coordinates are counted once.','',
        'Conditional and whole damage values, intervals and threshold classifications are provided side by side in the tables and separate figures. Their different target denominators can produce much larger conditional damage despite small whole means. For example, unpruned step2500 expert0 union e0_n1_8b18d0c66a236ba2 has whole future damage−5.998e-7 but future-native accepted-position damage about8.1e-5 nats/token. This retrospective illustration is not a selected prediction rule or a causal explanation. Conditional bootstrap samples with no accepted targets are undefined and recorded; whole means are still reported.','',
        '## Decision and limits','']
    primary=named[named.priority=='primary']
    adv=p[(p.subset=='whole')&p.coverage_qualified&~p.low_selection_unidentified&(p.rule=='current')]
    hw=int((hist.current_minus_rule_ci95_low>0).sum());hl=int((hist.current_minus_rule_ci95_high<0).sum())
    lines += [f'1. Early whole damage has named temporal ordering candidates in {len(primary)} primary equal-dose panels under the positive-interval/all-block-sign descriptive check (listed above). Other panels remain unresolved or are reversal candidates; Spearman alone with3/4 unions is fragile.',
        f'2. History is not uniformly better: {hw} covered identified history comparisons favor history and {hl} favor current by fixed-selection intervals; identical selections give no added evidence. Reselected intervals and selection frequencies show sensitivity to discovery noise.',
        f'3. Current low-score selection beats fixed-panel uniform selection by a positive fixed-selection interval in {int((adv.advantage_ci95_low>0).sum())} of {len(adv)} covered identified whole-outcome panels. This is a context-level description, not an independent-replication success rate.',
        '4. Read effect sizes and both fixed/reselected intervals together. Many differences are small relative to1e-4/1e-3; the threshold tables disclose ceilings rather than tightening tolerances. Conditional effects can be concealed by whole-probe averaging.',
        '5. Early low/absent traffic and tiny3/4-candidate overlapping panels limit interpretation; singletons provide no ordering information. Tokens, batches, groups, orderings and checkpoints are not independent model seeds.','',
        'Decision: practical temporal discrimination remains unresolved at this measurement budget. Improve selection stability before specifying a cross-seed confirmation of a usable temporal pruning rule. The fixed-ID primary signals are candidates to investigate, not sufficient support for a general rule; history adds no stable demonstrated value. No cross-seed or continuation experiment is launched.',
        '', 'This predicts later deletion damage along the original unmodified recovery trajectory. It does not establish that deleting early preserves subsequent learning, specialization as a cause, a winning-ticket-specific property, safe combinations of deletions or practical training savings.', '',
        'Reproduce using prepare.py (fresh output only), analyze.py, report.py and verify.py in scripts/expert_deletion_temporal_prediction_v1/. The frozen config/protocol and estimand clarification predate temporal associations. Only coordinate tensors were loaded; checkpoints were byte-hashed without model deserialization. Original sealed artifacts remain unchanged.']
    (D/'findings.md').write_text('\n'.join(lines)+'\n')
    # Compact exhaustive classifications for review; do not pool correlations.
    e.to_csv(D/'temporal_condition_decisions.csv',index=False)
    accounting=json.loads((D/'compute_accounting.json').read_text());accounting['report_wall_seconds']=time.perf_counter()-started
    accounting['storage_bytes']=sum(x.stat().st_size for x in D.rglob('*') if x.is_file());save(D/'compute_accounting.json',accounting)
    save(D/'report_provenance.json',dict(implementation_sha256=sha(Path(__file__)),figures=len(list((D/'figures').glob('*.png'))),
        named_conditions_rule='all covered whole current panels with positive fixed-selection interval and all4 future block contrasts positive; exhaustive exploratory listing',
        input_table_hashes={x.name:sha(x) for x in D.glob('*.csv')}))
    print('Temporal report and figures complete')

if __name__=='__main__':main()
