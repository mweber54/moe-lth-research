#!/usr/bin/env python3
"""Five-architecture descriptive map. Never trains or treats repeat runs as seeds."""
import json,math,itertools
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from run_16e_router_age_localization import ROOT,write_json,sha256_file
OUT=ROOT/'results/router_age_cross_architecture_map'
METRICS=['entropy','margin','load_cv','token_specialization','assignment_agreement_to_r100','js_distance_to_r100','cumulative_expert_exposure']
FEATURES=['normalized_entropy','margin','load_cv','specialization','assignment_agreement_to_r100','js_distance_to_r100','cumulative_expert_exposure']

def load_architecture(e):
    if e in (8,12):
        root=ROOT/('results/expert_count_replication' if e==8 else 'results/router_development_12expert_retry4_20260902')
        sparse=pd.read_csv(root/('sparse_results.csv' if e==8 else 'ticket_gap_seed_level.csv'))
        dense=pd.read_csv(root/'dense_baselines.csv')
        if e==8:sparse=sparse[sparse.expert_count==8];dense=dense[dense.expert_count==8]
        sparse=sparse[np.isclose(sparse.sparsity,.85)]
        rows=[]
        for s in sparse.itertuples():
            d=dense[(dense.seed==s.seed)&(dense.router_age_percent==s.router_age_percent)]
            assert len(d)==1;d=d.iloc[0]
            assert abs((s.final_validation_loss-d.final_validation_loss)-s.ticket_gap)<1e-10
            rows.append({'experts':e,'seed':int(s.seed),'router_age':int(s.router_age_percent),'dense_final_loss':d.final_validation_loss,'sparse_final_loss':s.final_validation_loss,'ticket_gap':s.ticket_gap,'pruning_shock':s.initial_validation_loss-d.initial_validation_loss,'dense_early_auc':d.early_auc,'sparse_early_auc':s.early_auc,'dense_initial_loss':d.initial_validation_loss,'sparse_initial_loss':s.initial_validation_loss,'dense_recovery_improvement':d.initial_validation_loss-d.final_validation_loss,'sparse_recovery_improvement':s.initial_validation_loss-s.final_validation_loss})
        frame=pd.DataFrame(rows)
        dev=pd.read_csv(ROOT/'results/r20_8_vs_r60_12/developmental_vectors.csv');dev=dev[dev.expert_count==e].copy()
        dev['entropy']=dev.normalized_entropy*math.log(e);dev['token_specialization']=dev.specialization*math.log(e)
        dev=dev.rename(columns={'assignment_agreement_to_R100':'assignment_agreement_to_r100','functional_js_to_R100':'js_distance_to_r100','cumulative_exposure_per_expert':'cumulative_expert_exposure'})
    else:
        root=ROOT/f'results/{e}e_router_age_localization'
        assert json.loads((root/'localization_decision.json').read_text())['status']=='complete'
        frame=pd.read_csv(root/'outcomes.csv').rename(columns={'immediate_pruning_shock':'pruning_shock'})
        dev=pd.read_csv(root/'developmental_metrics.csv')
    frame=frame.merge(dev[['seed','router_age']+list(dict.fromkeys(METRICS+FEATURES))],on=['seed','router_age'],validate='one_to_one')
    frame['matched_early_auc_gap']=frame.sparse_early_auc-frame.dense_early_auc
    frame['coarse_sweep']=frame.router_age.isin([0,20,40,60,80,100]);frame['refinement']=~frame.coarse_sweep
    frame['historical']=e<=16;frame['new']=e>16;frame['reproducibility_rerun']=False
    frame['source_root']=str(root)
    assert not frame.duplicated(['experts','seed','router_age']).any()
    assert frame.groupby('router_age').seed.nunique().eq(3).all()
    assert frame[METRICS].notna().all().all()
    return frame

def summarize(frame):
    summaries=[];winners=[];endpoints=[]
    for e,g in frame.groupby('experts'):
        means=g.groupby('router_age').ticket_gap.mean();best=int(means.idxmin());pivot=g.pivot(index='seed',columns='router_age',values='ticket_gap');sd=g.groupby('router_age').ticket_gap.std(ddof=1)
        ordered=sorted(means.index);idx=ordered.index(best);basin=[best]
        for direction in (-1,1):
            j=idx+direction
            while 0<=j<len(ordered):
                diff=pivot[ordered[j]]-pivot[best]
                if abs(diff.mean())>diff.std(ddof=1):break
                basin.append(int(ordered[j]));j+=direction
        basin=sorted(basin);local_winners={}
        for seed,h in g.groupby('seed'):
            top=h.sort_values(['ticket_gap','router_age']).iloc[:2];local_winners[int(seed)]=int(top.iloc[0].router_age)
            winners.append({'experts':int(e),'seed':int(seed),'best_router_age':int(top.iloc[0].router_age),'best_ticket_gap':top.iloc[0].ticket_gap,'second_best_router_age':int(top.iloc[1].router_age),'second_best_ticket_gap':top.iloc[1].ticket_gap,'difference':top.iloc[1].ticket_gap-top.iloc[0].ticket_gap})
        d0=pivot[best]-pivot[0];d100=pivot[best]-pivot[100];both=(d0<0)&(d100<0)
        consistent=all(a in basin for a in local_winners.values());monotonic=bool((np.diff(means)<=0).all() or (np.diff(means)>=0).all())
        if best in (0,100):category='MONOTONIC / NO INTERMEDIATE OPTIMUM' if monotonic else 'BOUNDARY OPTIMUM'
        elif not consistent or not both.all():category='SEED-CONDITIONED INTERMEDIATE EFFECT'
        elif len(basin)>1:category='BROAD INTERMEDIATE BASIN'
        else:category='CLEAR INTERMEDIATE OPTIMUM'
        summaries.append({'experts':int(e),'mean_winner':best,'basin':json.dumps(basin),'basin_width':max(basin)-min(basin),'mean_best_gap':means[best],'sd_at_mean_winner':sd[best],**{f'seed{s}_winner':a for s,a in local_winners.items()},'difference_vs_R0':d0.mean(),'difference_vs_R100':d100.mean(),'beats_R0_seed_count':int((d0<0).sum()),'beats_R100_seed_count':int((d100<0).sum()),'beats_both_seed_count':int(both.sum()),'seed_winner_spread':max(local_winners.values())-min(local_winners.values()),'mean_age_seed_sd':sd.mean(),'classification':category,'all_seed_winners_in_basin':consistent,'R0_winner_seed_count':sum(a==0 for a in local_winners.values()),'R100_winner_seed_count':sum(a==100 for a in local_winners.values())})
        endpoints.append({'experts':int(e),**{name:int(g.groupby('router_age')[col].mean().idxmin()) for name,col in [('pruning_shock_winner','pruning_shock'),('early_auc_winner','matched_early_auc_gap'),('sparse_loss_winner','sparse_final_loss'),('dense_loss_winner','dense_final_loss'),('ticket_gap_winner','ticket_gap')]}})
    return pd.DataFrame(summaries),pd.DataFrame(winners),pd.DataFrame(endpoints)

def main():
    OUT.mkdir(parents=True,exist_ok=True)
    repeat=json.loads((ROOT/'results/16e_seed7_reproducibility/reproducibility_summary.json').read_text());assert repeat['status']=='complete'
    frame=pd.concat([load_architecture(e) for e in (8,12,16,24,32)],ignore_index=True)
    assert not frame.duplicated(['experts','seed','router_age']).any() and not frame.reproducibility_rerun.any()
    frame.to_csv(OUT/'unified_outcomes.csv',index=False)
    summary,winners,endpoints=summarize(frame)
    summary.to_csv(OUT/'architecture_summary.csv',index=False);winners.to_csv(OUT/'all_seed_winners.csv',index=False);endpoints.to_csv(OUT/'endpoint_winner_comparison.csv',index=False)
    summary[['experts','classification','mean_winner','basin']].to_csv(OUT/'phenomenon_classification.csv',index=False)
    frame.groupby(['experts','router_age'])[METRICS+['ticket_gap']].agg(['mean','std']).to_csv(OUT/'developmental_metric_summary.csv')
    optimal=frame.merge(summary[['experts','mean_winner']],on='experts');optimal=optimal[optimal.router_age==optimal.mean_winner].copy()
    optimal.to_csv(OUT/'optimal_checkpoint_features_seed_level.csv',index=False)
    optimal.groupby(['experts','router_age'])[METRICS+['ticket_gap']].mean().reset_index().to_csv(OUT/'optimal_checkpoint_features.csv',index=False)
    # Freeze the historical coordinate ranges; compare concentration relative to all sampled age-pair distances.
    historical=frame[frame.experts.isin([8,12])];scales=(historical[FEATURES].max()-historical[FEATURES].min()).clip(lower=1e-12)
    distances=[]
    for seed in (7,17,29):
        for e1,e2 in itertools.combinations((8,12,16,24,32),2):
            first=frame[(frame.seed==seed)&(frame.experts==e1)];second=frame[(frame.seed==seed)&(frame.experts==e2)]
            for x in first.to_dict('records'):
                for y in second.to_dict('records'):
                    delta=np.array([x[f]-y[f] for f in FEATURES])/scales.to_numpy()
                    distances.append({'seed':seed,'first_experts':e1,'second_experts':e2,'first_age':x['router_age'],'second_age':y['router_age'],'feature_distance':float(np.sqrt(np.mean(delta**2))),'age_distance':abs(x['router_age']-y['router_age'])/100,'both_optimal':x['router_age']==int(summary.set_index('experts').loc[e1,'mean_winner']) and y['router_age']==int(summary.set_index('experts').loc[e2,'mean_winner'])})
    distances=pd.DataFrame(distances);distances.to_csv(OUT/'cross_architecture_distances.csv',index=False)
    concentration=[]
    for (seed,e1,e2),g in distances.groupby(['seed','first_experts','second_experts']):
        opt=g[g.both_optimal].iloc[0]
        concentration.append({'seed':seed,'first_experts':e1,'second_experts':e2,**{f'{metric}_optimal_to_all_ratio':opt[metric]/g[metric].mean() if g[metric].mean()>1e-12 else np.nan for metric in ('feature_distance','age_distance')}})
    concentration=pd.DataFrame(concentration);concentration.to_csv(OUT/'optimum_concentration.csv',index=False)
    correlations=[]
    for (e,seed),g in frame.groupby(['experts','seed']):
        for metric in ['router_age']+METRICS:
            correlations.append({'experts':e,'seed':seed,'metric':metric,'spearman_descriptive':g[metric].rank().corr(g.ticket_gap.rank()),'sampled_ages':len(g),'inference':'within-seed age trace; dependent observations, no p-value'})
    correlations=pd.DataFrame(correlations);correlations.to_csv(OUT/'exploratory_correlations.csv',index=False)
    trend=float(summary.experts.rank().corr(summary.mean_winner.rank()));nondecreasing=bool((np.diff(summary.mean_winner)>=0).all())
    feature_ratio=float(concentration.feature_distance_optimal_to_all_ratio.mean());age_ratio=float(concentration.age_distance_optimal_to_all_ratio.mean())
    figdir=OUT/'figures';figdir.mkdir(exist_ok=True)
    def save(name):
        plt.tight_layout()
        for ext in ('png','pdf'):plt.savefig(figdir/f'{name}.{ext}',dpi=160)
        plt.close()
    fig,axes=plt.subplots(1,5,figsize=(18,4))
    for ax,(e,g) in zip(axes,frame.groupby('experts')):
        stats=g.groupby('router_age').ticket_gap.agg(['mean','std']);ax.errorbar(stats.index,stats['mean'],yerr=stats['std'],fmt='o-');ax.set_title(f'{e}E');ax.set_xlabel('Router age');ax.set_ylabel('Ticket gap ± seed SD')
    save('01_mean_age_curves')
    fig,axes=plt.subplots(1,5,figsize=(18,4))
    for ax,(e,g) in zip(axes,frame.groupby('experts')):
        for seed,h in g.groupby('seed'):
            h=h.sort_values('router_age');ax.plot(h.router_age,h.ticket_gap,'.-',label=f'seed {seed}')
        ax.set_title(f'{e}E');ax.set_xlabel('Router age');ax.legend(fontsize=7)
    save('02_individual_seed_curves')
    for number,col,label in [(3,'mean_winner','Mean-winning router age'),(4,'basin_width','Observed basin width')]:
        plt.figure();plt.plot(summary.experts,summary[col],'o-');plt.xlabel('Experts');plt.ylabel(label);save(f'{number:02d}_{col}')
    plt.figure()
    for seed,g in winners.groupby('seed'):plt.plot(g.experts,g.best_router_age,'o-',label=f'seed {seed}')
    plt.xlabel('Experts');plt.ylabel('Seed-winning router age');plt.legend();save('05_seed_winning_ages')
    for number,metric in enumerate(['entropy','margin','load_cv','token_specialization','assignment_agreement_to_r100','cumulative_expert_exposure'],6):
        fig,axes=plt.subplots(1,5,figsize=(18,4))
        for ax,(e,g) in zip(axes,frame.groupby('experts')):
            for seed,h in g.groupby('seed'):
                h=h.sort_values('router_age');ax.plot(h[metric],h.ticket_gap,'.-',alpha=.7,label=f'seed {seed}')
            ax.set_title(f'{e}E');ax.set_xlabel(metric.replace('_',' '));ax.set_ylabel('Ticket gap');ax.legend(fontsize=7)
        save(f'{number:02d}_gap_vs_{metric}')
    for number,col in [(12,'pruning_shock_winner'),(13,'early_auc_winner')]:
        plt.figure();plt.plot([0,100],[0,100],'k:',alpha=.4)
        for row in endpoints.itertuples():plt.scatter(getattr(row,col),row.ticket_gap_winner);plt.annotate(f'{row.experts}E',(getattr(row,col),row.ticket_gap_winner))
        plt.xlabel(col.replace('_',' '));plt.ylabel('Final ticket-gap winner');save(f'{number}_{col}')
    audit=pd.read_csv(ROOT/'results/16e_seed7_reproducibility/original_vs_rerun.csv')
    plt.figure();plt.plot(audit.router_age,audit.original_ticket_gap,'o-',label='original seed 7');plt.plot(audit.router_age,audit.rerun_ticket_gap,'x--',label='repeat, not another seed');plt.xlabel('Router age');plt.ylabel('Ticket gap');plt.legend();save('14_seed7_repeat')
    fig,axes=plt.subplots(1,2,figsize=(13,4));means=optimal.groupby('experts')[FEATURES].mean();scaled=(means-historical[FEATURES].min())/scales
    im=axes[0].imshow(scaled.to_numpy(),aspect='auto');axes[0].set_yticks(range(5),[f'{e}E' for e in means.index]);axes[0].set_xticks(range(7),FEATURES,rotation=70,fontsize=7);fig.colorbar(im,ax=axes[0],label='Historical range-normalized feature')
    for seed,g in optimal.groupby('seed'):axes[1].plot(g.experts,g.router_age,'o-',label=f'seed {seed} at mean winner')
    axes[1].set_xlabel('Experts');axes[1].set_ylabel('Mean-winning age');axes[1].legend(fontsize=7);save('15_optimal_feature_representation')
    # Additional maturity-distance panel: all seven metrics are represented.
    fig,axes=plt.subplots(1,5,figsize=(18,4))
    for ax,(e,g) in zip(axes,frame.groupby('experts')):
        for seed,h in g.groupby('seed'):
            h=h.sort_values('router_age');ax.plot(h.js_distance_to_r100,h.ticket_gap,'.-',label=str(seed))
        ax.set_title(f'{e}E');ax.set_xlabel('JS distance to R100')
    save('16_gap_vs_js')
    interior=summary[(summary.mean_winner>0)&(summary.mean_winner<100)]
    clean=summary[summary.classification.isin(['CLEAR INTERMEDIATE OPTIMUM','BROAD INTERMEDIATE BASIN'])]
    conditioned=summary[summary.classification=='SEED-CONDITIONED INTERMEDIATE EFFECT']
    boundaries=summary[summary.mean_winner.isin([0,100])]
    shock_matches=int((endpoints.pruning_shock_winner==endpoints.ticket_gap_winner).sum());early_matches=int((endpoints.early_auc_winner==endpoints.ticket_gap_winner).sum());sparse_matches=int((endpoints.sparse_loss_winner==endpoints.ticket_gap_winner).sum())
    if len(interior)>=3:
        verdict='CROSS-ARCHITECTURE PHENOMENON SUPPORTED BUT SEED-CONDITIONED' if len(conditioned) else 'CROSS-ARCHITECTURE INTERMEDIATE TICKET PHENOMENON SUPPORTED'
    elif shock_matches<=1 and early_matches<=1 and len(interior)>=2:verdict='RECOVERY-EMERGENT PHENOMENON SUPPORTED'
    else:verdict='NO SINGLE CROSS-ARCHITECTURE PHENOMENON IDENTIFIED'
    reproducible=repeat['verdict'] in ['16E SEED7 EXACTLY REPRODUCED','16E SEED7 QUALITATIVELY REPRODUCED']
    statement=f'Under this fixed native 2500-step, 85% expert-pruning protocol, {len(interior)}/5 architecture-level sampled mean minima are interior; {len(clean)} architectures have seed-consistent local intermediate effects and {len(conditioned)} have seed-conditioned intermediate effects. This is a descriptive map, not a common causal mechanism.'
    questions=[]
    def q(text,answer):questions.append((text,str(answer)))
    q('Was the anomalous 16E seed-7 R100 optimum reproduced?',repeat['verdict']+f"; original winner R{repeat['original_winner']}, repeat R{repeat['rerun_winner']}.")
    q('Is 16E seed dependence robust to this rerun?',f'{reproducible}. One same-seed repeat measures run repeatability; it does not increase n=3 or replace the original observation.')
    for e in (24,32):
        row=summary.set_index('experts').loc[e]
        q(f'Does {e}E exhibit an intermediate ticket optimum?',row.classification)
        q('What is its mean optimum or basin?',f'R{row.mean_winner}; basin {row.basin}; gap {row.mean_best_gap:.8f} ± {row.sd_at_mean_winner:.8f}.')
        q('What are its seed-specific winners?',{s:int(row[f'seed{s}_winner']) for s in (7,17,29)})
    q('How often does an interior optimum occur?',f'{len(interior)}/5 mean winners; {int(winners.best_router_age.between(1,99).sum())}/15 seed-level winners. Different systems are not iid architecture replicates.')
    q('How often do all three seeds agree on an intermediate region?',f'{len(clean)}/5 architectures meet the descriptive basin consistency and both-endpoint criteria: {clean.experts.tolist()}.')
    q('Does optimal router age move later with expert count?',f'Sequence {dict(zip(summary.experts,summary.mean_winner))}; descriptive Spearman rank correlation {trend:.4f} with just five distinct systems. No scaling-law inference.')
    q('Is the trend monotonic?',f'Nondecreasing mean-winning age: {nondecreasing}. Seed-level curves and basin widths remain visible.')
    q('Is there evidence for an architecture threshold?',f'Interior mean cases {interior.experts.tolist()}, boundary cases {boundaries.experts.tolist()}. Five systems and adaptive grids cannot establish a sharp threshold.')
    for metric,label in [('entropy','entropy'),('margin','margin'),('load_cv','load balance'),('token_specialization','specialization'),('js_distance_to_r100','maturity distance'),('cumulative_expert_exposure','cumulative exposure')]:
        vals=optimal.groupby('experts')[metric].mean();q(f'Do optimal checkpoints share similar {label}?',f'Measured architecture means: {vals.to_dict()}. Ranges and seed values are reported; no prespecified equivalence margin establishes sameness.')
    q('Are optima closer in feature space than raw age?',f'Mean optimum-to-all-sampled-age-pair distance ratio: feature {feature_ratio:.4f}, normalized age {age_ratio:.4f}. Lower indicates tighter relative concentration; feature ratio is lower: {feature_ratio<age_ratio}. This is sampling-grid-sensitive and descriptive; it does not establish a shared causal developmental state.')
    q('Does pruning shock predict final optimum?',f'Exact sampled mean winner agreement in {shock_matches}/5 systems; full endpoint table retained.')
    q('Does early recovery predict final optimum?',f'Matched early-AUC winner agreement in {early_matches}/5 systems.')
    q('Does sparse final loss predict matched ticket optimum?',f'Winner agreement in {sparse_matches}/5 systems; selection always uses same-age sparse−dense gap.')
    q('How large is seed dependence relative to architecture dependence?',f'Within-architecture seed-winning age spreads: {dict(zip(summary.experts,summary.seed_winner_spread))}. Across-architecture mean winner spread: {int(summary.mean_winner.max()-summary.mean_winner.min())}. These are descriptive ranges, not a variance-components estimate.')
    q('Which architectures provide the cleanest intermediate phenomenon?',clean.experts.tolist())
    q('Which provide useful negative or boundary cases?',boundaries.experts.tolist())
    q('Is there enough evidence for a common cross-architecture phenomenon?',verdict)
    q('What is the narrowest defensible problem definition?',statement)
    q('What additional data are still required?', 'No additional experiment is launched. Unsampled ages, three seeds, logged exposure, and one repeat limit universality and equivalence claims; broader replication or complete exposure would require a separately specified study.')
    q('What mechanistic question is best justified?',f'Why does final matched sparse recoverability select different router ages from pruning shock and early recovery in the seed-consistent cases {clean.experts.tolist()}? A separately designed matched intervention should include both an intermediate case and a boundary/seed-conditioned comparator, with repeatability addressed first.')
    q('Was a follow-up launched?','No. This suite contains only the authorized repeatability and native localization runs.')
    assert len(questions)==31
    report='# Router-age cross-architecture phenomenon map\n\n'+statement+'\n\n```text\n'+summary.to_string(index=False)+'\n```\n\n'+'\n\n'.join(f'{i}. **{text}**\n\n{answer}' for i,(text,answer) in enumerate(questions,1))
    report+='\n\nMetric caveats: entropy uses natural logs; margin is probability top-1 minus top-2; load CV is layer-averaged population SD/mean; token specialization is historical token-ID/expert MI (log-expert normalization only in feature vectors); maturity uses the historical first probe batch per layer; exposure sums sampled logged assignments, not complete accepted-token updates. Original 8E/12E have sparser age coverage. All raw metric plots connect age-ordered traces within architecture and seed. Exploratory correlations are per-seed descriptive ranks with no p-values or independent-row inference. Feature ranges are frozen to historical 8E/12E coordinates. No metric is interpreted causally.\n\nPrimary verdict:\n\n'+verdict+'\n\nSecondary observations: endpoint agreement counts, descriptive age trend, feature concentration and seed heterogeneity above are separate from the primary classification. All seeds remain individually visible; sample SD uses n−1. The repeat is not averaged into 16E.\n'
    (OUT/'final_report.md').write_text(report)
    write_json(OUT/'mapping_summary.json',{'status':'complete','verdict':verdict,'optimal_age_sequence':dict(zip(summary.experts.astype(str),summary.mean_winner.astype(int))),'spearman_descriptive':trend,'nondecreasing':nondecreasing,'feature_concentration_ratio':feature_ratio,'age_concentration_ratio':age_ratio,'narrow_statement':statement,'independent_seeds':3})
    sourcefiles=[]
    for e in (16,24,32):sourcefiles.extend((ROOT/f'results/{e}e_router_age_localization').glob('*.csv'))
    sourcefiles.extend([ROOT/'results/expert_count_replication/sparse_results.csv',ROOT/'results/expert_count_replication/dense_baselines.csv',ROOT/'results/router_development_12expert_retry4_20260902/ticket_gap_seed_level.csv',ROOT/'results/router_development_12expert_retry4_20260902/dense_baselines.csv',ROOT/'results/r20_8_vs_r60_12/developmental_vectors.csv'])
    write_json(OUT/'artifact_integrity.json',{'status':'validated','unique_primary_rows':len(frame),'repeat_rows_in_primary':0,'source_sha256':{str(p):sha256_file(p) for p in sourcefiles},'output_sha256':{str(p.relative_to(OUT)):sha256_file(p) for p in OUT.rglob('*') if p.is_file() and p.name!='artifact_integrity.json' and p.suffix in ('.csv','.md','.png','.pdf')}})
    print('ROUTER-AGE CROSS-ARCHITECTURE MAPPING COMPLETE\n')
    print('16E seed-7 reproducibility:',repeat)
    print(summary.to_string(index=False))
    print(f'Optimal-age sequence: {dict(zip(summary.experts,summary.mean_winner))}\nInterior optimum architectures: {interior.experts.tolist()}\nSeed-consistent architectures: {clean.experts.tolist()}\nSeed-conditioned architectures: {conditioned.experts.tolist()}\nBoundary-optimum architectures: {boundaries.experts.tolist()}\nDoes optimum move later with expert count?: rho={trend}; nondecreasing={nondecreasing}\nDo optima share developmental statistics?: descriptive feature concentration ratio={feature_ratio}, age={age_ratio}\nDoes pruning shock predict final optimum?: {shock_matches}/5 exact winners\nDoes early recovery predict final optimum?: {early_matches}/5 exact winners\nNarrowest defensible phenomenon statement: {statement}\nRecommended next scientific question: {questions[-2][1]}\nFinal verdict: {verdict}\nPath to unified final_report.md: {OUT/"final_report.md"}',flush=True)
if __name__=='__main__':main()
