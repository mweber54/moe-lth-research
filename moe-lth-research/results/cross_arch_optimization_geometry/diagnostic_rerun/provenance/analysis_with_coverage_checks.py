#!/usr/bin/env python3
"""Descriptive seed-paired analysis of the preregistered diagnostic rerun."""
import json
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT=Path('results/cross_arch_optimization_geometry/diagnostic_rerun')
KEY=['experts','seed','age'];EK=KEY+['layer','expert'];SK=KEY+['step']
METRICS={'gradient_alignment_forced_sparse_routes':('cosine','forced_gradient',True),'gradient_alignment_native':('cosine','native_gradient',True),'update_alignment':('cosine','update_alignment',True),'functional_expert_similarity':('output_cosine','functional_similarity',True),'parameter_trajectory_distance':('distance','trajectory_distance',False)}

def save(df,name):df.to_csv(ROOT/f'{name}.csv',index=False)

def main():
    manifest=json.loads((ROOT/'run_manifest.json').read_text())
    repro=pd.read_csv(ROOT/'reproduction_check.csv')
    if len(repro)!=18:raise RuntimeError('All 18 pairs are required before final mechanism analysis')
    if len(pd.read_csv(ROOT/'snapshot_manifest.csv'))!=324:raise RuntimeError('Expected 324 recoverable snapshots')
    audit=pd.read_csv(ROOT/'mutation_audit.csv')
    if len(audit)!=324 or not audit[['model_unchanged','optimizer_unchanged','scaler_unchanged','rng_restored','training_iterator_not_accessed']].all().all():raise RuntimeError('Incomplete mutation audit')
    frames={name:pd.read_csv(ROOT/f'{name}.csv') for name in METRICS}
    for name,f in frames.items():
        expected=5760 if name=='update_alignment' else 6480
        if len(f)!=expected or f.duplicated(EK+['step']).any():
            raise RuntimeError(f'{name}: incomplete or duplicate expert measurements')
    coverage=[]
    for name,(value,label,higher) in METRICS.items():
        g=frames[name].groupby(SK)[value].agg(['size','count']).reset_index()
        g['metric']=label;g['fraction_available']=g['count']/g['size'];coverage.append(g)
    save(pd.concat(coverage,ignore_index=True),'diagnostic_coverage_summary')
    expert=[];layer=[];seed=[];time=[]
    for name,(value,label,higher) in METRICS.items():
        f=frames[name]
        es=f.groupby(EK)[value].agg(['mean','std','count']).reset_index().rename(columns={'mean':'value','std':'within_expert_time_sd','count':'valid_time_measurements'})
        es['metric']=label;expert.append(es)
        ls=f.groupby(KEY+['step','layer'])[value].agg(['mean','std','count']).reset_index().rename(columns={'mean':'value','std':'expert_sd','count':'valid_experts'})
        ls['metric']=label;layer.append(ls)
        ts=ls.groupby(SK)['value'].mean().reset_index();ts['metric']=label;time.append(ts)
        ss=ts.groupby(KEY)['value'].agg(['mean','count']).reset_index().rename(columns={'mean':'value','count':'valid_steps'});ss['metric']=label;ss['higher_is_better']=higher;seed.append(ss)
    expert=pd.concat(expert,ignore_index=True);layer=pd.concat(layer,ignore_index=True);seed=pd.concat(seed,ignore_index=True);time=pd.concat(time,ignore_index=True)
    save(time.groupby(['experts','age','step','metric']).value.agg(['mean','std','count']).reset_index().rename(columns={'std':'seed_sd','count':'valid_seeds'}),'architecture_timecourse_summary')
    seed['metric_rank']=seed.groupby(['experts','seed','metric'])['value'].rank(ascending=False,method='min')
    low=seed['metric']=='trajectory_distance';seed.loc[low,'metric_rank']=seed[low].groupby(['experts','seed','metric'])['value'].rank(ascending=True,method='min')
    repro['actual_ticket_gap_rank']=repro.groupby(['experts','seed'])['rerun_ticket_gap'].rank(ascending=True,method='min')
    seed=seed.merge(repro[KEY+['rerun_ticket_gap','actual_ticket_gap_rank']],on=KEY)
    save(expert,'expert_level_summary');save(layer,'layer_level_summary');save(seed,'seed_level_summary');save(time,'metric_timecourse_seed_level');save(seed,'optimization_geometry_vs_ticket_gap')
    cross=[];paired=[]
    for (e,metric),g in seed.groupby(['experts','metric']):
        target=20 if e==8 else 60;higher=metric!='trajectory_distance';w=g.pivot(index='seed',columns='age',values='value')
        for age in w:
            cross.append({'experts':e,'metric':metric,'age':age,'seed_n':int(w[age].count()),'mean':w[age].mean(),'seed_sd':w[age].std(ddof=1),'target_age':target})
            if age!=target:
                for s in w.index:
                    delta=w.loc[s,target]-w.loc[s,age]
                    paired.append({'experts':e,'seed':s,'metric':metric,'target_age':target,'control_age':age,'paired_difference':delta,'favors_target':bool(delta>0 if higher else delta<0)})
    save(pd.DataFrame(cross),'cross_architecture_summary');paired=pd.DataFrame(paired);save(paired,'paired_seed_differences')
    heterogeneity=[]
    for (e,seed_id,metric),g in expert.groupby(['experts','seed','metric']):
        target=20 if e==8 else 60
        w=g.pivot(index=['layer','expert'],columns='age',values='value')
        for control in w:
            if control==target:continue
            delta=w[target]-w[control]
            for layer_id,b in delta.groupby(level='layer'):
                valid=b.dropna();favorable=valid>0 if metric!='trajectory_distance' else valid<0
                heterogeneity.append({'experts':e,'seed':seed_id,'metric':metric,'layer':layer_id,'target_age':target,'control_age':control,'paired_expert_mean_difference':valid.mean(),'within_seed_expert_sd':valid.std(ddof=1),'valid_experts':len(valid),'fraction_experts_favoring_target':favorable.mean()})
    heterogeneity=pd.DataFrame(heterogeneity);save(heterogeneity,'expert_layer_paired_effects')
    # Historical evaluation schedule supplies the loss trajectory without new evaluations.
    curves=[]
    for row in repro.itertuples():
        base=ROOT/f'recovery/e{row.experts}/seed{row.seed}/R{row.age}'
        s=pd.read_json(base/'sparse/metrics.jsonl',lines=True);d=pd.read_json(base/'dense/metrics.jsonl',lines=True)
        joined=s.merge(d,on='step',suffixes=('_sparse','_dense'));joined['ticket_gap']=joined.loss_sparse-joined.loss_dense
        for k in KEY:joined[k]=getattr(row,k)
        curves.append(joined)
    curves=pd.concat(curves,ignore_index=True);save(curves,'validation_ticket_gap_timecourse')
    temporal=[]
    def first_separation(df,value,target,control,higher):
        p=df[df.age.isin([target,control])].pivot(index=['seed','step'],columns='age',values=value)
        delta=(p[target]-p[control])*(1 if higher else -1)
        valid=delta.unstack('seed');allfav=(valid>0).all(axis=1)&(valid.notna().sum(axis=1)==3)
        hits=allfav[allfav].index
        return int(hits[0]) if len(hits) else None
    for e in [8,12]:
        target=20 if e==8 else 60;controls=[0,100] if e==8 else [20,100]
        for control in controls:
            lossfirst=first_separation(curves[curves.experts==e],'ticket_gap',target,control,False)
            for metric in time.metric.unique():
                mf=first_separation(time[(time.experts==e)&(time.metric==metric)],'value',target,control,metric!='trajectory_distance')
                temporal.append({'experts':e,'target_age':target,'control_age':control,'metric':metric,'first_all_seed_metric_advantage_step':mf,'first_all_seed_ticket_gap_advantage_step':lossfirst,'metric_before_or_with_loss':mf<=lossfirst if mf is not None and lossfirst is not None else None,'interpretation':'descriptive first measured concurrence, not significance; update step is interval end'})
    temporal=pd.DataFrame(temporal);save(temporal,'temporal_mechanism_summary')
    fig=ROOT/'figures';fig.mkdir(exist_ok=True)
    plt.rcParams.update({'font.size':9,'axes.spines.top':False,'axes.spines.right':False})
    def finish(name):plt.tight_layout();plt.savefig(fig/f'{name}.png',dpi=180);plt.savefig(fig/f'{name}.pdf');plt.close()
    def plot_metric(ax,e,metric):
        g=time[(time.experts==e)&(time.metric==metric)]
        for age,a in g.groupby('age'):
            color={0:'tab:gray',20:'tab:blue',60:'tab:green',100:'tab:orange'}[age]
            for s,b in a.groupby('seed'):ax.plot(b.step,b.value,color=color,alpha=.45,lw=.8,linestyle={7:'-',17:'--',29:':'}[s])
            b=a.groupby('step').value.mean();ax.plot(b.index,b.values,color=color,lw=2,label=f'R{age}')
        ax.set(title=f'{e} experts: {metric.replace("_"," ")}',xlabel='Recovery step',ylabel=metric.replace('_',' '));ax.legend()
    for e in [8,12]:
        _,ax=plt.subplots(figsize=(6,4));plot_metric(ax,e,'forced_gradient');finish(f'{e}E_forced_gradient')
        _,axes=plt.subplots(1,2,figsize=(10,4));plot_metric(axes[0],e,'native_gradient');plot_metric(axes[1],e,'forced_gradient');finish(f'{e}E_native_vs_forced')
    for metric in ['update_alignment','trajectory_distance','functional_similarity']:
        _,axes=plt.subplots(1,2,figsize=(10,4))
        for ax,e in zip(axes,[8,12]):plot_metric(ax,e,metric)
        if metric=='update_alignment':
            for ax in axes:ax.set_xlabel('Interval end step (0–100, 100–250, …, 2000–2500)')
        finish(metric)
    _,axes=plt.subplots(2,3,figsize=(13,7),sharey=True)
    for ax,((e,s),g) in zip(axes.flat,seed.groupby(['experts','seed'])):
        for metric,b in g.groupby('metric'):ax.plot(b.age,b.metric_rank,'o-',label=metric)
        b=g.drop_duplicates('age');ax.plot(b.age,b.actual_ticket_gap_rank,'k*--',label='actual ticket gap',lw=2)
        ax.set(title=f'{e}E seed {s}',xlabel='Router age',ylabel='Within-seed rank (1 = best)',yticks=[1,2,3]);ax.invert_yaxis()
    axes.flat[-1].legend(fontsize=6);finish('seed_metric_vs_actual_winner')
    forced=frames['gradient_alignment_forced_sparse_routes'];_,axes=plt.subplots(2,3,figsize=(12,7))
    for i,e in enumerate([8,12]):
        for j,step in enumerate([0,500,2500]):
            ax=axes[i,j];g=forced[(forced.experts==e)&(forced.step==step)];ages=sorted(g.age.unique())
            ax.boxplot([g[g.age==a].cosine.dropna() for a in ages],tick_labels=[f'R{a}' for a in ages]);ax.set(title=f'{e}E step {step}',ylabel='Forced gradient cosine')
            for k,a in enumerate(ages):
                for s,sg in g[g.age==a].groupby('seed'):ax.scatter(np.full(len(sg),k+1+(s-17)/120),sg.cosine,s=3,alpha=.25)
    finish('expert_distributions')
    _,axes=plt.subplots(2,3,figsize=(12,7))
    for i,e in enumerate([8,12]):
        for j,metric in enumerate(['forced_gradient','update_alignment','functional_similarity']):
            ax=axes[i,j];g=layer[(layer.experts==e)&(layer.metric==metric)]
            for (age,s),b in g.groupby(['age','seed']):
                b=b.groupby('layer').value.mean();ax.plot(b.index,b.values,'o-',alpha=.55,label=f'R{age} s{s}')
            ax.set(title=f'{e}E {metric}',xlabel='MoE layer')
    axes.flat[-1].legend(fontsize=5);finish('layer_profiles')
    for metric in ['forced_gradient','update_alignment']:
        _,axes=plt.subplots(1,2,figsize=(10,4))
        for ax,e in zip(axes,[8,12]):
            g=seed[(seed.experts==e)&(seed.metric==metric)]
            for row in g.itertuples():ax.scatter(row.value,row.rerun_ticket_gap);ax.annotate(f'R{row.age}/s{row.seed}',(row.value,row.rerun_ticket_gap),fontsize=7)
            ax.set(title=f'{e}E',xlabel=metric,ylabel='Final sparse − dense loss')
        finish(f'ticket_gap_vs_{metric}')
    for e in [8,12]:
        _,axes=plt.subplots(3,1,figsize=(7,9));plot_metric(axes[0],e,'forced_gradient');plot_metric(axes[1],e,'update_alignment')
        for (age,s),g in curves[curves.experts==e].groupby(['age','seed']):axes[2].plot(g.step,g.ticket_gap,label=f'R{age}/s{s}',alpha=.65)
        axes[2].set(xlabel='Recovery step',ylabel='Sparse − dense validation loss');axes[2].legend(ncol=3,fontsize=7)
        finish(f'{e}E_temporal_alignment_and_loss')
    # Descriptive verdict, with seed failures taking precedence over architecture averages.
    support={m:bool(paired[paired.metric==m].favors_target.all()) for m in time.metric.unique()}
    reproduction=bool(repro.within_preregistered_tolerance.all())
    winners=seed[seed.metric_rank==1].copy();winners['matches_actual_winner']=winners.actual_ticket_gap_rank==1
    save(winners,'seed_specific_mechanism_winners')
    step0=time[time.step==0];zero=[]
    for e in [8,12]:
        target=20 if e==8 else 60
        g=step0[(step0.experts==e)&(step0.metric=='forced_gradient')].pivot(index='seed',columns='age',values='value')
        zero.append(all((g[target]>g[a]).all() for a in g if a!=target))
    precedes=temporal[(temporal.metric=='forced_gradient')].metric_before_or_with_loss.fillna(False).all()
    if not reproduction:verdict='DIAGNOSTIC RERUN FAILED TO REPRODUCE HISTORICAL PHENOMENON'
    elif support['forced_gradient'] and support['update_alignment'] and support['functional_similarity'] and precedes:
        verdict='COMMON SPARSE-EXPERT OPTIMIZATION GEOMETRY SUPPORTED' if all(zero) else 'EMERGENT CO-ADAPTATION MECHANISM SUPPORTED' if not any(zero) else 'SEED-CONDITIONED OPTIMIZATION EFFECT'
    elif support['native_gradient'] and not support['forced_gradient']:verdict='ROUTING-SELECTION MECHANISM SUPPORTED'
    elif (winners[winners.metric.isin(['forced_gradient','update_alignment','functional_similarity'])].groupby('metric').matches_actual_winner.sum()>=5).any():verdict='SEED-CONDITIONED OPTIMIZATION EFFECT'
    else:verdict='NO COMMON OPTIMIZATION GEOMETRY FOUND'
    report=['# Cross-architecture optimization geometry diagnostic rerun','',f'Completed 18/18 matched pairs, 36 recoveries of 2500 steps and 324 complete snapshots. Reproduction tolerance gate: {reproduction}. All snapshot mutation audits passed.','', 'Seed is the replication unit (n=3). Expert/token measurements describe within-seed structure. No significance or causality is inferred from these descriptive comparisons.','', '## Primary results','',pd.DataFrame(cross).to_csv(index=False),'','## Questions and interpretation','']
    questions=[
        ('8E:R20 forced alignment versus controls',paired[(paired.experts==8)&(paired.metric=='forced_gradient')].to_csv(index=False)),
        ('12E:R60 forced alignment versus controls',paired[(paired.experts==12)&(paired.metric=='forced_gradient')].to_csv(index=False)),
        ('Native pattern',f'All paired seed contrasts favor architecture targets: {support["native_gradient"]}.'),
        ('Effect after controlling token assignments',f'All forced contrasts favor targets: {support["forced_gradient"]}. Native gate amplitudes remain free; forcing does not isolate expert state from all router-mediated effects.'),
        ('Update support',f'All update contrasts favor targets: {support["update_alignment"]}.'),
        ('Parameter trajectories',f'All trajectory-distance contrasts favor targets: {support["trajectory_distance"]}.'),
        ('Functional similarity',f'All common-input output-cosine contrasts favor targets: {support["functional_similarity"]}.'),
        ('Step 0',f'Forced advantage across every seed and control at initialization: 8E={zero[0]}, 12E={zero[1]}.'),
        ('Emergence', 'Inspect step-0 and later trajectories separately; lack of a uniform step-0 effect does not itself prove co-adaptation.'),
        ('Temporal ordering',temporal.to_csv(index=False)),
        ('Seed-specific winners',winners[KEY+['metric','metric_rank','actual_ticket_gap_rank','matches_actual_winner']].to_csv(index=False)),
        ('Expert consistency',heterogeneity.groupby(['experts','metric','target_age','control_age']).fraction_experts_favoring_target.agg(['mean','min','max']).to_csv()+' Expert dispersion is within-seed structure, not independent replication.'),
        ('Layer consistency',heterogeneity.groupby(['experts','metric','layer']).paired_expert_mean_difference.mean().to_csv()+' See expert_layer_paired_effects.csv for every seed and control separately.'),
        ('Shared property',f'Descriptive evidence is classified as {verdict}; a gradient-only pattern is insufficient for universal geometry.'),
        ('Strongest mechanistic statement','These paired diagnostics test whether optimization/function alignment accompanies router-age recovery effects. Forced accepted tokens remove a selection difference on the fixed probe; they do not establish causal mediation of training outcomes.'),
        ('Unsupported claims','Universal architecture independence, causal mediation, and direct 8E↔12E expert equivalence remain unsupported. A late alignment advantage cannot explain an earlier loss advantage.')]
    for i,(q,a) in enumerate(questions,1):report.extend([f'{i}. **{q}.**', '',str(a),''])
    report.extend(['## Reproduction', '',repro.to_csv(index=False),'','## Limitations','','The probe contains eight fixed validation sequences. Forced-route gradient diagnostics use fp32 while scientific training remains historical fp16. Routing IDs and accepted token sets are fixed, but gate amplitudes and upstream representations may differ. Functional diagnostics separately fix expert inputs. Gradient coherence uses only two microbatches. Undefined cosines are excluded and their counts retained in diagnostic_coverage_summary.csv. Alignment means therefore describe experts with measurable nonzero gradients on this probe; router-dependent coverage must be considered alongside the means.','',verdict,'','**Most important implication for the paper:**','Architecture-specific favorable router ages must be discussed in terms of the observed paired seed effects, with initialization and later adaptation separated. The diagnostics provide a constrained mechanistic association, not proof of causal mediation or universal optimization geometry.','','**Most important next experiment:**','Run one matched intervention that holds the complete routed-token schedule fixed during sparse/dense recovery to test whether the observed alignment–recovery association survives removal of evolving token selection. Do not launch automatically.'])
    (ROOT/'final_report.md').write_text('\n'.join(report)+'\n')
    manifest.update(status='analysis_ready_for_review',provisional_verdict=verdict);(ROOT/'run_manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    print(f'Completed pairs: 18/18\nReproduction: {reproduction}\nForced gradient consistent: {support["forced_gradient"]}\nUpdate alignment consistent: {support["update_alignment"]}\nStep-0 forced effect (8E,12E): {zero}\nSeed consistency: see paired_seed_differences.csv\nFinal verdict: {verdict}\nReport: {ROOT / "final_report.md"}')

if __name__=='__main__':main()
