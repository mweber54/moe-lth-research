#!/usr/bin/env python3
"""Read-only native localization analysis plus historical fixed-probe measurements."""
import json
import math
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from run_16e_router_age_localization import OUT,ROOT,SEEDS,config_for,write_json,sha256_file
from analyze_r20_8_vs_r60_12 import collect,probe_batches,js,cumulative_exposure
from moe_lth.experiments import run_router_age_recovery as r
from moe_lth.utils import configure_device,resolve_device

FEATURES=['normalized_entropy','margin','load_cv','specialization','assignment_agreement_to_R100','functional_js_to_R100','cumulative_exposure_per_expert']

def developmental(outcomes):
    path=OUT/'developmental_metrics.csv'
    if path.exists():
        cached=pd.read_csv(path)
        if set(zip(cached.seed,cached.router_age))==set(zip(outcomes.seed,outcomes.router_age)):return cached
        raise RuntimeError('Partial developmental table requires explicit audit before reuse')
    rows=[];probe_hashes={}
    for seed in SEEDS:
        c=config_for(seed);device=resolve_device(c['device']);configure_device(device);batches=probe_batches(c)
        probe_hashes[str(seed)]=r._batch_sequence_hash([(x,x) for x in batches])
        root=OUT/f'references/seed_{seed}'
        mature_model=r.load_model_from_checkpoint(c['model'],str(root/'checkpoints/step_2500.pt'),device)
        mature=collect(mature_model,batches,device);del mature_model
        for age in sorted(outcomes.router_age.unique()):
            model=r.load_model_from_checkpoint(c['model'],str(root/f'checkpoints/step_{int(age)*25}.pt'),device)
            trace=collect(model,batches,device);del model
            probabilities=torch.cat([x[3] for x in trace]);top=probabilities.topk(2,-1).values
            entropy=float(-(probabilities*probabilities.clamp_min(1e-12).log()).sum(-1).mean())
            cvs=[];mis=[];agreement=[];distances=[]
            for layer in range(4):
                layer_rows=[x for x in trace if x[0]==layer]
                ids=torch.cat([x[4] for x in layer_rows]);tokens=torch.cat([x[2] for x in layer_rows])
                counts=torch.bincount(ids,minlength=16).float();util=counts/counts.sum();cvs.append(float(util.std(unbiased=False)/util.mean()))
                joint=torch.bincount(ids*256+tokens,minlength=16*256).reshape(16,256).double();joint/=joint.sum()
                independent=joint.sum(1,keepdim=True)*joint.sum(0,keepdim=True);nonzero=joint>0
                mis.append(float((joint[nonzero]*(joint[nonzero]/independent[nonzero]).log()).sum()))
                # Historical maturity implementation compares first probe batch per layer.
                a=next(x for x in trace if x[0]==layer);b=next(x for x in mature if x[0]==layer)
                agreement.append(float((a[4]==b[4]).float().mean()));distances.append(js(a[3],b[3]))
            exposure=cumulative_exposure(root,int(age)*25,16)
            if exposure is None:raise RuntimeError('Historical logged-assignment exposure unavailable')
            gap=float(outcomes.loc[(outcomes.seed==seed)&(outcomes.router_age==age),'ticket_gap'].iloc[0])
            rows.append({'experts':16,'expert_count':16,'seed':seed,'router_age':int(age),'entropy':entropy,'normalized_entropy':entropy/math.log(16),'margin':float((top[:,0]-top[:,1]).mean()),'load_cv':float(np.mean(cvs)),'token_specialization':float(np.mean(mis)),'specialization':float(np.mean(mis))/math.log(16),'assignment_agreement_to_r100':float(np.mean(agreement)),'assignment_agreement_to_R100':float(np.mean(agreement)),'js_distance_to_r100':float(np.mean(distances)),'functional_js_to_R100':float(np.mean(distances)),'cumulative_expert_exposure':float(np.mean(exposure)),'cumulative_exposure_per_expert':float(np.mean(exposure)),'ticket_gap':gap})
        del mature;torch.cuda.empty_cache()
    pd.DataFrame(rows).to_csv(path,index=False)
    write_json(OUT/'developmental_probe_manifest.json',{'probe_sequence_hashes':probe_hashes,'source_implementation':str(ROOT/'scripts/analyze_r20_8_vs_r60_12.py'),'source_sha256':sha256_file(ROOT/'scripts/analyze_r20_8_vs_r60_12.py'),'certainty_load_specialization_probe':'first 8 deterministic validation batches, full reference model','maturity_probe':'first validation batch per layer, exactly as historical implementation; not silently expanded to all eight','specialization':'token-ID/expert mutual information; normalized by log(experts) for developmental vector','exposure':'sum of historical logged train assignment counts at step 1 and every 25 steps, averaged over layers/experts; NOT complete cumulative accepted updates','distance_normalization':'Equal-weight root mean square of feature differences divided by historical 8E/12E pooled feature ranges. Frozen historical ranges preserve the prior coordinate system.'})
    return pd.DataFrame(rows)

def analyze():
    outcomes=pd.read_csv(OUT/'outcomes.csv');decision=json.loads((OUT/'localization_decision.json').read_text())
    expected=set(decision['stage_a_ages']+decision['stage_b_ages']);assert set(outcomes.router_age)==expected
    assert len(outcomes)==len(expected)*3 and outcomes.groupby('router_age').seed.nunique().eq(3).all()
    runs=pd.read_csv(OUT/'run_manifest.csv');assert len(runs)==2*len(outcomes)
    checks=[]
    for row in runs.itertuples():
        dest=Path(row.condition_dir);m=json.loads((dest/'metadata.json').read_text())
        assert m['integrity_checks_passed'] and m['optimizer_state_entries_at_start']==0 and m['recovery_steps']==2500 and m['scheduler']=='none' and m['precision']=='fp16' and m['router_mode']=='trainable' and not m['confidence_control']
        assert sha256_file(dest/'checkpoints/final_recovered.pt')==row.final_checkpoint_sha256
        curve=pd.read_json(dest/'metrics.jsonl',lines=True);assert curve.step.tolist()==list(range(0,2501,50))
        checks.append({'path':str(dest),'final_checkpoint_sha256':row.final_checkpoint_sha256,'validated':True})
    summaries=[]
    for age,g in outcomes.groupby('router_age'):
        row={'experts':16,'router_age':int(age),'num_seeds':3,'mean_pruning_shock':g.immediate_pruning_shock.mean()}
        for col in ['sparse_final_loss','dense_final_loss','ticket_gap']:
            row['mean_'+col]=g[col].mean();row['sd_'+col]=g[col].std(ddof=1)
        summaries.append(row)
    summary=pd.DataFrame(summaries).sort_values('router_age');summary.to_csv(OUT/'age_summary.csv',index=False)
    winners=[]
    for seed,g in outcomes.groupby('seed'):
        top=g.sort_values(['ticket_gap','router_age']).iloc[:2]
        winners.append({'seed':int(seed),'best_router_age':int(top.iloc[0].router_age),'best_ticket_gap':top.iloc[0].ticket_gap,'second_best_router_age':int(top.iloc[1].router_age),'second_best_ticket_gap':top.iloc[1].ticket_gap,'difference':top.iloc[1].ticket_gap-top.iloc[0].ticket_gap})
    winners=pd.DataFrame(winners);winners.to_csv(OUT/'seed_winners.csv',index=False)
    best=int(summary.loc[summary.mean_ticket_gap.idxmin(),'router_age']);bestrow=summary.set_index('router_age').loc[best]
    pivot=outcomes.pivot(index='seed',columns='router_age',values='ticket_gap')
    comparisons=[]
    for age in sorted(expected):
        diff=pivot[age]-pivot[best]
        comparisons.append({'router_age':age,'mean_gap_above_best':diff.mean(),'paired_seed_sd':diff.std(ddof=1),'within_one_paired_sd':bool(abs(diff.mean())<=diff.std(ddof=1))})
    comparison=pd.DataFrame(comparisons);comparison.to_csv(OUT/'local_uncertainty.csv',index=False)
    # Connected local basin on observed age grid; never use developmental features to choose it.
    ordered=sorted(expected);index=ordered.index(best);basin=[best]
    for direction in (-1,1):
        j=index+direction
        while 0<=j<len(ordered):
            age=ordered[j]
            if not comparison.set_index('router_age').loc[age,'within_one_paired_sd']:break
            basin.append(age);j+=direction
    basin=sorted(basin)
    vs0=pivot[best]-pivot[0];vs100=pivot[best]-pivot[100];both=(vs0<0)&(vs100<0)
    stable=all(int(a) in basin for a in winners.best_router_age)
    mean_curve=summary.mean_ticket_gap.to_numpy();monotonic=bool(np.all(np.diff(mean_curve)<=0) or np.all(np.diff(mean_curve)>=0))
    if not stable:verdict='16E OPTIMUM SEED-CONDITIONED'
    elif best in (0,100):verdict='16E NO INTERMEDIATE OPTIMUM' if monotonic else '16E BOUNDARY OPTIMUM'
    elif not both.all():verdict='16E OPTIMUM SEED-CONDITIONED'
    elif len(basin)>1:verdict='16E BROAD INTERMEDIATE TICKET BASIN'
    else:verdict='16E INTERMEDIATE TICKET OPTIMUM OBSERVED'
    metrics=developmental(outcomes)
    old=pd.read_csv(ROOT/'results/r20_8_vs_r60_12/developmental_vectors.csv')
    scales=(old[FEATURES].max()-old[FEATURES].min()).clip(lower=1e-12)
    distances=[]
    for seed in SEEDS:
        for experts,anchor_age in [(8,20),(12,60)]:
            anchor=old[(old.expert_count==experts)&(old.seed==seed)&(old.router_age==anchor_age)].iloc[0]
            for row in metrics[metrics.seed==seed].to_dict('records'):
                d=(np.array([row[f]-anchor[f] for f in FEATURES])/scales.to_numpy())
                distances.append({'seed':seed,'anchor':f'{experts}E_R{anchor_age}','router_age':row['router_age'],'distance':float(np.sqrt(np.mean(d*d)))})
    distances=pd.DataFrame(distances);distances.to_csv(OUT/'developmental_distances.csv',index=False)
    optimum_distances=[]
    for seed in SEEDS:
        anchors={f'{e}E_R{age}':old[(old.expert_count==e)&(old.seed==seed)&(old.router_age==age)].iloc[0] for e,age in [(8,20),(12,60)]}
        anchors[f'16E_R{best}']=metrics[(metrics.seed==seed)&(metrics.router_age==best)].iloc[0]
        labels=list(anchors)
        for i,first in enumerate(labels):
            for second in labels[i+1:]:
                delta=(anchors[first][FEATURES].to_numpy(dtype=float)-anchors[second][FEATURES].to_numpy(dtype=float))/scales.to_numpy()
                optimum_distances.append({'seed':seed,'first':first,'second':second,'normalized_distance':float(np.sqrt(np.mean(delta*delta)))})
    pd.DataFrame(optimum_distances).to_csv(OUT/'cross_architecture_optimum_distances.csv',index=False)
    closest={a:int(g.groupby('router_age').distance.mean().idxmin()) for a,g in distances.groupby('anchor')}
    cross=[]
    for experts,age in [(8,20),(12,60),(16,best)]:
        if experts==16:g=outcomes[outcomes.router_age==age];dev=metrics[metrics.router_age==age];gaps=g.ticket_gap
        else:
            source=ROOT/('results/expert_count_replication/ticket_gap_seed_level.csv' if experts==8 else 'results/router_development_12expert_retry4_20260902/ticket_gap_seed_level.csv')
            g=pd.read_csv(source)
            if 'expert_count' in g:g=g[g.expert_count==experts]
            age_column='router_age_percent' if 'router_age_percent' in g else 'router_age'
            g=g[(g[age_column]==age)&np.isclose(g.sparsity,.85)]
            assert len(g)==3
            gaps=g.ticket_gap;dev=old[(old.expert_count==experts)&(old.router_age==age)]
        cross.append({'architecture':f'{experts}E','optimal_router_age':age,'mean_ticket_gap':gaps.mean(),'seed_sd':gaps.std(ddof=1),'entropy':dev.normalized_entropy.mean()*math.log(experts),'margin':dev.margin.mean(),'load_cv':dev.load_cv.mean(),'token_specialization':dev.specialization.mean()*math.log(experts),'assignment_agreement':dev.assignment_agreement_to_R100.mean(),'js_distance_to_maturity':dev.functional_js_to_R100.mean(),'cumulative_expert_exposure':dev.cumulative_exposure_per_expert.mean()})
    pd.DataFrame(cross).to_csv(OUT/'cross_architecture_optima.csv',index=False)
    figure_dir=OUT/'figures';figure_dir.mkdir(exist_ok=True)
    def save(name):
        plt.tight_layout()
        for ext in ('png','pdf'):plt.savefig(figure_dir/f'{name}.{ext}',dpi=160)
        plt.close()
    def seedplot(frame,x,y):
        for seed,g in frame.groupby('seed'):
            g=g.sort_values(x);plt.plot(g[x],g[y],'.-',alpha=.65,label=f'seed {seed}')
    plt.figure();seedplot(outcomes,'router_age','ticket_gap');plt.errorbar(summary.router_age,summary.mean_ticket_gap,yerr=summary.sd_ticket_gap,color='black',label='mean ± seed SD');plt.xlabel('Router age (%)');plt.ylabel('Matched ticket gap');plt.legend();save('01_ticket_gap')
    plt.figure()
    for kind in ('sparse','dense'):
        for seed,g in outcomes.groupby('seed'):plt.plot(g.sort_values('router_age').router_age,g.sort_values('router_age')[kind+'_final_loss'],'.-' if kind=='sparse' else '.--',label=f'{kind} seed {seed}')
    plt.xlabel('Router age (%)');plt.ylabel('Final validation loss');plt.legend(fontsize=7);save('02_sparse_dense_loss')
    plt.figure();seedplot(outcomes,'router_age','immediate_pruning_shock');plt.xlabel('Router age (%)');plt.ylabel('Initial sparse − dense loss');plt.legend();save('03_pruning_shock')
    plt.figure()
    for kind in ('sparse','dense'):
        for seed,g in outcomes.groupby('seed'):
            g=g.sort_values('router_age');plt.plot(g.router_age,g[kind+'_early_auc'],'.-' if kind=='sparse' else '.--',label=f'{kind} seed {seed}')
    plt.xlabel('Router age (%)');plt.ylabel('Historical early-recovery AUC');plt.legend(fontsize=7);save('04_early_auc')
    fig,axes=plt.subplots(1,3,figsize=(12,3))
    for ax,(seed,g) in zip(axes,outcomes.groupby('seed')):
        g=g.sort_values('router_age');ax.plot(g.router_age,g.ticket_gap,'.-');ax.set_title(f'seed {seed}');ax.set_xlabel('Router age');ax.set_ylabel('Ticket gap')
    save('05_seed_curves')
    for number,metric in [(6,'entropy'),(7,'margin'),(8,'load_cv'),(9,'cumulative_expert_exposure')]:
        plt.figure()
        for seed,g in metrics.groupby('seed'):
            plt.scatter(g[metric],g.ticket_gap,label=f'seed {seed}')
            for row in g.itertuples():plt.annotate(f'R{row.router_age}',(getattr(row,metric),row.ticket_gap),fontsize=6)
        plt.xlabel(metric.replace('_',' '));plt.ylabel('Ticket gap');plt.legend();save(f'{number:02d}_gap_vs_{metric}')
    fig,axes=plt.subplots(1,2,figsize=(10,4))
    for ax,(anchor,g) in zip(axes,distances.groupby('anchor')):
        for seed,h in g.groupby('seed'):ax.plot(h.router_age,h.distance,'.-',label=f'seed {seed}')
        ax.axvline(best,color='black',linestyle=':',label='ticket-gap mean winner');ax.set_title(f'Distance to {anchor}');ax.set_xlabel('16E router age');ax.set_ylabel('Historical range-normalized RMS distance');ax.legend(fontsize=7)
    save('10_developmental_position')
    shock_best=int(outcomes.groupby('router_age').immediate_pruning_shock.mean().idxmin())
    aucgap=outcomes.assign(auc_gap=outcomes.sparse_early_auc-outcomes.dense_early_auc).groupby('router_age').auc_gap.mean();auc_best=int(aucgap.idxmin())
    sparse_best=int(summary.loc[summary.mean_sparse_final_loss.idxmin(),'router_age'])
    disagreements=[int(s) for s in both.index[~both]]
    later=best>60 and best not in (0,100) and 'SEED-CONDITIONED' not in verdict
    recommend=f'R{best}' if both.all() and stable and 0<best<100 else 'No stable intermediate target recommended'
    confidence='All three seeds favor the candidate over both endpoints; local resolution remains limited to sampled ages.' if both.all() and stable else 'Limited: seed disagreement or boundary optimum prevents a stable intermediate target.'
    questions=[
        ('Does 16E exhibit an intermediate router-age ticket optimum?',verdict),
        ('What router age or basin minimizes mean ticket gap?',f'Mean winner R{best}; local uncertainty basin {basin}, using the preregistered paired-SD descriptive rule.'),
        ('What is the mean ± seed SD?',f'{bestrow.mean_ticket_gap:.8f} ± {bestrow.sd_ticket_gap:.8f} (n=3 seeds).'),
        ('Does the optimum beat R0?',f'Mean paired difference {vs0.mean():+.8f}; negative favors candidate. Seed differences: {vs0.to_dict()}.'),
        ('Does the optimum beat R100?',f'Mean paired difference {vs100.mean():+.8f}; seed differences: {vs100.to_dict()}.'),
        ('Does it beat both in all three seeds?',str(bool(both.all()))),
        ('Which seeds disagree?',str(disagreements)),
        ('Is the optimum sharp or broad?',f'Observed local basin {basin}; inspect local_uncertainty.csv for raw differences and paired seed SD. No significance claim or unsampled-age precision.'),
        ('Does immediate pruning shock show the same optimum?',f'Minimum mean shock R{shock_best}; matches candidate: {shock_best==best}. Shock is measured on matched rewound E0/shared, age-router initial states.'),
        ('Does early recovery show the same optimum?',f'Minimum matched sparse−dense early AUC at R{auc_best}; matches: {auc_best==best}. Individual sparse/dense AUCs are retained.'),
        ('Is the result driven by sparse loss or matched recoverability?',f'The selection uses matched ticket gap only. Sparse-loss-only winner is R{sparse_best}; matched winner is R{best}. These descriptive results do not identify a causal optimization mechanism.'),
        ('How do developmental statistics evolve?', 'Age means are tabulated below. Entropy uses natural logs, margin uses probabilities, load CV uses population SD per layer, specialization uses historical token-ID/expert MI. Assignment/JS maturity uses the historical first-batch-per-layer implementation. Exposure is sampled logged assignments, not complete accepted updates.'),
        ('Which 16E checkpoint is closest to 8E R20?',f'R{closest["8E_R20"]}, by mean historical-range-normalized distance; individual seed distances retained.'),
        ('Which is closest to 12E R60?',f'R{closest["12E_R60"]}.'),
        ('Is the ticket optimum developmentally close to those optima?',f'It coincides with the closest sampled checkpoint to 8E R20: {best==closest["8E_R20"]}; to 12E R60: {best==closest["12E_R60"]}. Distance is descriptive; nearest does not establish a common developmental mechanism.'),
        ('Does increasing expert count shift the optimum later?',f'The observed mean sequence is 8E R20 → 12E R60 → 16E R{best}. A stable later interior optimum is supported descriptively: {later}. Three architectures do not establish a scaling law.'),
        ('What architecture-dependence claim is justified?',f'The measured matched ticket-gap curves support {verdict.lower()}. Only these model sizes, training budget, data, sparsity and three seeds are covered; developmental similarity is secondary.'),
        ('Should 16E enter the next gate-amplitude causal experiment?',recommend+'. Any future experiment requires a separate instruction.')]
    report='# 16-expert router-age ticket localization\n\n'+ '\n\n'.join(f'{i}. **{q}**\n\n{answer}' for i,(q,answer) in enumerate(questions,1))
    report+='\n\nAge summary (mean and sample SD across independent seeds):\n\n```text\n'+summary.to_string(index=False)+'\n```\n\nSeed winners:\n\n```text\n'+winners.to_string(index=False)+'\n```\n\nDevelopmental age means:\n\n```text\n'+metrics.groupby('router_age')[['entropy','margin','load_cv','token_specialization','assignment_agreement_to_r100','js_distance_to_r100','cumulative_expert_exposure']].mean().to_string()+'\n```\n\nAll primary readouts are native, with matched router-age dense recoveries. No replay interventions were run. Historical 8E/12E artifacts were reused without retraining. Logged exposure and first-batch maturity limitations preserve comparability but constrain developmental interpretation. No token, layer, expert, checkpoint or recovery step is treated as an independent replicate.\n\n'+verdict+'\n\nRecommended 16E target checkpoint for future mechanism experiments: '+recommend+'\n\nRecommended control checkpoints: R0 and R100; neighboring basin checkpoints if a future protocol requires local controls.\n\nConfidence based on seed consistency: '+confidence+'\n'
    (OUT/'final_report.md').write_text(report)
    decision.update({'status':'complete','candidate_optimum':best,'basin':basin,'verdict':verdict,'seed_winners':winners.to_dict('records'),'recommended_target':recommend,'developmentally_closest':closest});write_json(OUT/'localization_decision.json',decision)
    write_json(OUT/'artifact_integrity.json',{'status':'validated','invalid_runs':0,'recoveries':len(runs),'pairs':len(outcomes),'run_checks':checks,'files':{str(p.relative_to(OUT)):sha256_file(p) for p in OUT.iterdir() if p.is_file() and p.suffix in ('.csv','.md','.json') and p.name!='artifact_integrity.json'},'figures':{p.name:sha256_file(p) for p in figure_dir.iterdir()}})
    print('16E ROUTER-AGE LOCALIZATION COMPLETE\n')
    print('Stage A ages:',decision['stage_a_ages'],'\nStage B refinement ages:',decision['stage_b_ages'],'\nStage C confirmation ages:',decision['stage_c_ages'])
    print(summary[['router_age','mean_ticket_gap','sd_ticket_gap']].to_string(index=False))
    print(f'Candidate optimum: R{best}\nMean ticket gap: {bestrow.mean_ticket_gap}\nSeed SD: {bestrow.sd_ticket_gap}\nSeed-level winners: {winners.to_dict("records")}\nDifference vs R0: {vs0.mean()}\nDifference vs R100: {vs100.mean()}\n8E known optimum: R20\n12E known optimum: R60\n16E observed optimum/basin: {basin}\nDevelopmentally closest checkpoints: {closest}\nDoes ticket optimum coincide with developmental similarity?: {[best==a for a in closest.values()]}\nFinal verdict: {verdict}\nRecommended checkpoint for mechanism experiments: {recommend}\nPath to final_report.md: {OUT/"final_report.md"}',flush=True)

if __name__=='__main__':analyze()
