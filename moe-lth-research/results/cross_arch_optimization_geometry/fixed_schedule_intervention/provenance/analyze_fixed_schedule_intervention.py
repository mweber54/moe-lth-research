"""Prespecified paired analysis of a single complete trajectory schedule intervention."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=Path('results/cross_arch_optimization_geometry/fixed_schedule_intervention')
POLICIES=['baseline_native','baseline_fixed','clamped_native','clamped_fixed']

def table(frame):
    def cell(v):
        if pd.isna(v):return '—'
        return f'{v:.5f}' if isinstance(v,float) else str(v)
    return '\n'.join(['| '+' | '.join(frame.columns)+' |','| '+' | '.join(['---']*len(frame.columns))+' |']+['| '+' | '.join(cell(v) for v in row)+' |' for row in frame.itertuples(index=False,name=None)])

def main():
    m=json.loads((ROOT/'run_manifest.json').read_text());out=pd.read_csv(ROOT/'outcomes.csv')
    assert m['completed_donors']==6 and m['completed_pairs']==18 and m['completed_recipient_recoveries']==36
    assert len(out)==18 and not out.duplicated(['experts','seed','age']).any()
    audit=pd.read_csv(ROOT/'replay_audit.csv');assert len(audit)==36 and audit.all_assignments_and_accepted_tokens_identical.all() and audit.training_forwards.eq(2500).all()
    assert audit.groupby(['experts','seed']).training_schedule_hash.nunique().eq(1).all() and audit.groupby(['experts','seed']).validation_schedule_hash.nunique().eq(1).all()
    donors=pd.read_csv(ROOT/'donor_reproduction.csv');self_replay=pd.read_csv(ROOT/'self_replay_control.csv');assert len(donors)==len(self_replay)==6 and donors.bit_identical_to_history.all() and self_replay.bit_identical_to_donor.all()
    snaps=pd.read_csv(ROOT/'snapshot_manifest.csv');assert len(snaps)==378 and snaps.path.map(lambda p:Path(p).is_file()).all()
    mutations=pd.read_csv(ROOT/'snapshot_mutation_audit.csv');assert len(mutations)==378 and mutations[['model_unchanged','optimizer_unchanged','scaler_unchanged','rng_restored']].all().all()
    assert np.isfinite(out.select_dtypes('number').values).all()
    long=[];ranks=[];contrasts=[]
    for row in out.itertuples():
        for policy in POLICIES:long.append(dict(experts=row.experts,seed=row.seed,age=row.age,policy=policy,ticket_gap=getattr(row,policy+'_ticket_gap')))
    long=pd.DataFrame(long);long['rank']=long.groupby(['experts','seed','policy']).ticket_gap.rank(method='min',ascending=True);long.to_csv(ROOT/'seed_rankings.csv',index=False)
    mean=long.groupby(['experts','age','policy']).ticket_gap.agg(['mean','std','count']).reset_index().rename(columns={'std':'seed_sd','count':'seeds'});mean.to_csv(ROOT/'architecture_summary.csv',index=False)
    for (e,s),g in long.groupby(['experts','seed']):
        rr={'experts':e,'seed':s}
        for policy,b in g.groupby('policy'):rr[policy]=' > '.join('R'+str(a) for a in b.sort_values('rank').age)
        ranks.append(rr)
        target=20 if e==8 else 60;controls=[0,100] if e==8 else [20,100];w=g.pivot(index='age',columns='policy',values='ticket_gap')
        for control in controls:
            baseline_fixed=w.loc[target,'baseline_fixed']-w.loc[control,'baseline_fixed'];clamped_fixed=w.loc[target,'clamped_fixed']-w.loc[control,'clamped_fixed']
            baseline_native=w.loc[target,'baseline_native']-w.loc[control,'baseline_native'];clamped_native=w.loc[target,'clamped_native']-w.loc[control,'clamped_native']
            contrasts.append(dict(experts=e,seed=s,target_age=target,control_age=control,baseline_fixed_contrast=baseline_fixed,clamped_fixed_contrast=clamped_fixed,primary_training_interaction=clamped_fixed-baseline_fixed,primary_pairwise_order_flipped=bool(baseline_fixed*clamped_fixed<0),baseline_native_contrast=baseline_native,clamped_native_contrast=clamped_native,secondary_native_readout_interaction=clamped_native-baseline_native,secondary_pairwise_order_flipped=bool(baseline_native*clamped_native<0),evaluation_only_pairwise_order_flipped=bool(baseline_native*baseline_fixed<0)))
    ranks=pd.DataFrame(ranks);ranks.to_csv(ROOT/'ranking_comparison.csv',index=False)
    contrasts=pd.DataFrame(contrasts);contrasts.to_csv(ROOT/'paired_contrasts.csv',index=False)
    summary=contrasts.groupby(['experts','target_age','control_age']).agg(interaction_mean=('primary_training_interaction','mean'),interaction_seed_sd=('primary_training_interaction','std'),positive_interactions=('primary_training_interaction',lambda s:int((s>0).sum())),negative_interactions=('primary_training_interaction',lambda s:int((s<0).sum())),ordering_flips=('primary_pairwise_order_flipped','sum'),native_readout_ordering_flips=('secondary_pairwise_order_flipped','sum'),evaluation_only_flips=('evaluation_only_pairwise_order_flipped','sum')).reset_index();summary.to_csv(ROOT/'paired_effect_summary.csv',index=False)
    curves=[]
    for row in out.itertuples():
        base=ROOT/f'recovery/e{row.experts}/seed{row.seed}/R{row.age}'
        s=pd.read_json(base/'sparse/metrics.jsonl',lines=True);d=pd.read_json(base/'dense/metrics.jsonl',lines=True);g=s.merge(d,on='step',suffixes=('_sparse','_dense'));g['ticket_gap']=g.loss_sparse-g.loss_dense
        g['experts']=row.experts;g['seed']=row.seed;g['age']=row.age;curves.append(g)
    curves=pd.concat(curves,ignore_index=True);curves.to_csv(ROOT/'fixed_evaluation_recovery_curves.csv',index=False)
    figs=ROOT/'figures';figs.mkdir(exist_ok=True)
    def finish(name):plt.tight_layout();plt.savefig(figs/f'{name}.png',dpi=180);plt.savefig(figs/f'{name}.pdf');plt.close()
    _,axes=plt.subplots(2,2,figsize=(11,8))
    colors={'baseline_native':'gray','baseline_fixed':'tab:blue','clamped_native':'tab:orange','clamped_fixed':'tab:green'}
    for i,e in enumerate([8,12]):
        for j,(policies,title) in enumerate([(['baseline_fixed','clamped_fixed'],'Primary: shared fixed evaluation'),(['baseline_native','clamped_native'],'Secondary: native evaluation')]):
            ax=axes[i,j]
            for policy in policies:
                for s,g in long[(long.experts==e)&(long.policy==policy)].groupby('seed'):ax.plot(g.age,g.ticket_gap,color=colors[policy],alpha=.35,ls={7:'-',17:'--',29:':'}[s])
                g=mean[(mean.experts==e)&(mean.policy==policy)];ax.errorbar(g.age,g['mean'],yerr=g.seed_sd,marker='o',color=colors[policy],label=policy)
            ax.set(title=f'{e}E — {title}',xlabel='Router initialization age',ylabel='Sparse − dense loss');ax.legend(fontsize=8)
    finish('matched_ticket_gaps')
    _,axes=plt.subplots(1,2,figsize=(11,4))
    for ax,e in zip(axes,[8,12]):
        g=contrasts[contrasts.experts==e]
        for s,b in g.groupby('seed'):ax.plot(b.control_age,b.primary_training_interaction,'o-',label=f'seed {s}')
        ax.axhline(0,color='black',ls='--');ax.set(title=f'{e}E: change in target-minus-control gap',xlabel='Control router age',ylabel='Clamped contrast − native-trained contrast');ax.legend()
    finish('paired_training_interactions')
    _,axes=plt.subplots(2,3,figsize=(12,7))
    for ax,((e,s),g) in zip(axes.flat,long.groupby(['experts','seed'])):
        for policy,b in g.groupby('policy'):ax.plot(b.age,b['rank'],'o-',label=policy,color=colors[policy])
        ax.set(title=f'{e}E seed {s}',xlabel='Router age',ylabel='Gap rank (1 = best)',yticks=[1,2,3]);ax.invert_yaxis()
    axes.flat[-1].legend(fontsize=6);finish('seed_rankings')
    _,axes=plt.subplots(1,2,figsize=(11,4))
    for ax,e in zip(axes,[8,12]):
        for (age,s),g in curves[curves.experts==e].groupby(['age','seed']):ax.plot(g.step,g.ticket_gap,label=f'R{age}/s{s}')
        ax.set(title=f'{e}E: clamped recovery, fixed evaluation',xlabel='Recovery step',ylabel='Sparse − dense validation loss');ax.legend(ncol=3,fontsize=6)
    finish('recovery_ticket_gaps')
    # Preregistered descriptive classification, not a significance test.
    if summary.ordering_flips.eq(3).any():verdict='CONSISTENT PAIRWISE ORDERING CHANGE UNDER ROUTE REPLAY'
    elif summary.ordering_flips.gt(0).any():verdict='SEED-CONDITIONED ORDERING CHANGE UNDER ROUTE REPLAY'
    else:verdict='PAIRWISE ORDERING PRESERVED UNDER ROUTE REPLAY'
    integrity={'donors_exactly_reproduce_history':6,'dense_R100_self_replays_exact':6,'matched_pairs':18,'complete_recipient_replays':36,'snapshots':378,'snapshot_mutation_audits':378,'all_primary_values_finite':True,'all_architecture_seed_schedule_hashes_identical_across_recipients':True}
    (ROOT/'artifact_integrity.json').write_text(json.dumps(integrity,indent=2)+'\n')
    report=['# Complete recovery-trajectory routing intervention','',f'**{verdict}.** Six R100 dense donor captures and all 18 matched intervention pairs completed. Every donor reproduced its historical final model exactly; every dense R100 self-replay reproduced its donor exactly. All 36 recipient replays passed full assignment, accepted-token, input-identity, and chronological schedule-hash checks.','',
    '## Intervention and primary comparison','',
    'For each architecture/seed, all three router ages and both sparse/dense models replay the same complete R100 dense recovery trajectory. This fixes endogenous token selection to a shared pre-recorded sequence, including capacity decisions; the sequence itself can change with recovery step. Native differentiable gate amplitudes, trainable routers, shared parameters, the original initialization/masks, data order, fresh AdamW, fp16/GradScaler, clipping, and the 2500-step budget are retained.','',
    'The primary outcome compares native-trained historical checkpoints with clamped-trained checkpoints under the SAME replayed donor validation routes. The interaction is (clamped target-minus-control ticket gap) minus (native-trained target-minus-control ticket gap). Negative values shift the comparison toward the architecture-specific target. Native evaluation of both arms is reported separately. Changes caused solely by changing evaluation routes are not evidence of a training effect.','',
    '## Paired results','',table(summary),'',table(contrasts),'',
    '## Seed rankings and architecture means','',
    'Rankings run from lowest (best) to highest ticket gap, always within the prespecified router grid.','',table(ranks),'',table(mean),'',
    '## Interpretation','',
    f'The primary fixed-readout comparison contains {int(summary.ordering_flips.sum())} seed-level pairwise ordering flips across the 12 preregistered target/control comparisons. {int(summary.ordering_flips.eq(3).sum())} of the four architecture/control contrasts flip in all three seeds. The secondary native-readout comparison contains {int(summary.native_readout_ordering_flips.sum())} flips. Merely reevaluating unchanged native-trained models under donor routes produces {int(summary.evaluation_only_flips.sum())} flips; these evaluation-only effects must be separated from the training intervention.','',
    'This is a controlled intervention on one donor-specific token-selection schedule. A training-arm difference at a common readout supports an effect of that schedule substitution in these matched runs. It does not show that all natural router-age effects are mediated by token selection, or that any alternative common schedule would have the same effect. Gates and upstream expert inputs remain endogenous; forcing them as well would answer a different question. With only three seeds per architecture, report paired effects and SD without claiming statistical significance or architecture independence.','',
    '## Audit and storage','',table(pd.DataFrame([integrity])),'',
    'Primary data: outcomes.csv, paired_contrasts.csv, paired_effect_summary.csv, ranking_comparison.csv, and architecture_summary.csv. Full replay evidence: donor_reproduction.csv, self_replay_control.csv, replay_audit.csv, schedule manifests, and snapshot audits. Native routing-stat probes in recovery logs describe unforced proposals and are not the routes used for parameter updates. Historical recovery-fraction/time-to-threshold fields use the old helper denominator and are not used in this intervention analysis.','',
    f'Bulk schedules, recovery artifacts and checkpoints are stored at `{m["bulk_storage"]}` and linked from this result directory. The historical-results filesystem was nearly full; preserve that local directory together with this report. Nothing has been automatically deleted.','',
    '## Verdict','',verdict,'',
    '**Most important implication:** The conclusion applies to the measured ordering changes under this specific complete R100 donor replay, with training and evaluation effects separated. The previously observed optimization-geometry correlations are not substituted for this intervention result. No additional experiment was launched.']
    (ROOT/'final_report.md').write_text('\n'.join(report)+'\n')
    m.update(status='analysis_ready_for_review',provisional_verdict=verdict);(ROOT/'run_manifest.json').write_text(json.dumps(m,indent=2)+'\n')
    text=f'Donors: 6/6 exact. Self-replay controls: 6/6 exact. Intervention pairs: 18/18.\nPrimary seed-level ordering flips: {int(summary.ordering_flips.sum())}/12.\nVerdict: {verdict}\nReport: {(ROOT/"final_report.md").resolve()}'
    (ROOT/'terminal_summary.txt').write_text(text+'\n');print(text)

if __name__=='__main__':main()
