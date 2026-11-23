"""Review frozen measurement gate. Failure produces explicit stopped-study artifacts."""
from common import *
from measure import correlation
import csv
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def rows(path):
    with Path(path).open() as f:return list(csv.DictReader(f))


def main():
    reg=frozen();rule=reg['reliability'];reliability=[];summaries=[];exposure=[];checks={};coverage=[]
    assert not (D/'completion.json').exists(),'Review already completed'
    for seed in SEEDS:
        status=read(D/'status'/f'measure_seed{seed}.json');assert status['status']=='COMPLETE'
        p=D/status['attempt'];reliability+=rows(p/'score_reliability.csv');sr=rows(p/'tensor_reliability_summary.csv');summaries+=sr;exposure+=rows(p/'exposure_summary.csv')
        checks[str(seed)]=read(p/'measurement_checks.json')
        for x in sr:x['eligible']=x['eligible']=='True'
        eligible=sum(x['eligible'] for x in sr)
        strata=[]
        for l in range(4):
            for projection in ('fc1','fc2'):
                k=sum(x['eligible'] for x in sr if int(x['layer'])==l and x['projection']==projection)
                strata.append(dict(layer=l,projection=projection,eligible=k,total=32,passed=k>=rule['minimum_eligible_tensors_per_layer_projection']))
        coverage.append(dict(seed=seed,eligible=eligible,total=len(sr),fraction=eligible/len(sr),
          passed=eligible/len(sr)>=rule['minimum_eligible_tensors_fraction_per_seed'] and all(x['passed'] for x in strata),layer_projection=strata))
    table(D/'score_reliability.csv',reliability);table(D/'tensor_reliability_summary.csv',summaries);table(D/'exposure_summary.csv',exposure)
    passed=all(x['passed'] for x in coverage)
    gate={'status':'PASS' if passed else 'STOP_RELIABILITY','all_five_seeds_required':True,'coverage':coverage,
      'selection_on_full32_permitted':passed,'registered_thresholds':rule,
      'review_uses_intervention_losses':False}
    save(D/'reliability_review.json',gate)
    assoc=[]
    for seed in SEEDS:
        for layer in range(4):
            for projection in ('fc1','fc2'):
                rs=[x for x in summaries if int(x['seed'])==seed and int(x['layer'])==layer and x['projection']==projection and x['median_stratum_rho']!='']
                rho=correlation([float(x['accepted_total']) for x in rs],[float(x['median_stratum_rho']) for x in rs])
                assoc.append(dict(seed=seed,layer=layer,projection=projection,experts=len(rs),accepted_vs_reliability_spearman=rho,interpretation='descriptive expert association, not training replications'))
    table(D/'exposure_reliability_association.csv',assoc)
    if passed:
        print('PASS: proceed to registered matching, no intervention permitted yet',flush=True)
        return
    status='NOT_RUN_RELIABILITY_GATE'
    for filename in ('matched_group_summary.csv','magnitude_growth_balance.csv','structural_balance.csv','first_order_prediction_checks.csv'):
        table(D/filename,[dict(seed=s,branch=b,status=status,reason='Full-32 selection prohibited by split-half reliability failure') for s in SEEDS for b in ('H','L')])
    table(D/'per_branch_results.csv',[dict(seed=s,branch=b,status=status,immediate_validation_CE=None,final_validation_CE=None,attempted_updates=0,applied_updates=0,AMP_skips=0,control_reused=False) for s in SEEDS for b in ('C','H','L')])
    table(D/'learning_curves.csv',[dict(seed=s,branch=b,continuation_step=t,validation_CE=None,excess_CE=None,status=status) for s in SEEDS for b in ('C','H','L') for t in reg['endpoints']['validation_grid']])
    table(D/'paired_seed_effects.csv',[dict(seed=s,immediate_H_minus_L=None,integrated_H_minus_L=None,final_H_minus_L=None,H_minus_C=None,L_minus_C=None,status=status) for s in SEEDS])
    save(D/'masks'/'manifest.json',{'status':status,'coordinate_lists_created':False,'masks':{},'reason':'No full-32 selection after failed reliability gate; no zero-effect interpretation'})
    measured=['calibration_batch_token_target_hashes_verified','coordinate_schema_verified','weight_hash_preserved','RNG_preserved','module_modes_preserved','existing_gradients_preserved','TF32_flags_preserved']
    numerical={'stage':'MEASUREMENT_ONLY','measurement_checks':{str(s):{k:checks[str(s)][k] for k in measured} for s in SEEDS},
      'synthetic_streaming_checks':checks[str(SEEDS[0])]['synthetic_checks'],
      'no_intervention_or_validation_or_production':all(checks[str(s)]['no_intervention_loss_or_validation_or_training'] for s in SEEDS),
      'downstream_checks':{k:status for k in ['finite_differences_native_model','route_boundary_diagnostics','matching_counts_balance_disjointness','surviving_parameter_equality','fresh_optimizer_membership_reset','persistent_zeros_gradient_flow','continuation_hashes_runtime','interruption_resumption','control_reuse_compatibility']},
      'why_not_run':'Registered measurement gate failed. No masked groups or production branch exists; downstream checks are not marked passed.',
      'calibration_only_attempt_isolation_locks_heartbeat_atomic_status':True}
    save(D/'numerical_checks.json',numerical)
    figdir=D/'figures';figdir.mkdir(exist_ok=True)
    plt.rcParams.update({'font.size':10})
    fig,axes=plt.subplots(1,2,figsize=(12,4.5))
    for i,s in enumerate(SEEDS):
        rs=[r for r in summaries if int(r['seed'])==s and r['median_stratum_rho']!='']
        vals=[float(r['median_stratum_rho']) for r in rs]
        axes[0].scatter(np.full(len(vals),i)+np.random.default_rng(s).uniform(-.15,.15,len(vals)),vals,s=6,alpha=.35)
        axes[1].scatter([float(r['accepted_total']) for r in rs],vals,s=6,alpha=.25,label=str(s))
    axes[0].axhline(rule['minimum_median_stratum_rho'],ls='--',color='black',label='registered threshold')
    axes[0].set(xticks=range(5),xticklabels=SEEDS,xlabel='Source seed',ylabel='Median within-magnitude-stratum Spearman',title='Independent 16-batch sensitivity halves');axes[0].legend()
    axes[1].set(xlabel='Accepted calibration tokens per expert',ylabel='Median stratum Spearman',title='Exposure and score reliability');axes[1].legend(title='Seed')
    fig.tight_layout();fig.savefig(figdir/'sensitivity_reliability.png',dpi=180);fig.savefig(figdir/'sensitivity_reliability.pdf');plt.close(fig)
    for filename,title,subtitles in [
      ('magnitude_growth_structural_balance','Matching was not performed',['Magnitude balance','Growth and variation balance','Row and column degree balance']),
      ('immediate_final_paired_effects','No pruning effects were measured',['Immediate H minus L','Final H minus L']),
      ('continuation_CE_excess_loss','No continuations were run',['Validation task CE','Excess loss relative to control'])]:
        fig,axs=plt.subplots(1,len(subtitles),figsize=(5*len(subtitles),3.5),squeeze=False)
        for ax,subtitle in zip(axs[0],subtitles):
            ax.set(xticks=range(5),xticklabels=SEEDS,xlabel='Source seed',title=subtitle,yticks=[])
            for i in range(5):ax.text(i,.5,'NOT RUN',ha='center',va='center',fontsize=8)
            ax.set(xlim=(-.5,4.5),ylim=(0,1))
        fig.suptitle(title+' — registered reliability gate failed');fig.tight_layout();fig.savefig(figdir/(filename+'.png'),dpi=160);plt.close(fig)
    lines=['# Gradient sensitivity pruning: stopped at reliability gate','',
      'The fixed 32-batch calibration design failed the registered split-half reliability requirements. No masks were selected, no intervention losses were evaluated, no controls were reused, and no production training was launched. This is a measurement-feasibility result; it cannot answer whether sensitivity predicts persistent pruning damage.','',
      '| Source seed | Eligible tensors / 256 | Median tensor stratum rank agreement | Median high overlap | Median low overlap |','|---|---:|---:|---:|---:|']
    for s in SEEDS:
        rs=[r for r in summaries if int(r['seed'])==s]
        def med(key):return float(np.median([float(r[key]) for r in rs if r[key]!='']))
        c=next(x for x in coverage if x['seed']==s)
        lines.append(f"| {s} | {c['eligible']} / 256 | {med('median_stratum_rho'):.4f} | {med('median_high_overlap'):.4f} | {med('median_low_overlap'):.4f} |")
    failure_counts={k:sum(k in r['failed_criteria'].split(';') for r in summaries) for k in ('traffic','valid_strata','rank','high_overlap','low_overlap','cross_separation')}
    lines+=['',f"Failed tensor criteria across all five sources (1280 tensors; descriptive, not independent replications): {failure_counts}.",
      '', 'Each tensor required median stratum rank correlation >=0.60, median high and low quartile overlap >=0.50, >=80% valid strata and >=80% bidirectional held-half separation (H at least twice L), plus registered accepted-traffic thresholds. Each seed required >=90% eligible tensors and at least 26/32 in every layer/projection. No threshold, batch count, score, or source was changed after measurement.',
      '', 'Rank ties use average ranks. Constant strata are undefined even if coordinate tie breaks produce apparent set overlap. Missing or inadequate accepted exposure is not evidence of unimportance. Exposure associations are descriptive and are reported within seed/layer/projection in exposure_reliability_association.csv. The primary score uses raw batch-averaged gradients without exposure normalization.',
      '', 'Gradient means, second moments, sample batch variances, both half scores, full mean absolute scores, signed scores, absolute mean scores, sign consistency and nonzero counts are retained per coordinate in scores/. Scores are fixed-checkpoint measurements; variance is across complete batches, not tokens or training history. Second moments are not Hessian diagonals.',
      '', 'Matching, signed group Taylor checks, native finite differences, persistent masks, optimizer/reset checks and resumption tests are NOT_RUN because the registered gate stopped before intervention. The numerical artifact distinguishes verified measurement checks from these downstream checks. Empty outcome fields and annotated plots are unavailable results, never zero effects. The pilot was not reached. All 15 registered branch conditions remain visible in per_branch_results.csv.',
      '', 'No source-seed pruning effect or n=5 effect summary exists. Weight/expert/batch statistics do not supply training replications. There is no basis to recommend a full pruning-policy experiment from this study, to claim equivalence, to infer a routing mechanism, or to claim novelty for weight-times-gradient scoring. Any different measurement design would require separately authorized work; no further experiment was run.']
    (D/'findings.md').write_text('\n'.join(lines)+'\n')
    # Verify preservation of source checkpoints, all sealed code/data, and every retained score file.
    frozen();prov=read(D/'checkpoint_and_calibration_provenance.json')
    for s in SEEDS:
        for c in prov['seeds'][str(s)]['checkpoints'].values():assert sha(ROOT/c['path'])==c['sha256']
        for p,h in checks[str(s)]['score_files'].items():assert sha(D/p)==h
    output_files=[p for p in D.rglob('*') if p.is_file() and not any(x in p.parts for x in ('scores','calibration','locks')) and p.name not in ('heartbeat.json','progress.json','completion.json')]
    save(D/'completion.json',{'status':'STOPPED_RELIABILITY_GATE','study_execution_complete':True,'production_experiment_complete':False,
      'measurement_seeds':5,'planned_production_branches':15,'production_continuations':0,'control_reuses':0,
      'intervention_loss_evaluations':0,'masks_created':0,'reliability_passed':False,'coverage':coverage,
      'registration_sha256':sha(D/'registration.json'),'output_sha256':{str(p.relative_to(D)):sha(p) for p in output_files},
      'score_manifest_files':{str(s):str((D/read(D/'status'/f'measure_seed{s}.json')['attempt']/'measurement_checks.json').relative_to(D)) for s in SEEDS},
      'verdict':'This calibration design is inadequate under the frozen gate. Persistent pruning damage was not tested. No full policy recommended.'})
    print('STOPPED_RELIABILITY_GATE',[(c['seed'],c['eligible']) for c in coverage],flush=True)
if __name__=='__main__':main()
