"""Report a registered matching stop without manufacturing pruning outcomes."""
from common import *
import csv
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def rows(p):
    with Path(p).open() as f:return list(csv.DictReader(f))


def main():
    frozen();review=read(D/'matching_review.json');assert review['status']=='STOP_MATCHING'
    assert not (D/'completion.json').exists()
    assert sha(HERE/'match.py')==read(D/'matching_implementation_seal_attempt2.json')['implementation_sha256']
    status='NOT_RUN_MATCHING_GATE';summaries=rows(D/'matched_group_summary.csv');balance=rows(D/'magnitude_growth_balance.csv');struct=rows(D/'structural_balance.csv')
    reliability=rows(D/'tensor_reliability_summary.csv')
    # A zero-pair tensor has undefined feature moments, not an omitted successful comparison.
    existing={(r['seed'],r['parameter'],r['feature']) for r in balance}
    for r in balance:r['status']='DEFINED_DIAGNOSTIC_CANDIDATES'
    for r in summaries:
        for feature in ('magnitude','slope','variation'):
            if (r['seed'],r['parameter'],feature) not in existing:
                balance.append({**{k:r[k] for k in ('seed','parameter','layer','expert','projection')},'feature':feature,'H_mean':None,'L_mean':None,'SMD':None,'KS':None,'status':'UNDEFINED_NO_PAIRS'})
    table(D/'magnitude_growth_balance.csv',balance)
    table(D/'first_order_prediction_checks.csv',[dict(seed=s,branch=b,predicted_calibration_change=None,actual_calibration_change=None,validation_change=None,status=status) for s in SEEDS for b in ('H','L')])
    table(D/'per_branch_results.csv',[dict(seed=s,branch=b,status=status,immediate_validation_CE=None,final_validation_CE=None,attempted_updates=0,applied_updates=0,AMP_skips=0,control_reused=False) for s in SEEDS for b in ('C','H','L')])
    table(D/'learning_curves.csv',[dict(seed=s,branch=b,continuation_step=t,validation_CE=None,excess_CE=None,status=status) for s in SEEDS for b in ('C','H','L') for t in (0,25,100,250,500,1000,1500)])
    table(D/'paired_seed_effects.csv',[dict(seed=s,immediate_H_minus_L=None,integrated_H_minus_L=None,final_H_minus_L=None,H_minus_C=None,L_minus_C=None,status=status) for s in SEEDS])
    save(D/'masks'/'manifest.json',{'status':status,'intervention_masks_sealed':False,'candidate_lists_sealed':True,'candidate_sha256':review['candidate_coordinate_file_sha256'],'reason':'Registered matching coverage/dose/balance failed. Candidates are diagnostics, never applied.'})
    measurement={}
    keys=['weight_hash_preserved','RNG_preserved','module_modes_preserved','existing_gradients_preserved','TF32_flags_preserved','calibration_batch_token_target_hashes_verified','coordinate_schema_verified','no_intervention_loss_or_validation_or_training','synthetic_checks']
    score_manifests={}
    for s in SEEDS:
        p=D/read(D/'status'/f'measure_seed{s}.json')['attempt']/'measurement_checks.json';c=read(p)
        measurement[str(s)]={k:c[k] for k in keys};score_manifests[str(s)]=str(p.relative_to(D))
        for path,h in c['score_files'].items():assert sha(D/path)==h
    for path,h in review['candidate_coordinate_file_sha256'].items():assert sha(D/path)==h
    # Reconstruct exact row/column counts from every candidate list, including exclusions.
    identity_checks={};prov=read(D/'checkpoint_and_calibration_provenance.json')
    for s in SEEDS:
        for cp in prov['seeds'][str(s)]['checkpoints'].values():assert sha(ROOT/cp['path'])==cp['sha256']
        cand=load(D/'candidate_coordinate_lists'/f'seed{s}.pt');degrees=load(D/'degree_distributions'/f'seed{s}.pt');schema=prov['seeds'][str(s)]['checkpoints']['1000']['expert_schema']
        count=0
        for n,h in cand['H'].items():
            l=cand['L'][n];shape=schema[n][0];assert h.numel()==l.numel();assert h.unique().numel()==h.numel() and l.unique().numel()==l.numel()
            assert not np.intersect1d(h.numpy(),l.numpy()).size
            for branch,idx in [('H',h),('L',l)]:
                assert torch.all((idx>=0)&(idx<math.prod(shape)))
                r=torch.bincount(idx//shape[1],minlength=shape[0]);c=torch.bincount(idx%shape[1],minlength=shape[1])
                assert torch.equal(r,degrees[n]['rows_'+branch]);assert torch.equal(c,degrees[n]['columns_'+branch])
            assert torch.equal(degrees[n]['rows_H'],degrees[n]['rows_L']);assert torch.equal(degrees[n]['columns_H'],degrees[n]['columns_L']);count+=h.numel()
        identity_checks[str(s)]={'all_candidate_tensors':len(cand['H']),'candidate_pairs':count,'unique_and_disjoint':True,'exact_rows_and_columns':True,'degree_arrays_reconstructed':True}
    save(D/'numerical_checks.json',{'stage':'MEASUREMENT_AND_MATCHING_ONLY','measurement':measurement,'matching_coordinate_and_degree_checks':identity_checks,
      'matching_algorithm_fixture_checks':read(D/'matching_algorithm_fixture_checks.json'),
      'native_configuration_audit':read(D/'native_configuration_audit.json'),
      'score_algebra_audit':read(D/'score_algebra_audit.json') if (D/'score_algebra_audit.json').exists() else 'PENDING_SEPARATE_AUDIT',
      'Jensen_comparison_tolerance':{'relative':1e-12,'absolute':1e-25,'amendment':'engineering_code_amendment.json'},
      'registered_matching_checks_executed':True,'all_registered_matching_checks_passed':False,
      'downstream_checks':{k:status for k in ('native_finite_differences','route_boundary_diagnostics','initial_surviving_equality','fresh_optimizer_scaler_RNG','persistent_zero_gradient_flow','continuation_batch_hashes_runtime','interruption_resumption','control_reuse_compatibility')},
      'no_intervention_or_validation_or_production':True,'engineering_failures_preserved':['engineering_history.json','matching_transport_amendment.json'],
      'downstream_reason':'Registered matching failed before any intervention; unavailable checks are not marked passed.'})
    figdir=D/'figures';figdir.mkdir(exist_ok=True)
    fig,axes=plt.subplots(1,2,figsize=(12,4.5))
    for i,s in enumerate(SEEDS):
        rs=[r for r in reliability if int(r['seed'])==s and r['median_stratum_rho']!=''];vals=[float(r['median_stratum_rho']) for r in rs]
        axes[0].scatter(np.full(len(vals),i)+np.random.default_rng(s).uniform(-.15,.15,len(vals)),vals,s=5,alpha=.3)
        axes[1].scatter([float(r['accepted_total']) for r in rs],vals,s=5,alpha=.25,label=str(s))
    axes[0].axhline(.6,ls='--',color='black');axes[0].set(xticks=range(5),xticklabels=SEEDS,xlabel='Source seed',ylabel='Median stratum Spearman',title='Independent 16-batch score halves')
    axes[1].set(xlabel='Accepted calibration tokens / expert',ylabel='Median stratum Spearman',title='Exposure and sensitivity reliability');axes[1].legend(title='Seed')
    fig.tight_layout();fig.savefig(figdir/'sensitivity_reliability.png',dpi=180);fig.savefig(figdir/'sensitivity_reliability.pdf');plt.close(fig)
    fig,axes=plt.subplots(1,4,figsize=(16,4))
    for j,feature in enumerate(('magnitude','slope','variation')):
        for i,s in enumerate(SEEDS):
            vals=[float(r['SMD']) for r in balance if int(r['seed'])==s and r['feature']==feature and r['SMD'] not in ('',None)]
            axes[j].scatter(np.full(len(vals),i)+np.random.default_rng(s).uniform(-.15,.15,len(vals)),vals,s=5,alpha=.3)
        axes[j].axhline(.05,ls='--',color='black');axes[j].axhline(-.05,ls='--',color='black');axes[j].set(title=feature+' balance',ylabel='H minus L SMD',xticks=range(5),xticklabels=SEEDS,xlabel='Source seed')
    for i,s in enumerate(SEEDS):
        rs=[r for r in struct if int(r['seed'])==s];axes[3].scatter([i]*len(rs),[int(r['max_column_count_difference']) for r in rs],s=6)
    axes[3].set(title='Exact structural balance',ylabel='Maximum column count difference',xticks=range(5),xticklabels=SEEDS,xlabel='Source seed')
    fig.suptitle('Registered matching candidates (including rejected tensors); no masks applied');fig.tight_layout();fig.savefig(figdir/'magnitude_growth_structural_balance.png',dpi=180);plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(11,4))
    for i,s in enumerate(SEEDS):
        rs=[r for r in summaries if int(r['seed'])==s]
        axes[0].scatter(np.full(len(rs),i)+np.random.default_rng(s).uniform(-.2,.2,len(rs)),[int(r['cycle_pairs']) for r in rs],s=6,alpha=.4)
        c=next(x for x in review['coverage'] if x['seed']==s);axes[1].bar(i,c['overall_dose']*100)
    axes[0].axhline(2621,ls='--',color='black',label='target');axes[0].axhline(656,ls=':',color='red',label='minimum per included tensor');axes[0].legend(fontsize=8)
    axes[0].set(ylabel='Pairs per tensor',title='Feasible directed-cycle pairs')
    axes[1].axhline(.5,ls='--',color='red',label='minimum whole-seed dose');axes[1].legend(fontsize=8);axes[1].set(ylabel='Percent of all expert coordinates',title='Retained dose after all balance checks')
    for ax in axes:ax.set(xticks=range(5),xticklabels=SEEDS,xlabel='Source seed')
    fig.tight_layout();fig.savefig(figdir/'matching_feasibility.png',dpi=180);plt.close(fig)
    for filename,title,labels in [('immediate_final_paired_effects','No pruning effects measured',['Immediate H minus L','Final H minus L']),('continuation_CE_excess_loss','No continuations run',['Validation CE','Excess CE relative to control'])]:
        fig,axes=plt.subplots(1,2,figsize=(11,3.5))
        for ax,label in zip(axes,labels):
            ax.set(title=label,xticks=range(5),xticklabels=SEEDS,xlabel='Source seed',yticks=[],xlim=(-.5,4.5),ylim=(0,1))
            for i in range(5):ax.text(i,.5,'NOT RUN',ha='center',va='center',fontsize=8)
        fig.suptitle(title+' — registered matching gate failed');fig.tight_layout();fig.savefig(figdir/(filename+'.png'),dpi=160);plt.close(fig)
    lines=['# Gradient sensitivity pruning: stopped at matching gate','',
      'Sensitivity passed the registered split-half reliability gate. The registered matching algorithm failed to obtain acceptable feature balance, sensitivity contrast, structural balance and retained dose/coverage jointly in all five sources. No selected coordinates were applied to a model, no intervention or validation losses were measured, no controls were reused, and no production training was launched.','',
      '| Source seed | Reliability eligible / 256 | Median stratum rank agreement | Matching included / 256 | Retained dose per branch | Matching passed |','|---|---:|---:|---:|---:|---|']
    reliability_review=read(D/'reliability_review.json')
    for c in review['coverage']:
        r=next(x for x in reliability_review['coverage'] if x['seed']==c['seed'])
        rho=float(np.median([float(x['median_stratum_rho']) for x in reliability if int(x['seed'])==c['seed'] and x['median_stratum_rho']!='']))
        lines.append(f"| {c['seed']} | {r['eligible']} / 256 | {rho:.4f} | {c['included_tensors']} / 256 | {100*c['overall_dose']:.4f}% | {c['passed']} |")
    lines += ['', '| Source seed | Median row-matched candidate pairs | Median column-cycle pairs | Column-cycle pair range |', '|---|---:|---:|---:|']
    for seed in SEEDS:
        rs=[r for r in summaries if int(r['seed'])==seed];raw=[int(r['candidate_pairs']) for r in rs];cyc=[int(r['cycle_pairs']) for r in rs]
        lines.append(f"| {seed} | {float(np.median(raw)):.1f} | {float(np.median(cyc)):.1f} | {min(cyc)}–{max(cyc)} |")
    reasons={reason:sum(r['excluded_reason']==reason for r in summaries) for reason in ('insufficient_cycles','registered_balance_failure')}
    lines+=['',f'Excluded tensor diagnostics across sources: {reasons}. These are descriptive tensor counts, not independent training replications.',
      '', 'Matching used the frozen bottom-half/50-stratum population, exact paired sign and row identity, fixed magnitude/growth/variation calipers, then directed column cycles. Every reported candidate has exact equal H/L counts in each row and column. This controls degree distributions, not exact connectivity. Candidate coordinate lists, degree arrays and hashes are retained for excluded as well as included tensors. No alternative matcher, dose sweep, relaxed caliper, batch expansion or score change was attempted.',
      '', 'A tensor required at least 656 pairs, absolute SMD <=0.05 and KS <=0.05 for magnitude, recent slope and absolute variation, removed squared-magnitude relative difference <=0.005, and mean high sensitivity at least twice low. Every source required >=80% tensors, >=26/32 per layer/projection and >=0.5% of all expert coordinates removed in each branch. A stop demonstrates a limitation of this registered matching algorithm/design, not impossibility of every conceivable matching procedure.',
      '', 'Reliability statistics, accepted/proposed/dropped traffic and gate summaries are retained. Scores are batchwise mean absolute weight-times-task-gradients in FP32 native routing, accumulated in FP64, with signed means, gradient second moments, sample batch variances and sign consistency. Missing traffic was never equated with unimportance. Second moments are not Hessian diagonals. Descriptive exposure/reliability associations do not establish a routing mechanism.',
      '', 'The native configuration audit independently verified model/data/routing/training identity across all historical checkpoints and the sealed native reference, along with original training/continuation/validation hashes and separate train/validation text files. This is not a control-reuse compatibility claim. The numerical checks verify the measurement preservation contract, tiny native full-versus-streaming gradients, cancellation distinction, coordinate identity, candidate uniqueness/disjointness and reconstructed degree equality. Two first calibration attempts failed an overly exact floating-point Jensen audit; they are preserved. The documented engineering correction only changed comparison tolerance; scientific definitions and gates stayed fixed. Pre-scoring fixture corrections and original sealed code snapshots are also preserved. The first matching attempt hit the process file-descriptor limit while transferring Torch degree tensors; the preserved engineering correction changes only interprocess transport to NumPy arrays. SciPy emitted an exact-p-value fallback warning for some KS calls; only the unchanged empirical KS statistic is used for balance, with no p-values or altered thresholds.',
      '', 'Signed group Taylor checks, individual native finite differences and route-boundary diagnostics, fresh-reset/persistent-mask checks, the seed17 production pilot and interruption/resumption tests are NOT_RUN after the registered matching stop. They are not marked passed. All 15 planned branch conditions and all seed effects remain visible with unavailable outcomes. Annotated outcome figures show NOT RUN, not zero effects.',
      '', 'The final research question remains unanswered: reproducible sensitivity alone does not establish information beyond magnitude/growth/structure about persistent pruning damage. There are zero source-seed pruning replications and no meaningful mean/SD/SE of pruning effects. No equivalence, novelty, causality or general-policy claim follows. A full pruning-policy test is not recommended on this evidence and was not implemented.']
    (D/'findings.md').write_text('\n'.join(lines)+'\n')
    frozen()
    files=[p for p in D.rglob('*') if p.is_file() and not any(x in p.parts for x in ('scores','calibration','locks')) and p.name not in ('heartbeat.json','progress.json','completion.json')]
    save(D/'completion.json',{'status':'STOPPED_MATCHING_GATE','study_execution_complete':True,'production_experiment_complete':False,'measurement_seeds':5,
      'planned_production_branches':15,'production_continuations':0,'control_reuses':0,'intervention_loss_evaluations':0,
      'reliability_passed':True,'matching_passed':False,'coverage':review['coverage'],'registration_sha256':sha(D/'registration.json'),
      'output_sha256':{str(p.relative_to(D)):sha(p) for p in files},'score_manifest_files':score_manifests,
      'verdict':'Reliable sensitivity was measured, but the registered matched contrast was infeasible. Persistent pruning damage was not tested. No full policy recommended.'})
    print('STOPPED_MATCHING_GATE',flush=True)
if __name__=='__main__':main()
