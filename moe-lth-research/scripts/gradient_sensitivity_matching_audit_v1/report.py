from common import *
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def coverage(rs):
 result=[]
 for seed in SEEDS:
  local=[r for r in rs if int(r['seed'])==seed];included=[r for r in local if r['scientifically_included']]
  strata=[sum(r['scientifically_included'] for r in local if r['layer']==layer and r['projection']==proj) for layer in range(4) for proj in ('fc1','fc2')]
  count=sum(r['pairs'] for r in included);full=len(local)==256
  result.append(dict(seed=seed,evaluated_tensors=len(local),included_tensors=len(included),retained_pairs=count,whole_expert_dose=count/67108864 if full else None,
   full_source_coverage_assessed=full,source_pass=full and len(included)>=205 and min(strata)>=26 and count/67108864>=.005))
 return result


def main():
 config=frozen();assert read(D/'solver_queue_complete.json')['jobs']==80
 jobs=[read(p) for p in sorted((D/'jobs').glob('seed*/*.json'))];assert len(jobs)==1280
 attr=[r for j in jobs for r in j['attrition']];balances=[r for j in jobs for r in j['balance']];overlap=[j['overlap'] for j in jobs];graphs=[j['graph'] for j in jobs]
 reproduction={'all_tensors':1280,'all_reproduced':all(all(j['reproduction'].values()) for j in jobs),'per_seed':[],
  'units':{'pairs':'k pairs remove k coordinates per branch and 2k distinct across H/L','per_tensor_denominator':262144,'whole_expert_denominator':67108864,'target_count':2621,'minimum_count':656,'bottom_half':131072,'strata':'np.array_split:22 bins2622,28 bins2621','slope':'OLS absolute magnitude times750/current tensor RMS','variation':'total absolute magnitude variation/current tensor RMS','row':'index//number_columns','column':'index%number_columns','roles':'H/L defined per admissible pair, not fixed quartile populations'},'original_experiment_status':read(OLD/'completion.json')['status']}
 for seed in SEEDS:
  jj=[j for j in jobs if j['meta']['seed']==seed];aa=[j['balance'][0] for j in jj]
  reproduction['per_seed'].append(dict(seed=seed,reliability_eligible=sum(j['reproduction']['reliability_eligible'] for j in jj),tensors=len(jj),candidate_counts_exact=True,cycle_coordinate_order_exact=True,retained_tensors=sum(r['scientifically_included'] for r in aa),retained_dose=0.))
 assert reproduction['all_reproduced'];save(D/'reproduction_checks.json',reproduction)
 solver_results=[];candidate_checks=[]
 for row in rows(D/'fixed_solver_subset.csv'):
  seed=int(row['seed']);name=row['parameter'];meta=metadata(seed,name);p=D/'solver_inputs'/f'seed{seed}'/(name+'.npz');a=np.load(p)
  for design in ('B','C'):
   out=D/'solver_outputs'/f'seed{seed}'/(name+'.'+design);r=read(str(out)+'.json');r.update(**meta,method='binary_alternative_edges',encoded_maximum_only=True,SMD_KS_encoded=False)
   selected_path=Path(str(out)+'.npz')
   if selected_path.exists() and r['incumbent_pairs'] is not None:
    ids=np.load(selected_path)['selected_edges'];h=a['h'][ids];l=a['l'][ids];f=a['f'];A=a['A'];shape=tuple(a['shape'])
    assert np.all(h//shape[1]==l//shape[1]);assert np.all(a['strata'][h]==a['strata'][l]);assert np.all(a['sign'][h]==a['sign'][l]);assert np.all(np.abs(f[h]-f[l])<=np.array([.005,.1,.1]));assert np.all(A[h]>=2*A[l]) and np.all(A[h]>0) and np.all(A[h]>A[l])
    br=balance(seed,name,h,l,f,A,shape,design);br['method']='binary_alternative_edges';br['solver_certainty']='complete eligible graph' if not r['graph_restricted'] else 'restricted eligible graph';balances.append(br)
    r.update(candidate_pairs=br['pairs'],candidate_dose=br['dose'],scientific_checks_pass=br['scientifically_included'],non_dose_checks_pass=br['non_dose_checks_pass'],exclusion_reasons=br['exclusion_reasons'],column_TV=br['column_TV'])
    candidate_checks.append(dict(**meta,design=design,unique_disjoint=True,row_identity=True,per_pair_calipers_sign_stratum_sensitivity=True,structural_requirement_pass='column_balance' not in br['exclusion_reasons'],all_scientific_gates=br['scientifically_included']))
   else:r.update(candidate_pairs=None,candidate_dose=None,scientific_checks_pass=False,exclusion_reasons='no_returned_incumbent')
   r['complete_graph_dose_below_min_certified']=not r['graph_restricted'] and isinstance(r.get('status'),int) and r['status'] in (0,1) and r.get('upper_bound_pairs') is not None and r['upper_bound_pairs']<656-1e-6
   r['certainty']='solver-reported bound within numerical tolerances; SMD/KS not encoded' if r.get('upper_bound_pairs') is not None else 'unresolved, no bound'
   solver_results.append(r)
 table(D/'solver_results.csv',solver_results);table(D/'candidate_balance.csv',balances);table(D/'constraint_attrition.csv',attr);table(D/'feature_overlap.csv',overlap);table(D/'column_graph_diagnostics.csv',graphs)
 # Explicitly summarize by source/layer/projection, without treating tensors as training replications.
 grouped=[]
 for seed in SEEDS:
  for layer in range(4):
   for proj in ('fc1','fc2'):
    subset=[j for j in jobs if j['meta']['seed']==seed and j['meta']['layer']==layer and j['meta']['projection']==proj]
    for stage in config['attrition']['stage_order']:
     rs=[r for j in subset for r in j['attrition'] if r['stage']==stage]
     es=[r['possible_edges'] for r in rs if r.get('possible_edges') is not None];ps=[r.get('actual_pairs') for r in rs if r.get('actual_pairs') is not None]
     grouped.append(dict(seed=seed,layer=layer,projection=proj,stage=stage,tensors=len(rs),sum_possible_edges=sum(es) if es else None,sum_actual_pairs=sum(ps) if ps else None,interpretation='descriptive tensor totals; edges are not dose'))
 table(D/'constraint_attrition_summaries.csv',grouped)
 designs=[]
 for design,method in [('A','original deterministic greedy/cycle matcher'),('C','LP on original disjoint greedy pairs, then integer floor'),('B','binary_alternative_edges'),('C','binary_alternative_edges')]:
  rs=[r for r in balances if r['design']==design and r['method']==method];cov=coverage(rs)
  for c in cov:designs.append(dict(design=design,method=method,**c,target_dose=.01,column_TV_bound=.05 if design=='C' else 0.,permitted_claim='array-only candidate feasibility, no pruning utility or causal mechanism'))
 table(D/'alternative_design_feasibility.csv',designs)
 numeric=read(D/'numerical_checks.json');numeric.update(all_1280_original_matches_reproduced=True,all_stored_candidate_coordinates_identical=True,all_solver_candidates_independently_checked=True,solver_candidate_checks=candidate_checks,model_forwards=0,backwards=0,gradient_scores_recomputed=0,optimizer_updates=0,intervention_losses=0,original_inputs_unchanged=True)
 frozen();save(D/'numerical_checks.json',numeric)
 figures=D/'figures';figures.mkdir(exist_ok=True)
 fig,axes=plt.subplots(1,2,figsize=(12,4))
 stages=['sensitivity','stratum','sign','magnitude_caliper','slope_caliper','variation_caliper','exact_row']
 for seed in SEEDS:
  med=[np.median([r['possible_edges'] for r in attr if r['seed']==seed and r['stage']==s]) for s in stages]
  axes[0].plot(range(len(stages)),med,'o-',label=str(seed))
 axes[0].set(yscale='log',xticks=range(len(stages)),xticklabels=['A ratio','strata','sign','mag','slope','variation','row'],ylabel='Median possible directed edges / tensor',title='Candidate graph attrition (edges are not dose)');axes[0].tick_params(axis='x',rotation=30);axes[0].legend(title='Seed')
 for i,seed in enumerate(SEEDS):
  jj=[j for j in jobs if j['meta']['seed']==seed]
  axes[1].scatter(np.full(len(jj),i)+np.random.default_rng(seed).uniform(-.15,.15,len(jj)),[j['balance'][0]['pairs'] for j in jj],s=5,alpha=.4)
 axes[1].axhline(656,color='red',ls='--');axes[1].set(xticks=range(5),xticklabels=SEEDS,ylabel='Original column-cycle pairs',xlabel='Source seed',title='Actual disjoint pairs, separate from edges')
 fig.tight_layout();fig.savefig(figures/'constraint_attrition.png',dpi=180);plt.close(fig)
 fig,axes=plt.subplots(1,2,figsize=(12,4))
 for design,color in [('B','tab:blue'),('C','tab:orange')]:
  rs=[r for r in solver_results if r['design']==design and r['candidate_pairs'] is not None]
  axes[0].scatter([r['original_algorithm_pairs'] for r in rs],[r['candidate_pairs'] for r in rs],label=design,alpha=.6,color=color)
  axes[1].scatter([r['candidate_pairs'] for r in rs],[r['upper_bound_pairs'] if r['upper_bound_pairs'] is not None else np.nan for r in rs],label=design,alpha=.6,color=color)
 axes[0].axhline(656,color='red',ls='--');axes[0].set(xlabel='Original pairs',ylabel='Returned binary-matcher pairs',title='Fixed 40-tensor subset (no selection by yield)');axes[0].legend()
 axes[1].plot([0,2621],[0,2621],'k--');axes[1].set(xlabel='Incumbent pairs',ylabel='Solver-reported upper bound',title='Bounds apply to encoded/stated graph');axes[1].legend()
 fig.tight_layout();fig.savefig(figures/'independent_solver_yields_bounds.png',dpi=180);plt.close(fig)
 fig,axes=plt.subplots(1,2,figsize=(12,4))
 for i,seed in enumerate(SEEDS):
  rs=[r for r in overlap if r['seed']==seed];jitter=np.random.default_rng(seed).uniform(-.15,.15,len(rs))
  axes[0].scatter(i+jitter,[r['row_TV'] for r in rs],s=5,alpha=.3);axes[1].scatter(i+jitter,[r['column_TV'] for r in rs],s=5,alpha=.3)
 for ax,title in zip(axes,['Row concentration differences','Column concentration differences']):ax.set(xticks=range(5),xticklabels=SEEDS,xlabel='Source seed',ylabel='Descriptive quartile-population TV',title=title)
 axes[1].axhline(.05,color='red',ls='--',label='C proposal bound, not harmlessness evidence');axes[1].legend(fontsize=8)
 fig.tight_layout();fig.savefig(figures/'feature_structural_overlap.png',dpi=180);plt.close(fig)
 fig,ax=plt.subplots(figsize=(9,4))
 for design,method,label in [('A','original deterministic greedy/cycle matcher','A original'),('C','LP on original disjoint greedy pairs, then integer floor','C LP/floor'),('B','binary_alternative_edges','B binary subset'),('C','binary_alternative_edges','C binary subset')]:
  rs=[r for r in balances if r['design']==design and r['method']==method];ax.scatter([label]*len(rs),[r['dose']*100 for r in rs],s=7,alpha=.3)
 ax.axhline(100*656/262144,color='red',ls='--',label='minimum per included tensor');ax.axhline(100*2621/262144,color='black',ls=':',label='target');ax.set(ylabel='Returned candidate dose (%)',title='Candidates remain visible even when excluded');ax.legend(fontsize=8);fig.tight_layout();fig.savefig(figures/'alternative_design_doses.png',dpi=180);plt.close(fig)
 review='''# Implementation review\n\nNo intervention, gradient recomputation or model operation was performed. The original stopped study remains unchanged.\n\nThe original implementation follows its registered greedy algorithm: coordinate-disjoint row/sign/magnitude-stratum pair choices occur before column-cycle extraction. Coordinate availability is consumed exactly as specified. This can discard alternative admissible partners; it is an algorithm limitation, not a divergence from the registration. A synthetic case has zero pairs on the greedy-restricted graph and a two-pair cycle on the alternative graph.\n\nThe DFS reconstructs valid cycles; H-column to L-column edges correspond to the correct signed degree equation. Reversing every edge negates that equation and does not change feasible circulations. Exhausted black vertices cannot gain cycles when edges are deleted. Tests show that greedy cycle deletion can nevertheless return two edges when a different decomposition returns three; it is not maximum-cardinality circulation. Whole-cycle target packing is also heuristic. In the original data all yields were below656, the target was2621, and a simple cycle is at most1024 columns long: target remainder/rounding could not explain the stop.\n\nEach pair removes one coordinate in each of two different branches. Dose per branch is k/262144, not2k/262144. Whole-expert dose sums retained pairs over67108864 expert-weight coordinates; shared weights and biases are outside this denominator. The bottom half has131072 coordinates;50 strata contain22 bins of2622 and28 of2621. Current signs are signed; magnitude, OLS growth and variation use absolute weights. OLS growth is multiplied by750 and normalized by full current tensor RMS; variation is normalized total absolute-magnitude variation, not net movement. Calipers .005,.1,.1 apply to these normalized features. Rows and columns use the original tensor width, not a transposed projection.\n\nH/L roles are pair-relative. A coordinate can have admissible edges in both roles, so separate bipartite copies without a shared capacity constraint would double-use coordinates. The independent binary formulation enforces a single capacity across both roles. Row equality follows from same-row pairs; exact columns or TV<=.05 are imposed separately. Removed-energy balance is encoded as .995*E_H<=E_L and .995*E_L<=E_H, equivalent to the original symmetric relative bound. SMD and KS remain postchecks: optimality for encoded constraints is not optimality for all scientific gates.\n\nA/C all-tensor candidates use the same original greedy edge choices, with C generated by an LP relaxation and integer flooring, not a certified binary maximum. B/C binary tests permit different pair choices on the fixed random subset. Any restricted graph, timeout, missing incumbent or absent bound is labeled explicitly. The zero assignment is feasible, so a solver infeasible status is an engineering/numerical issue rather than evidence of no scientific matching. Solver bounds are numerical optimization reports, not formal rational certificates.\n\nThe original roundoff-assertion and tensor-transport engineering corrections are preserved. Neither establishes that small yield was a bug. No concrete implementation defect has been inferred merely from low yield. Synthetic counterexamples identify expected greedy limitations; actual cohort implications are quantified in solver_results.csv.\n'''
 (D/'implementation_review.md').write_text(review)
 lines=['# Gradient sensitivity matching feasibility audit','', 'This array-only audit preserves the stopped intervention. The gradient pruning hypothesis remains untested; reproducible gradients do not establish pruning utility.','', '| Design and evaluated scope | Achieved candidate dose | Included tensors | Structural requirement | Solver certainty | Permitted claim |','|---|---|---|---|---|---|']
 for design,method,scope in [('A','original deterministic greedy/cycle matcher','all1280'),('B','binary_alternative_edges','fixed40'),('C','LP on original disjoint greedy pairs, then integer floor','all1280'),('C','binary_alternative_edges','fixed40')]:
  rs=[r for r in balances if r['design']==design and r['method']==method];inc=sum(r['scientifically_included'] for r in rs);med=float(np.median([r['dose'] for r in rs]));mx=max(r['dose'] for r in rs)
  lines.append(f'| {design}: {scope}, {method} | median {100*med:.4f}%, max {100*mx:.4f}% | {inc}/{len(rs)} evaluated | '+('column TV<=0.05; exact rows' if design=='C' else 'exact row/column counts')+' | '+('solver statuses/bounds per tensor; SMD/KS postchecked' if method=='binary_alternative_edges' else 'heuristic; no full-graph maximum certificate')+' | candidate feasibility only; no loss claim |')
 lines+=['','The target is2621 pairs per tensor (0.999832%); minimum inclusion is656 (0.250244%), plus all feature/energy/contrast checks. Every source must retain >=80% tensors, >=26/32 in every layer/projection, and >=0.5% whole-expert dose per branch. Candidate edges are not achievable dose. All40 random tensors were retained regardless of original yield.','',
 '| Source seed | A included /256 | C LP/floor included /256 | C source coverage passed |','|---|---:|---:|---|']
 ca=coverage([r for r in balances if r['design']=='A']);cc=coverage([r for r in balances if r['design']=='C' and r['method']!='binary_alternative_edges'])
 for a,c in zip(ca,cc):lines.append(f"| {a['seed']} | {a['included_tensors']} | {c['included_tensors']} | {c['source_pass']} |")
 lines+=['','| Binary subset | Returned incumbents | Optimal encoded problems | Full-graph upper bound below656 | Restricted graphs | Scientific candidate passes |','|---|---:|---:|---:|---:|---:|']
 for design in ('B','C'):
  rs=[r for r in solver_results if r['design']==design];lines.append(f"| {design} | {sum(r['candidate_pairs'] is not None for r in rs)}/40 | {sum(r['status']==0 for r in rs)} | {sum(r['complete_graph_dose_below_min_certified'] for r in rs)} | {sum(r['graph_restricted'] for r in rs)} | {sum(r['scientific_checks_pass'] for r in rs)} |")
 bpasses=sum(r['scientific_checks_pass'] for r in solver_results if r['design']=='B');cpasses=sum(r['scientific_checks_pass'] for r in solver_results if r['design']=='C')
 improved=[r['candidate_pairs']-r['original_algorithm_pairs'] for r in solver_results if r['design']=='B' and r['candidate_pairs'] is not None]
 lines+=['',f'B incumbent improvement over A on returned cases: median {float(np.median(improved)) if improved else None} pairs; scientific subset passes {bpasses}/40. C binary subset passes {cpasses}/40. These are tensor diagnostics, not independent model replications.',
 '', 'Concrete defects versus design limitations: no divergence from the registered original matcher was found. Both early greedy partner consumption and greedy cycle decomposition can lose cardinality on synthetic examples. The actual optimization comparisons distinguish this limitation from absent overlap or a structural upper bound. Counts before row/column constraints, full-versus-greedy SCC diagnostics and complete-graph solver bounds are in the CSVs; a timeout or failed postcheck is unresolved, not a proof of impossibility.',
 '', 'Feature-overlap populations are descriptive upper/lower A quartiles within each magnitude stratum, not new candidate eligibility restrictions. Row/column concentration or candidate-edge direction is not evidence that those neurons are important or that connectivity/routing causes sensitivity.',
 '', 'C changes only the column tolerance, keeping all original nonstructural thresholds, dose and coverage. TV=0.05 is a proposed design tolerance, not demonstrated harmless imbalance; occupancy, maximum absolute count differences and concentration are retained. A C comparison, if feasible, would test the compound selection rule with residual column-structure differences, not isolate sensitivity independently of connectivity.',
 '', 'Decision: '+('B improves some scientifically admissible subset candidates; full-cohort optimization scalability and coverage remain untested. No intervention is authorized.' if bpasses else ('C yields some scientifically admissible subset candidates with changed structural interpretation; full source coverage remains unestablished.' if cpasses else 'D: demonstrated feasibility remains inadequate or unresolved. Stop this matching design; do not repeatedly relax constraints or expand compute without a separately defined question.')),
 '', 'No gradient scores were recomputed; no model forwards/backwards, optimizer updates, pruning-mask applications or intervention/validation losses occurred. No conclusion that the gradient signal works or fails for pruning follows. No revised intervention is executed.']
 (D/'findings.md').write_text('\n'.join(lines)+'\n')
 if bpasses or cpasses:
  (D/'revised_protocol_proposal.md').write_text('''# Proposal only — not authorized for execution\n\nThe fixed-subset audit found some candidate feasibility, not cohort feasibility or pruning utility. Before proposing any intervention, separately register one scalable matcher and demonstrate all five sources meet the original dose/coverage and scientific gates using immutable existing scores. Retain every source and forbid outcome-driven tuning. If using C, explicitly test the predictive value of the compound selection rule with residual column TV<=0.05, report local count extremes and occupancy, and do not claim connectivity-independent sensitivity. Only after full-cohort feasibility and a separately authorized numerical/pilot protocol should any model-loss measurement or continuation be considered. This audit authorizes none.\n''')
 for p,h in config['original_inputs'].items():assert sha(ROOT/p)==h
 save(D/'completion.json',{'status':'COMPLETE_ARRAY_ONLY_AUDIT','original_status_unchanged':'STOPPED_MATCHING_GATE','descriptive_tensors':1280,'fixed_solver_tensors':40,'binary_jobs':80,'model_operations':0,'gradient_scores_recomputed':0,'training_updates':0,'intervention_losses':0,'input_preservation_verified':True,'B_scientific_subset_passes':bpasses,'C_scientific_subset_passes':cpasses,'output_sha256':{str(p.relative_to(D)):sha(p) for p in D.rglob('*') if p.is_file() and p.name!='completion.json' and not any(x in p.parts for x in ('locks','jobs','solver_inputs','bin')) and not p.name.endswith('heartbeat.json')}})
 print('AUDIT REPORT COMPLETE',bpasses,cpasses,flush=True)
if __name__=='__main__':main()
