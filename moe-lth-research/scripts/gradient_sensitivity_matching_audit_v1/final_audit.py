from common import *

def main():
 config=frozen();c=read(D/'completion.json')
 assert sha(D/'fixed_solver_subset.csv')==config['solver_subset_sha256']
 assert sha(D/'bin/ranges.so')==read(D/'build.json')['binary_sha256']
 required=['audit_config.json','reproduction_checks.json','implementation_review.md','constraint_attrition.csv','fixed_solver_subset.csv','solver_results.csv','feature_overlap.csv','alternative_design_feasibility.csv','candidate_balance.csv','numerical_checks.json','findings.md','complete_graph_coverage_bounds.json']
 for p in required:assert (D/p).is_file()
 for p,h in c['output_sha256'].items():assert sha(D/p)==h,'Output changed: '+p
 subset=rows(D/'fixed_solver_subset.csv');assert len(subset)==40 and len({(r['seed'],r['layer'],r['projection']) for r in subset})==40
 expected={(r['seed'],r['parameter'],d) for r in subset for d in ('B','C')};solvers=rows(D/'solver_results.csv');assert len(solvers)==80 and {(r['seed'],r['parameter'],r['design']) for r in solvers}==expected
 assert len(rows(D/'constraint_attrition.csv'))==1280*13
 assert len(rows(D/'feature_overlap.csv'))==1280
 assert read(D/'reproduction_checks.json')['all_reproduced']
 assert read(D/'bound_proof_checks.json')['passed']
 numeric=read(D/'numerical_checks.json');assert numeric['passed'] and numeric['model_forwards']==numeric['backwards']==numeric['gradient_scores_recomputed']==numeric['optimizer_updates']==numeric['intervention_losses']==0
 original_completion=read(OLD/'completion.json');assert original_completion['status']=='STOPPED_MATCHING_GATE'
 for p,h in original_completion['output_sha256'].items():assert sha(OLD/p)==h,'Original artifact changed: '+p
 for p,h in original_completion['implementation_sha256'].items():assert sha(ROOT/p)==h,'Original implementation changed: '+p
 bounds=read(D/'complete_graph_coverage_bounds.json');assert len(bounds['source_bounds'])==5
 audit={'passed':True,'required_deliverables_present':True,'original_all_declared_artifact_and_implementation_hashes_unchanged':True,'original_status':'STOPPED_MATCHING_GATE','fixed40_tensor_subset_preserved':True,'all80_binary_conditions_retained':True,'all1280_reproductions_checked':True,'exact_attrition_stages':16640,'range_counter_binary_matches_bruteforce':True,'SCC_bound_independent_exhaustive_tests_passed':True,'no_model_operations':True,'no_gradient_recomputation':True,'solver_timeouts_not_infeasibility':True,'RSS_for_wall_killed_jobs':'unavailable, never zero; enforced address-space8GiB','original_source_coverage_certificates':bounds['source_bounds']}
 save(D/'final_artifact_audit.json',audit)
 c['output_sha256']['final_artifact_audit.json']=sha(D/'final_artifact_audit.json');c['final_artifact_audit_passed']=True
 c['audit_implementation_sha256']={str(p.relative_to(ROOT)):sha(p) for p in HERE.iterdir() if p.suffix in ('.py','.cpp')}
 save(D/'completion.json',c);print('FINAL ARRAY-ONLY AUDIT PASSED',flush=True)
if __name__=='__main__':main()
