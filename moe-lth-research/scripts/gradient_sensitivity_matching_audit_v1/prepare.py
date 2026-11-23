from common import *

def main():
 assert not (D/'audit_config.json').exists();D.mkdir(parents=True,exist_ok=True)
 g.frozen();c=read(OLD/'completion.json')
 for p,h in c['output_sha256'].items():assert sha(OLD/p)==h,p
 for p,h in c['implementation_sha256'].items():assert sha(ROOT/p)==h,p
 subset=[];rng=np.random.default_rng(4041917)
 for seed in SEEDS:
  for layer in range(4):
   for proj in ('fc1','fc2'):
    expert=int(rng.integers(0,32));name=f'blocks.{layer}.moe.experts.{expert}.{proj}.weight'
    subset.append(dict(seed=seed,layer=layer,projection=proj,expert=expert,parameter=name))
 table(D/'fixed_solver_subset.csv',subset)
 paths=[OLD/'registration.json',OLD/'completion.json',OLD/'matching_review.json',OLD/'matched_group_summary.csv',OLD/'tensor_reliability_summary.csv',OLD/'checkpoint_and_calibration_provenance.json',OLD/'numerical_checks.json',OLD/'engineering_code_amendment.json',OLD/'matching_transport_amendment.json',ROOT/'scripts/gradient_sensitivity_pruning_v1/match.py',ROOT/'results/magnitude_growth_pruning_v1/findings.md']
 config={'audit':'gradient_sensitivity_matching_audit_v1','seeds':list(SEEDS),'prohibited':['model forwards','backwards','gradient recomputation','mask application to models','optimizer updates','intervention or validation losses'],
 'original_inputs':{str(p.relative_to(ROOT)):sha(p) for p in paths},'solver_subset_sha256':sha(D/'fixed_solver_subset.csv'),'subset_rng':'numpy.default_rng(4041917), sequential seed/layer/fc1-fc2 draws, one expert per cell, all 40 retained',
 'designs':{'A':'Exact reproduction of original row-greedy and column-cycle matcher, all 1280 tensors',
 'B':'Binary maximum-cardinality admissible edge selection on fixed 40 tensors; coordinate capacity one in either role, exact column equality, exact per-pair row/sign/stratum/calipers/A_H>=2*A_L, target cap2621; exact linear energy bound; evaluate SMD/KS afterward',
 'C':'Same nonstructural rules and dose/coverage, exact rows; only column equality replaced by TV<=0.05. All tensors: LP on original coordinate-disjoint greedy edges plus conservative integer flooring; evaluate all gates without repair. Fixed subset: binary solver on alternative admissible edges as in B with TV bound. No additional relaxation'},
 'column_TV':'sum(abs(column_H-column_L))/(2*pairs)','column_tolerance':.05,'column_tolerance_interpretation':'proposed design tolerance, not demonstrated harmless structural imbalance',
 'solver':{'binary_time_limit_seconds':30,'external_wall_limit_seconds':45,'address_space_limit_GiB':8,'node_limit':20000,'threads':1,'mip_relative_gap':0.,'max_admissible_graph_edges':300000,'graph_restriction':'If complete eligible edges exceed300000 retain all original greedy pairs then lowest(feature_cost,H,L) remaining up to cap; restricted infeasibility does not certify complete graph infeasibility',
 'C_all_tensor_LP_time_limit_seconds':3,'C_all_tensor_LP_memory':'bounded by worker address space24GiB; <=65536 original greedy edges',
 'maximum_binary_jobs':80,'maximum_parallel_binary_jobs':2,'maximum_descriptive_workers':4},
 'attrition':{'edge_counts':'exact directed admissible pair counts from order statistics/orthogonal range counting; not attainable dose',
 'coordinate_counts':'unique union of coordinates having any H or L partner at stage; roles can overlap until disjoint selection',
 'stage_order':['currently_small','sensitivity','stratum','sign','magnitude_caliper','slope_caliper','variation_caliper','exact_row','disjoint_greedy','exact_column_A','aggregate_balance','sensitivity_contrast','dose_coverage'],
 'certification':'Binary solver upper bounds only apply to stated graph and encoded constraints; SMD/KS checked separately, timeouts unresolved; LP-floor candidates have no maximum certificate'},
 'overlap_populations':'Descriptive within-stratum lower/upper quartiles of existing A on bottom half; ascending A then flat index, first/last ceil25%; not new matcher eligibility',
 'replication_unit':'five original source seeds; tensor/edge/coordinate counts descriptive',
 'scope':'No threshold/dose/score/architecture/checkpoint search; no intervention authorized; original stopped experiment remains unchanged'}
 save(D/'audit_config.json',config);(D/'audit_config.json').chmod(0o444);(D/'fixed_solver_subset.csv').chmod(0o444)
 save(D/'audit_seal.json',{'config_sha256':sha(D/'audit_config.json'),'implementation':{str(p.relative_to(ROOT)):sha(p) for p in HERE.iterdir() if p.suffix in ('.py','.cpp')}});(D/'audit_seal.json').chmod(0o444)
 print('AUDIT CONFIG SEALED: 40 tensors, binary limits30s/8GiB, no model operations',flush=True)
if __name__=='__main__':main()
