"""Post-audit combinatorial interpretation; no new matcher or solver settings."""
from common import *

def main():
 frozen()
 from enrich import main as enrich
 enrich()
 data=rows(D/'column_graph_diagnostics.csv');assert len(data)==1280
 certificates=[]
 for seed in SEEDS:
  rs=[r for r in data if int(r['seed'])==seed]
  eligible=[r for r in rs if int(r['full_exact_column_coordinate_upper_bound'])>=656]
  cells=[]
  for layer in range(4):
   for proj in ('fc1','fc2'):
    n=sum(int(r['layer'])==layer and r['projection']==proj for r in eligible);cells.append(dict(layer=layer,projection=proj,possibly_eligible=n,required=26))
  maxdose=sum(int(r['full_exact_column_coordinate_upper_bound']) for r in eligible)/67108864
  impossible=len(eligible)<205 or any(x['possibly_eligible']<26 for x in cells) or maxdose<.005
  certificates.append(dict(seed=seed,complete_graph_tensors=256,SCC_bound_below_minimum=sum(int(r['full_exact_column_coordinate_upper_bound'])<656 for r in rs),possibly_eligible_tensors=len(eligible),maximum_retained_whole_expert_dose_bound=maxdose,source_coverage_proven_impossible=impossible,cells=cells))
 save(D/'complete_graph_coverage_bounds.json',{'method':'Every selected edge of a column-balanced circulation belongs to a directed cycle and therefore one column SCC in the complete admissible graph. Distinct-coordinate capacity then bounds pairs by min(unique cyclic H,unique cyclic L,floor(cyclic union/2),2621). These are necessary bounds, independent of greedy choices and aggregate balance.', 'full_candidate_graphs':True,'applies_to_designs':['A','B'],'does_not_apply_to_C_TV_relaxation':True,'source_bounds':certificates})
 table(D/'certified_graph_bounds.csv',[{k:v for k,v in r.items() if k!='cells'} for r in certificates])
 results=rows(D/'solver_results.csv');overlap=rows(D/'feature_overlap.csv');attr=rows(D/'constraint_attrition.csv')
 lines=['','## Reviewed diagnosis','', 'The original matching implementation reproduced its registration; no concrete implementation defect was found. The synthetic examples establish genuine greedy limitations, but those examples alone do not explain the actual cohort stop.','',
 '| Source seed | Complete-graph SCC bound below656 | Potentially includable tensors at most | Source coverage impossible under exact columns |','|---|---:|---:|---|']
 for r in certificates:lines.append(f"| {r['seed']} | {r['SCC_bound_below_minimum']}/256 | {r['possibly_eligible_tensors']} | {r['source_coverage_proven_impossible']} |")
 if all(r['source_coverage_proven_impossible'] for r in certificates):
  lines+=['', '**The original exact-column design cannot satisfy the registered source-coverage gates in any of the five models, even if its greedy matcher is replaced.** This follows from complete admissible graphs and necessary SCC/coordinate bounds, before imposing aggregate SMD/KS or energy requirements. It is a combinatorial statement about these registered candidate populations and constraints, not evidence against the gradient pruning hypothesis.']
 else:lines+=['','The complete-graph bounds certify only the source failures explicitly marked above; other sources remain unresolved by this bound.']
 lines+=['', '| Source seed | Median admissible same-row edges | Median greedy disjoint pairs before columns | Median cycle pairs after exact columns | Quartile row TV median | Quartile column TV median |','|---|---:|---:|---:|---:|---:|']
 for seed in SEEDS:
  def med(stage,key):return float(np.median([float(r[key]) for r in attr if int(r['seed'])==seed and r['stage']==stage and r.get(key,'')!='']))
  op=[r for r in overlap if int(r['seed'])==seed]
  lines.append(f"| {seed} | {med('exact_row','possible_edges'):.0f} | {med('disjoint_greedy','actual_pairs'):.0f} | {med('exact_column_A','actual_pairs'):.1f} | {float(np.median([float(r['row_TV']) for r in op])):.3f} | {float(np.median([float(r['column_TV']) for r in op])):.3f} |")
 lines+=['', 'Thousands of actual disjoint row-matched pairs can exist before column balancing. This distinguishes raw/nonstructural overlap from the scarcity of column-balanced cycles. The complete-graph bound uses all admissible pair choices, so this conclusion is stronger than infeasibility on the original greedy-restricted graph. Aggregate balance can impose further limits; descriptive population overlap alone does not certify group feasibility.',
 '', 'Wall-killed solver jobs have unavailable peak RSS and no returned bound; these are not zero measurements. The8GiB address-space and45s external wall limits remained enforced. Time-limit returns with a valid complete-graph bound below656 can rule out the registered minimum even when the exact maximum is unresolved.',
 '', 'Design C is a different scientific question. Its all-tensor LP/floor generator is restricted to original greedy pairs and cannot certify full-graph infeasibility. The fixed-subset C binary solves use alternative admissible partners, but their bounds/timeouts apply only to that subset and stated graph. At fewer than20 pairs, TV<=0.05 itself forces exact columns because the smallest nonzero TV is1/k. This does not license treating four or five pairs as adequate dose.',
 '', '**Decision D for the intended intervention:** stop the original matching design. Greedy limitations exist, but replacing the matcher cannot rescue its exact-column cohort under the certified coverage bounds above. The evaluated C candidates and bounded subset optimization do not establish a revised full-cohort comparison. Do not repeatedly relax settings or expand compute without a separately defined question. The0.05 tolerance is proposed, not demonstrated harmless; no model-loss or pruning utility claim is permitted.']
 local=rows(D/'candidate_balance.csv');passing=[r for r in local if r['method']=='binary_alternative_edges' and r['scientifically_included']=='True']
 lines+=['', '### Per-tensor gains versus cohort feasibility', '', 'A returned candidate passing the per-tensor thresholds does not pass source coverage or the whole-expert dose gate. Every such candidate on the fixed subset is listed below; none is a pruning outcome.', '', '| Design | Seed / parameter | Original pairs | Returned pairs | Column TV | Local checks |', '|---|---|---:|---:|---:|---|']
 for r in passing:
  sr=next(x for x in results if x['seed']==r['seed'] and x['parameter']==r['parameter'] and x['design']==r['design'])
  lines.append(f"| {r['design']} | {r['seed']} / {r['parameter']} | {sr['original_algorithm_pairs']} | {r['pairs']} | {float(r['column_TV']):.6f} | passed locally; source gate not met |")
 lines+=['', 'Thus B/C may improve individual candidates while D remains the decision for the intended full-cohort intervention. These local gains do not override the exact-column source-coverage certificate, and C does not establish adequate full-cohort feasibility. No continued training or model-loss measurement is recommended or authorized from these examples.']
 proposal=D/'revised_protocol_proposal.md'
 if proposal.exists():
  proposal.write_text(proposal.read_text()+"\nCoverage review: scaling B cannot rescue the original exact-column cohort under the certified bounds. Local C feasibility changes the structural interpretation but does not demonstrate the required source coverage. This is a contingency document, not a justified revised intervention registration or a recommendation to expand compute. No further constraint relaxation is proposed.\n")
 p=D/'findings.md';text=p.read_text().replace('a timeout or failed postcheck is unresolved, not a proof of impossibility.', 'A timeout alone or a failed postcheck is not an infeasibility certificate; a valid complete-graph upper bound below656 can still rule out the dose without resolving the exact maximum.')
 text=text.replace('B improves some scientifically admissible subset candidates; full-cohort optimization scalability and coverage remain untested. No intervention is authorized.', 'B improves individual subset candidates, but the reviewed complete-graph bounds below rule out the original cohort coverage. D remains the decision for the intended intervention.')
 p.write_text(text+'\n'.join(lines)+'\n')
 numeric=read(D/'numerical_checks.json');numeric['SCC_bound_exhaustive_checks']=read(D/'bound_proof_checks.json');numeric['scientific_gate_scope']='Candidate pass flags concern per-tensor checks only; source coverage and whole-expert dose are separate gates.';numeric['concentration_reporting_checks']=read(D/'concentration_reporting_checks.json');save(D/'numerical_checks.json',numeric)
 c=read(D/'completion.json');c['exact_column_coverage_bounds']=certificates;c['reviewed_decision']='D: original exact-column matching design cannot meet the certified source coverage; C remains inadequate or unresolved' if any(r['source_coverage_proven_impossible'] for r in certificates) else 'D: feasibility remains inadequate or unresolved under the bounded audit';c['original_inputs_unchanged']=True
 for p in [D/'findings.md',D/'complete_graph_coverage_bounds.json',D/'certified_graph_bounds.csv',D/'feature_overlap.csv',D/'feature_overlap_summaries.csv',D/'candidate_balance.csv',D/'concentration_reporting_checks.json',D/'numerical_checks.json']:c['output_sha256'][str(p.relative_to(D))]=sha(p)
 if proposal.exists():c['output_sha256']['revised_protocol_proposal.md']=sha(proposal)
 c['audit_implementation_sha256']={str(p.relative_to(ROOT)):sha(p) for p in HERE.glob('*') if p.suffix in ('.py','.cpp')}
 save(D/'completion.json',c);print('REVIEWED COMPLETE-GRAPH COVERAGE BOUNDS',[(r['seed'],r['possibly_eligible_tensors']) for r in certificates],flush=True)
if __name__=='__main__':main()
