import os
"""Independent output verification; reads raw endpoint JSON and exported tables.
Run after analyze.py. Writes independent_verification.json only if absent.
"""
import json,math,itertools
from pathlib import Path
import numpy as np,pandas as pd
from scipy.stats import t
O=Path(os.environ.get('DATASET_A_ANALYSIS_OUT',str(Path(__file__).resolve().parents[1])))
f=pd.read_csv(O/'canonical_lower_sparsity.csv');s=pd.read_csv(O/'setting_summary.csv');c=pd.read_csv(O/'canonical_combined.csv');d=pd.read_csv(O/'dense_references.csv');counts=pd.read_csv(O/'replicated_candidate_counts.csv');checks={}
checks['exact_grid']=len(f)==1100 and set(map(tuple,f[['num_experts','seed','router_age','sparsity']].values))==set(itertools.product([8,12,16,24,32],[7,17,29,43,61],range(0,101,10),[.5,.55,.6,.65]))
checks['unique_combined']=len(c)==2750 and not c.duplicated(['num_experts','seed','router_age','sparsity']).any()
checks['unique_dense']=len(d)==275 and d.condition_id.is_unique
checks['finite_losses']=np.isfinite(f[['sparse_final_validation_loss','dense_final_validation_loss','ticket_gap','perplexity_ratio']]).all().all()
checks['gap_arithmetic']=np.allclose(f.ticket_gap,f.sparse_final_validation_loss-f.dense_final_validation_loss,atol=1e-12,rtol=0)
checks['ppl_arithmetic']=np.allclose(f.perplexity_ratio,np.exp(f.ticket_gap),atol=1e-12,rtol=0)
raw=[]
for r in f.to_dict('records'):
 z=json.loads((Path(r['artifact_path'])/'validated_result.json').read_text());raw.append(abs(z['final_validation_loss']-r['sparse_final_validation_loss'])<1e-12)
checks['raw_endpoint_roundtrip']=all(raw)
checks['audit_checks_pass']=pd.read_csv(O/'final_verification_checks.csv').passed.all()
checks['five_seeds']=len(s)==220 and s.n.eq(5).all()
statok=True
for r in s.to_dict('records'):
 h=f[(f.num_experts==r['num_experts'])&(f.router_age==r['router_age'])&(f.sparsity==r['sparsity'])].sort_values('seed');x=h.ticket_gap.values;se=np.std(x,ddof=1)/math.sqrt(5)
 statok &= abs(x.mean()-r['mean_gap'])<1e-12 and abs(np.std(x,ddof=1)-r['sd_gap'])<1e-12 and abs(x.mean()+t.ppf(.95,4)*se-r['upper95_one_sided'])<1e-12
checks['independent_paired_statistics']=statok
for name,tol in [('strict',0),('ppl_1pct',math.log(1.01)),('ppl_2pct',math.log(1.02)),('ppl_5pct',math.log(1.05))]:
 r=counts[counts.threshold==name].iloc[0];checks['count_'+name]=int((f.ticket_gap<=tol).sum())==r.seed_matches and int((s[name+'_passing_seeds']==5).sum())==r.all_five
checks['figures']=all((O/'figures'/f'{name}.{ext}').stat().st_size>1000 for name in ['ticket_gap_vs_age','absolute_endpoints','combined_gap_heatmaps','matching_seed_counts','strongest_candidates_seed_views','all_seed_age_curves'] for ext in ['pdf','png'])
result=dict(status='PASS' if all(checks.values()) else 'FAIL',checks={k:bool(v) for k,v in checks.items()})
p=O/os.environ.get('DATASET_A_VERIFICATION_NAME','independent_verification.json')
with p.open('x') as h:json.dump(result,h,indent=2)
print(json.dumps(result,indent=2));raise SystemExit(0 if all(checks.values()) else 1)
