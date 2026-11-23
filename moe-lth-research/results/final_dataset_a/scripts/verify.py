"""Standalone integrity verification. Exit 1 means finalization is blocked, never freeze."""
import sys,json,itertools,hashlib
from pathlib import Path
import pandas as pd,numpy as np
ROOT=Path.cwd();OUT=ROOT/'results/final_dataset_a'
def main():
 f=pd.read_csv(OUT/'canonical/dataset_a_canonical.csv',float_precision='round_trip');dense=pd.read_csv(OUT/'canonical/dataset_a_dense.csv',float_precision='round_trip');source=pd.read_csv(OUT/'canonical/dataset_a_sources.csv');audit=pd.read_csv(OUT/'audits/dataset_a_completion_audit.csv');status=json.loads((OUT/'dataset_a_finalization_status.json').read_text());tests=[]
 def check(name,ok,detail=''):tests.append({'test':name,'passed':bool(ok),'detail':detail})
 E=[8,12,16,24,32];S=[7,17,29,43,61];A=list(range(0,101,10));P=[.7,.75,.8,.85,.9,.95]
 check('sparse_cartesian_product',set(f[['num_experts','seed','router_age','sparsity']].itertuples(index=False,name=None))==set(itertools.product(E,S,A,P)) and len(f)==1650)
 check('dense_cartesian_product',set(dense[['architecture','seed','router_age']].itertuples(index=False,name=None))==set(itertools.product(E,S,A)) and len(dense)==275)
 check('source_cartesian_product',set(source[['num_experts','seed']].itertuples(index=False,name=None))==set(itertools.product(E,S)) and len(source)==25)
 check('all_expected_keys_audited',len(audit)==1950 and audit.condition_id.nunique()==1950)
 check('every_required_cell_valid',audit.condition_valid.all())
 check('all_endpoints_finite',np.isfinite(f[['sparse_final_validation_loss','dense_final_validation_loss','ticket_gap']]).all().all())
 raw_ok=True
 for r in f.itertuples():
  sp=json.loads((Path(r.sparse_artifact_path)/'metrics.jsonl').read_text().splitlines()[-1]);dp=json.loads((Path(r.dense_artifact_path)/'metrics.jsonl').read_text().splitlines()[-1]);raw_ok &= sp['step']==dp['step']==2500 and abs(sp['loss']-dp['loss']-r.ticket_gap)<=1e-10
 check('independent_raw_endpoint_reopen',raw_ok)
 pq=pd.read_parquet(OUT/'canonical/dataset_a_canonical.parquet');check('csv_parquet_numeric_roundtrip',np.array_equal(f.ticket_gap.to_numpy(),pq.ticket_gap.to_numpy()))
 t=pd.read_parquet(OUT/'trajectories/dataset_a_recovery_trajectories.parquet');check('trajectory_no_duplicate_steps',not t.duplicated(['condition_id','recovery_step']).any());check('dense_not_duplicated_by_sparsity',t[t.model_type=='dense'].condition_id.nunique()==len(dense));check('no_forbidden_frozen_marker',not (OUT/'DATASET_A_FROZEN.json').exists() if status['status']=='BLOCKED' else (OUT/'DATASET_A_FROZEN.json').exists())
 checks=pd.DataFrame(tests);p=OUT/'audits/dataset_a_independent_verification.csv';tmp=p.with_suffix('.csv.tmp');checks.to_csv(tmp,index=False);tmp.replace(p);print(checks.to_string(index=False));print('FINALIZATION GATE:', 'PASS' if checks.passed.all() else 'BLOCKED')
 # Update output hashes after writing test artifact.
 hp=OUT/'dataset_a_output_hashes.json';h=json.loads(hp.read_text());h[str(p.relative_to(OUT))]=hashlib.sha256(p.read_bytes()).hexdigest();tmp=hp.with_suffix('.json.tmp');tmp.write_text(json.dumps(h,indent=2)+'\n');tmp.replace(hp)
 return 0 if checks.passed.all() else 1
if __name__=='__main__':sys.exit(main())
