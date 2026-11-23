"""Read-only archived-code, snapshot availability and original-output preservation audit."""
from common import *

def run():
    seal=read(BASE/'launch_seal.json');code=[]
    for rel,expected in seal['code'].items():
        if '/models/' not in rel:continue
        p=ROOT/rel;actual=sha(p) if p.exists() else None
        code.append(dict(path=rel,expected_sha256=expected,actual_sha256=actual,archived_code_matches=actual==expected))
    frame(OUT/'deployed_model_code_audit.csv',code)
    assert all(r['archived_code_matches'] for r in code),'Deployed model differs from archived implementation'
    jobs=pd.read_csv(BASE/'run_manifest.csv');coverage=[]
    for _,j in jobs.iterrows():
        for step in STEPS:
            p=Path(j['attempt'])/'geometry'/f'step_{step:04d}'/'diagnostic_model_state.pt'
            weights=p.parent/'weights.csv';meta=p.parent/'metadata.json'
            coverage.append(dict(condition_id=j.condition_id,seed=int(j.seed),router_age=int(j.router_age),sparsity=j.sparsity,mask_type=j.mask_type,
                                 step=step,weights_path=str(p),weights_available=p.is_file(),svd_available=weights.is_file(),metadata_available=meta.is_file()))
    frame(OUT/'expected_recovery_coverage.csv',coverage)
    config=read(OUT/'implementation_config.json');preserved=[]
    for rel,h in config['original_analysis_hashes'].items():preserved.append(dict(path=rel,original_sha256=h,current_sha256=sha(BASE/rel),preserved=h==sha(BASE/rel)))
    frame(OUT/'original_output_preservation.csv',preserved)
    assert all(r['preserved'] for r in preserved)
    assert sha(BASE/'run_manifest.csv')==config['original_manifest_sha256']
    save(OUT/'input_audit.json',dict(passed=True,archived_model_modules=len(code),available_recovery_weights=sum(r['weights_available'] for r in coverage),expected_recovery_weights=len(coverage),
       missing=[r for r in coverage if not (r['weights_available'] and r['svd_available'] and r['metadata_available'])],
       original_outputs_preserved=True,original_manifest_preserved=True,
       checkpoint_hash_scope='Source, masks, probes and native endpoints have prior expected hashes; intermediate snapshots receive new identities and structural/provenance checks.'))
    print('INPUT/PRESERVATION AUDIT PASS',flush=True)
if __name__=='__main__':run()
