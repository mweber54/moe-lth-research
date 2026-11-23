"""Validate the completed result bundle without loading all neuron rows."""
import argparse
import hashlib
import json
from pathlib import Path
import pandas as pd
import pyarrow.parquet as pq
ROOT=Path(__file__).resolve().parents[2]


def sha(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(4*1024*1024),b''):h.update(b)
    return h.hexdigest()


def validate(out):
    c=json.loads((out/'completion.json').read_text())
    assert c['completed_profiles']==75 and c['historical_recoveries']==30 and not c['training_launched']
    registry=json.loads((out/'registration.json').read_text())
    for path,digest in registry['source_hashes'].items():
        assert sha(ROOT/path)==digest, f'Source registration differs: {path}'
    amendment=json.loads((out/'engineering_amendment.json').read_text())
    prior=(out/'profiling_source_snapshot/ramp.py').read_text().split('def allocate_weight_mask')[0]
    current=(ROOT/'src/moe_lth/analysis/ramp.py').read_text().split('def allocate_weight_mask')[0]
    assert current==prior and hashlib.sha256(current.encode()).hexdigest()==amendment['unchanged_profiling_prefix_sha256']
    neuron_rows=0
    for marker in sorted(out.glob('e*_seed*/*/complete.json')):
        payload=json.loads(marker.read_text())
        for filename,digest in payload['files'].items():
            path=marker.parent/filename
            assert sha(path)==digest, f'Result hash mismatch: {path}'
            if filename.startswith('neurons_'):
                experts=int(marker.parents[1].name.split('_')[0][1:])
                n=pq.read_metadata(path).num_rows
                assert n==experts*4*1024
                neuron_rows+=n
    assert neuron_rows==11468800
    runs=pd.read_csv(out/'mask_run_results.csv')
    assert len(runs)==450 and not runs.duplicated(['architecture','seed','sparsity','profile_name','domain']).any()
    assert (runs.groupby(['architecture','sparsity','profile_name','domain']).seed.nunique()==5).all()
    reconstructed=json.loads((out/'reconstruction_validation.json').read_text())
    assert len(reconstructed)==15 and all(r['reconstructed_data_match'] and len(r['profile_payloads_verified'])==5 for r in reconstructed)
    supplement=json.loads((out/'supplement_manifest.json').read_text())
    assert len(supplement['recovered_artifacts'])==30
    for artifact in supplement['recovered_artifacts']:
        assert sha(Path(artifact['path']))==artifact['sha256']
    controls=pd.read_csv(out/'historical_mask_controls.csv')
    degree=controls[controls.condition=='degree']
    assert len(degree)==75 and degree.exact_neuron_retention_preserved.all() and (degree.ramp_weighted_retention_minus_imp==0).all()
    figures=json.loads((out/'figures/figure_manifest.json').read_text())
    for figure in figures:
        for path in figure['underlying_artifacts']:assert Path(path).is_file()
        for ext in ('pdf','png'):assert (out/'figures'/f"{figure['figure']}.{ext}").is_file()
    result=dict(status='PASS',profiles=75,historical_recoveries=30,pooled_neuron_pairing_rows=neuron_rows,
        mask_profile_domain_rows=450,independent_source_groups=15,canonical_streams_reconstructed=15,
        profile_checkpoint_payloads_verified=75,recovered_supports_verified=30,
        exact_degree_control_masks=25,figure_pairs=len(figures),unresolved_validation_failures=0,
        training_launched=False,profiling_source_unchanged_by_h3_amendment=True)
    (out/'validation_summary.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,default=ROOT/'results/ramp_imp_correspondence_v1')
    validate(p.parse_args().output.resolve())
