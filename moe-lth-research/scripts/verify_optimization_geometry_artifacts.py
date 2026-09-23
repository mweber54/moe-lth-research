"""Read-only scientific artifact checks; writes only new verification evidence."""
from pathlib import Path
import csv,json,hashlib
import torch
from PIL import Image
from moe_lth.experiments.optimization_geometry_diagnostics import digest
R=Path('results/cross_arch_optimization_geometry/diagnostic_rerun')
m=json.loads((R/'run_manifest.json').read_text());assert len(m['conditions'])==18
snap=list(csv.DictReader((R/'snapshot_manifest.csv').open()));assert len(snap)==324
assert len({(x['experts'],x['seed'],x['age'],x['kind'],x['step']) for x in snap})==324
assert all(Path(x['path']).is_file() and Path(x['path']).stat().st_size>1_000_000 for x in snap)
audit=list(csv.DictReader((R/'mutation_audit.csv').open()));assert len(audit)==324
flags=['model_unchanged','optimizer_unchanged','scaler_unchanged','rng_restored','training_iterator_not_accessed']
assert all(all(x[f]=='True' for f in flags) for x in audit)
final=[]
for row in m['conditions']:
 for kind in ['sparse','dense']:
  old=Path(row[kind])/'checkpoints/final_recovered.pt'
  new=R/f"recovery/e{row['experts']}/seed{row['seed']}/R{row['age']}/{kind}/checkpoints/final_recovered.pt"
  a=torch.load(old,map_location='cpu',weights_only=False);b=torch.load(new,map_location='cpu',weights_only=False)
  same=digest(a['model'])==digest(b['model'])
  final.append(dict(experts=row['experts'],seed=row['seed'],age=row['age'],kind=kind,bit_identical_model=same,historical_model_hash=digest(a['model']),rerun_model_hash=digest(b['model'])))
  old_meta=json.loads((old.parent.parent/'metadata.json').read_text());new_meta=json.loads((new.parent.parent/'metadata.json').read_text())
  for key in ['seed','router_step','router_age_percent','expert_state_hash','shared_state_hash','mask_hash','initial_router_state_hash','training_batch_sequence_hash','validation_batch_sequence_hash','recovery_steps','recovery_eval_interval','optimizer','scheduler','precision','router_mode']:
   assert old_meta[key]==new_meta[key],(row,kind,key)
  del a,b
 print(f"Verified final states and protocol metadata: e{row['experts']} seed{row['seed']} R{row['age']}",flush=True)
with (R/'full_final_state_reproduction.csv').open('w',newline='') as f:
 w=csv.DictWriter(f,fieldnames=final[0]);w.writeheader();w.writerows(final)
reload=[]
for e,age in [(8,20),(12,60)]:
 for kind in ['sparse','dense']:
  for step in [0,2500]:
   path=R/f'snapshots/e{e}/seed7/R{age}/{kind}/step_{step:04d}.pt'
   s=torch.load(path,map_location='cpu',weights_only=False)
   required=['model','optimizer','grad_scaler','scheduler','lr','step','rng','config','seed','mask','historical_metadata','data_sequence_identity']
   assert all(k in s for k in required)
   assert digest(s['model'])==s['parameter_hash'] and digest(s['optimizer'])==s['optimizer_hash']
   assert s['step']==s['data_sequence_identity']['next_batch_index']==step
   assert bool(s['optimizer']['state'])==(step>0)
   assert s['mask_applied']==(kind=='sparse')
   if kind=='sparse':
    assert all(not s['model'][name][~mask.bool()].count_nonzero() for name,mask in s['mask'].items())
   reload.append(str(path));del s
figs=list((R/'figures').glob('*.png'));assert len(figs)>=14
for path in figs:
 with Image.open(path) as im:im.verify()
 assert path.with_suffix('.pdf').read_bytes().startswith(b'%PDF')
counts={}
for name,expected in [('gradient_alignment_native',6480),('gradient_alignment_forced_sparse_routes',6480),('functional_expert_similarity',6480),('update_alignment',5760),('parameter_trajectory_distance',6480),('sparse_dense_routing_overlap',6480),('gradient_coherence',6480),('gradient_norms',12960)]:
 rows=list(csv.DictReader((R/f'{name}.csv').open()));assert len(rows)==expected
 counts[name]={'rows':len(rows),'nonfinite_rows':sum(x.get('finite')=='False' for x in rows),'unavailable_rows':sum(x.get('status','ok')!='ok' for x in rows)}
result={'completed_pairs':18,'completed_recoveries':36,'complete_unique_snapshot_paths':324,'snapshot_mutation_audits_passed':324,'all_final_metadata_protocol_checks_passed':True,'bit_identical_final_models':sum(x['bit_identical_model'] for x in final),'snapshot_reloads_verified':reload,'all_snapshot_files_exist_and_nontrivial_size':True,'figure_png_pdf_pairs':len(figs),'diagnostic_row_checks':counts,'snapshot_hash_scope':'All snapshots hashed when saved; representative eight snapshots reloaded here. All 36 final model states independently compared to history.'}
(R/'artifact_integrity.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps(result,indent=2),flush=True)
