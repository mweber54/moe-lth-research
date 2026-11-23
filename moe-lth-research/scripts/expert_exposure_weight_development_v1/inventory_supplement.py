"""Enrich excluded-family metadata without loading their weight checkpoints."""
from audit import *
def main():
 rows=list(csv.DictReader((O/'inventory.csv').open()));extra=[];coverage=list(csv.DictReader((O/'exposure_coverage.csv').open()))
 af=json.loads((ROOT/'results/imp_adaptation_factorial_v1/manifest.json').read_text())
 for v in af['variants']:
  if v['seed'] not in SEEDS:continue
  for offset in (0,2500):
   d=Path(v['full_attempt'])/f'segment_{offset}';p=d/'record.json'
   if not p.exists():continue
   r=json.loads(p.read_text());fam='unpruned_recovery_segment' if not v['sparsity'] else 'sparse_recovery_segment'
   extra.append(dict(seed=v['seed'],architecture='32E',family=fam,protocol=r.get('condition','native recovery'),checkpoint_step=f'{offset} to {offset+2500}',path=str(d/'checkpoints/final_recovered.pt'),sha256='',hash_verified=False,selected=False,attempted_updates=r.get('optimizer_step_attempts',''),applied_updates=r.get('optimizer_steps_applied',''),optimizer_reset_boundary=offset,reason='excluded protocol family; metadata rehashed; endpoint weights not read',metadata_path=str(p),metadata_sha256=sha(p),amp_skips=r.get('amp_overflow_step_count',''),weight_decay=v['config']['training']['weight_decay'],mask_hash=r.get('mask_hash',''),checkpoint_paths=json.dumps([str(q) for q in sorted((d/'checkpoints').glob('*.pt'))])))
   coverage.append(dict(seed=v['seed'],trajectory=f'{fam}_{v["id"]}_{offset}',log_path=str(d/'gradient_stats'),sha256='',first_step=offset,last_step=offset+2500,logged_steps='periodic gradients/routing; not audited as complete exposure',attempted_updates=r.get('optimizer_step_attempts',''),applied_updates=r.get('optimizer_steps_applied',''),proposed='sampled diagnostic only',accepted='not established as complete',gate_weighted=False,accepted_token_composition=False,hidden_input_statistics=False,domain_labels=False))
 for row in rows:
  if row['family']!='IMP_discovery_round':continue
  d=Path(row['path']);p=d/'stage.json'
  if p.exists():
   r=json.loads(p.read_text());row.update(attempted_updates=r.get('optimizer_step_attempts',''),applied_updates=r.get('optimizer_steps_applied',''),amp_skips=r.get('amp_overflow_step_count',''),metadata_path=str(p),metadata_sha256=sha(p),sparsity=r.get('sparsity',''),checkpoint_paths=json.dumps([str(q) for q in sorted((d/'checkpoints').glob('*.pt'))]))
 table(O/'inventory.csv',rows+extra);table(O/'exposure_coverage.csv',coverage)
 save(O/'data_provenance.json',dict(train_path=str(ROOT/'data/multidomain_balanced/multidomain_train.txt'),train_sha256=sha(ROOT/'data/multidomain_balanced/multidomain_train.txt'),validation_path=str(ROOT/'data/multidomain_balanced/multidomain_validation.txt'),validation_sha256=sha(ROOT/'data/multidomain_balanced/multidomain_validation.txt'),fixed_probe_identity='common unchanged validation data/config; deterministic unshuffled validation loader and matching saved batch shapes; archived assignment files lack token hashes',exposure_logging_code_sha256=sha(ROOT/'src/moe_lth/training/train.py'),route_serialization_code_sha256=sha(ROOT/'src/moe_lth/routing/route_history.py'),pruning_code_sha256=sha(ROOT/'src/moe_lth/pruning/magnitude_prune.py')))
if __name__=='__main__':main()
