"""Supplementary raw audit of existing arm-A random-mask controls only."""
import os,sys,json,hashlib
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import pandas as pd,torch
ROOT=Path(__file__).resolve().parents[4];O=Path(os.environ.get('DATASET_A_ANALYSIS_OUT',str(ROOT/'results/dataset_a_full_lower_sparsity/analysis_final')))
sys.path.insert(0,str(ROOT/'src'))
from moe_lth.pruning.router_age import state_dict_hash,parameter_group
r=pd.read_csv(O/'random_controls.csv') if (O/'random_controls.csv').exists() else pd.read_csv(ROOT/'results/dataset_b_lower_sparsity/manifests/run_manifest.csv')
if 'random_loss' not in r:
 r=r[(r.job_type=='sparse')&(r.mask_type=='random')&(r.arm=='A')].copy()
 r['random_loss']=[json.loads((Path(p)/'validated_result.json').read_text())['final_validation_loss'] for p in r.artifact_path]
sources=json.loads((ROOT/'results/dataset_b_lower_sparsity/runtime_sources.json').read_text());torch.set_num_threads(2)
def read(p):return json.loads(Path(p).read_text())
def sha(p):
 with Path(p).open('rb') as h:return hashlib.file_digest(h,'sha256',_bufsize=8*1024*1024).hexdigest()
def audit(row):
 out={'condition_id':row['condition_id']}
 try:
  e,s,a,sp=[int(row[k]) if k!='sparsity' else float(row[k]) for k in ['num_experts','seed','router_age','sparsity']];p=Path(row['artifact_path']);info=sources[f'source_e32_seed{s}_T2500'];rec=info['masks'][str(sp)];mask=torch.load(rec['random_path'],map_location='cpu',weights_only=True);init=torch.load(info['checkpoints']['0']['path'],map_location='cpu',weights_only=False);ck=torch.load(p/'checkpoints/final_recovered.pt',map_location='cpu',weights_only=False);meta=read(p/'metadata.json');done=read(p/'pilot_completed.json');res=read(p/'validated_result.json');base={n:v*mask[n] if n in mask else v for n,v in init['model'].items()};group=lambda k:{n:v for n,v in base.items() if parameter_group(n)==k};curve=[json.loads(x) for x in (p/'metrics.jsonl').read_text().splitlines()]
  out.update(source_R0_hash_valid=sha(info['checkpoints']['0']['path'])==info['checkpoints']['0']['sha256'],random_mask_file_valid=sha(rec['random_path'])==rec['random_file_sha256'],initial_expert_valid=state_dict_hash(group('expert'))==meta['expert_state_hash'],initial_shared_valid=state_dict_hash(group('shared'))==meta['shared_state_hash'],checkpoint_finite=all(bool(torch.isfinite(v).all()) for v in ck['model'].values()),pruned_weights_zero=all(not bool(ck['model'][n][~m.bool()].any()) for n,m in mask.items()),endpoint_valid=ck['step']==2500 and done['condition_id']==row['condition_id'] and abs(curve[-1]['loss']-res['final_validation_loss'])<1e-10 and abs(done['final_validation_loss']-row['random_loss'])<1e-10 and [x['step'] for x in curve]==list(range(0,2501,50)),optimizer_valid=meta['optimizer']=='fresh_AdamW' and meta['optimizer_state_entries_at_start']==0 and meta['optimizer_steps_applied']+meta['amp_overflow_step_count']==2500 and ck['config']['training']['learning_rate']==.0003 and ck['config']['training']['weight_decay']==.01 and ck['config']['training']['grad_clip']==1)
  out['passed']=all(out[k] for k in out if k!='condition_id')
 except Exception as ex:out.update(passed=False,error=repr(ex))
 return out
with ThreadPoolExecutor(max_workers=4) as pool:rows=list(pool.map(audit,r.to_dict('records')))
p=O/'random_control_verification.csv'
if p.exists():raise FileExistsError(p)
pd.DataFrame(rows).to_csv(p,index=False)
print('Random raw audit:',sum(x['passed'] for x in rows),'/',len(rows))
raise SystemExit(0 if all(x['passed'] for x in rows) else 1)
