from gcommon import *
import socket

def inventory():
 result=[]
 for r in rows(OUT/'input_inventory.csv'):
  p=Path(r['path']);actual=sha(p) if p.is_file() else '';result.append(dict(host=socket.getfqdn(),kind=r['kind'],condition=r['source_id'],path=str(p),expected_sha256=r['expected_sha256'],actual_sha256=actual,status='VERIFIED' if actual==r['expected_sha256'] else 'MISSING_OR_MISMATCH'))
 seen=set()
 def endpoint(cid,p,expected):
  if str(p) in seen:return
  seen.add(str(p));actual=sha(p) if p.is_file() else '';result.append(dict(host=socket.getfqdn(),kind='historical_endpoint',condition=cid,path=str(p),expected_sha256=expected,actual_sha256=actual,status='TABLE_ENDPOINT_ONLY' if not actual and 'dense_table_references' in str(p) else 'MISSING_OPTIONAL_ENDPOINT' if not actual else 'VERIFIED' if not expected or actual==expected else 'MISMATCH'))
 for package in ('dataset_b_lower_sparsity','dataset_a_full_lower_sparsity'):
  for r in rows(ROOT/'results'/package/'manifests/run_manifest.csv'):
   if r.get('num_experts')!='32' or int(float(r.get('seed',-1))) not in SEEDS or int(float(r.get('router_age',-1))) not in AGES:continue
   if r.get('job_type')=='sparse' and float(r.get('sparsity',0)) not in SP:continue
   if r['status'] in ('COMPLETE','REUSED'):endpoint(r['condition_id'],Path(r['artifact_path'])/'checkpoints/final_recovered.pt',r.get('artifact_sha256',''))
 for r in rows(ROOT/'results/final_dataset_a/inputs/dense_recovery_manifest.csv'):
  if r['num_experts']=='32' and int(r['seed']) in SEEDS and int(r['router_age']) in AGES:endpoint(r['condition_id'],Path(r['path'])/'checkpoints/final_recovered.pt',r['final_checkpoint_sha256'])
 table(OUT/'inventory'/f'{socket.getfqdn()}.csv',result)
 if any(r['status'] in ('MISMATCH','MISSING_OR_MISMATCH') for r in result):raise RuntimeError('Inventory integrity failure')
 print(socket.getfqdn(),'verified',sum(r['status']=='VERIFIED' for r in result),'optional missing',sum(r['status']=='MISSING_OPTIONAL_ENDPOINT' for r in result),flush=True)
if __name__=='__main__':inventory()
