"""Seal validated code/inputs and permit a durable queue, not GPU oversubscription."""
from pilot_common import *
import subprocess,shutil,time,socket

def main():
 loss=read(OUT/'audits/initial_loss_verification.json');assert loss['passed'],'Initialization/metric semantics unresolved'
 log=(OUT/'logs/tests.log').read_text();assert '\nOK\n' in log and 'SMOKE ' in log
 smoke=[p for p in (OUT/'smoke').glob('cpu_*/smoke_pass.json') if read(p)['passed']];assert smoke
 f=pd.read_csv(OUT/'manifests/initial_manifest.csv');assert len(f)==225 and f.condition_id.nunique()==225
 assert f.groupby('job_type').size().to_dict()=={'dense':60,'source':5,'sparse':160}
 assert int(f.status.eq('REUSED').sum())==45 and int(f.status.eq('PENDING').sum())==180
 assert read(AD/'dataset_a_finalization_status.json')['status']=='PASS';assert len(pd.read_csv(AD/'canonical/dataset_a_canonical.csv'))==1650
 long=pd.read_csv(OUT/'audits/long_source_reuse_search.csv');assert not long.compatible_long_source_candidate.any()
 for rel,digest in read(OUT/'manifests/immutable_hashes.json').items():assert sha(OUT/rel)==digest
 inputs=read(OUT/'manifests/input_provenance.json')
 for path,record in inputs.items():
  p=Path(path);st=p.stat()
  if st.st_size!=record['bytes'] or st.st_mtime_ns!=record['mtime_ns']:assert sha(p)==record['sha256'],'Changed source/reuse input'
 extra={}
 for r in f[f.status=='REUSED'].itertuples():
  for name in ['metadata.json','summary.json','metrics.jsonl']:
   p=Path(r.artifact_path)/name;extra[str(p)]={'sha256':sha(p),'bytes':p.stat().st_size,'mtime_ns':p.stat().st_mtime_ns}
 jsave(OUT/'manifests/reuse_raw_input_hashes.json',extra)
 free=shutil.disk_usage(OUT).free/1024**3;estimate=read(OUT/'audits/resource_estimate.json');required=sum(estimate['expected_storage_gib'].values());assert free>required
 assert torch.__version__.startswith('2.7.0') and torch.cuda.is_available()
 code=[p for p in (ROOT/'scripts/nominal_exposure_dataset').glob('*.py')]+[ROOT/n for n in ['scripts/fixed_budget_dataset/common.py','src/moe_lth/experiments/run_router_age_recovery.py','src/moe_lth/training/train.py','src/moe_lth/training/evaluate.py','src/moe_lth/training/checkpoint.py','src/moe_lth/pruning/router_age.py','src/moe_lth/pruning/magnitude_prune.py','src/moe_lth/pruning/masks.py','src/moe_lth/pruning/rewind.py','src/moe_lth/data.py','src/moe_lth/models/transformer.py','src/moe_lth/models/moe_layer.py','src/moe_lth/models/router.py','src/moe_lth/utils.py']]
 digests={str(p.relative_to(ROOT)):sha(p) for p in code}
 for p in code:atomic(OUT/'validated_code'/p.relative_to(ROOT),p.read_bytes())
 evidence={str(p.relative_to(OUT)):sha(p) for p in (OUT/'analysis').glob('*') if p.is_file() and not p.name.startswith('pilot_')};evidence.update({str(p.relative_to(OUT)):sha(p) for p in (OUT/'audits').glob('*') if p.is_file() and p.name!='preflight.json'});evidence['PROTOCOL.md']=sha(OUT/'PROTOCOL.md');evidence['manifests/reuse_raw_input_hashes.json']=sha(OUT/'manifests/reuse_raw_input_hashes.json')
 gpu=subprocess.check_output(['nvidia-smi','--id=0','--query-compute-apps=pid,process_name,used_memory','--format=csv,noheader,nounits'],text=True).strip()
 status={'status':'PASS','permission':'User explicitly authorized full staged 32E pilot after checks','queue_admission':'READY','production_device_state':'WAITING_FOR_EXTERNAL_GPU_PROCESS' if gpu else 'IDLE','external_gpu_processes':gpu.splitlines() if gpu else [],'free_storage_gib':free,'planned_storage_including_reserve_gib':required,'new_jobs':180,'reused_jobs':45,'host':socket.gethostname(),'timestamp':time.time(),'initial_loss_verified':True,'focused_tests_passed':8,'synthetic_smoke_passed':True,'code_hashes':digests,'evidence_hashes':evidence,'cuda_build':torch.version.cuda,'resume_policy':'preserve attempts; never continue model-only historical checkpoints'}
 jsave(OUT/'audits/preflight.json',status);print(json.dumps({k:v for k,v in status.items() if k not in ['code_hashes','evidence_hashes']},indent=2))
if __name__=='__main__':main()
