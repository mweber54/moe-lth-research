"""Sequential authorized workflow: capture gate -> compile -> validate -> recover -> analyze."""
from pathlib import Path
import json,time,subprocess,sys,traceback,hashlib
ROOT=Path('results/gate_amplitude_pattern_decomposition')

def save(m):
 path=ROOT/'workflow_manifest.json';tmp=path.with_suffix('.tmp');tmp.write_text(json.dumps(m,indent=2)+'\n');tmp.replace(path)

def main():
 path=ROOT/'workflow_manifest.json'
 if path.exists():raise FileExistsError('Workflow already exists; do not start duplicate recoveries')
 m={'status':'waiting_for_all_sources','required_dense_sources':18,'required_decomposition_recoveries':252,'conditions':['global_mean','layer_mean','expert_layer_mean','token_conditioned','temporal','temporal_shuffle','token_shuffle'],'primary_readout':'primary_r100_common_readout','extra_experiments_authorized':False};save(m);last=None
 while True:
  cap=json.loads((ROOT/'capture_run_manifest.json').read_text());count=cap['validated_sources']
  if count!=last:print(f'[workflow] validated canonical dense sources {count}/18',flush=True);last=count
  if cap['status']=='failed':raise RuntimeError('Capture validation failed; decomposition is prohibited')
  if cap['status']=='all_sources_validated':
   assert count==18;break
  time.sleep(30)
 import pandas as pd
 sources=pd.read_csv(ROOT/'validated_amplitude_sources.csv');assert len(sources)==18 and (sources.kind=='dense').all();sources['source_role']='canonical age/seed dense amplitude source tensor';sources['statistical_replicate']=False;sources.to_csv(ROOT/'amplitude_source_manifest.csv',index=False)
 for stage,script in [('building_transforms','scripts/build_amplitude_pattern_schedules.py'),('checking_compiled_forwards','scripts/validate_compiled_amplitude_patterns.py'),('running_decomposition','scripts/run_amplitude_pattern_decomposition.py')]:
  m.update(status=stage,stage_script=script,stage_source_sha256=hashlib.sha256(Path(script).read_bytes()).hexdigest());save(m);print('[workflow]',stage,flush=True)
  subprocess.run([sys.executable,'-u',script],check=True)
 m.update(status='analysis_ready_for_review');save(m)
if __name__=='__main__':
 try:main()
 except Exception:
  (ROOT/'workflow_failure.txt').write_text(traceback.format_exc())
  p=ROOT/'workflow_manifest.json'
  if p.exists():
   m=json.loads(p.read_text());m.update(status='failed');save(m)
  raise
