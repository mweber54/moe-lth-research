"""Synthetic file fixtures only; no model, optimizer, forward or training."""
from pathlib import Path
import tempfile,importlib.util,json,csv
import pandas as pd
ROOT=Path.cwd();OUT=Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location('synthetic_report',OUT/'report.py');r=importlib.util.module_from_spec(spec);spec.loader.exec_module(r)
legacy=pd.read_csv(ROOT/'results/imp_initial_value_placement_v1/trajectory_losses.csv')
with tempfile.TemporaryDirectory(prefix='stage1_synthetic_no_training_') as td:
 target=Path(td);r.OUT=target;(target/'status').mkdir();(target/'workers').mkdir();(target/'jobs').mkdir()
 manifest=[]
 for seed in (7,17,29):
  for order in ('A','B'):
   for draw in (1,2,3):
    for mask in ('imp','random1','random2'):
     job=f'synthetic_{seed}_{order}_{draw}_{mask}';path=target/'jobs'/job;path.mkdir()
     original='imp_original' if mask=='imp' else mask+'_original';base=legacy[(legacy.seed==seed)&(legacy.order==order)&(legacy.variant==original)].set_index('step')
     with (path/'attempt_ledger.jsonl').open('w') as f:
      for n in range(1,2501):f.write(json.dumps({'attempt':n})+'\n')
     with (path/'common_evaluations.jsonl').open('w') as f:
      for step in (0,250,500,1000,1500,2000,2500):
       delta=(.02 if step<=250 else .10)+.001*draw if mask=='imp' else .001*draw if mask=='random1' else .002*draw
       f.write(json.dumps(dict(step=step,cross_entropy=float(base.loc[step,'cross_entropy'])+delta))+'\n')
     (target/'status'/f'{job}.json').write_text(json.dumps(dict(status='COMPLETE',attempted_updates=2500,applied_updates=2499,amp_skipped_updates=1,host='synthetic_no_training',slot_seconds=600,native_segment_seconds=590,attempt=str(path))))
     manifest.append(dict(job_id=job,seed=seed,order=order,draw=draw,mask_type=mask,initial_state_hash=f'synthetic_{seed}_{mask}_{draw}',realized_distance=1.4,frac_sign_flipped=.5))
 with (target/'job_manifest.csv').open('w') as f:w=csv.DictWriter(f,fieldnames=list(manifest[0]));w.writeheader();w.writerows(manifest)
 (target/'registration.json').write_text(json.dumps(dict(preparation_elapsed_seconds=0)))
 (target/'workers/fixture_accounting.json').write_text(json.dumps(dict(worker_slot_seconds=32400)))
 r.main()
 draws=pd.read_csv(target/'individual_paired_draw_values.csv');assert len(draws)==9 and (abs(draws.R-.08)<1e-12).all()
 gates=pd.read_csv(target/'per_seed_gate.csv');assert gates.joint_pass.all()
 summary=pd.read_csv(target/'per_seed_order_summary.csv');assert len(summary)==6 and (abs(summary.R-.08)<1e-12).all()
 with (OUT/'report_check.csv').open('w') as f:
  w=csv.DictWriter(f,fieldnames=['check','passed','actual_training_updates']);w.writeheader();w.writerow(dict(check='synthetic complete-report pairing and values',passed=True,actual_training_updates=0))
 print('synthetic report test passed; zero actual training updates')
