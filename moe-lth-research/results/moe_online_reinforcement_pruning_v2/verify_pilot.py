"""Confirm the three completed pilot branches are comparable."""
import json,math,os,time,uuid
from pathlib import Path
import status
HERE=Path(__file__).resolve().parent
root=HERE/'branches'/'seed7_A';arms=status.ARMS
for arm in arms:
 if status.status(root/arm)!='COMPLETED':raise RuntimeError(f'pilot endpoint unverified: {arm}')
records={arm:json.loads((root/arm/'complete.json').read_text()) for arm in arms}
for key in ('prefix_sha256','train_hash','validation_hash'):
 if len({records[a]['identity'][key] for a in arms})!=1:raise RuntimeError(f'pilot identity mismatch: {key}')
pre=[records[a]['losses']['pre_mask_250'] for a in arms]
if max(pre)-min(pre)>1e-12:raise RuntimeError('pilot pre-mask losses differ')
for arm in arms:
 d=records[arm]
 if d['identity']['arm']!=arm or d['global_step']!=2500 or not 0<=d['applied_since_branch']<=2250:raise RuntimeError('invalid pilot endpoint')
 if any(not math.isfinite(v) for v in d['losses'].values()):raise RuntimeError('nonfinite pilot loss')
path=HERE/'pilot_verified.json';tmp=path.with_name('.pilot.'+uuid.uuid4().hex);tmp.write_text(json.dumps({'verified_unix':time.time(),'records':records},indent=2)+'\n');os.replace(tmp,path)
print('Pilot triplet verified: same prefix, data streams, evaluation tokens and pre-intervention CE')
