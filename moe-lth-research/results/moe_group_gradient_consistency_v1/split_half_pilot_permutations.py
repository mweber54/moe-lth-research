"""Continue the verified pilot to 20 label permutations under admitted.py."""
import json,subprocess,sys
from pathlib import Path
P=Path(__file__).resolve().parent
v=json.loads((P/'split_half_pilot_verification.json').read_text())
assert v['bitwise_repeat_passed'] and v['original_integer_accumulators_match_archived'] and v['model_and_rng_unchanged']
for i in range(1,20):
 for h in (1,2):
  dest=P/f'split_half_gradients_seed17_B_250_h{h}_perm{i}.pt'
  if not dest.exists():
   subprocess.run([sys.executable,str(P/'split_half_replay.py'),'17','B','250',str(h),'--permutation',str(i)],check=True)
  status={'status':'pilot_permutations_running','cell':'seed17_B_250','last_completed_permutation':i,'last_completed_half':h,'target_permutations':20,'non_180_cells':'batch rule unresolved'}
  (P/'split_half_replay_status.json').write_text(json.dumps(status,indent=2)+'\n')
subprocess.run([sys.executable,str(P/'split_half_analysis.py'),'--permutations','20'],check=True)
(P/'split_half_replay_status.json').write_text(json.dumps({'status':'pilot_permutations_complete','cell':'seed17_B_250','permutations':20,'full_study':'not started; non-180-cell batch rule unresolved'},indent=2)+'\n')
