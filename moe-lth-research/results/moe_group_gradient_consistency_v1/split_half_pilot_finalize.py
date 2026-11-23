"""Add unconditional results to the complete pilot control without replaying gradients."""
import ast,csv,json
from pathlib import Path
import numpy as np,torch
P=Path(__file__).resolve().parent; out=P/'split_half_cell_analysis/seed17_B_250_first_180';out.mkdir(parents=True,exist_ok=True)
tree=ast.parse((P/'split_half_analysis.py').read_text());funcs=[n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name in ('arrays','calc')];ns={'np':np};exec(compile(ast.Module(body=funcs,type_ignores=[]),'analysis','exec'),ns)
pair=[torch.load(P/f'split_half_gradients_seed17_B_250_h{h}_original.pt',map_location='cpu',weights_only=False) for h in (1,2)]
obs={(str(t),'reverse' if rev else 'primary'):ns['calc'](pair,t,rev,list(range(32))) for t in (0.6,0.8,0.9) for rev in (False,True)}
def read(name):
 with (P/name).open(newline='') as f:return list(csv.DictReader(f))
def write(name,rows):
 keys=list(dict.fromkeys(k for r in rows for k in r))
 with (out/name).open('w',newline='') as f:
  w=csv.DictWriter(f,keys);w.writeheader();w.writerows(rows)
rows=read('split_half_reproducibility.csv')
for r in rows:
 r['analysis']='first_180';v=obs[(r['threshold'],r['direction'])]
 for k in ('eligible_both_halves','qualify_both_same_directions_count','qualify_both_same_directions_fraction'):r[k]=v[k]
write('split_half_reproducibility.csv',rows)
null=read('split_half_permutation_null.csv');assert len(null)==120
for r in null:
 assert r['matched_experts']=='32' and r['null_selected']=='0' and r['null_unknown_rate']=='0.0'
 r['analysis']='first_180';v=obs[(r['threshold'],r['direction'])]
 for k in ('eligible_both_halves','qualify_both_same_directions_count','qualify_both_same_directions_fraction'):r['observed_'+k]=v[k]
 r['null_eligible_both_halves']=int(r['null_total_weights']);r['null_qualify_both_same_directions_count']=0;r['null_qualify_both_same_directions_fraction']=0.0
 for k in list(r):
  if k.startswith('null_'):r['all_experts_'+k[5:]]=r[k]
write('split_half_permutation_null.csv',null)
coverage=read('split_half_coverage.csv');orig={(r['half'],r['domain'],r['expert']):r for r in coverage if r['permutation']=='-1'}
for r in coverage:
 r['analysis']='first_180';o=orig[(r['half'],r['domain'],r['expert'])]
 for key,short in (('accepted_occurrences','occurrence'),('contributing_batches','contributing_batch')):
  x=int(r[key]);y=int(o[key]);r['original_'+key]=y;r[short+'_shift']=x-y;r[short+'_ratio']=x/y if y else ''
write('split_half_coverage.csv',coverage)
comparison=read('split_half_permutation_comparison.csv')
for r in comparison:r['analysis']='first_180'
for (t,d),v in obs.items():comparison.append({'cell':'seed17_B_250','analysis':'first_180','threshold':t,'direction':d,'metric':'qualify_both_same_directions_fraction','usable_permutations':20,'null_at_least_observed':0 if v['qualify_both_same_directions_fraction']>0 else 20,'empirical_upper_tail_p':1/21 if v['qualify_both_same_directions_fraction']>0 else 1})
write('split_half_permutation_comparison.csv',comparison)
(out/'complete.json').write_text(json.dumps({'cell':'seed17_B_250','analysis':'first_180','primary_reconstruction_verified':True,'model_rng_unchanged':True,'permutations':20,'note':'unconditional control reconstructed exactly from verified zero selection counts'})+'\n')
# Publish the verified pilot until the running supervisor merges further cells.
for f in out.glob('*.csv'):(P/f.name).write_bytes(f.read_bytes())
print('pilot outputs refreshed for 20 zero-selection permutations')
