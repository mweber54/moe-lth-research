import sys,json,copy
from pathlib import Path
import pandas as pd
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts/high_sparsity_coupling'))
from review_stats import paired_contrast,cache_identity

def test_common_population_uses_four_cells():
 def frame(vals):return pd.DataFrame({'pearson':vals,'n':[10,10]},index=pd.MultiIndex.from_tuples([(0,0),(0,1)]))
 a,b,c,d=[frame(v) for v in ([.1,.9],[.3,.9],[.1,None],[.4,None])]
 coverage,equal,weighted=paired_contrast([a,b,c,d]);assert coverage['n_valid_units']==1
 assert abs(equal['sparse_change']-.2)<1e-12 and abs(equal['dense_change']-.3)<1e-12
 assert abs(equal['difference_in_changes']+.1)<1e-12
 # Previously, separate sparse/dense populations give .1-.3=-.2.
 assert abs((b.pearson.mean()-a.pearson.mean())-(d.pearson.mean()-c.pearson.mean())+.2)<1e-12

def test_identity_changes_with_all_provenance_axes():
 base=dict(track='historical',condition='e32_seed7_dense',step=5000,checkpoint={'path':'a','sha256':'1'},components={'expert':'2'},references={'same_step_dense':'3'},probe={'tokens':'4','positions':'5'},pairs={'initial':'6'},implementation='7',configuration={'tf32':False})
 key,payload=cache_identity(**base);assert cache_identity(**copy.deepcopy(base))[0]==key
 for field in base:
  other=copy.deepcopy(base);other[field]={'changed':True} if isinstance(base[field],dict) else ('rerun' if field=='track' else 999 if field=='step' else 'changed')
  assert cache_identity(**other)[0]!=key,field
