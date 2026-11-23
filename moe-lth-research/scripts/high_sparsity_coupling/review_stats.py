"""Paired common-population contrasts and explicit immutable diagnostic identities."""
import hashlib,json
import numpy as np
import pandas as pd

def cache_identity(*,track,condition,step,checkpoint,components,references,probe,pairs,implementation,configuration):
 payload=dict(schema=2,track=track,condition=condition,global_recovery_step=step,checkpoint=checkpoint,components=components,references=references,probe=probe,fixed_pair_identities=pairs,implementation=implementation,configuration=configuration)
 return hashlib.sha256(json.dumps(payload,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest(),payload

def paired_contrast(frames,metric='pearson'):
 """Each frame is indexed by (layer,expert); equal expert population in all 4 cells."""
 ids=set.intersection(*(set(f.index[f[metric].notna() & f['n'].ge(8)]) for f in frames))
 ids=sorted(ids);values=[];counts=[]
 for f in frames:
  z=f.loc[ids];values.append(z[metric].to_numpy());counts.append(z['n'].to_numpy())
 def summary(weighted):
  means=[float(np.average(v,weights=n)) if len(v) and n.sum() else None for v,n in zip(values,counts)] if weighted else [float(v.mean()) if len(v) else None for v in values]
  a,b,c,d=means
  return dict(sparse_initial=a,sparse_final=b,dense_initial=c,dense_final=d,sparse_change=b-a if a is not None else None,dense_change=d-c if c is not None else None,endpoint_difference=b-d if b is not None else None,difference_in_changes=(b-a)-(d-c) if a is not None else None)
 return dict(n_valid_units=len(ids),n_excluded_units=128-len(ids),valid_units=json.dumps(ids),**{f'n_pairs_{label}':int(v.sum()) for label,v in zip(('sparse_initial','sparse_final','dense_initial','dense_final'),counts)}),summary(False),summary(True)
