"""Prespecified paired contrasts. Missing arms remain unavailable; R50 descriptive."""
from pilot_common import *
import itertools,math

def main():
 jobs=pd.read_csv(OUT/'manifests/run_manifest.csv');lookup=jobs.set_index('condition_id').to_dict('index');rows=[]
 for cid,r in lookup.items():
  if r['job_type']!='sparse' or r['status'] not in ('REUSED','COMPLETE'):continue
  dense=lookup[r['dense_condition_id']]
  if dense['status'] not in ('REUSED','COMPLETE'):continue
  sparse=read(Path(r['artifact_path'])/'summary.json')['final_validation_loss'];dl=read(Path(dense['artifact_path'])/'summary.json')['final_validation_loss'];rows.append({'condition_id':cid,**{k:r[k] for k in ['arm','seed','router_age','sparsity','mask_type']},'sparse_loss':sparse,'dense_loss':dl,'ticket_gap':sparse-dl})
 cols=['condition_id','arm','seed','router_age','sparsity','mask_type','sparse_loss','dense_loss','ticket_gap'];f=pd.DataFrame(rows,columns=cols);csv(OUT/'analysis/pilot_observed_endpoints.csv',f);values={(r.arm,int(r.seed),int(r.router_age),float(r.sparsity),r.mask_type):r for r in f.itertuples()};contrasts=[]
 def record(name,seed,metric,terms,age=None,sp=None,arm=None,role='primary'):
  missing=[key for coef,key in terms if key not in values];v=sum(coef*getattr(values[key],metric) for coef,key in terms) if not missing else None;contrasts.append(dict(contrast=name,seed=seed,metric=metric,router_age=age,sparsity=sp,arm=arm,role=role,value=v,available=not missing,missing_endpoints=json.dumps(missing)))
 for seed,age,sp,metric in itertools.product(SEEDS,AGES,SP,['sparse_loss','dense_loss','ticket_gap']):
  keys={arm:(arm,seed,age,sp,'learned') for arm in ARMS};role='descriptive_R50' if age==50 else 'primary'
  for name,pos,neg in [('C_minus_A','C','A'),('B_minus_A','B','A'),('D_minus_B','D','B')]:record(name,seed,metric,[(1,keys[pos]),(-1,keys[neg])],age,sp,role=role)
  record('source_x_recovery_interaction',seed,metric,[(1,keys['D']),(-1,keys['B']),(-1,keys['C']),(1,keys['A'])],age,sp,role=role)
 for seed,arm,metric in itertools.product(SEEDS,ARMS,['sparse_loss','dense_loss','ticket_gap']):
  for sp in SP:record('R100_minus_R0',seed,metric,[(1,(arm,seed,100,sp,'learned')),(-1,(arm,seed,0,sp,'learned'))],sp=sp,arm=arm)
  record('age_effect_95_minus_75',seed,metric,[(1,(arm,seed,100,.95,'learned')),(-1,(arm,seed,0,.95,'learned')),(-1,(arm,seed,100,.75,'learned')),(1,(arm,seed,0,.75,'learned'))],arm=arm)
 for seed,arm,sp in itertools.product(SEEDS,['A','D'],SP):
  for age in [0,100]:record('random_minus_learned_mask_advantage',seed,'sparse_loss',[(1,(arm,seed,age,sp,'random')),(-1,(arm,seed,age,sp,'learned'))],age,sp,arm)
  record('mask_advantage_R100_minus_R0',seed,'sparse_loss',[(1,(arm,seed,100,sp,'random')),(-1,(arm,seed,100,sp,'learned')),(-1,(arm,seed,0,sp,'random')),(1,(arm,seed,0,sp,'learned'))],sp=sp,arm=arm)
 c=pd.DataFrame(contrasts);csv(OUT/'analysis/pilot_paired_contrasts.csv',c);summary=[]
 for key,g in c.groupby(['contrast','metric','router_age','sparsity','arm','role'],dropna=False):
  v=g.value.dropna();n=len(v);sd=v.std(ddof=1);sem=sd/math.sqrt(n) if n else np.nan;mean=v.mean();summary.append(dict(zip(['contrast','metric','router_age','sparsity','arm','role'],key))|{'n_paired_seeds':n,'mean':mean,'sample_sd':sd,'sem':sem,'positive_seeds':int((v>0).sum()),'negative_seeds':int((v<0).sum()),'zero_seeds':int((v==0).sum()),'full_five_seed_contrast':n==5,'reference_t95_lower':mean-2.776445105*sem if n==5 else None,'reference_t95_upper':mean+2.776445105*sem if n==5 else None})
 csv(OUT/'analysis/pilot_contrast_summary.csv',summary);print('Paired contrast analysis:',len(f),'observed sparse endpoints;',int(c.available.sum()),'available seed contrasts')
if __name__=='__main__':main()
