from common import *
from graph import *
import argparse,concurrent.futures,threading,resource,traceback,warnings

EXPOSURE=None;EXPECTED=None;COORDINATES=None;SUBSET=None

def overlap(seed,name,f,A,shape,small,strata,sign):
 h=[];l=[]
 for idx in np.array_split(small,50):
  order=idx[np.lexsort((idx,A[idx]))];k=int(math.ceil(len(idx)*.25));l.extend(order[:k]);h.extend(order[-k:])
 h=np.array(h);l=np.array(l);r=balance(seed,name,h,l,f,A,shape,'OVERLAP_QUARTILES')
 r['population']='descriptive within-magnitude-stratum A quartiles; not solver restrictions';r['positive_sign_fraction_H']=float(np.mean(sign[h]>0));r['positive_sign_fraction_L']=float(np.mean(sign[l]>0))
 rh=np.bincount(h//shape[1],minlength=shape[0]);rl=np.bincount(l//shape[1],minlength=shape[0]);r['row_TV']=float(np.abs(rh-rl).sum()/(2*len(h)))
 for label,idx in [('H',h),('L',l)]:
  for q in (.1,.5,.9):r[f'A_{label}_q{q}']=float(np.quantile(A[idx],q))
 return r

def job(task):
 seed,name,values,A,A0,A1=task;started=time.monotonic();meta=metadata(seed,name)
 shape,v,f,rms,small,strata,sign=features(values)
 relrows,reliability=measurement.reliability(seed,name,v[2],A0,A1,EXPOSURE,read(OLD/'registration.json')['reliability'])
 old_r=EXPECTED['reliability'][name];assert reliability['eligible']==(old_r['eligible']=='True')
 assert abs(reliability['median_stratum_rho']-float(old_r['median_stratum_rho']))<1e-12
 captured=[];real=original.cycles_from_pairs
 def capture(pairs,columns,target):captured.extend(pairs);return real(pairs,columns,target)
 original.cycles_from_pairs=capture
 try:_,h,l,summary,_,_,_=original.tensor_match((seed,name,values,A))
 finally:original.cycles_from_pairs=real
 old=EXPECTED['matching'][name];assert int(old['candidate_pairs'])==len(captured)==summary['candidate_pairs'];assert int(old['cycle_pairs'])==len(h)==summary['cycle_pairs'];assert (old['included']=='True')==summary['included']
 assert np.array_equal(h,COORDINATES['H'][name]) and np.array_equal(l,COORDINATES['L'][name])
 gh=np.array([x['h'] for x in captured],dtype=np.int64);gl=np.array([x['l'] for x in captured],dtype=np.int64)
 assert len(np.union1d(gh,gl))==2*len(gh)
 attr,fh,fl,cost=attrition_and_graph(seed,name,f,A,shape,small,strata,sign)
 assert np.all(strata[fh]==strata[fl]) and np.all(sign[fh]==sign[fl]) and np.all(fh//shape[1]==fl//shape[1])
 assert np.all(A[fh]>=2*A[fl]) and np.all(A[fh]>A[fl]) and np.all(np.abs(f[fh]-f[fl])<=np.array([.005,.1,.1]))
 edgeset=set(zip(fh.tolist(),fl.tolist()));assert all((int(a),int(b)) in edgeset for a,b in zip(gh,gl))
 a=balance(seed,name,h,l,f,A,shape,'A');a['method']='original deterministic greedy/cycle matcher'
 assert a['scientifically_included']==summary['included'];a['solver_certainty']='heuristic, no maximum certificate'
 with warnings.catch_warnings(record=True) as ws:ch,cl,lp=lp_C(gh,gl,f,shape)
 c=balance(seed,name,ch,cl,f,A,shape,'C');c['method']='LP on original disjoint greedy pairs, then integer floor';c['solver_certainty']='restricted-graph LP bound; floored candidate, no binary maximum certificate';c['LP_status']=lp['status'];c['LP_upper_bound_pairs']=lp['upper_bound'];c['LP_fractional_variables']=lp['fractional_variables'];c['LP_seconds']=lp['seconds']
 aggregate=bool(len(h) and all(a.get(x+'_SMD') is not None and abs(a[x+'_SMD'])<=.05 and a[x+'_KS']<=.05 for x in ('magnitude','slope','variation')) and a.get('energy_relative_imbalance') is not None and a['energy_relative_imbalance']<=.005)
 retained_after_aggregate=len(h) if aggregate else 0
 retained_after_contrast=retained_after_aggregate if a['sensitivity_pass'] else 0
 for stage,k,kind in [('disjoint_greedy',len(gh),'actual_disjoint_row_matching'),('exact_column_A',len(h),'actual_disjoint_column_balanced_matching'),('aggregate_balance',retained_after_aggregate,'A_candidate_cumulative_gate'),('sensitivity_contrast',retained_after_contrast,'A_candidate_cumulative_gate'),('dose_coverage',len(h) if a['scientifically_included'] else 0,'A_retained_per_branch')]:
  attr.append(dict(**meta,stage=stage,possible_edges=None,unique_high_coordinates=None,unique_low_coordinates=None,unique_union_coordinates=2*k,actual_pairs=k,count_kind=kind))
 full_diag=component_diagnostics(fh,fl,shape[1]);greedy_diag=component_diagnostics(gh,gl,shape[1])
 graph_diag=dict(**meta,complete_admissible_edges=len(fh),greedy_disjoint_pairs=len(gh),**{'full_'+k:v for k,v in full_diag.items()},**{'greedy_'+k:v for k,v in greedy_diag.items()})
 if (seed,name) in SUBSET:
  sh,sl,sc,restricted=restricted_graph(fh,fl,cost,gh,gl)
  p=D/'solver_inputs'/f'seed{seed}'/(name+'.npz');p.parent.mkdir(parents=True,exist_ok=True)
  np.savez_compressed(p,h=sh,l=sl,eh=f[sh,0]**2,el=f[sl,0]**2,columns=shape[1],shape=np.array(shape),f=f,A=A,strata=strata,sign=sign,complete_edges=len(fh),restricted=restricted,original_pairs=len(h),original_h=h,original_l=l,full_h=fh,full_l=fl)
  p.chmod(0o444);graph_diag['solver_input_sha256']=sha(p);graph_diag['solver_graph_restricted']=restricted;graph_diag['solver_edges']=len(sh)
 p=D/'coordinates'/f'seed{seed}'/(name+'.npz');p.parent.mkdir(parents=True,exist_ok=True);np.savez_compressed(p,A_h=h,A_l=l,C_h=ch,C_l=cl,greedy_h=gh,greedy_l=gl);p.chmod(0o444)
 result=dict(meta=meta,reproduction={'reliability_eligible':reliability['eligible'],'reliability_rho_equal':True,'candidate_pairs_equal':True,'cycle_pairs_equal':True,'coordinates_order_exact':True,'exclusion_equal':True,'production_dose_equal':True},
   attrition=attr,balance=[a,c],overlap=overlap(seed,name,f,A,shape,small,strata,sign),graph=graph_diag,seconds=time.monotonic()-started,maximum_RSS_MiB=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024,coordinate_sha256=sha(p))
 save(D/'jobs'/f'seed{seed}'/(name+'.json'),result)
 return meta

def main():
 global EXPOSURE,EXPECTED,COORDINATES,SUBSET
 ap=argparse.ArgumentParser();ap.add_argument('--seeds',nargs='+',type=int,required=True);args=ap.parse_args();config=frozen()
 resource.setrlimit(resource.RLIMIT_AS,(24*1024**3,24*1024**3))
 host=socket.gethostname();hold=g.lock(D/'locks'/f'descriptive_{host}.lock',False)
 SUBSET={(int(r['seed']),r['parameter']) for r in rows(D/'fixed_solver_subset.csv')}
 provenance=read(OLD/'checkpoint_and_calibration_provenance.json');allmatch=rows(OLD/'matched_group_summary.csv');allrel=rows(OLD/'tensor_reliability_summary.csv');allexp=rows(OLD/'exposure_summary.csv')
 stop=threading.Event()
 def beat():
  while not stop.wait(15):save(D/'workers'/f'{host}.heartbeat.json',{'host':host,'time':time.time(),'pid':os.getpid()})
 threading.Thread(target=beat,daemon=True).start()
 try:
  for seed in args.seeds:
   states=[]
   for step in (250,500,1000):
    cp=provenance['seeds'][str(seed)]['checkpoints'][str(step)];assert sha(ROOT/cp['path'])==cp['sha256'];states.append(load(ROOT/cp['path'])['model'])
   EXPOSURE=[{k:(int(v) if k in ('seed','layer','expert','half','accepted_tokens') else v) for k,v in r.items()} for r in allexp if int(r['seed'])==seed]
   EXPECTED={'matching':{r['parameter']:r for r in allmatch if int(r['seed'])==seed},'reliability':{r['parameter']:r for r in allrel if int(r['seed'])==seed}}
   raw=load(OLD/'candidate_coordinate_lists'/f'seed{seed}.pt');COORDINATES={b:{n:t.numpy() for n,t in raw[b].items()} for b in ('H','L')}
   status=read(OLD/'status'/f'measure_seed{seed}.json');checks=read(OLD/status['attempt']/'measurement_checks.json')
   def tasks():
    for name in g.names(states[0]):
     if (D/'jobs'/f'seed{seed}'/(name+'.json')).exists():continue
     p=OLD/'scores'/f'seed{seed}'/(name+'.pt');assert sha(p)==checks['score_files'][str(p.relative_to(OLD))];s=load(p)
     yield seed,name,np.stack([st[name].numpy() for st in states]),s['A'].numpy().reshape(-1),s['A_half0'].numpy().reshape(-1),s['A_half1'].numpy().reshape(-1)
   with concurrent.futures.ProcessPoolExecutor(max_workers=2) as pool:
    iterator=iter(tasks());pending=[]
    for _ in range(4):
     try:pending.append(pool.submit(job,next(iterator)))
     except StopIteration:break
    count=0
    while pending:
     meta=pending.pop(0).result();count+=1
     save(D/'workers'/f'{host}.progress.json',{'seed':seed,'done_this_attempt':count,'parameter':meta['parameter'],'time':time.time()})
     if count%32==0:print('Audited',seed,count,flush=True)
     try:pending.append(pool.submit(job,next(iterator)))
     except StopIteration:pass
   print('SOURCE COMPLETE',seed,flush=True)
  save(D/'workers'/f'{host}.finished.json',{'seeds':args.seeds,'time':time.time()})
 except BaseException:
  save(D/'workers'/f'{host}.failed.json',{'error':traceback.format_exc(),'time':time.time()});raise
 finally:stop.set()
if __name__=='__main__':main()
