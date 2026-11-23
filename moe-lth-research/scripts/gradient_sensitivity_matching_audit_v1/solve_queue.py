from common import *
import concurrent.futures,traceback

def solve(item):
 seed,name,design=item;p=D/'solver_inputs'/f'seed{seed}'/(name+'.npz');meta=read(D/'jobs'/f'seed{seed}'/(name+'.json'))
 assert sha(p)==meta['graph']['solver_input_sha256']
 out=D/'solver_outputs'/f'seed{seed}'/(name+'.'+design);out.parent.mkdir(parents=True,exist_ok=True)
 assert not out.with_suffix(out.suffix+'.json').exists()
 start=time.monotonic()
 try:
  result=subprocess.run([str(ROOT/'.venv-highres/bin/python'),str(HERE/'solver.py'),str(p),design,str(out)],cwd=ROOT,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True,timeout=45,env={**os.environ,'OMP_NUM_THREADS':'1','OPENBLAS_NUM_THREADS':'1'})
  (Path(str(out)+'.log')).write_text(result.stdout)
  if not Path(str(out)+'.json').exists():save(Path(str(out)+'.json'),{'status':'PROCESS_FAILURE','returncode':result.returncode,'incumbent_pairs':None,'upper_bound_pairs':None,'message':result.stdout[-4000:]})
 except subprocess.TimeoutExpired as e:
  (Path(str(out)+'.log')).write_text(str(e.stdout or ''))
  save(Path(str(out)+'.json'),{'status':'WALL_TIMEOUT','incumbent_pairs':None,'upper_bound_pairs':None,'message':'45s parent wall limit; no infeasibility conclusion'})
 record=read(Path(str(out)+'.json'));record.update(seed=seed,parameter=name,design=design,input_sha256=sha(p),wall_seconds=time.monotonic()-start,graph_restricted=meta['graph']['solver_graph_restricted'],complete_admissible_edges=meta['graph']['complete_admissible_edges'],original_algorithm_pairs=int(np.load(p)['original_pairs']))
 if str(record['status'])=='2':record['warning']='Zero assignment is always feasible; infeasible status is an engineering/numerical issue, not a scientific certificate.'
 save(Path(str(out)+'.json'),record)
 print('SOLVER',seed,name,design,record['status'],record['incumbent_pairs'],record['upper_bound_pairs'],flush=True)
 return seed,name,design

def main():
 config=frozen();hold=g.lock(D/'locks/solvers.lock',False);subset=rows(D/'fixed_solver_subset.csv')
 waiting=[(int(r['seed']),r['parameter'],d) for r in subset for d in ('B','C')];done=[];pending={};start=time.monotonic()
 with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
  while waiting or pending:
   assert time.monotonic()-start<10800,'Fixed audit queue deadline exceeded'
   for item in waiting.copy():
    if len(pending)>=2:break
    seed,name,design=item
    if (D/'jobs'/f'seed{seed}'/(name+'.json')).exists():pending[pool.submit(solve,item)]=item;waiting.remove(item)
   for future in list(pending):
    if future.done():done.append(future.result());pending.pop(future)
   save(D/'solver_queue_status.json',{'done':len(done),'waiting':len(waiting),'running':len(pending),'heartbeat':time.time(),'pid':os.getpid()})
   if waiting or pending:time.sleep(2)
 save(D/'solver_queue_complete.json',{'jobs':len(done),'status':'COMPLETE','all_selected_tensors_retained':True})
if __name__=='__main__':main()
