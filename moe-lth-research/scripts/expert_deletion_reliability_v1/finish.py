"""Finish only this authorized study: wait for32-batch completion, analyze, report, audit."""
import hashlib,json,os,subprocess,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];D=ROOT/'results/expert_deletion_reliability_v1'


def save(path,x):
    q=path.with_suffix('.tmp');q.write_text(json.dumps(x,indent=2)+'\n');q.replace(path)


def main():
    started=time.time();env=dict(os.environ,OPENBLAS_NUM_THREADS='2',OMP_NUM_THREADS='2')
    while not (D/'measurement_completion.json').exists():
        save(D/'finisher_status.json',dict(stage='waiting_for_registered_measurements',unix=time.time(),pid=os.getpid()))
        time.sleep(10)
    assert json.loads((D/'measurement_completion.json').read_text())['complete']
    accounting=json.loads((D/'compute_accounting.json').read_text())
    transition=json.loads((D/'execution_transition.json').read_text());compat=json.loads((D/'host_compatibility.json').read_text())
    accounting['execution_hosts']=['viper.cs.kent.edu','adder.cs.kent.edu']
    accounting['viper_interrupted_worker_elapsed_seconds']=transition['viper_elapsed_seconds_at_interrupt']
    accounting['adder_resumed_measurement_wall_seconds']=accounting['measure_wall_seconds']
    accounting['host_compatibility_wall_seconds']=compat['wall_seconds']
    accounting['combined_main_worker_elapsed_seconds']=transition['viper_elapsed_seconds_at_interrupt']+accounting['measure_wall_seconds']
    accounting['host_assignment']={'viper_completed_cache_files':transition['existing_cache_files'],'adder_completed_cache_files':'all remaining primary cache files; registered profile also rechecked on Adder'}
    accounting['cross_host_CE_and_sequence_statistics_identical']=compat['passed']
    save(D/'compute_accounting.json',accounting)
    for stage in ('analyze','functional_uncertainty','report','verify'):
        save(D/'finisher_status.json',dict(stage=stage,unix=time.time(),pid=os.getpid()))
        with (D/(stage+'.log')).open('w') as output:
            subprocess.run([sys.executable,str(Path(__file__).with_name(stage+'.py'))],cwd=ROOT,env=env,stdout=output,stderr=subprocess.STDOUT,check=True)
    save(D/'finisher_status.json',dict(stage='complete',wall_seconds=time.time()-started,unix=time.time(),pid=os.getpid()))

if __name__=='__main__':main()
