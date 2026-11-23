"""100-attempt seed17 timing run; engineering only, never a scientific endpoint."""
from common import *
from engine import run_segment
def main():
 torch.set_num_threads(4);R.configure_device(torch.device('cuda'));host=socket.getfqdn()
 assert host in HOSTS and torch.cuda.mem_get_info()[0]>24*1024**3
 v=next(v for v in read(OUT/'manifest.json')['variants'] if v['seed']==17);train,val=batches(v)
 initial=state(load_model_from_checkpoint(v['config']['model'],v['initial'],torch.device('cpu')))
 attempt=OUT/'benchmarks'/host/f'attempt_{time.time_ns()}';attempt.mkdir(parents=True)
 torch.cuda.reset_peak_memory_stats();start=time.monotonic()
 r=run_segment(v,'dense',initial,v['initial'],train[:100],val[:2],attempt/'segment',0,load(v['schedule']))
 result=dict(host=host,seed=17,engineering_only=True,attempt=str(attempt),attempts=100,wall_seconds=time.monotonic()-start,
    native_segment_seconds=r['wall_seconds'],diagnostic_seconds=r['diagnostic_seconds'],evaluation_seconds=r['common_evaluation_seconds'],
    peak_allocated_bytes=torch.cuda.max_memory_allocated(),peak_reserved_bytes=torch.cuda.max_memory_reserved(),unix=time.time())
 save(OUT/'benchmarks'/f'{host}.json',result);print(json.dumps(result),flush=True)
if __name__=='__main__':main()
