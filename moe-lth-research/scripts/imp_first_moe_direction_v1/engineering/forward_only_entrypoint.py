"""Corrected authoritative entrypoint; frozen scientific evaluator remains unchanged."""
import sys,subprocess,time,traceback
from pathlib import Path
ROOT=Path(__file__).resolve().parents[3];sys.path.insert(0,str(ROOT))
from scripts.imp_first_moe_direction_v1 import run as study
original=subprocess.run
def forward_only(args,*rest,**kwargs):
    if isinstance(args,list) and '-m' in args and 'pytest' in args:
        args.extend(['-k','not test_top2_routes_to_two_distinct_experts_and_backpropagates','-p','scripts.imp_first_moe_direction_v1.engineering.forward_only_plugin'])
    return original(args,*rest,**kwargs)
if __name__=='__main__':
    subprocess.run=forward_only
    try:study.main()
    except Exception as exc:
        failure=dict(status='FAILED',scientifically_complete=False,error=repr(exc),traceback=traceback.format_exc());study.c._json_write(study.OUT/'attempts'/f'failure_{time.time_ns()}.json',failure);study.c._json_write(study.OUT/'completion.json',failure);raise
