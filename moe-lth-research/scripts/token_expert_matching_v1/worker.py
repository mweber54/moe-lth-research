"""Two-host admission and pilot-gated execution; no automatic retry or job termination."""
from run import *
from report import analyze,pilot_decision
import threading
import contextlib
import shutil


def check_seal():
    seal=read(DEST/'seal.json')
    if sha(DEST/'protocol.md')!=seal['protocol_sha256'] or sha(DEST/'registration.json')!=seal['registration_sha256']:raise RuntimeError('Frozen design changed')
    for path,expected in seal['implementation'].items():
        if sha(ROOT/path)!=expected:raise RuntimeError('Sealed implementation changed '+path)
    if sha(OUT/'input_hashes.json')!=seal['source_input_inventory_sha256']:raise RuntimeError('Source inventory changed')
    frozen()


def worker():
    host=socket.getfqdn();torch.set_num_threads(4);R.configure_device(torch.device('cuda'))
    if host not in HOSTS or not torch.__version__.startswith('2.7.0'):raise RuntimeError('Canonical host/runtime required')
    if not read(DEST/'validation.json')['production_ready']:raise RuntimeError('Validation incomplete')
    manifest=read(DEST/'registration.json');active={'job':None};event=threading.Event()
    def heartbeat():
        while not event.is_set():
            save(DEST/'workers'/f'{host}.heartbeat.json',dict(host=host,pid=os.getpid(),active=active['job'],unix=time.time()))
            event.wait(15)
    with lock(DEST/'workers'/f'{host}.lock',False):
        thread=threading.Thread(target=heartbeat,daemon=True);thread.start()
        try:
            while not (DEST/'STOP.json').exists():
                check_seal()
                selected=None
                with lock(DEST/'dispatch.lock'):
                    decision=pilot_decision()
                    if decision is not None and not decision['expand']:
                        save(DEST/'STOP.json',dict(reason='Pilot identity replay changes learning regime beyond registered feasibility limits',scientific_stop=True,pilot=decision));break
                    if decision is not None and not (DEST/'expansion_estimate.json').exists():
                        pilot_results=[read(DEST/'attempts'/x['id']/'result.json') for x in manifest['jobs'] if x['pilot']]
                        measured=statistics.mean(x['wall_seconds'] for x in pilot_results)
                        save(DEST/'expansion_estimate.json',dict(measured_pilot_runs=6,mean_run_seconds=measured,remaining_runs=69,remaining_gpu_hours=69*measured/3600,minimum_free_storage_gib=read(DEST/'resource_estimate.json')['suggested_free_storage_gib'],free_bytes=shutil.disk_usage(DEST).free))
                    if decision is not None and shutil.disk_usage(DEST).free<read(DEST/'resource_estimate.json')['suggested_free_storage_gib']*1024**3:raise RuntimeError('Expansion storage admission failed')
                    if torch.cuda.mem_get_info()[0]>=24*1024**3:
                        for job in manifest['jobs']:
                            if not job['pilot'] and decision is None:continue
                            if (DEST/'claims'/f'{job["id"]}.json').exists():continue
                            if job['kind']!='donor' and not (DEST/'attempts'/f'{job["variant_id"]}_donor'/'result.json').exists():continue
                            selected=job
                            save(DEST/'claims'/f'{job["id"]}.json',dict(host=host,pid=os.getpid(),unix=time.time()))
                            break
                    if selected is None and all((DEST/'attempts'/x['id']/'result.json').exists() for x in manifest['jobs']):break
                if selected is None:event.wait(15);continue
                active['job']=selected['id']
                variant=next(x for x in manifest['variants'] if x['id']==selected['variant_id'])
                for path in (variant['initial'],variant['mask']):
                    if path and sha(path)!=read(OUT/'input_hashes.json')[path]:raise RuntimeError('Changed source '+path)
                log=DEST/'logs'/f'{selected["id"]}.log';log.parent.mkdir(exist_ok=True)
                with log.open('a',buffering=1) as stream,contextlib.redirect_stdout(stream),contextlib.redirect_stderr(stream):run(variant,selected['kind'],selected['draw'])
                with lock(DEST/'report.lock'):analyze()
                torch.cuda.empty_cache();active['job']=None
        except BaseException as error:
            save(DEST/'STOP.json',dict(reason=repr(error),host=host,active=active['job'],traceback=traceback.format_exc()))
            raise
        finally:
            with lock(DEST/'report.lock'):analyze()
            event.set();thread.join(timeout=2)
            save(DEST/'workers'/f'{host}.finished.json',dict(pid=os.getpid(),unix=time.time(),stopped=(DEST/'STOP.json').exists()))


if __name__=='__main__':
    import statistics
    worker()
