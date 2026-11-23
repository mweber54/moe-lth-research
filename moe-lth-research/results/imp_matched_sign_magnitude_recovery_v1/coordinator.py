"""CPU coordinator: select one eligible preparation host and dispatch once."""
from pathlib import Path
import fcntl
import json
import os
import shlex
import socket
import subprocess
import sys
import time
import traceback

sys.dont_write_bytecode = True
OUT = Path(__file__).resolve().parent
ROOT = Path.cwd()
sys.path.insert(0, str(OUT))
import gpu_policy
import run

SSH = ['ssh', '-i', str(Path.home()/'.ssh/dataset_b_cluster'),
       '-o', 'BatchMode=yes', '-o', 'ConnectTimeout=15']


def on_host(host, args, **kwargs):
    env_args = ['env', f'{gpu_policy.ENV}={os.environ.get(gpu_policy.ENV, "{}")}', *args]
    if host == socket.getfqdn():
        return subprocess.run(env_args, cwd=ROOT, **kwargs)
    return subprocess.run([*SSH, host, 'cd ' + shlex.quote(str(ROOT)) + ' && ' +
                           shlex.join(env_args)], **kwargs)


def select_host():
    for host in gpu_policy.HOSTS:
        if (OUT/'STOP.json').exists():
            return None
        args = [str(ROOT/'.venv-highres/bin/python'), str(OUT/'gpu_policy.py'),
                'probe', 'preparation_selection']
        try:
            result = on_host(host, args, capture_output=True, text=True, timeout=30)
            if result.returncode:
                raise RuntimeError(f"probe exit {result.returncode}: {result.stderr.strip()}")
            decision = json.loads(result.stdout)
            if decision['host'] != host:
                raise RuntimeError(f"host identity mismatch: {decision['host']}")
            if decision['accepted']:
                return host
        except (OSError, ValueError, KeyError, subprocess.SubprocessError, RuntimeError) as exc:
            gpu_policy.unavailable(host, 'preparation_selection', repr(exc))
            run.save(OUT/'coordinator_status.json', dict(
                status='HOST_PROBE_FAILED', host=host, checked_utc=run.utc(), reason=repr(exc)))
    return None


def main():
    (OUT/'locks').mkdir(exist_ok=True)
    with (OUT/'locks/coordinator.lock').open('a') as lock:
        try:
            fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise SystemExit('Another coordinator holds the lock; no STOP marker written')
        try:
            if (OUT/'STOP.json').exists():
                return
            run.check_authority()
            gpu_policy.policy()
            assert not any((OUT/name).exists() for name in ('preparation_started.json','registration.json','job_manifest.csv','variants','launch.json','jobs')), 'Prior preparation or dispatch requires manual review; no retry'
            while not (OUT/'STOP.json').exists():
                host = select_host()
                if host is None:
                    run.save(OUT/'coordinator_status.json', dict(
                        status='WAITING_FOR_ELIGIBLE_HOST', host=socket.getfqdn(), pid=os.getpid(),
                        checked_utc=run.utc(), new_training_runs=0))
                    time.sleep(30)
                    continue
                if (OUT/'STOP.json').exists():
                    return
                run.storage_check('before_preparation')
                args = [str(ROOT/'.venv-highres/bin/python'), '-u', str(OUT/'run.py'), 'prepare']
                result = on_host(host, args)
                if result.returncode == 75:
                    continue  # Admission changed before preparation began.
                if result.returncode:
                    raise RuntimeError(f'Preparation on {host} failed with {result.returncode}; no retry')
                if (OUT/'STOP.json').exists():
                    return
                subprocess.run([sys.executable, '-u', str(OUT/'dispatch.py')], cwd=ROOT, check=True)
                run.save(OUT/'coordinator_status.json', dict(
                    status='DISPATCHED_REGISTERED_QUEUE', host=socket.getfqdn(),
                    preparation_host=host, pid=os.getpid(), finished_utc=run.utc(),
                    continuation_authorized=False))
                return
        except BaseException as exc:
            if not (OUT/'STOP.json').exists():
                run.save(OUT/'STOP.json', dict(reason='Preparation/coordinator failure: '+repr(exc),
                    traceback=traceback.format_exc(), utc=run.utc(), no_automatic_retry=True))
            raise


if __name__ == '__main__':
    main()
