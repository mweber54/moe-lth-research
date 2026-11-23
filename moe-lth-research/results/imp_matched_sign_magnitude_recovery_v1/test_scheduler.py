"""CPU-only scheduler dry-run; all files, hosts and workers are simulated."""
import contextlib
import fcntl
import io
import json
import os
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest import mock

import coordinator
import dispatch
import gpu_policy
import run


class SchedulerTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.out = Path(self.tmp.name)
        self.stack = contextlib.ExitStack()
        for module in (coordinator, dispatch, run):
            self.stack.enter_context(mock.patch.object(module, 'OUT', self.out))
        self.addCleanup(self.tmp.cleanup)
        self.addCleanup(self.stack.close)
        self.stack.enter_context(mock.patch.dict(os.environ, {}, clear=False))
        os.environ.pop(gpu_policy.ENV, None)

    def test_idle_policy_and_shared_requires_committed_authorization(self):
        config = gpu_policy.policy()
        busy = dict(memory_total_mib=95830, memory_used_mib=1000,
                    available_vram_mib=94830, utilization_percent=0,
                    external_compute_pids=[123])
        self.assertEqual(gpu_policy.decide(gpu_policy.HOSTS[0], config, busy),
                         (False, 'exclusive_allocation_not_authorized'))
        config['exclusive_authorized_hosts'] = list(gpu_policy.HOSTS)
        self.assertEqual(gpu_policy.decide(gpu_policy.HOSTS[0], config, busy),
                         (False, 'exclusive_gpu_not_idle'))
        os.environ[gpu_policy.ENV] = '{"allow_shared":true}'
        with self.assertRaises(ValueError):
            gpu_policy.policy()

    def test_resource_threshold_and_audit(self):
        config = gpu_policy.policy()
        config['min_free_mib'][gpu_policy.HOSTS[0]] = 20000
        gpu = dict(memory_total_mib=95830, memory_used_mib=90000,
                   available_vram_mib=5830, utilization_percent=0, external_compute_pids=[])
        result = gpu_policy.admit('test', out=self.out, host=gpu_policy.HOSTS[0],
                                  config=config, query=lambda: gpu)
        self.assertEqual(result['reason'], 'insufficient_free_vram')
        saved = list((self.out/'gpu_admission').glob('*.json'))
        self.assertEqual(len(saved), 1)
        self.assertEqual(json.loads(saved[0].read_text()), result)

    def test_shared_policy_uses_explicit_committed_attestation(self):
        authorization = {
            'experiment': gpu_policy.OUT.name,
            'cluster_gpu_sharing_authorized': True,
            'preregistration_amended_for_shared_gpu': True,
            'preregistration_commit': 'b4e9451bd9de867bdca8e6f996bf93f62955de33',
            'cluster_authorization_reference': 'test approval',
            'protocol_amendment_text': 'test amendment',
            'hosts': [gpu_policy.HOSTS[0]],
        }
        data = json.dumps(authorization).encode()
        (self.out/'authorization.json').write_bytes(data)
        os.environ[gpu_policy.ENV] = json.dumps({
            'allow_shared': True, 'authorization_path': 'authorization.json',
            'authorization_commit': 'a'*40, 'min_free_mib': {gpu_policy.HOSTS[0]: 53248}})
        def git(args, **kwargs):
            return str(self.out)+'\n' if 'rev-parse' in args else data
        with mock.patch.object(gpu_policy, 'ROOT', self.out), \
             mock.patch.object(gpu_policy.subprocess, 'check_output', side_effect=git):
            config = gpu_policy.policy()
        gpu = dict(memory_total_mib=95830, memory_used_mib=1000, available_vram_mib=94830,
                   utilization_percent=50, external_compute_pids=[123])
        self.assertTrue(gpu_policy.decide(gpu_policy.HOSTS[0], config, gpu)[0])
        self.assertEqual(gpu_policy.decide(gpu_policy.HOSTS[1], config, gpu)[1],
                         'host_not_authorized_for_sharing')

    def test_shared_admission_requires_five_safe_samples(self):
        config = gpu_policy.policy()
        config.update(allow_shared=True, authorization={'hosts': [gpu_policy.HOSTS[0]]},
                      min_free_mib={h: 53248 for h in gpu_policy.HOSTS})
        def check(utils, free=60000):
            samples = iter([dict(memory_total_mib=95830, memory_used_mib=1000,
                                 available_vram_mib=free, utilization_percent=u,
                                 external_compute_pids=[123]) for u in utils])
            sleeps = []
            result = gpu_policy.admit('test', out=self.out, host=gpu_policy.HOSTS[0],
                                      config=config, query=lambda: next(samples),
                                      sleep=lambda t: sleeps.append(t))
            self.assertEqual(len(result['samples']), 5)
            self.assertEqual(sleeps, [2, 2, 2, 2])
            return result
        self.assertTrue(check([40, 50, 60, 70, 80])['accepted'])
        self.assertEqual(check([20, 20, 20, 20, 91])['reason'], 'shared_utilization_limit')
        self.assertEqual(check([61, 61, 61, 61, 61])['reason'], 'shared_mean_utilization_limit')
        self.assertEqual(check([0, 0, 0, 0, 0], free=53247)['reason'], 'insufficient_free_vram')

    def test_query_failure_is_rejected_and_logged(self):
        def fail():
            raise OSError('nvidia-smi unavailable')
        result = gpu_policy.admit('test', out=self.out, host=gpu_policy.HOSTS[0],
                                  config=gpu_policy.policy(), query=fail)
        self.assertFalse(result['accepted'])
        self.assertIn('gpu_query_failed', result['reason'])

    def test_query_uses_reported_free_memory_and_excludes_own_pid(self):
        responses = [subprocess.CompletedProcess([], 0, '1000, 100, 850, 5\n', ''),
                     subprocess.CompletedProcess([], 0, f'{os.getpid()}\n123\n', '')]
        with mock.patch.object(gpu_policy.subprocess, 'run', side_effect=responses):
            gpu = gpu_policy.query_gpu()
        self.assertEqual(gpu['available_vram_mib'], 850)
        self.assertEqual(gpu['external_compute_pids'], [123])

    def test_selection_does_not_depend_on_viper(self):
        def remote(host, *args, **kwargs):
            return subprocess.CompletedProcess([], 0, json.dumps(
                dict(host=host, accepted=host == gpu_policy.HOSTS[0])), '')
        with mock.patch.object(coordinator, 'on_host', side_effect=remote):
            self.assertEqual(coordinator.select_host(), gpu_policy.HOSTS[0])

    def test_selection_skips_blocked_adder(self):
        def remote(host, *args, **kwargs):
            return subprocess.CompletedProcess([], 0, json.dumps(
                dict(host=host, accepted=host == gpu_policy.HOSTS[1])), '')
        with mock.patch.object(coordinator, 'on_host', side_effect=remote):
            self.assertEqual(coordinator.select_host(), gpu_policy.HOSTS[1])

    def test_concurrent_coordinator_does_not_write_stop(self):
        (self.out/'locks').mkdir()
        with (self.out/'locks/coordinator.lock').open('a') as held:
            fcntl.flock(held, fcntl.LOCK_EX | fcntl.LOCK_NB)
            with self.assertRaises(SystemExit):
                coordinator.main()
        self.assertFalse((self.out/'STOP.json').exists())

    def test_restart_after_registration_is_not_a_retry(self):
        (self.out/'registration.json').write_text('{}')
        with mock.patch.object(run, 'check_authority'):
            with self.assertRaises(AssertionError):
                coordinator.main()
        self.assertTrue((self.out/'STOP.json').exists())

    def test_restart_after_preparation_started_is_not_a_retry(self):
        (self.out/'preparation_started.json').write_text('{}')
        with mock.patch.object(run, 'check_authority'), \
             mock.patch.object(coordinator, 'select_host') as select:
            with self.assertRaises(AssertionError):
                coordinator.main()
            select.assert_not_called()
        self.assertTrue((self.out/'STOP.json').exists())

    def test_deferred_preparation_selects_next_host_before_dispatch(self):
        completed = [subprocess.CompletedProcess([], 75), subprocess.CompletedProcess([], 0)]
        with mock.patch.object(run, 'check_authority'), \
             mock.patch.object(run, 'storage_check'), \
             mock.patch.object(run, 'save', side_effect=self.save), \
             mock.patch.object(coordinator, 'select_host', side_effect=gpu_policy.HOSTS), \
             mock.patch.object(coordinator, 'on_host', side_effect=completed) as prepare, \
             mock.patch.object(coordinator.subprocess, 'run') as dispatch_process:
            coordinator.main()
        self.assertEqual(prepare.call_count, 2)
        dispatch_process.assert_called_once()
        self.assertEqual(json.loads((self.out/'coordinator_status.json').read_text())['preparation_host'],
                         gpu_policy.HOSTS[1])

    def test_coordinator_respects_existing_stop(self):
        marker = self.out/'STOP.json'
        marker.write_text('{"reason":"existing"}')
        with mock.patch.object(run, 'check_authority') as authority:
            coordinator.main()
        authority.assert_not_called()
        self.assertEqual(marker.read_text(), '{"reason":"existing"}')

    def test_preparation_lock_and_admission_precede_preparation(self):
        with mock.patch.object(gpu_policy, 'admit', return_value={'accepted': False, 'reason': 'busy'}), \
             mock.patch.object(run, '_prepare') as prepare:
            with self.assertRaises(run.PreparationDeferred):
                run.prepare()
            prepare.assert_not_called()
        (self.out/'STOP.json').write_text('{}')
        with mock.patch.object(gpu_policy, 'admit') as admit:
            with self.assertRaises(RuntimeError):
                run.prepare()
            admit.assert_not_called()

    def test_preparation_lock_and_marker_prevent_concurrent_or_replayed_work(self):
        (self.out/'locks').mkdir()
        with (self.out/'locks/preparation.lock').open('a') as held:
            fcntl.flock(held, fcntl.LOCK_EX | fcntl.LOCK_NB)
            with self.assertRaises(BlockingIOError):
                run.prepare()
        with mock.patch.object(gpu_policy, 'admit', return_value={'accepted': True}), \
             mock.patch.object(run, '_prepare', side_effect=RuntimeError('interrupted')) as prepare:
            with self.assertRaises(RuntimeError):
                run.prepare()
            self.assertTrue((self.out/'preparation_started.json').exists())
            with self.assertRaises(RuntimeError):
                run.prepare()
            prepare.assert_called_once()

    def test_worker_rechecks_before_every_claim(self):
        jobs = [{'job_id': 'a'}, {'job_id': 'b'}]
        decisions = iter([{'accepted': False}, {'accepted': True}, {'accepted': True}])
        admissions = []
        def admit(stage):
            decision = next(decisions)
            admissions.append(decision['accepted'])
            if not decision['accepted']:
                self.assertFalse((self.out/'claims').exists())
            return decision
        def fake_one(row, cache):
            self.assertTrue((self.out/'claims'/row['job_id']/'claim.json').exists())
        with mock.patch.object(run, 'verify', return_value={'amendment_commit': 'test'}), \
             mock.patch.object(run, 'rows', return_value=jobs), \
             mock.patch.object(gpu_policy, 'admit', side_effect=admit), \
             mock.patch.object(run, 'storage_check'), \
             mock.patch.object(run, 'run_one', side_effect=fake_one), \
             mock.patch.object(run.torch.cuda, 'set_device'), \
             mock.patch.object(run.time, 'sleep'), \
             mock.patch.object(run, 'save', side_effect=self.save):
            # worker checks the registered queue length.
            with mock.patch.object(run, 'len', create=True, side_effect=lambda x: 108 if x is jobs else len(x)):
                with contextlib.redirect_stdout(io.StringIO()):
                    run.worker()
        self.assertEqual(admissions, [False, True, True])

    @staticmethod
    def save(path, value):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(value))

    def test_worker_stop_during_wait_does_not_claim_or_replace_marker(self):
        jobs = [{'job_id': 'a'}]
        marker = self.out/'STOP.json'
        def sleep(_):
            marker.write_text('{"reason":"operator stop"}')
        with mock.patch.object(run, 'verify', return_value={'amendment_commit': 'test'}), \
             mock.patch.object(run, 'rows', return_value=jobs), \
             mock.patch.object(gpu_policy, 'admit', return_value={'accepted': False}), \
             mock.patch.object(run.time, 'sleep', side_effect=sleep), \
             mock.patch.object(run, 'save', side_effect=self.save), \
             mock.patch.object(run, 'len', create=True, side_effect=lambda x: 108 if x is jobs else len(x)):
            with self.assertRaises(RuntimeError):
                run.worker()
        self.assertFalse((self.out/'claims').exists())
        self.assertEqual(json.loads(marker.read_text())['reason'], 'operator stop')

    def exercise_partial_dispatch(self, launch_failure=False):
        manifest = [dict(seed=s, order=o, mask_type=m, draw=d, arm=a, cap_updates='2500',
                         passed='True', realized_distance='0', target_distance='0',
                         job_id=f'{s}{o}{m}{d}{a}')
                    for s in (7,17,29) for o in ('A','B') for m in ('imp','random1','random2')
                    for d in (1,2,3) for a in ('M','Bm')]
        def rows(path):
            name = Path(path).name
            if name == 'job_manifest.csv': return manifest
            if name == 'report_check.csv': return [{'actual_training_updates': '0'}]
            if name == 'matrix_invariants.csv': return [{'absolute_distance_mismatch': '0'}]*13824
            return [{'passed': 'True'}]
        def remote(host, args, **kwargs):
            if host == gpu_policy.HOSTS[1]:
                if not launch_failure:
                    return subprocess.CompletedProcess(args, 255, '', 'unreachable')
                if 'subprocess,os,json' in args[-1]:
                    raise subprocess.CalledProcessError(255, args)
            output = json.dumps({'host': host, 'pid': 42}) if 'subprocess,os,json' in args[-1] else 'verified'
            if host == gpu_policy.HOSTS[1]:
                output = 'verified'
            return subprocess.CompletedProcess(args, 0, output, '')
        (self.out/'permutations').mkdir()
        with mock.patch.object(run, 'verify', return_value={'amendment_commit': 'test'}), \
             mock.patch.object(run, 'rows', side_effect=rows), \
             mock.patch.object(run, 'read', return_value={'completed_runs': 54, 'attempted_updates': 135000}), \
             mock.patch.object(run, 'sha', return_value='hash'), \
             mock.patch.object(run, 'table'), \
             mock.patch.object(run, 'storage_check', return_value={}), \
             mock.patch.object(run, 'save', side_effect=self.save), \
             mock.patch.object(coordinator, 'on_host', side_effect=remote), \
             mock.patch.object(gpu_policy, 'unavailable'), \
             mock.patch.object(dispatch.subprocess, 'Popen', return_value=mock.Mock(pid=43)):
            with contextlib.redirect_stdout(io.StringIO()):
                if launch_failure:
                    with self.assertRaises(subprocess.CalledProcessError):
                        dispatch.entry()
                else:
                    dispatch.entry()
        launched = json.loads((self.out/'launch.json').read_text())
        self.assertEqual([x['host'] for x in launched['workers']], [gpu_policy.HOSTS[0]])
        if launch_failure:
            self.assertTrue(json.loads((self.out/'STOP.json').read_text())['no_automatic_retry'])

    def test_partial_dispatch_keeps_eligible_adder(self):
        self.exercise_partial_dispatch()

    def test_partial_launch_failure_stops_without_retry(self):
        self.exercise_partial_dispatch(launch_failure=True)

    def test_dispatch_failure_stops_without_retry_or_replacing_stop(self):
        with mock.patch.object(dispatch, 'main', side_effect=RuntimeError('ambiguous worker launch')) as main, \
             mock.patch.object(run, 'save', side_effect=self.save):
            with self.assertRaises(RuntimeError):
                dispatch.entry()
        main.assert_called_once()
        stop = json.loads((self.out/'STOP.json').read_text())
        self.assertTrue(stop['no_automatic_retry'])
        (self.out/'STOP.json').write_text('{"reason":"operator stop"}')
        with mock.patch.object(dispatch, 'main', side_effect=RuntimeError('failure')), \
             mock.patch.object(run, 'save', side_effect=self.save):
            with self.assertRaises(RuntimeError):
                dispatch.entry()
        self.assertEqual(json.loads((self.out/'STOP.json').read_text())['reason'], 'operator stop')


if __name__ == '__main__':
    unittest.main(verbosity=2)
