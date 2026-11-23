import unittest, tempfile, json
from pathlib import Path
from support import *

class Tests(unittest.TestCase):
    def records(self): return [{'condition_id':str(i),'status':'COMPLETE'} for i in range(225)]
    def test_complete_parent(self): self.assertTrue(parent_complete(self.records()))
    def test_running_blocks(self):
        r=self.records(); r[0]['status']='RUNNING'; self.assertFalse(parent_complete(r))
    def test_failure_blocks(self):
        r=self.records(); r[0]['status']='FAILED_UNCERTAIN'; self.assertFalse(parent_complete(r))
    def test_incomplete_blocks(self): self.assertFalse(parent_complete(self.records()[:-1]))
    def test_duplicate_blocks(self):
        r=self.records(); r[0]=r[1]; self.assertFalse(parent_complete(r))
    def test_grid(self):
        ids=expected_ids(); self.assertEqual(len(ids),320)
        self.assertEqual(sum('_random_' in x for x in ids),80)
        self.assertEqual(sum('_learned_' in x for x in ids),240)
    def test_tampering_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'x'; p.write_text('original'); expected={'x':sha(p)}
            verify_map(Path(d),expected); p.write_text('changed')
            with self.assertRaises(RuntimeError): verify_map(Path(d),expected)
    def test_installed_plan(self):
        if not (OUT/'manifests/job_plan.csv').exists(): self.skipTest('No prepared server plan here')
        plan=rows(OUT/'manifests/job_plan.csv')
        self.assertEqual(len(plan),320)
        self.assertEqual({r['condition_id'] for r in plan},expected_ids())
    def test_exclusive_lock(self):
        import fcntl
        with tempfile.TemporaryDirectory() as d:
            with open(Path(d)/'lock','a') as a,open(Path(d)/'lock','a') as b:
                fcntl.flock(a,fcntl.LOCK_EX|fcntl.LOCK_NB)
                with self.assertRaises(BlockingIOError): fcntl.flock(b,fcntl.LOCK_EX|fcntl.LOCK_NB)
    def test_sparse_worker_target(self):
        s=(Path(__file__).parent/'coordinator.py').read_text()
        self.assertIn('scripts/lower_sparsity_extension/worker_adapter.py',s)
        self.assertNotIn('scripts/nominal_exposure_dataset/worker.py',s)

if __name__=='__main__': unittest.main(verbosity=2)
