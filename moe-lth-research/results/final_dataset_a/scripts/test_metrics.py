"""Metric regression tests use the existing implementation; no model execution."""
import tempfile,json,unittest
from pathlib import Path
from build import metrics
from analyze import win
import pandas as pd
class MetricTests(unittest.TestCase):
 def test_exact_auc_and_endpoint(self):
  with tempfile.TemporaryDirectory() as d:
   p=Path(d);curve=[{'step':s,'loss':2+s/2500} for s in range(0,2501,50)]
   (p/'metadata.json').write_text('{}');(p/'summary.json').write_text(json.dumps({'final_validation_loss':3}));(p/'metrics.jsonl').write_text('\n'.join(map(json.dumps,curve)))
   m,s,initial,final,auc=metrics(p);self.assertEqual(initial,2);self.assertEqual(final,3);self.assertAlmostEqual(auc,2812.5)
 def test_duplicate_curve_step_rejected(self):
  with tempfile.TemporaryDirectory() as d:
   p=Path(d);curve=[{'step':s,'loss':2.} for s in range(0,2501,50)];curve.insert(2,curve[1]);(p/'metadata.json').write_text('{}');(p/'summary.json').write_text('{"final_validation_loss":2}');(p/'metrics.jsonl').write_text('\n'.join(map(json.dumps,curve)))
   with self.assertRaises(AssertionError):metrics(p)
 def test_numerical_ties_preserved(self):
  f=pd.DataFrame({'router_age':[0,10,20],'ticket_gap':[.1,.1+1e-13,.2]});self.assertEqual(win(f,'ticket_gap'),[0,10])
 def test_dense_pairing_changes_gap(self):
  sparse=2.;same_age_dense=1.4;different_age_dense=1.2;self.assertAlmostEqual(sparse-same_age_dense,.6);self.assertNotEqual(sparse-same_age_dense,sparse-different_age_dense)
if __name__=='__main__':unittest.main()
