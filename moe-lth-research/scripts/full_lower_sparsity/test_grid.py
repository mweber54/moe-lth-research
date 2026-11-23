"""Pure Python exact-grid tests, including uploaded source/dense inventory checks."""
import ast,csv,itertools,unittest
from pathlib import Path
from support import ROOT

def load_grid():
    # Execute only the pure grid definitions, without importing torch.
    tree=ast.parse((Path(__file__).parent/'prepare.py').read_text())
    keep=[]
    for node in tree.body:
        if isinstance(node,ast.Assign) and all(isinstance(t,ast.Name) and t.id in ('ES','SEEDS','AGES','SP') for t in node.targets): keep.append(node)
        if isinstance(node,ast.FunctionDef) and node.name in ('cid','grid'): keep.append(node)
    env={};exec(compile(ast.Module(body=keep,type_ignores=[]),'grid','exec'),env);return env['grid']()
class Tests(unittest.TestCase):
    def setUp(self): self.r=load_grid()
    def test_exact_product(self):
        actual={(x['num_experts'],x['seed'],x['router_age'],x['sparsity']) for x in self.r}
        expected=set(itertools.product((8,12,16,24,32),(7,17,29,43,61),range(0,101,10),(.5,.55,.6,.65)))
        self.assertEqual(actual,expected);self.assertEqual(len(self.r),1100)
    def test_unique_ids(self): self.assertEqual(len({r['condition_id'] for r in self.r}),1100)
    def test_all_learned(self): self.assertTrue(all(r['mask_type']=='learned' for r in self.r))
    def test_budgets(self): self.assertTrue(all(r['source_attempts']==r['recovery_attempts']==2500 for r in self.r))
    def test_checkpoints(self): self.assertTrue(all(r['router_checkpoint_step']==25*r['router_age'] for r in self.r))
    def test_dense_sharing(self): self.assertEqual(len({r['dense_condition_id'] for r in self.r}),275)
    def test_reuse_counts(self):
        reused=[r for r in self.r if r['num_experts']==32 and r['router_age'] in (0,50,100)]
        self.assertEqual(len(reused),60);self.assertEqual(len(self.r)-len(reused),1040)
    def test_inventories(self):
        base=ROOT/'results/final_dataset_a'
        for rel,field in [('inputs/source_checkpoint_manifest.csv','num_experts'),('canonical/dataset_a_dense.csv','architecture')]:
            p=base/rel
            if not p.exists(): self.skipTest('Server inventories not present')
            with p.open() as h: rows=list(csv.DictReader(h))
            keys={(int(r[field]),int(r['seed']),int(r['router_age'])) for r in rows}
            self.assertEqual(keys,set(itertools.product((8,12,16,24,32),(7,17,29,43,61),range(0,101,10))))
            self.assertEqual(len(rows),275)
if __name__=='__main__': unittest.main(verbosity=2)
