import unittest
import numpy as np
from numerics import spectrum,compare,geometry,cka,associations
class Tests(unittest.TestCase):
 def test_gram_scale(self):
  g,_,_=spectrum(np.diag([2.,1.]));r=compare(g,4*g)
  self.assertAlmostEqual(r['gram_distance'],0);self.assertAlmostEqual(r['trace_ratio'],.25)
 def test_gram_rotation_neurons(self):
  w=np.array([[2.,1.],[1.,-3.]]);q=np.array([[0.,1.],[-1.,0.]])
  self.assertTrue(np.allclose(spectrum(w)[0],spectrum(q@w)[0]))
 def test_cka(self):
  x=np.random.default_rng(1).normal(size=(30,4));self.assertAlmostEqual(cka(x,3*x+2),1)
 def test_fixed_pairs(self):
  x=np.arange(20.);s=np.column_stack([x,-x]);a=s.copy();ids=np.zeros((20,1),dtype=int)
  r=associations(s,a,{'fixed':(ids,np.ones_like(ids,dtype=bool))})
  self.assertAlmostEqual(r[0]['pearson'],1);self.assertEqual(r[1]['n'],0);self.assertIsNone(r[1]['pearson'])
 def test_undefined(self):
  g=np.zeros((2,2));self.assertIsNone(compare(g,g)['gram_distance'])
if __name__=='__main__':unittest.main()
