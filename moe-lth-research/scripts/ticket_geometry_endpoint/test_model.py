"""CPU integration gate to execute inside the installed project environment."""
import unittest
import torch
import run as R
from moe_lth.models import TinyMoELanguageModel
class Integration(unittest.TestCase):
 def test_shapes_preservation_and_identical_dense(self):
  torch.set_num_threads(2);torch.manual_seed(3)
  R.torch=torch;R.TinyMoELanguageModel=TinyMoELanguageModel
  cfg=dict(vocab_size=16,max_seq_len=8,num_layers=1,num_heads=2,d_model=64,num_experts=2,expert_hidden_size=96,dropout=0.,top_k=1,capacity_factor=1.25)
  model=TinyMoELanguageModel(cfg).eval();before={k:v.clone() for k,v in model.state_dict().items()}
  tokens=torch.randint(0,16,(4,8));targets=tokens.roll(-1,1);inputs,assign=R.capture(model,tokens);R.dense_inputs_global=inputs
  fixed=R.stateselection(model,inputs);rows=[]
  with torch.no_grad(): layers=R.analyze_state(model,model,model,model,inputs,assign,inputs,assign,fixed,tokens,targets,'toy',0,rows)
  for r in rows:
   if r['kind']=='geometry':self.assertAlmostEqual(r['dense_gram_distance'],0)
   if r['kind']=='function':self.assertAlmostEqual(r['dense_relative_output_error'],0)
  self.assertTrue(any(r.get('selection')=='initial_proposed' for r in rows))
  self.assertTrue(all(torch.equal(v,model.state_dict()[k]) for k,v in before.items()))
  self.assertAlmostEqual(layers[0]['dense_hidden_cka'],1)
if __name__=='__main__':unittest.main()
