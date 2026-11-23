"""Focused CPU-only preflight tests and segregated native-runner smoke test."""
from pilot_common import *
import unittest,tempfile,copy
from moe_lth.models import TinyMoELanguageModel
from moe_lth.pruning.magnitude_prune import expert_local_magnitude_masks
from moe_lth.pruning.masks import apply_masks_
from moe_lth.pruning.rewind import register_mask_gradient_hooks
from moe_lth.experiments import run_router_age_recovery as R
from worker import protected

class PilotTests(unittest.TestCase):
 def test_budgets_and_age_mapping(self):
  self.assertEqual([attempts(e) for e in (8,12,16,24,32)],[2500,3750,5000,7500,10000]);self.assertEqual(router_step(10000,50),5000);self.assertEqual(router_step(2500,50),1250)
 def test_grid_controls_and_sharing(self):
  g=grid();r=[x for x in g if x['job_type']!='source'];self.assertEqual(len(r),220);self.assertEqual(len({x['condition_id'] for x in g}),225);self.assertEqual(sum(x['job_type']=='dense' for x in r),60);self.assertEqual(sum(x['mask_type']=='random' for x in r),40)
  lookup={x['condition_id']:x for x in r}
  for x in r:
   d=lookup[x['dense_condition_id']];self.assertEqual((d['source_id'],d['router_checkpoint_step'],d['recovery_attempts']),(x['source_id'],x['router_checkpoint_step'],x['recovery_attempts']))
  for arm1,arm2 in [('A','C'),('B','D')]:self.assertEqual(lookup[jobid(arm1,7,0,'dense')]['source_mask_family'],lookup[jobid(arm2,7,100,'dense')]['source_mask_family'])
 def test_mask_counts_and_independent_rng(self):
  m={'a':torch.tensor([[1,0,1],[0,0,1]],dtype=torch.bool),'b':torch.tensor([0,1,0,0],dtype=torch.bool)};before=torch.get_rng_state().clone();r=random_mask(m,'source',.75);self.assertTrue(torch.equal(before,torch.get_rng_state()));self.assertTrue(check_masks(m,r));self.assertEqual(state_dict_hash(r),state_dict_hash(random_mask(m,'source',.75)))
 def test_reuse_rejects_protocol_and_unsafe_resume(self):
  c=A.config(32,7);c['training']['learning_rate']=.001;self.assertTrue(A.protocol_errors(c,32,7))
  with self.assertRaises(ValueError):reject_model_only_resume({'model':{}})
  with tempfile.TemporaryDirectory() as d:
   p=safe_attempt(d);p.mkdir();(p/'preserve').write_text('x');self.assertNotEqual(p,safe_attempt(d));self.assertTrue((p/'preserve').exists())
 def test_checkpoint_exposure_not_final_substitution(self):
  with tempfile.TemporaryDirectory() as d:
   p=Path(d)/'x.jsonl';p.write_text('\n'.join(json.dumps({'step':s,'proposed':{'e0':10,'e1':10},'accepted':{'e0':s,'e1':s+1},'optimizer_step_applied':s!=2}) for s in range(1,5)));r=exposure_at(p,[2,4],True);self.assertEqual(r[0]['accepted']['e0'],3);self.assertEqual(r[1]['accepted']['e0'],10);self.assertEqual(r[0]['applied_updates'],1);zero=exposure_at(p,[0,2],True)[0];self.assertEqual(zero['accepted'],{'e0':0,'e1':0});self.assertIsNone(zero['accepted_load_cv'])
   with self.assertRaises(ValueError):exposure_at(p,[5],True)
 def test_rng_and_mode_preservation(self):
  model=torch.nn.Linear(2,2);state=torch.get_rng_state().clone()
  def random_probe(m):m.eval();return torch.rand(3)
  protected(random_probe)(model);self.assertTrue(model.training);self.assertTrue(torch.equal(state,torch.get_rng_state()))
 def test_persistent_sparsity_and_optimizer_moments(self):
  model=TinyMoELanguageModel(toy_config()['model']);mask=expert_local_magnitude_masks(model,.75);apply_masks_(model,mask);handles=register_mask_gradient_hooks(model,mask);opt=torch.optim.AdamW(model.parameters(),lr=.0003,weight_decay=.01)
  for step in range(3):
   out=model(torch.arange(16).reshape(2,8)%16);loss=out.logits.square().mean()+.1*out.auxiliary_loss;opt.zero_grad();loss.backward();opt.step();apply_masks_(model,mask)
  for n,p in model.named_parameters():
   if n in mask:self.assertFalse(bool(p[~mask[n]].any()));self.assertFalse(bool(opt.state[p]['exp_avg'][~mask[n]].any()))
  for h in handles:h.remove()
 def test_paired_gap_arithmetic(self):
  sparse={'A':2.1,'B':2.,'C':1.8,'D':1.6};dense={'A':1.5,'B':1.4,'C':1.3,'D':1.2};gap={k:sparse[k]-dense[k] for k in sparse};self.assertAlmostEqual((gap['D']-gap['B'])-(gap['C']-gap['A']),-.1)

def toy_config():
 return {'seed':7,'model':{'vocab_size':16,'max_seq_len':8,'num_layers':1,'num_heads':2,'d_model':8,'num_experts':2,'expert_hidden_size':16,'dropout':0.,'top_k':1,'capacity_factor':1.25},'training':{'learning_rate':.0003,'weight_decay':.01,'grad_clip':1.,'precision':'fp32'},'routing':{'mode':'learned','aux_loss_weight':.1}}
def smoke():
 dest=OUT/'smoke'/f'cpu_{os.getpid()}';dest.mkdir();c=toy_config();torch.manual_seed(7);model=TinyMoELanguageModel(c['model']);cp=dest/'source.pt';torch.save({'model':model.state_dict(),'config':c,'step':0},cp);learned=expert_local_magnitude_masks(model,.75);mask=random_mask(learned,'toy',.75);base={n:v*mask[n] if n in mask else v for n,v in model.state_dict().items()};batches=[(torch.randint(0,16,(2,8)),torch.randint(0,16,(2,8))) for _ in range(8)];val=batches[:2];probe=[b[0] for b in val];ref=R.selected_experts_per_batch(model,probe,torch.device('cpu'));old=R.RECOVERY_EVAL_INTERVAL;R.RECOVERY_EVAL_INTERVAL=2
 try:
  result=R._run_recovery_condition(config=c,condition_name='segregated_cpu_random_mask_smoke',pruned_base_state=base,router_checkpoint=str(cp),router_age_percent=0,router_step=0,masks=mask,expert_hash=state_dict_hash(A.group(base,'expert')),shared_hash=state_dict_hash(A.group(base,'shared')),mask_hash=state_dict_hash(mask),reference_selected=ref,calibration_batches=probe,train_batches=batches,validation_batches=val,train_batch_hash=R._batch_sequence_hash(batches),validation_batch_hash=R._batch_sequence_hash(val),device=torch.device('cpu'),recovery_steps=8,dense_loss=R._evaluate_condition_loss(model,val,torch.device('cpu'),False,1.),output_dir=dest/'recovery',confidence_control=False,target_confidence=None,seed=7,sparsity=.75,diagnostic_steps=(0,4,8))
  assert np.isfinite(result['final_validation_loss']);jsave(dest/'smoke_pass.json',{'passed':True,'synthetic':True,'not_production':True,'steps':8});return str(dest)
 finally:R.RECOVERY_EVAL_INTERVAL=old
if __name__=='__main__':
 torch.set_num_threads(2);result=unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(PilotTests))
 if not result.wasSuccessful():sys.exit(1)
 print('SMOKE',smoke(),flush=True)
