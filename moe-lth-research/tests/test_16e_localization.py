"""Small deterministic implementation checks; not experimental replicates."""
import copy
import importlib.util
from pathlib import Path
import tempfile
import pandas as pd
import torch
from moe_lth.config import load_config
from moe_lth.training.train import train_from_config
from moe_lth.pruning.router_age import state_dict_hash

spec=importlib.util.spec_from_file_location('localization',Path(__file__).resolve().parents[1]/'scripts/run_16e_router_age_localization.py')
module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)

def test_refinement():
    def frame(values):return pd.DataFrame([{'seed':s,'router_age':a,'ticket_gap':v+(s-17)*.00001} for s in (7,17,29) for a,v in zip(module.COARSE,values)])
    assert module.select_refinement(frame([6,5,4,1,3,5]))[0]==[50,70]
    assert module.select_refinement(frame([0,1,2,3,4,5]))[0]==[10]
    assert module.select_refinement(frame([5,4,3,2,1,0]))[0]==[90]
    ambiguous=frame([6,5,4,1,1,5]);assert module.select_refinement(ambiguous)[0]==[50,70,90]

def test_checkpoint_only_preserves_trajectory():
    with tempfile.TemporaryDirectory() as tmp:
        c=load_config('configs/multidomain_gpu.yaml');c['device']='cpu'
        c['data'].update({'train_path':None,'validation_path':None,'path':None,'seq_len':8,'validation_blocks':1})
        c['model'].update({'d_model':8,'expert_hidden_size':16,'num_layers':1,'num_heads':2,'num_experts':2,'max_seq_len':8})
        c['training'].update({'steps':4,'batch_size':2,'precision':'fp32','checkpoint_steps':[0,4],'eval_interval':4,'log_interval':4})
        hashes=[];logs=[]
        for extra in ([],[1,2,3]):
            cfg=copy.deepcopy(c);cfg['output_dir']=str(Path(tmp)/str(len(extra)));cfg['training']['checkpoint_only_steps']=extra
            train_from_config(cfg)
            final=torch.load(Path(cfg['output_dir'])/'checkpoints/step_4.pt',weights_only=False)
            hashes.append(state_dict_hash(final['model']));logs.append((Path(cfg['output_dir'])/'logs/validation_metrics.jsonl').read_text())
            assert len(list((Path(cfg['output_dir'])/'checkpoints').glob('*.pt')))==2+len(extra)
        assert hashes[0]==hashes[1] and logs[0]==logs[1]

if __name__=='__main__':
    test_refinement();test_checkpoint_only_preserves_trajectory();print('16E localization implementation checks passed')
