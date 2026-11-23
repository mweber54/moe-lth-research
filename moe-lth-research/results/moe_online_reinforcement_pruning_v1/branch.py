"""Branch continuation from step 250 to 2500 with sealed pruning selections."""
from __future__ import annotations
import argparse, copy, fcntl, json, math, os, sys, time, traceback
from pathlib import Path
from study_common import *
from torch.optim.optimizer import register_optimizer_step_pre_hook, register_optimizer_step_post_hook

class BranchStop(Exception): pass

def branch_args(seed, order, arm, output, checkpoint, state, config, batches, validation,
                callback, selections, forward=None):
    device = torch.device('cuda:0')
    with N._preserved_rng():
        model = TinyMoELanguageModel(copy.deepcopy(config['model'])).to(device); model.load_state_dict(state); model.eval()
        reference = selected_experts_per_batch(model, N.recovery._calibration_batches(validation), device)
    del model
    names = expert_names(state); masks = {}
    if arm != 'untouched':
        sel_data = torch.load(selections, map_location='cpu', weights_only=True)
        arm_selections = sel_data[arm]
        for expert_name in names:
            mask = torch.ones(state[expert_name].shape, dtype=torch.bool)
            indices = arm_selections[expert_name]
            flat_idx = indices.numpy() if hasattr(indices, 'numpy') else indices
            rows, cols = flat_idx // state[expert_name].shape[1], flat_idx % state[expert_name].shape[1]
            mask[rows, cols] = False; masks[expert_name] = mask
    return dict(config=config, condition_name=f'online_reinforcement_branch_{seed}_{order}_{arm}',
        pruned_base_state=state, router_checkpoint=str(Path(entry(seed, order)['base']['initial_checkpoint'])),
        router_age_percent=0, router_step=0, masks=masks,
        expert_hash=state_dict_hash({n: t for n, t in state.items() if parameter_group(n)=='expert'}),
        shared_hash=state_dict_hash({n: t for n, t in state.items() if parameter_group(n)=='shared'}),
        mask_hash=N.recovery._mask_hash(masks),
        reference_selected=reference, calibration_batches=N.recovery._calibration_batches(validation),
        train_batches=batches[250:2501], validation_batches=validation,
        train_batch_hash=N.recovery._batch_sequence_hash(batches[250:2501]),
        validation_batch_hash=N.recovery._batch_sequence_hash(validation),
        device=device, recovery_steps=2501, dense_loss=1., output_dir=output,
        confidence_control=False, target_confidence=None, seed=seed, sparsity=0.,
        router_mode='trainable',         diagnostic_steps=(0,251,2500,2501),
        state_snapshot_callback=callback, training_forward_callback=forward,
        interruption_checkpoint=checkpoint, interruption_interval=2500)

def run_branch(seed, order, arm):
    root = OUT/'prefixes'/f'seed{seed}_{order}'; prefix_cp = root/'native'/'interruption.pt'
    if not prefix_cp.exists(): return False
    prefix_payload = torch.load(prefix_cp, map_location='cpu', weights_only=False)
    assert prefix_payload['step']==250
    state = prefix_payload['model']; config, batches, validation = material(seed, order)
    branch_root = OUT/'branches'/f'seed{seed}_{order}'/arm; branch_root.mkdir(parents=True, exist_ok=True)
    done = branch_root/'final_step_2500.json'
    if done.exists(): return True
    with (branch_root/'lock').open('a') as lock:
        try: fcntl.flock(lock, fcntl.LOCK_EX|fcntl.LOCK_NB)
        except IOError: return None
    adm = admission(); save_json(branch_root/'admission.json', adm)
    measurements = {}
    def callback(model, opt, scaler, step):
        if step==251:
            measurements['step_251'] = {step:step, timestamp:time.time(),
                validation_ce_before_mask: 0., validation_ce_after_mask: 0.}
        if step==2500:
            measurements['step_2500'] = {step:step, timestamp:time.time()}
        if step==2501: raise BranchStop()
    selections_file = OUT/'matching'/f'seed{seed}_{order}'/'selections.pt'
    output = branch_root/'native'; checkpoint = output/'interruption.pt'
    try:
        try: N.recovery._run_recovery_condition(**branch_args(seed, order, arm, output, checkpoint, state, config, batches, validation, callback, selections_file))
        except BranchStop: pass
        if checkpoint.exists():
            payload = torch.load(checkpoint, map_location='cpu', weights_only=False)
            if payload['step']==2500:
                save_json(done, dict(seed=seed, order=order, arm=arm, completed_step=2500, measurements=measurements, status='COMPLETED'))
                return True
    except Exception as e:
        print(f'seed{seed}_{order} {arm}: error {e}', file=sys.stderr)
        traceback.print_exc()
        return False
    finally: (branch_root/'lock').unlink(missing_ok=True)
    return False

if __name__=='__main__':
    a = argparse.ArgumentParser()
    a.add_argument('seed', type=int, choices=SEEDS)
    a.add_argument('order', choices=ORDERS)
    a.add_argument('arm', choices=['untouched','random-small','high-reinforcement-small'])
    z = a.parse_args(); result = run_branch(z.seed, z.order, z.arm); sys.exit(0 if result else 1)
