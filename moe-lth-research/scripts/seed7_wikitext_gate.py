from __future__ import annotations

from copy import deepcopy
from pathlib import Path

import torch

from moe_lth.config import load_config
from moe_lth.routing.deconfounded import assert_counts_preserved, deconfounded_identity_shuffle_flat
from moe_lth.routing.rich_trace import RichRouteHistory
from moe_lth.routing.route_history import RouteHistory
from moe_lth.training.train import train_from_config


ROOT = Path('results/wikitext_reference_gate/seed7_protocol')
ROOT.mkdir(parents=True, exist_ok=True)


def prep(base: dict, output_dir: str) -> dict:
    cfg = deepcopy(base)
    cfg['device'] = 'cpu'
    cfg['training']['precision'] = 'fp32'
    cfg['training']['steps'] = 5
    cfg['training']['eval_interval'] = 5
    cfg['training']['log_interval'] = 1
    cfg['training']['checkpoint_steps'] = [0, 5]
    cfg['training']['record_train_routes'] = True
    cfg['training']['record_rich_routes'] = True
    cfg['data']['validation_blocks'] = 1
    cfg['output_dir'] = output_dir
    cfg['routing']['replay_path'] = None
    return cfg


base = load_config('configs/wikitext103_gpu.yaml')
learned_dir = ROOT / 'learned'
learned_cfg = prep(base, str(learned_dir))
learned_cfg['routing']['mode'] = 'learned'
learned_summary = train_from_config(learned_cfg)
print('LEARNED', learned_summary)

hist_path = learned_dir / 'logs' / 'train_route_history.npz'
rich_path = learned_dir / 'logs' / 'rich_train_route_history.npz'
print('HISTORY_EXISTS', hist_path.exists(), rich_path.exists())
assert hist_path.exists(), hist_path
assert rich_path.exists(), rich_path

rich = RichRouteHistory.load(str(rich_path))
rich.verify_integrity()
print('TRACE_FIELDS', sorted(rich.metadata.keys()))
first = next(iter(rich.traces.values()))
print('TRACE_SAMPLE', first.selected_expert_ids.shape, first.gate_values.shape, first.accepted_mask.shape)

for mode, label in [('replay', 'replay'), ('shuffled_usage', 'legacy_shuffle'), ('deconfounded_shuffle', 'deconfounded_shuffle')]:
    run_dir = ROOT / label
    cfg = prep(base, str(run_dir))
    cfg['routing']['mode'] = mode
    cfg['routing']['replay_path'] = str(hist_path)
    cfg['training']['record_train_routes'] = False
    cfg['training']['record_rich_routes'] = True
    summary = train_from_config(cfg)
    print(label.upper(), summary)

    if mode == 'deconfounded_shuffle':
        routes = RouteHistory.load(str(hist_path)).get(1, 0, torch.device('cpu'))
        shuffled = deconfounded_identity_shuffle_flat(
            routes,
            step=1,
            layer_id=0,
            seed=base['seed'],
            num_experts=base['model']['num_experts'],
        )
        assert_counts_preserved(routes, shuffled, base['model']['num_experts'])
        before = torch.bincount(routes.flatten(), minlength=base['model']['num_experts'])
        after = torch.bincount(shuffled.flatten(), minlength=base['model']['num_experts'])
        print('DECONFOUNDED_INVARIANT_OK', before.tolist(), after.tolist())
