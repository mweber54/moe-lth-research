from __future__ import annotations

import json
import argparse
from copy import deepcopy
from pathlib import Path

from moe_lth.config import load_config, save_config


def _jsonable(value):
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_jsonable(v) for v in value]
    if isinstance(value, tuple):
        return [_jsonable(v) for v in value]
    return value
from moe_lth.models import TinyMoELanguageModel
from moe_lth.pruning.magnitude_prune import expert_local_magnitude_masks
from moe_lth.pruning.masks import save_masks
from moe_lth.pruning.train_ticket_v2 import compute_matched_rewind
from moe_lth.training.checkpoint import load_checkpoint

parser = argparse.ArgumentParser()
parser.add_argument('--sparsity', type=float, default=0.8)
args = parser.parse_args()
sparsity = args.sparsity

ROOT = Path('results/wikitext_reference_gate')
ROOT.mkdir(parents=True, exist_ok=True)

base = load_config('configs/wikitext103_gpu.yaml')
normal_dir = ROOT / 'reference_suite' / 'normal'
checkpoint = max(
    (normal_dir / 'checkpoints').glob('step_*.pt'),
    key=lambda p: int(p.stem.split('_')[-1]),
)

model = TinyMoELanguageModel(base['model'])
load_checkpoint(str(checkpoint), model, map_location='cpu')
probe_name = f'tight_rewind_probe_{sparsity:.1f}'
mask_path = ROOT / probe_name / f'learned_mask_{sparsity:.1f}.pt'
mask_path.parent.mkdir(parents=True, exist_ok=True)
mask = expert_local_magnitude_masks(model, sparsity)
save_masks(mask, mask_path)

config = deepcopy(base)
config['routing']['mode'] = 'learned'
config['output_dir'] = str(ROOT / probe_name)
config['training']['record_train_routes'] = False
config['training']['record_rich_routes'] = False
save_config(config, ROOT / probe_name / 'resolved_config.yaml')

results = []
for fraction in [0.0, 0.10, 0.25]:
    rewind_step = int(round(config['training']['steps'] * fraction))
    run_dir = ROOT / probe_name / f'rewind_{fraction:.2f}'
    run_dir.mkdir(parents=True, exist_ok=True)
    run_cfg = deepcopy(config)
    run_cfg['output_dir'] = str(run_dir)
    result = compute_matched_rewind(
        run_cfg,
        str(checkpoint),
        str(mask_path),
        rewind_step=rewind_step,
        total_steps=int(config['training']['steps']),
        random_reinitialize_experts=False,
    )
    results.append({
        'rewind_fraction': fraction,
        'rewind_step': rewind_step,
        'protocol': result['protocol'],
        'loss': result['loss'],
        'perplexity': result['perplexity'],
        'actual_sparse_steps': result['actual_sparse_steps'],
        'total_dense_steps': result['total_dense_steps'],
        'output_dir': str(run_dir),
    })

print(json.dumps(_jsonable(results), indent=2))
