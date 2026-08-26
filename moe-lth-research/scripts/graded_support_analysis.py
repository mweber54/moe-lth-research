import json
import re
from pathlib import Path

from moe_lth.config import load_config
from moe_lth.models import TinyMoELanguageModel
from moe_lth.pruning.magnitude_prune import expert_local_magnitude_masks
from moe_lth.pruning.masks import mask_jaccard
from moe_lth.training.checkpoint import load_checkpoint

root = Path('results/wikitext_reference_gate/graded_sweep')
base_cfg = load_config('configs/wikitext103_gpu.yaml')

family_runs = {}
for run in sorted(root.iterdir()):
    if not run.is_dir():
        continue
    match = re.search(r'_(0\.00|0\.10|0\.25|0\.50|0\.75|1\.00)$', run.name)
    if not match:
        continue
    family = run.name[: match.start()]
    frac = float(run.name[match.start() + 1 :])
    family_runs.setdefault(family, {})[frac] = run

rows = []
for family in sorted(family_runs):
    normal_run = family_runs[family].get(0.0)
    if normal_run is None:
        continue
    normal_ckpt = max(
        (normal_run / 'checkpoints').glob('step_*.pt'),
        key=lambda p: int(p.stem.split('_')[-1]),
    )
    normal_model = TinyMoELanguageModel(base_cfg['model'])
    load_checkpoint(str(normal_ckpt), normal_model, map_location='cpu')
    normal_masks = expert_local_magnitude_masks(normal_model, 0.8)

    for frac in sorted(family_runs[family]):
        run = family_runs[family][frac]
        ckpt = max(
            (run / 'checkpoints').glob('step_*.pt'),
            key=lambda p: int(p.stem.split('_')[-1]),
        )
        model = TinyMoELanguageModel(base_cfg['model'])
        load_checkpoint(str(ckpt), model, map_location='cpu')
        masks = expert_local_magnitude_masks(model, 0.8)
        summary = json.loads((run / 'summary.json').read_text()) if (run / 'summary.json').exists() else {}
        rows.append(
            {
                'family': family,
                'corruption_fraction': frac,
                'loss': float(summary.get('final_validation_loss', float('nan'))),
                'mask_jaccard_to_normal': float(mask_jaccard(normal_masks, masks)),
            }
        )

by_frac = {}
for row in rows:
    by_frac.setdefault(row['corruption_fraction'], []).append(row)

print('MEAN_BY_FRACTION')
for frac in sorted(by_frac):
    vals = by_frac[frac]
    avg_loss = sum(v['loss'] for v in vals) / len(vals)
    avg_jacc = sum(v['mask_jaccard_to_normal'] for v in vals) / len(vals)
    print(f'frac={frac} loss={avg_loss:.6f} mask_jaccard_to_normal={avg_jacc:.6f}')

print('DETAIL')
for row in rows:
    print(json.dumps(row, sort_keys=True))
