from __future__ import annotations

import hashlib
import json
from copy import deepcopy
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch

from moe_lth.config import load_config
from moe_lth.experiments.analyze import analyze_suite
from moe_lth.pruning.evaluate_pruning import evaluate_pruning
from moe_lth.pruning.magnitude_prune import expert_local_magnitude_masks
from moe_lth.pruning.masks import mask_jaccard
from moe_lth.routing.deconfounded import (
    assert_acceptance_preserved,
    assert_counts_preserved,
    assert_gate_values_preserved,
    deconfounded_identity_shuffle,
    measure_corruption_statistics,
)
from moe_lth.routing.rich_trace import RichRouteHistory
from moe_lth.training.checkpoint import load_checkpoint
from moe_lth.training.train import train_from_config
from moe_lth.models import TinyMoELanguageModel


ROOT = Path('results/wikitext_reference_gate')
ROOT.mkdir(parents=True, exist_ok=True)


def _prep_reference_config(base: dict, output_dir: str | Path) -> dict:
    config = deepcopy(base)
    config['training']['record_train_routes'] = True
    config['training']['record_rich_routes'] = True
    config['output_dir'] = str(Path(output_dir))
    config['routing']['replay_path'] = None
    return config


def _summarize_mask_shift(normal_model_path: str, run_model_path: str, sparsity: float = 0.8) -> float:
    normal_model = TinyMoELanguageModel(load_config('configs/wikitext103_gpu.yaml')['model'])
    run_model = TinyMoELanguageModel(load_config('configs/wikitext103_gpu.yaml')['model'])
    load_checkpoint(normal_model_path, normal_model, map_location='cpu')
    load_checkpoint(run_model_path, run_model, map_location='cpu')
    normal_masks = expert_local_magnitude_masks(normal_model, sparsity)
    run_masks = expert_local_magnitude_masks(run_model, sparsity)
    return float(mask_jaccard(normal_masks, run_masks))


def _assert_exact_reference_gate(learned_summary: dict, replay_summary: dict, deconfounded_summary: dict, learned_hist: RichRouteHistory, replay_hist: RichRouteHistory, deconfounded_hist: RichRouteHistory) -> dict:
    learned_loss = float(learned_summary['final_validation_loss'])
    if not (1.60 <= learned_loss <= 1.80):
        raise AssertionError(f"Learned baseline mismatch: expected ~1.68 and got {learned_loss}")

    if replay_summary['final_validation_loss'] != learned_summary['final_validation_loss']:
        raise AssertionError(
            f"Replay must match learned loss exactly. Learned={learned_loss}, Replay={replay_summary['final_validation_loss']}"
        )

    learned_trace = learned_hist.get(1, 0)
    replay_trace = replay_hist.get(1, 0)
    deconf_trace = deconfounded_hist.get(1, 0)

    # Route agreement for exact replay must be 1.0.
    learned_ids = learned_trace.selected_expert_ids[:, 0]
    replay_ids = replay_trace.selected_expert_ids[:, 0]
    route_agreement = float((learned_ids == replay_ids).mean())
    if route_agreement != 1.0:
        raise AssertionError(f"Exact replay route agreement must be 1.0, got {route_agreement}")

    # Deconfounded shuffle must be nontrivial, deterministic, and preserve count + gate + acceptance invariants.
    shuffled_ids = deconf_trace.selected_expert_ids[:, 0]
    if torch.equal(torch.from_numpy(learned_ids), torch.from_numpy(shuffled_ids)):
        raise AssertionError("Deconfounded shuffle degenerated to the identity mapping.")
    disagreement = float((learned_ids != shuffled_ids).mean())
    if disagreement <= 0.0:
        raise AssertionError(f"Deconfounded shuffle must change assignments; disagreement={disagreement}")

    learned_hash = hashlib.sha256(learned_ids.tobytes()).hexdigest()
    deconf_hash = hashlib.sha256(shuffled_ids.tobytes()).hexdigest()
    if learned_hash == deconf_hash:
        raise AssertionError("Deconfounded shuffle route hash must differ from the learned route hash.")

    learned_counts = torch.bincount(torch.from_numpy(learned_ids), minlength=learned_hist.metadata.get('num_experts', 8))
    deconf_counts = torch.bincount(torch.from_numpy(shuffled_ids), minlength=learned_hist.metadata.get('num_experts', 8))
    if not torch.equal(learned_counts, deconf_counts):
        raise AssertionError(f"Deconfounded shuffle must preserve expert counts: {learned_counts.tolist()} vs {deconf_counts.tolist()}")

    assert_gate_values_preserved(
        torch.from_numpy(learned_trace.gate_values[:, 0]),
        torch.from_numpy(deconf_trace.gate_values[:, 0]),
    )
    assert_acceptance_preserved(
        torch.from_numpy(learned_trace.accepted_mask[:, 0]),
        torch.from_numpy(deconf_trace.accepted_mask[:, 0]),
    )

    return {
        'learned_validation_loss': learned_loss,
        'replay_validation_loss': float(replay_summary['final_validation_loss']),
        'route_agreement_to_learned': route_agreement,
        'deconfounded_disagreement_fraction': disagreement,
        'deconfounded_hash_differs': learned_hash != deconf_hash,
        'counts_preserved': True,
        'gate_preserved': True,
        'acceptance_preserved': True,
    }


def run_reference_gate() -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required to execute the historical WikiText reference gate. This environment has no CUDA runtime.")

    base = load_config('configs/wikitext103_gpu.yaml')
    assert base['device'] == 'cuda', f"Reference config must use CUDA; got {base['device']}"
    assert base['training']['precision'] == 'fp16', f"Reference config must use fp16; got {base['training']['precision']}"
    assert base['training']['steps'] == 2500, f"Reference config must use 2500 steps; got {base['training']['steps']}"
    print('RESOLVED_REFERENCE_CONFIG', json.dumps(base, sort_keys=True, indent=2))

    suite_dir = ROOT / 'reference_suite'
    suite_dir.mkdir(parents=True, exist_ok=True)

    normal_dir = suite_dir / 'normal'
    normal_cfg = _prep_reference_config(base, normal_dir)
    normal_cfg['routing']['mode'] = 'learned'
    normal_summary = train_from_config(normal_cfg)
    print('NORMAL', normal_summary)

    rich_path = Path(normal_dir) / 'logs' / 'rich_train_route_history.npz'
    if not rich_path.exists():
        legacy_history_path = Path(normal_dir) / 'logs' / 'train_route_history.npz'
        if not legacy_history_path.exists():
            raise FileNotFoundError(f"No saved route-history artifact found in {normal_dir / 'logs'}")
        rich = RichRouteHistory.upgrade_legacy_history(RouteHistory.load(legacy_history_path))
    else:
        rich = RichRouteHistory.load(rich_path)
    rich.verify_integrity()
    print('RICH_TRACE_FIELDS', sorted(rich.metadata.keys()))
    print('TRACE_COUNT', len(rich.traces))
    sample = next(iter(rich.traces.values()))
    print('TRACE_SAMPLE', sample.selected_expert_ids.shape, sample.gate_values.shape, sample.accepted_mask.shape)

    replay_history_path = rich_path if rich_path.exists() else Path(normal_dir) / 'logs' / 'train_route_history.npz'
    replay_cfg = _prep_reference_config(base, suite_dir / 'replay')
    replay_cfg['routing']['mode'] = 'replay'
    replay_cfg['routing']['replay_path'] = str(replay_history_path)
    replay_cfg['training']['record_train_routes'] = False
    replay_cfg['training']['record_rich_routes'] = True
    replay_summary = train_from_config(replay_cfg)
    print('REPLAY', replay_summary)

    deconf_cfg = _prep_reference_config(base, suite_dir / 'deconfounded_shuffle')
    deconf_cfg['routing']['mode'] = 'deconfounded_shuffle'
    deconf_cfg['routing']['replay_path'] = str(replay_history_path)
    deconf_cfg['training']['record_train_routes'] = False
    deconf_cfg['training']['record_rich_routes'] = True
    deconf_summary = train_from_config(deconf_cfg)
    print('DECONFOUNDED', deconf_summary)

    replay_output = Path(replay_cfg['output_dir'])
    deconf_output = Path(deconf_cfg['output_dir'])
    replay_rich = RichRouteHistory.load(replay_output / 'logs' / 'rich_train_route_history.npz')
    deconf_rich = RichRouteHistory.load(deconf_output / 'logs' / 'rich_train_route_history.npz')

    report = _assert_exact_reference_gate(
        normal_summary,
        replay_summary,
        deconf_summary,
        rich,
        replay_rich,
        deconf_rich,
    )
    print('REFERENCE_GATE_OK', json.dumps(report, indent=2))

    synthetic = torch.tensor([[0, 1, 1, 0], [1, 0, 0, 1]], dtype=torch.long)
    from moe_lth.routing.deconfounded import deconfounded_identity_shuffle_flat
    shuffled = deconfounded_identity_shuffle_flat(synthetic, step=1, layer_id=0, seed=7, num_experts=2)
    assert_counts_preserved(synthetic, shuffled, 2)
    stats = measure_corruption_statistics(synthetic, shuffled, 2)
    print('COUNT_INVARIANT_OK', torch.bincount(synthetic.flatten(), minlength=2).tolist(), torch.bincount(shuffled.flatten(), minlength=2).tolist())
    print('STATISTICS', stats)

    model = TinyMoELanguageModel(base['model']).to(torch.device('cuda'))
    assert next(model.parameters()).device.type == 'cuda', 'Instantiated model parameters are not on CUDA.'
    return report


def run_graded_sweep() -> None:
    configs = [
        'configs/wikitext103_gpu.yaml',
        'configs/wikitext103_gpu_seed17.yaml',
        'configs/wikitext103_gpu_seed29.yaml',
    ]
    fractions = [0.0, 0.1, 0.25, 0.5, 0.75, 1.0]
    root = ROOT / 'graded_sweep'
    root.mkdir(parents=True, exist_ok=True)
    rows = []

    for config_path in configs:
        base = load_config(config_path)
        suite_dir = Path(base['output_dir']).parent / (Path(base['output_dir']).name + '_suite')
        normal_dir = suite_dir / 'normal'
        if not normal_dir.exists():
            normal_cfg = _prep_reference_config(base, str(normal_dir))
            normal_cfg['routing']['mode'] = 'learned'
            normal_summary = train_from_config(normal_cfg)
            print('NORMAL_FOR_GRADED', config_path, normal_summary)
        for frac in fractions:
            output_dir = root / f"{Path(config_path).stem}_{frac:.2f}"
            cfg = _prep_reference_config(base, str(output_dir))
            cfg['routing']['mode'] = 'graded_corruption'
            cfg['routing']['corruption_fraction'] = frac
            cfg['routing']['replay_path'] = str(normal_dir / 'logs' / 'train_route_history.npz')
            cfg['training']['record_train_routes'] = False
            cfg['training']['record_rich_routes'] = True
            summary = train_from_config(cfg)
            run_model = str(output_dir / 'checkpoints' / 'step_25.pt')
            mask_shift = _summarize_mask_shift(str(normal_dir / 'checkpoints' / 'step_25.pt'), run_model, sparsity=0.8)
            rows.append({
                'seed': base['seed'],
                'config': Path(config_path).name,
                'corruption_fraction': frac,
                'loss': summary['final_validation_loss'],
                'mask_jaccard_to_normal': mask_shift,
            })
            print('GRADED', config_path, 'frac', frac, 'loss', summary['final_validation_loss'], 'mask_shift', mask_shift)

    rows_path = root / 'graded_corruption_summary.json'
    rows_path.write_text(json.dumps(rows, indent=2), encoding='utf-8')

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    by_frac = sorted({row['corruption_fraction'] for row in rows})
    losses = []
    shifts = []
    for frac in by_frac:
        values = [row for row in rows if row['corruption_fraction'] == frac]
        losses.append(sum(v['loss'] for v in values) / len(values))
        shifts.append(sum(v['mask_jaccard_to_normal'] for v in values) / len(values))
    axes[0].plot(by_frac, losses, marker='o')
    axes[0].set_xlabel('Routing corruption fraction')
    axes[0].set_ylabel('Validation loss')
    axes[0].set_title('Dense-model degradation vs. routing corruption')
    axes[1].plot(by_frac, shifts, marker='o', color='tab:green')
    axes[1].set_xlabel('Routing corruption fraction')
    axes[1].set_ylabel('Mask Jaccard to normal')
    axes[1].set_title('Mask/support shift vs. routing corruption')
    fig.tight_layout()
    fig.savefig(root / 'corruption_vs_dense_loss_and_mask_shift.png', dpi=160)
    plt.close(fig)


if __name__ == '__main__':
    run_reference_gate()
    run_graded_sweep()
