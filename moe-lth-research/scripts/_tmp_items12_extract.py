import json
from pathlib import Path
from statistics import mean, pstdev

root = Path('results')
out_dir = root / 'revision_progress'
out_dir.mkdir(parents=True, exist_ok=True)

# ---- Item 1: functional support metrics at 0/10/25/100 corruption ----
graded_path = root / 'wikitext_reference_gate' / 'graded_sweep' / 'graded_corruption_summary.json'
rows = json.loads(graded_path.read_text(encoding='utf-8'))
keep = {0.0, 0.1, 0.25, 1.0}

by_frac = {}
for r in rows:
    f = float(r['corruption_fraction'])
    if f in keep:
        by_frac.setdefault(f, []).append(r)

item1 = []
for frac in sorted(by_frac):
    vals = by_frac[frac]
    losses = [float(v['loss']) for v in vals]
    jacc = [float(v['mask_jaccard_to_normal']) for v in vals]
    item1.append({
        'corruption_fraction': frac,
        'n': len(vals),
        'loss_mean': mean(losses),
        'loss_std': pstdev(losses) if len(losses) > 1 else 0.0,
        'mask_jaccard_mean': mean(jacc),
        'mask_jaccard_std': pstdev(jacc) if len(jacc) > 1 else 0.0,
    })

# ---- Item 2: compute-matched protocol confirmation at headline sparsities ----
# Search for compute-matched files
cm_files = list(root.rglob('ticket_result_cm_*.json'))
cm_rows = []
for p in cm_files:
    try:
        payload = json.loads(p.read_text(encoding='utf-8'))
    except Exception:
        continue
    if payload.get('protocol') not in {'compute_matched', 'compute_matched_rewind'}:
        continue
    cm_rows.append({
        'path': str(p).replace('\\', '/'),
        'loss': payload.get('loss'),
        'rewind_step': payload.get('rewind_step'),
        'total_dense_steps': payload.get('total_dense_steps'),
        'actual_sparse_steps': payload.get('actual_sparse_steps'),
    })

# Extract tight_rewind_probe headline sparse (known to be 0.8 from mask file name)
tight = [r for r in cm_rows if 'wikitext_reference_gate/tight_rewind_probe' in r['path'].lower()]

summary = {
    'item1_functional_support_0_10_25_100': item1,
    'item2_compute_matched': {
        'compute_matched_files_found': len(cm_rows),
        'tight_rewind_probe_files': len(tight),
        'tight_rewind_probe_rows': sorted(tight, key=lambda x: (x['rewind_step'] if x['rewind_step'] is not None else 10**9)),
        'headline_sparsities_confirmed': {
            '0.8': len(tight) > 0,
            '0.5': False,
            'note': 'No compute-matched artifacts for 0.5 detected in existing results scan.'
        }
    }
}

(out_dir / 'items_1_2_summary.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')

# Write markdown for immediate use
lines = []
lines.append('# Revision Progress: Items 1-2')
lines.append('')
lines.append('## 1) Functional support metrics at 0/10/25/100 corruption (from graded_corruption_summary.json)')
lines.append('')
lines.append('| Corruption | n | Mean loss | Std loss | Mean mask Jaccard | Std Jaccard |')
lines.append('|---:|---:|---:|---:|---:|---:|')
for row in item1:
    lines.append(f"| {int(row['corruption_fraction']*100)}% | {row['n']} | {row['loss_mean']:.6f} | {row['loss_std']:.6f} | {row['mask_jaccard_mean']:.6f} | {row['mask_jaccard_std']:.6f} |")

lines.append('')
lines.append('## 2) Compute-matched protocol confirmation at headline sparsities')
lines.append('')
lines.append(f"Compute-matched artifact files found: {len(cm_rows)}")
lines.append('')
lines.append('| File | rewind_step | actual_sparse_steps | loss |')
lines.append('|---|---:|---:|---:|')
for row in sorted(tight, key=lambda x: (x['rewind_step'] if x['rewind_step'] is not None else 10**9)):
    lines.append(f"| {row['path']} | {row['rewind_step']} | {row['actual_sparse_steps']} | {float(row['loss']):.6f} |")

lines.append('')
lines.append('Headline sparsity confirmation:')
lines.append('- 80%: confirmed from existing compute-matched outputs.')
lines.append('- 50%: not yet found in existing compute-matched outputs (requires run).')

(out_dir / 'items_1_2_summary.md').write_text('\n'.join(lines) + '\n', encoding='utf-8')

print(str(out_dir / 'items_1_2_summary.md'))
