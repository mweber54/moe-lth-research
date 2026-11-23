"""Comprehensive analysis and visualization of MoE Lottery Ticket Initializations,
IMP Masks, and Recovery Performance across E8, E16, and E32.
"""

import os
import sys
import csv
import json
import math
from pathlib import Path
from collections import defaultdict
import numpy as np
from scipy import stats

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import torch

ROOT = Path(__file__).resolve().parents[2]
os.chdir(ROOT)

OUT_DIR = ROOT / 'results/initialization_recovery_analysis_v1'
FIG_DIR = OUT_DIR / 'figures'
OUT_DIR.mkdir(exist_ok=True, parents=True)
FIG_DIR.mkdir(exist_ok=True, parents=True)

# Set style
plt.rcParams['font.sans-serif'] = 'DejaVu Sans'
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['figure.dpi'] = 200
plt.rcParams['savefig.dpi'] = 200

# Color palette for architectures and seeds
ARCH_COLORS = {'E8': '#1f77b4', 'E16': '#ff7f0e', 'E32': '#2ca02c'}
SEED_COLORS = {7: '#4e79a7', 17: '#f28e2b', 29: '#e15759', 43: '#76b7b2', 61: '#59a14f'}

print("--- Step 1: Loading comparison.csv metadata ---")
comp_path = ROOT / 'results/imp_ticket_audit_v1/comparison.csv'
with open(comp_path) as f:
    comp_rows = list(csv.DictReader(f))

print(f"Loaded {len(comp_rows)} primary recovery experiments.")

# Sort comp_rows by (architecture, sparsity, source_seed)
def arch_key(a):
    return int(a.replace('E', ''))

comp_rows.sort(key=lambda r: (arch_key(r['architecture']), float(r['sparsity']), int(r['source_seed'])))

print("\n--- Step 2: Processing Checkpoints, Masks, and Weight Statistics ---")

# Data structures to hold aggregated data
# For each of the 15 unique (arch, seed) initializations:
init_stats_table = []
# For each of the 30 primary experiments:
post_mask_stats_table = []
recovery_summary_table = []

# Histograms cache for plotting
# Keyed by (arch, seed): { 'fc1_raw': hist, 'fc2_raw': hist, 'norm_raw': hist, ... }
init_hist_cache = {}
# Keyed by (arch, seed, sparsity): { 'w_all': ..., 'w_ret': ..., 'w_pru': ..., 'w_masked': ... }
post_mask_hist_cache = {}

# Process each unique initialization first
unique_inits = {}
for r in comp_rows:
    key = (r['architecture'], int(r['source_seed']))
    if key not in unique_inits:
        unique_inits[key] = r['initialization_path']

print(f"Processing {len(unique_inits)} unique step-0 initializations...")

init_weights_cache = {}

for (arch, seed), init_rel_path in unique_inits.items():
    init_path = ROOT / init_rel_path
    print(f"  Loading init {arch} seed {seed} from {init_path.name}...")
    payload = torch.load(init_path, map_location='cpu', weights_only=True)
    model = payload['model']
    
    fc1_tensors = []
    fc2_tensors = []
    all_expert_tensors = []
    
    for k, v in model.items():
        if 'moe.experts' in k and 'weight' in k:
            w = v.float().numpy()
            all_expert_tensors.append(w.reshape(-1))
            if 'fc1' in k:
                fc1_tensors.append(w.reshape(-1))
            elif 'fc2' in k:
                fc2_tensors.append(w.reshape(-1))
                
    fc1_flat = np.concatenate(fc1_tensors)
    fc2_flat = np.concatenate(fc2_tensors)
    all_flat = np.concatenate(all_expert_tensors)
    
    # Normalized weights: fc1 has fan_in=256, fc2 has fan_in=1024
    fc1_norm = fc1_flat * np.sqrt(256.0)
    fc2_norm = fc2_flat * np.sqrt(1024.0)
    all_norm = np.concatenate([fc1_norm, fc2_norm])
    
    # Cache flat arrays for subsequent mask processing
    init_weights_cache[(arch, seed)] = {
        'model': model,
        'fc1': fc1_flat,
        'fc2': fc2_flat,
        'all': all_flat,
        'fc1_norm': fc1_norm,
        'fc2_norm': fc2_norm,
        'all_norm': all_norm
    }
    
    # Compute initial statistics
    for mtype, w_arr, norm_arr, fan_in in [('fc1', fc1_flat, fc1_norm, 256),
                                          ('fc2', fc2_flat, fc2_norm, 1024),
                                          ('pooled_experts', all_flat, all_norm, np.nan)]:
        mean_val = float(np.mean(w_arr))
        std_val = float(np.std(w_arr))
        rms_val = float(np.sqrt(np.mean(w_arr**2)))
        fan_in_rms = rms_val * np.sqrt(fan_in) if not np.isnan(fan_in) else float(np.sqrt(np.mean(norm_arr**2)))
        
        q25, q50, q75, q90, q95, q99 = np.percentile(np.abs(w_arr), [25, 50, 75, 90, 95, 99])
        pos_frac = float(np.mean(w_arr > 0))
        neg_frac = float(np.mean(w_arr < 0))
        zero_count = int(np.count_nonzero(w_arr == 0))
        
        init_stats_table.append({
            'architecture': arch,
            'source_seed': seed,
            'matrix_type': mtype,
            'fan_in': fan_in if not np.isnan(fan_in) else 'mixed',
            'weight_count': len(w_arr),
            'mean': f"{mean_val:.8f}",
            'std': f"{std_val:.8f}",
            'rms': f"{rms_val:.8f}",
            'fan_in_scaled_rms': f"{fan_in_rms:.8f}",
            'min': f"{float(np.min(w_arr)):.8f}",
            'max': f"{float(np.max(w_arr)):.8f}",
            'abs_q25': f"{float(q25):.8f}",
            'abs_q50_median': f"{float(q50):.8f}",
            'abs_q75': f"{float(q75):.8f}",
            'abs_q90': f"{float(q90):.8f}",
            'abs_q95': f"{float(q95):.8f}",
            'abs_q99': f"{float(q99):.8f}",
            'positive_fraction': f"{pos_frac:.6f}",
            'negative_fraction': f"{neg_frac:.6f}",
            'exact_zeros_count': zero_count
        })

print(f"Recorded {len(init_stats_table)} initialization statistics rows.")

print("\n--- Step 3: Processing IMP Masks and Post-Mask Distributions ---")

for idx, r in enumerate(comp_rows, 1):
    arch = r['architecture']
    seed = int(r['source_seed'])
    sparsity = float(r['sparsity'])
    mask_path = ROOT / r['mask_path']
    
    print(f"  [{idx:2d}/30] Processing {arch} seed {seed} s{sparsity:.2f} ({mask_path.name})...")
    mask_payload = torch.load(mask_path, map_location='cpu', weights_only=True)
    model = init_weights_cache[(arch, seed)]['model']
    
    w_ret_list = []
    w_pru_list = []
    w_all_list = []
    
    w_ret_norm_list = []
    w_pru_norm_list = []
    w_all_norm_list = []
    
    for k, v in model.items():
        if 'moe.experts' in k and 'weight' in k:
            w = v.float().numpy().reshape(-1)
            m = mask_payload[k].bool().numpy().reshape(-1)
            
            fan_in = 256.0 if 'fc1' in k else 1024.0
            scale = np.sqrt(fan_in)
            
            w_ret_list.append(w[m])
            w_pru_list.append(w[~m])
            w_all_list.append(w)
            
            w_ret_norm_list.append(w[m] * scale)
            w_pru_norm_list.append(w[~m] * scale)
            w_all_norm_list.append(w * scale)
            
    w_ret = np.concatenate(w_ret_list)
    w_pru = np.concatenate(w_pru_list)
    w_all = np.concatenate(w_all_list)
    
    w_ret_norm = np.concatenate(w_ret_norm_list)
    w_pru_norm = np.concatenate(w_pru_norm_list)
    w_all_norm = np.concatenate(w_all_norm_list)
    
    # Store in cache for plotting
    post_mask_hist_cache[(arch, seed, sparsity)] = {
        'w_ret': w_ret,
        'w_pru': w_pru,
        'w_all': w_all,
        'w_ret_norm': w_ret_norm,
        'w_pru_norm': w_pru_norm,
        'w_all_norm': w_all_norm
    }
    
    # Quantify post-mask statistics
    total_count = len(w_all)
    ret_count = len(w_ret)
    pru_count = len(w_pru)
    actual_sparsity = pru_count / total_count
    
    ret_abs_mean = float(np.mean(np.abs(w_ret)))
    pru_abs_mean = float(np.mean(np.abs(w_pru)))
    all_abs_mean = float(np.mean(np.abs(w_all)))
    ratio_ret_pru = ret_abs_mean / pru_abs_mean if pru_abs_mean > 0 else np.nan
    
    ret_rms = float(np.sqrt(np.mean(w_ret**2)))
    all_rms = float(np.sqrt(np.mean(w_all**2)))
    ret_rms_ratio = ret_rms / all_rms if all_rms > 0 else np.nan
    
    tot_energy = float(np.sum(w_all**2))
    ret_energy = float(np.sum(w_ret**2))
    energy_frac = ret_energy / tot_energy if tot_energy > 0 else np.nan
    
    ret_pos_frac = float(np.mean(w_ret > 0))
    pru_pos_frac = float(np.mean(w_pru > 0))
    
    # Mean percentile rank of retained weights in the absolute magnitude distribution
    # Subsample to 50k points for rapid rank computation if necessary
    sub_all = np.abs(w_all)
    sub_ret = np.abs(w_ret)
    if len(sub_all) > 50000:
        sub_all_eval = np.random.choice(sub_all, size=50000, replace=False)
        sub_ret_eval = np.random.choice(sub_ret, size=20000, replace=False)
    else:
        sub_all_eval = sub_all
        sub_ret_eval = sub_ret
    sub_all_sorted = np.sort(sub_all_eval)
    ranks = np.searchsorted(sub_all_sorted, sub_ret_eval) / len(sub_all_sorted)
    ret_rank_mean = float(np.mean(ranks))
    
    dense_ce = float(r['absolute_dense_ce'])
    sparse_ce = float(r['absolute_sparse_ce'])
    gap = float(r['sparse_minus_dense_ce'])
    rel_gap = gap / dense_ce
    
    post_mask_stats_table.append({
        'architecture': arch,
        'source_seed': seed,
        'sparsity': sparsity,
        'total_weights': total_count,
        'retained_weights': ret_count,
        'pruned_weights': pru_count,
        'actual_sparsity': f"{actual_sparsity:.6f}",
        'retained_abs_mean': f"{ret_abs_mean:.6f}",
        'pruned_abs_mean': f"{pru_abs_mean:.6f}",
        'all_abs_mean': f"{all_abs_mean:.6f}",
        'retained_over_pruned_abs_ratio': f"{ratio_ret_pru:.6f}",
        'retained_rms': f"{ret_rms:.6f}",
        'whole_rms': f"{all_rms:.6f}",
        'retained_over_whole_rms': f"{ret_rms_ratio:.6f}",
        'total_initial_energy': f"{tot_energy:.6f}",
        'retained_initial_energy': f"{ret_energy:.6f}",
        'energy_fraction': f"{energy_frac:.6f}",
        'retained_positive_fraction': f"{ret_pos_frac:.6f}",
        'pruned_positive_fraction': f"{pru_pos_frac:.6f}",
        'retained_magnitude_rank_mean': f"{ret_rank_mean:.6f}",
        'dense_validation_ce': f"{dense_ce:.6f}",
        'sparse_validation_ce': f"{sparse_ce:.6f}",
        'sparse_minus_dense_gap': f"{gap:+.6f}",
        'relative_gap': f"{rel_gap:+.6f}",
        'candidate_winner': "True" if gap <= 0 else "False"
    })

print(f"Recorded {len(post_mask_stats_table)} post-mask statistics rows.")

print("\n--- Step 4: Computing Condition Rankings and Performance Summaries ---")

# Group experiments by condition (arch, sparsity) to compute ranks
cond_groups = defaultdict(list)
for r in comp_rows:
    cond = (r['architecture'], float(r['sparsity']))
    cond_groups[cond].append(r)

seed_ranks_table = []
cond_descriptive_stats = []

for cond, group in cond_groups.items():
    arch, sparsity = cond
    # Rank by gap (lowest/most negative gap = rank 1)
    by_gap = sorted(group, key=lambda x: float(x['sparse_minus_dense_ce']))
    # Rank by sparse CE (lowest CE = rank 1)
    by_sparse = sorted(group, key=lambda x: float(x['absolute_sparse_ce']))
    # Rank by dense CE (lowest CE = rank 1)
    by_dense = sorted(group, key=lambda x: float(x['absolute_dense_ce']))
    
    gap_rank_map = {int(r['source_seed']): rank for rank, r in enumerate(by_gap, 1)}
    sparse_rank_map = {int(r['source_seed']): rank for rank, r in enumerate(by_sparse, 1)}
    dense_rank_map = {int(r['source_seed']): rank for rank, r in enumerate(by_dense, 1)}
    
    gaps = np.array([float(r['sparse_minus_dense_ce']) for r in group])
    dense_ces = np.array([float(r['absolute_dense_ce']) for r in group])
    sparse_ces = np.array([float(r['absolute_sparse_ce']) for r in group])
    
    cond_descriptive_stats.append({
        'architecture': arch,
        'sparsity': sparsity,
        'n_seeds': len(group),
        'mean_gap': float(np.mean(gaps)),
        'std_gap': float(np.std(gaps, ddof=1)),
        'median_gap': float(np.median(gaps)),
        'min_gap': float(np.min(gaps)),
        'max_gap': float(np.max(gaps)),
        'range_gap': float(np.max(gaps) - np.min(gaps)),
        'mean_dense_ce': float(np.mean(dense_ces)),
        'std_dense_ce': float(np.std(dense_ces, ddof=1)),
        'mean_sparse_ce': float(np.mean(sparse_ces)),
        'std_sparse_ce': float(np.std(sparse_ces, ddof=1)),
        'winners_count': int(np.sum(gaps <= 0))
    })
    
    for r in group:
        s = int(r['source_seed'])
        d_val = float(r['absolute_dense_ce'])
        sp_val = float(r['absolute_sparse_ce'])
        g_val = float(r['sparse_minus_dense_ce'])
        recovery_summary_table.append({
            'architecture': arch,
            'source_seed': s,
            'sparsity': sparsity,
            'dense_validation_ce': f"{d_val:.6f}",
            'sparse_validation_ce': f"{sp_val:.6f}",
            'sparse_minus_dense_gap': f"{g_val:+.6f}",
            'relative_gap': f"{g_val/d_val:+.6%}",
            'rank_gap': gap_rank_map[s],
            'rank_sparse_ce': sparse_rank_map[s],
            'rank_dense_ce': dense_rank_map[s],
            'is_candidate_winner': "True" if g_val <= 0 else "False"
        })

recovery_summary_table.sort(key=lambda r: (arch_key(r['architecture']), float(r['sparsity']), r['rank_gap']))

# Seed ranking consistency across all 6 conditions
seeds_list = [7, 17, 29, 43, 61]
conditions_list = [('E8', 0.8), ('E8', 0.85), ('E16', 0.8), ('E16', 0.85), ('E32', 0.8), ('E32', 0.85)]

for s in seeds_list:
    ranks_gap = []
    ranks_sparse = []
    ranks_dense = []
    gaps_by_cond = []
    
    for cond in conditions_list:
        match = next(r for r in recovery_summary_table if r['architecture'] == cond[0] and float(r['sparsity']) == cond[1] and r['source_seed'] == s)
        ranks_gap.append(match['rank_gap'])
        ranks_sparse.append(match['rank_sparse_ce'])
        ranks_dense.append(match['rank_dense_ce'])
        gaps_by_cond.append(float(match['sparse_minus_dense_gap']))
        
    seed_ranks_table.append({
        'source_seed': s,
        'mean_rank_gap': f"{np.mean(ranks_gap):.2f}",
        'std_rank_gap': f"{np.std(ranks_gap, ddof=1):.2f}",
        'min_rank_gap': min(ranks_gap),
        'max_rank_gap': max(ranks_gap),
        'rank_E8_80': ranks_gap[0],
        'rank_E8_85': ranks_gap[1],
        'rank_E16_80': ranks_gap[2],
        'rank_E16_85': ranks_gap[3],
        'rank_E32_80': ranks_gap[4],
        'rank_E32_85': ranks_gap[5],
        'mean_rank_sparse_ce': f"{np.mean(ranks_sparse):.2f}",
        'mean_rank_dense_ce': f"{np.mean(ranks_dense):.2f}",
        'mean_gap_value': f"{np.mean(gaps_by_cond):+.6f}"
    })

print("\n--- Step 5: Computing Property Correlations across Conditions ---")

corr_features = [
    ('retained_over_pruned_abs_ratio', 'Retained / Pruned Abs Magnitude Ratio'),
    ('retained_over_whole_rms', 'Retained RMS / Whole Initialization RMS'),
    ('energy_fraction', 'Retained Energy Fraction (Sum w^2)'),
    ('retained_magnitude_rank_mean', 'Retained Magnitude Percentile Rank Mean'),
    ('retained_positive_fraction', 'Retained Positive Sign Fraction'),
    ('whole_rms', 'Whole Initial Weight RMS')
]

corr_table = []
for cond in conditions_list:
    arch, sparsity = cond
    sub_pm = [r for r in post_mask_stats_table if r['architecture'] == arch and float(r['sparsity']) == sparsity]
    sub_pm.sort(key=lambda x: int(x['source_seed']))
    
    gaps = [float(r['sparse_minus_dense_gap']) for r in sub_pm]
    sp_ces = [float(r['sparse_validation_ce']) for r in sub_pm]
    dn_ces = [float(r['dense_validation_ce']) for r in sub_pm]
    
    for feat_key, feat_name in corr_features:
        vals = [float(r[feat_key]) for r in sub_pm]
        
        rho_gap, p_gap = stats.spearmanr(vals, gaps)
        rho_sp, p_sp = stats.spearmanr(vals, sp_ces)
        rho_dn, p_dn = stats.spearmanr(vals, dn_ces)
        
        corr_table.append({
            'architecture': arch,
            'sparsity': sparsity,
            'feature': feat_key,
            'feature_name': feat_name,
            'n_seeds': len(vals),
            'spearman_rho_gap': f"{rho_gap:+.4f}",
            'p_value_gap': f"{p_gap:.4f}",
            'spearman_rho_sparse_ce': f"{rho_sp:+.4f}",
            'p_value_sparse_ce': f"{p_sp:.4f}",
            'spearman_rho_dense_ce': f"{rho_dn:+.4f}",
            'p_value_dense_ce': f"{p_dn:.4f}"
        })

print("\n--- Step 6: Writing Machine-Readable CSV Tables ---")

def write_csv_dict(path, rows):
    if not rows:
        return
    fields = list(dict.fromkeys(k for r in rows for k in r))
    with open(path, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)

write_csv_dict(OUT_DIR / 'recovery_performance_summary.csv', recovery_summary_table)
write_csv_dict(OUT_DIR / 'initialization_weight_statistics.csv', init_stats_table)
write_csv_dict(OUT_DIR / 'post_mask_weight_statistics.csv', post_mask_stats_table)
write_csv_dict(OUT_DIR / 'seed_rankings_by_condition.csv', seed_ranks_table)
write_csv_dict(OUT_DIR / 'property_correlations.csv', corr_table)

print("Saved all 5 machine-readable CSV summary tables.")

print("\n--- Step 7: Generating Visualizations (Deliverables 1 to 6) ---")

# Figure 1: Seed vs. Performance Plots
fig, axes = plt.subplots(2, 3, figsize=(14, 8), sharex='col')
archs = ['E8', 'E16', 'E32']

for col_idx, arch in enumerate(archs):
    # Top row: Absolute Validation Cross-Entropy
    ax_top = axes[0, col_idx]
    # Bottom row: Sparse-minus-Dense Gap
    ax_bot = axes[1, col_idx]
    
    seeds = [7, 17, 29, 43, 61]
    x = np.arange(len(seeds))
    
    # Get values for 80% and 85%
    rows_80 = {int(r['source_seed']): r for r in recovery_summary_table if r['architecture'] == arch and float(r['sparsity']) == 0.80}
    rows_85 = {int(r['source_seed']): r for r in recovery_summary_table if r['architecture'] == arch and float(r['sparsity']) == 0.85}
    
    dense_vals = [float(rows_80[s]['dense_validation_ce']) for s in seeds]
    sp_80_vals = [float(rows_80[s]['sparse_validation_ce']) for s in seeds]
    sp_85_vals = [float(rows_85[s]['sparse_validation_ce']) for s in seeds]
    
    gap_80_vals = [float(rows_80[s]['sparse_minus_dense_gap']) for s in seeds]
    gap_85_vals = [float(rows_85[s]['sparse_minus_dense_gap']) for s in seeds]
    
    # Top plot: absolute lines and points
    ax_top.plot(x, dense_vals, 'k--o', label='Dense baseline', lw=1.8, markersize=6, alpha=0.85)
    ax_top.plot(x, sp_80_vals, 's-', color='#1f77b4', label='Sparse 80%', lw=1.8, markersize=6)
    ax_top.plot(x, sp_85_vals, '^-', color='#d62728', label='Sparse 85%', lw=1.8, markersize=6)
    
    ax_top.set_title(f"{arch} Absolute Performance", fontsize=11, fontweight='bold')
    ax_top.set_ylabel("Validation Cross-Entropy" if col_idx == 0 else "")
    ax_top.grid(True, linestyle=':', alpha=0.6)
    if col_idx == 0:
        ax_top.legend(loc='upper right', fontsize=8.5)
        
    # Bottom plot: Gaps
    w = 0.35
    b1 = ax_bot.bar(x - w/2, gap_80_vals, w, label='Gap 80%', color='#1f77b4', alpha=0.85, edgecolor='black', linewidth=0.5)
    b2 = ax_bot.bar(x + w/2, gap_85_vals, w, label='Gap 85%', color='#d62728', alpha=0.85, edgecolor='black', linewidth=0.5)
    
    ax_bot.axhline(0, color='black', lw=1.0)
    ax_bot.set_xticks(x)
    ax_bot.set_xticklabels([f"Seed {s}" for s in seeds], fontsize=9.5, fontweight='bold')
    ax_bot.set_title(f"{arch} Sparse − Dense Gap", fontsize=11, fontweight='bold')
    ax_bot.set_ylabel("Validation CE Gap" if col_idx == 0 else "")
    ax_bot.set_xlabel("Random Seed (Categorical)")
    ax_bot.grid(True, axis='y', linestyle=':', alpha=0.6)
    if col_idx == 0:
        ax_bot.legend(loc='upper left', fontsize=8.5)
        
    # Annotate negative gaps (winners)
    for idx_s, g in enumerate(gap_80_vals):
        if g <= 0:
            ax_bot.text(idx_s - w/2, g - 0.005, f"{g:+.3f}", ha='center', va='top', fontsize=7.5, fontweight='bold', color='#1f77b4')
    for idx_s, g in enumerate(gap_85_vals):
        if g <= 0:
            ax_bot.text(idx_s + w/2, g - 0.005, f"{g:+.3f}", ha='center', va='top', fontsize=7.5, fontweight='bold', color='#d62728')

fig.suptitle("Seed vs. Recovery Performance Across MoE Architectures (Step 2,500 Final Readout)", fontsize=13, fontweight='bold')
fig.tight_layout()
fig.savefig(FIG_DIR / 'fig1_seed_vs_performance.png')
plt.close(fig)
print("Saved figures/fig1_seed_vs_performance.png")

# Figure 2: Summary of Gap Distributions Across Conditions
fig, ax = plt.subplots(figsize=(9, 4.8))
cond_labels = [f"{c[0]} ({int(c[1]*100)}%)" for c in conditions_list]
cond_gap_data = []
for cond in conditions_list:
    g = [float(r['sparse_minus_dense_gap']) for r in recovery_summary_table if r['architecture'] == cond[0] and float(r['sparsity']) == cond[1]]
    cond_gap_data.append(g)

bp = ax.boxplot(cond_gap_data, patch_artist=True, widths=0.45,
                boxprops=dict(facecolor='#e0ecf4', color='#1f77b4', alpha=0.7),
                medianprops=dict(color='#d62728', lw=2),
                whiskerprops=dict(color='#1f77b4', lw=1.2),
                capprops=dict(color='#1f77b4', lw=1.2))

# Overlay individual seed points
for idx_c, g in enumerate(cond_gap_data, 1):
    x_jit = np.random.normal(idx_c, 0.04, size=len(g))
    ax.scatter(x_jit, g, color='#2b5c8f', edgecolors='black', s=45, zorder=3, alpha=0.9)
    # Add seed labels next to points
    sub_r = [r for r in recovery_summary_table if r['architecture'] == conditions_list[idx_c-1][0] and float(r['sparsity']) == conditions_list[idx_c-1][1]]
    sub_r.sort(key=lambda x: int(x['source_seed']))
    for pt_idx, r in enumerate(sub_r):
        s_val = r['source_seed']
        gap_val = float(r['sparse_minus_dense_gap'])
        ax.annotate(f"s{s_val}", (x_jit[pt_idx], gap_val), textcoords="offset points", xytext=(5, -2), fontsize=7.5, alpha=0.8)

ax.axhline(0, color='black', lw=1.2, linestyle='--', label='Dense Baseline Parity (Gap = 0)')
ax.set_xticks(range(1, len(cond_labels)+1))
ax.set_xticklabels(cond_labels, fontsize=10, fontweight='bold')
ax.set_ylabel("Sparse − Dense Validation CE Gap", fontsize=10, fontweight='bold')
ax.set_title("Distribution of Recovery Gaps Across Source Seeds by Condition (n=5 per box)", fontsize=11, fontweight='bold')
ax.grid(True, axis='y', linestyle=':', alpha=0.6)
ax.legend(loc='upper right', fontsize=9)
fig.tight_layout()
fig.savefig(FIG_DIR / 'fig2_gap_distribution_summary.png')
plt.close(fig)
print("Saved figures/fig2_gap_distribution_summary.png")

# Figure 3: Initial Weight Distributions (Signed, Separated by fc1 & fc2 & Normalized)
fig, axes = plt.subplots(2, 3, figsize=(14, 8), sharey='row')

for col_idx, arch in enumerate(archs):
    ax_top = axes[0, col_idx]  # Raw signed weights separated by fc1 and fc2
    ax_bot = axes[1, col_idx]  # Fan-in normalized weights (pooled) across seeds
    
    # Let's take Seed 17 as representative for raw fc1 vs fc2 comparison
    fc1_ex = init_weights_cache[(arch, 17)]['fc1']
    fc2_ex = init_weights_cache[(arch, 17)]['fc2']
    
    bins_raw = np.linspace(-0.08, 0.08, 80)
    ax_top.hist(fc1_ex, bins=bins_raw, density=True, alpha=0.6, color='#1f77b4', label='fc1 (fan-in 256, bounds ±0.0625)')
    ax_top.hist(fc2_ex, bins=bins_raw, density=True, alpha=0.6, color='#ff7f0e', label='fc2 (fan-in 1024, bounds ±0.03125)')
    
    ax_top.set_title(f"{arch} Step-0 Weights (fc1 vs. fc2, Seed 17)", fontsize=10.5, fontweight='bold')
    ax_top.set_xlabel("Signed Raw Weight Value")
    ax_top.set_ylabel("Probability Density" if col_idx == 0 else "")
    ax_top.grid(True, linestyle=':', alpha=0.6)
    if col_idx == 0:
        ax_top.legend(loc='upper right', fontsize=8)
        
    # Bottom: Normalized weights (w * sqrt(fan_in)) across all 5 seeds
    bins_norm = np.linspace(-1.2, 1.2, 60)
    for s in seeds:
        w_norm = init_weights_cache[(arch, s)]['all_norm']
        ax_bot.hist(w_norm, bins=bins_norm, density=True, histtype='step', lw=1.5,
                    color=SEED_COLORS[s], label=f"Seed {s}")
        
    # Theoretical uniform density: height = 1 / (1 - (-1)) = 0.5 on [-1, 1]
    ax_bot.plot([-1, -1, 1, 1], [0, 0.5, 0.5, 0], 'k--', lw=1.2, label='Theoretical Uniform [-1, 1]')
    ax_bot.set_title(f"{arch} Fan-In Normalized Weights Across Seeds", fontsize=10.5, fontweight='bold')
    ax_bot.set_xlabel(r"Normalized Weight ($w \cdot \sqrt{\mathrm{fan\_in}}$)")
    ax_bot.set_ylabel("Probability Density" if col_idx == 0 else "")
    ax_bot.grid(True, linestyle=':', alpha=0.6)
    if col_idx == 0:
        ax_bot.legend(loc='lower center', fontsize=7.5, ncol=3)

fig.suptitle("Initial Step-0 Expert Weight Distributions: Projection Scale Separation and Seed Invariance", fontsize=12, fontweight='bold')
fig.tight_layout()
fig.savefig(FIG_DIR / 'fig3_initial_weight_distributions_signed.png')
plt.close(fig)
print("Saved figures/fig3_initial_weight_distributions_signed.png")

# Figure 4: Initial Absolute Magnitude Distributions and Quantiles
fig, axes = plt.subplots(1, 3, figsize=(13, 4.2), sharey=True)

for col_idx, arch in enumerate(archs):
    ax = axes[col_idx]
    bins_mag = np.linspace(0, 1.1, 55)
    
    for s in seeds:
        w_mag = np.abs(init_weights_cache[(arch, s)]['all_norm'])
        ax.hist(w_mag, bins=bins_mag, density=True, histtype='step', lw=1.6,
                color=SEED_COLORS[s], label=f"Seed {s}")
        
    # Theoretical uniform absolute magnitude on [0, 1]: density is flat = 1.0
    ax.plot([0, 1, 1], [1.0, 1.0, 0], 'k--', lw=1.2, label='Uniform Null [0, 1]')
    
    # Mark median and 90th percentile
    ax.axvline(0.5, color='gray', linestyle=':', lw=1, label='Median (0.50)')
    ax.axvline(0.9, color='gray', linestyle='-.', lw=1, label='90th Pct (0.90)')
    
    ax.set_title(f"{arch} Absolute Initial Magnitudes", fontsize=11, fontweight='bold')
    ax.set_xlabel(r"Normalized Magnitude ($|w| \cdot \sqrt{\mathrm{fan\_in}}$)")
    ax.set_ylabel("Probability Density" if col_idx == 0 else "")
    ax.grid(True, linestyle=':', alpha=0.6)
    if col_idx == 0:
        ax.legend(loc='upper right', fontsize=8)

fig.suptitle("Step-0 Normalized Magnitude Distributions across Architectures (Perfect Overlap across Seeds)", fontsize=12, fontweight='bold')
fig.tight_layout()
fig.savefig(FIG_DIR / 'fig4_initial_weight_distributions_magnitude.png')
plt.close(fig)
print("Saved figures/fig4_initial_weight_distributions_magnitude.png")

# Figure 5: Retained vs. Pruned Initial Weight Distributions (80% Sparsity)
fig, axes = plt.subplots(2, 3, figsize=(14, 8), sharey='row')

for col_idx, arch in enumerate(archs):
    ax_top = axes[0, col_idx]  # Seed 17 (Strong/Candidate Winner)
    ax_bot = axes[1, col_idx]  # Seed 7 or Seed 29 (Losing/Weaker Seed)
    
    # Candidate/Strong Seed: Seed 17
    s_win = 17
    data_win = post_mask_hist_cache[(arch, s_win, 0.8)]
    w_ret_win = data_win['w_ret_norm']
    w_pru_win = data_win['w_pru_norm']
    w_all_win = data_win['w_all_norm']
    
    # Weaker Seed: Seed 29 (E16, E32) or Seed 7 (E8)
    s_lose = 7 if arch == 'E8' else 29
    data_lose = post_mask_hist_cache[(arch, s_lose, 0.8)]
    w_ret_lose = data_lose['w_ret_norm']
    w_pru_lose = data_lose['w_pru_norm']
    w_all_lose = data_lose['w_all_norm']
    
    bins_dn = np.linspace(-1.15, 1.15, 70)
    
    # Top: Winning Seed
    ax_top.hist(w_all_win, bins=bins_dn, density=True, histtype='step', lw=1.4, color='black', linestyle='--', label='A. All Initial Weights')
    ax_top.hist(w_ret_win, bins=bins_dn, density=True, alpha=0.6, color='#2ca02c', label='B. Retained Initial Weights')
    ax_top.hist(w_pru_win, bins=bins_dn, density=True, alpha=0.45, color='#d62728', label='C. Pruned Initial Weights')
    
    # Get outcome gap
    r_win = next(r for r in recovery_summary_table if r['architecture'] == arch and float(r['sparsity']) == 0.8 and r['source_seed'] == s_win)
    ax_top.set_title(f"{arch} Seed {s_win} (80% Sparsity, Gap: {float(r_win['sparse_minus_dense_gap']):+.4f})", fontsize=10.5, fontweight='bold')
    ax_top.set_ylabel("Probability Density" if col_idx == 0 else "")
    ax_top.grid(True, linestyle=':', alpha=0.6)
    if col_idx == 0:
        ax_top.legend(loc='upper center', fontsize=8)
        
    # Bottom: Losing Seed
    ax_bot.hist(w_all_lose, bins=bins_dn, density=True, histtype='step', lw=1.4, color='black', linestyle='--', label='A. All Initial Weights')
    ax_bot.hist(w_ret_lose, bins=bins_dn, density=True, alpha=0.6, color='#2ca02c', label='B. Retained Initial Weights')
    ax_bot.hist(w_pru_lose, bins=bins_dn, density=True, alpha=0.45, color='#d62728', label='C. Pruned Initial Weights')
    
    r_lose = next(r for r in recovery_summary_table if r['architecture'] == arch and float(r['sparsity']) == 0.8 and r['source_seed'] == s_lose)
    ax_bot.set_title(f"{arch} Seed {s_lose} (80% Sparsity, Gap: {float(r_lose['sparse_minus_dense_gap']):+.4f})", fontsize=10.5, fontweight='bold')
    ax_bot.set_xlabel(r"Normalized Weight Value ($w \cdot \sqrt{\mathrm{fan\_in}}$)")
    ax_bot.set_ylabel("Probability Density" if col_idx == 0 else "")
    ax_bot.grid(True, linestyle=':', alpha=0.6)

fig.suptitle("Comparison of Original (A), Retained (B), and Pruned (C) Initial Weights (IMP 80%)", fontsize=12, fontweight='bold')
fig.tight_layout()
fig.savefig(FIG_DIR / 'fig5_retained_vs_pruned_distributions.png')
plt.close(fig)
print("Saved figures/fig5_retained_vs_pruned_distributions.png")

# Figure 6: Full Masked Tensor Distribution (Highlighting the Point Mass at Exact Zero)
fig, axes = plt.subplots(1, 3, figsize=(14, 4.2), sharey=True)

for col_idx, arch in enumerate(archs):
    ax = axes[col_idx]
    
    # E.g. Seed 17, 80% vs 85%
    data_80 = post_mask_hist_cache[(arch, 17, 0.8)]
    data_85 = post_mask_hist_cache[(arch, 17, 0.85)]
    
    # Full masked tensor has 80% or 85% zeros
    w_ret_80 = data_80['w_ret_norm']
    n_pru_80 = len(data_80['w_pru_norm'])
    masked_80 = np.concatenate([w_ret_80, np.zeros(n_pru_80)])
    
    w_ret_85 = data_85['w_ret_norm']
    n_pru_85 = len(data_85['w_pru_norm'])
    masked_85 = np.concatenate([w_ret_85, np.zeros(n_pru_85)])
    
    bins_m = np.linspace(-1.15, 1.15, 81)
    
    # Histogram of full masked tensor
    ax.hist(masked_80, bins=bins_m, density=True, alpha=0.55, color='#1f77b4', label=f'Full Masked 80% (80% delta at 0)')
    ax.hist(masked_85, bins=bins_m, density=True, histtype='step', lw=1.6, color='#d62728', label=f'Full Masked 85% (85% delta at 0)')
    
    ax.set_title(f"{arch} Full Masked Tensor (Seed 17)", fontsize=11, fontweight='bold')
    ax.set_xlabel(r"Weight Value ($w \cdot \sqrt{\mathrm{fan\_in}}$)")
    ax.set_ylabel("Probability Density (Point Mass + Tails)" if col_idx == 0 else "")
    ax.grid(True, linestyle=':', alpha=0.6)
    if col_idx == 0:
        ax.legend(loc='upper right', fontsize=8.5)

fig.suptitle("Full Masked Tensor Distributions: Point Mass Spike at Zero vs. Continuous Retained Tails", fontsize=12, fontweight='bold')
fig.tight_layout()
fig.savefig(FIG_DIR / 'fig6_full_masked_distributions.png')
plt.close(fig)
print("Saved figures/fig6_full_masked_distributions.png")

# Figure 7: Initialization & Post-Mask Statistics vs. Recovery Gap Scatterplots
fig, axes = plt.subplots(2, 2, figsize=(11, 9))
features_scatter = [
    ('retained_over_pruned_abs_ratio', 'Retained / Pruned Magnitude Ratio', axes[0, 0]),
    ('energy_fraction', 'Retained Initial Energy Fraction', axes[0, 1]),
    ('retained_magnitude_rank_mean', 'Retained Magnitude Percentile Rank', axes[1, 0]),
    ('retained_over_whole_rms', 'Retained RMS / Whole Initialization RMS', axes[1, 1])
]

markers_sp = {0.8: 'o', 0.85: 's'}

for feat_key, feat_title, ax in features_scatter:
    for arch in archs:
        for s in [0.8, 0.85]:
            sub = [r for r in post_mask_stats_table if r['architecture'] == arch and float(r['sparsity']) == s]
            x_vals = [float(r[feat_key]) for r in sub]
            y_vals = [float(r['sparse_minus_dense_gap']) for r in sub]
            
            ax.scatter(x_vals, y_vals, c=ARCH_COLORS[arch], marker=markers_sp[s], s=55,
                       alpha=0.85, edgecolors='black', linewidth=0.5,
                       label=f"{arch} {int(s*100)}%" if feat_key == 'retained_over_pruned_abs_ratio' else None)
            
    ax.axhline(0, color='black', lw=1.0, linestyle='--')
    ax.set_title(feat_title, fontsize=10.5, fontweight='bold')
    ax.set_xlabel(feat_title)
    ax.set_ylabel("Sparse − Dense Validation CE Gap")
    ax.grid(True, linestyle=':', alpha=0.6)

axes[0, 0].legend(loc='upper right', fontsize=7.5, ncol=2)
fig.suptitle("Candidate Initial/Mask Metrics vs. Recovery Performance Gap Across Architectures", fontsize=12, fontweight='bold')
fig.tight_layout()
fig.savefig(FIG_DIR / 'fig7_property_vs_recovery_scatter.png')
plt.close(fig)
print("Saved figures/fig7_property_vs_recovery_scatter.png")

# Figure 8: Retained RMS Sign Reversal Deep-Dive
fig, ax = plt.subplots(figsize=(8, 5))

for arch in archs:
    sub = [r for r in post_mask_stats_table if r['architecture'] == arch and float(r['sparsity']) == 0.8]
    sub.sort(key=lambda x: int(x['source_seed']))
    x_vals = np.array([float(r['retained_over_whole_rms']) for r in sub])
    y_vals = np.array([float(r['sparse_minus_dense_gap']) for r in sub])
    
    # Linear regression
    slope, intercept, r_val, p_val, std_err = stats.linregress(x_vals, y_vals)
    rho, _ = stats.spearmanr(x_vals, y_vals)
    
    ax.scatter(x_vals, y_vals, c=ARCH_COLORS[arch], s=70, edgecolors='black', linewidth=0.6,
               label=f"{arch} (80%): Spearman $\\rho = {rho:+.2f}$ (slope ${slope:+.2f}$)")
    
    x_line = np.linspace(x_vals.min() - 0.01, x_vals.max() + 0.01, 50)
    ax.plot(x_line, slope * x_line + intercept, color=ARCH_COLORS[arch], lw=1.8, linestyle='--')
    
    for idx_pt, r in enumerate(sub):
        s_val = r['source_seed']
        ax.annotate(f"s{s_val}", (x_vals[idx_pt], y_vals[idx_pt]), textcoords="offset points", xytext=(6, -2), fontsize=8)

ax.axhline(0, color='black', lw=1.0)
ax.set_title("Architecture-Dependent Sign Inversion of Retained RMS Association (80% Sparsity)", fontsize=11, fontweight='bold')
ax.set_xlabel("Retained Initial RMS / Whole Initial RMS")
ax.set_ylabel("Sparse − Dense Validation CE Gap")
ax.grid(True, linestyle=':', alpha=0.6)
ax.legend(loc='upper right', fontsize=9)
fig.tight_layout()
fig.savefig(FIG_DIR / 'fig8_retained_rms_reversal_analysis.png')
plt.close(fig)
print("Saved figures/fig8_retained_rms_reversal_analysis.png")

# Figure 9: Seed Ranking Consistency Heatmaps
fig, axes = plt.subplots(1, 3, figsize=(15, 5))
col_labels = [f"{c[0]}\n{int(c[1]*100)}%" for c in conditions_list]
row_labels = [f"Seed {s}" for s in seeds_list]

matrix_gap = np.zeros((len(seeds_list), len(conditions_list)))
matrix_sp = np.zeros((len(seeds_list), len(conditions_list)))
matrix_dn = np.zeros((len(seeds_list), len(conditions_list)))

matrix_gap_ranks = np.zeros((len(seeds_list), len(conditions_list)), dtype=int)
matrix_sp_ranks = np.zeros((len(seeds_list), len(conditions_list)), dtype=int)
matrix_dn_ranks = np.zeros((len(seeds_list), len(conditions_list)), dtype=int)

for r_idx, s in enumerate(seeds_list):
    for c_idx, cond in enumerate(conditions_list):
        match = next(r for r in recovery_summary_table if r['architecture'] == cond[0] and float(r['sparsity']) == cond[1] and r['source_seed'] == s)
        matrix_gap[r_idx, c_idx] = float(match['sparse_minus_dense_gap'])
        matrix_sp[r_idx, c_idx] = float(match['sparse_validation_ce'])
        matrix_dn[r_idx, c_idx] = float(match['dense_validation_ce'])
        matrix_gap_ranks[r_idx, c_idx] = int(match['rank_gap'])
        matrix_sp_ranks[r_idx, c_idx] = int(match['rank_sparse_ce'])
        matrix_dn_ranks[r_idx, c_idx] = int(match['rank_dense_ce'])

# Panel 1: Gap
im0 = axes[0].imshow(matrix_gap, cmap='coolwarm', aspect='auto')
axes[0].set_title("A. Sparse − Dense Gap (Values & Ranks)", fontsize=10.5, fontweight='bold')
fig.colorbar(im0, ax=axes[0], fraction=0.046, pad=0.04)

# Panel 2: Sparse CE
im1 = axes[1].imshow(matrix_sp, cmap='YlOrRd', aspect='auto')
axes[1].set_title("B. Absolute Sparse CE (Values & Ranks)", fontsize=10.5, fontweight='bold')
fig.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04)

# Panel 3: Dense CE
im2 = axes[2].imshow(matrix_dn, cmap='YlOrRd', aspect='auto')
axes[2].set_title("C. Dense Baseline CE (Values & Ranks)", fontsize=10.5, fontweight='bold')
fig.colorbar(im2, ax=axes[2], fraction=0.046, pad=0.04)

for ax_idx, (ax, mat, mat_ranks) in enumerate([(axes[0], matrix_gap, matrix_gap_ranks),
                                              (axes[1], matrix_sp, matrix_sp_ranks),
                                              (axes[2], matrix_dn, matrix_dn_ranks)]):
    ax.set_xticks(range(len(col_labels)))
    ax.set_xticklabels(col_labels, fontsize=9, fontweight='bold')
    ax.set_yticks(range(len(row_labels)))
    ax.set_yticklabels(row_labels, fontsize=9.5, fontweight='bold')
    
    for i in range(len(seeds_list)):
        for j in range(len(conditions_list)):
            val = mat[i, j]
            rnk = mat_ranks[i, j]
            txt = f"Rank {rnk}\n({val:+.3f})" if ax_idx == 0 else f"Rank {rnk}\n({val:.3f})"
            color = 'black' if abs(val) < 0.08 or (ax_idx > 0 and val < 1.50) else 'white'
            ax.text(j, i, txt, ha='center', va='center', fontsize=7.5, fontweight='bold', color=color)

fig.suptitle("Seed Consistency Heatmaps: Recovery Gap vs. Absolute Sparse vs. Dense Baseline", fontsize=12, fontweight='bold')
fig.tight_layout()
fig.savefig(FIG_DIR / 'fig9_seed_ranking_heatmap.png')
plt.close(fig)
print("Saved figures/fig9_seed_ranking_heatmap.png")

# Figure 10: Absolute vs. Gap Seed Rankings Comparison
fig, ax = plt.subplots(figsize=(8.5, 4.8))

for s in seeds_list:
    r_sp = [int(r['rank_sparse_ce']) for r in recovery_summary_table if r['source_seed'] == s]
    r_gp = [int(r['rank_gap']) for r in recovery_summary_table if r['source_seed'] == s]
    
    mean_sp_rank = np.mean(r_sp)
    mean_gp_rank = np.mean(r_gp)
    
    ax.plot([0, 1], [mean_sp_rank, mean_gp_rank], 'o-', lw=2.2, markersize=8,
            color=SEED_COLORS[s], label=f"Seed {s} (Sparse rank: {mean_sp_rank:.1f} → Gap rank: {mean_gp_rank:.1f})")

ax.set_xlim(-0.15, 1.15)
ax.set_xticks([0, 1])
ax.set_xticklabels(["Mean Absolute Sparse CE Rank\n(Lower = Better Loss)", "Mean Sparse − Dense Gap Rank\n(Lower = Better Relative Recovery)"], fontsize=10, fontweight='bold')
ax.set_ylabel("Rank (1 = Best, 5 = Worst)", fontsize=10, fontweight='bold')
ax.set_yticks(range(1, 6))
ax.invert_yaxis()
ax.grid(True, linestyle=':', alpha=0.6)
ax.set_title("Divergence Between Absolute Performance and Matched Gap Rankings", fontsize=11, fontweight='bold')
ax.legend(loc='lower left', fontsize=8.5)
fig.tight_layout()
fig.savefig(FIG_DIR / 'fig10_absolute_vs_gap_rankings.png')
plt.close(fig)
print("Saved figures/fig10_absolute_vs_gap_rankings.png")

print("\n--- All 10 figures and 5 CSV summary tables generated successfully! ---")
