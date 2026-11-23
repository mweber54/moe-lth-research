# Initialization, IMP Mask, and Recovery Performance Analysis (v1)

## Overview
This directory contains the complete artifacts, tables, and visualization suite analyzing whether random initializations consistently produce superior sparse MoE lottery tickets, and whether measurable statistical properties of initial weights or iterative magnitude pruning (IMP) masks explain or predict recovery performance.

The primary cohort consists of **30 matched experiments** across E8, E16, and E32 architectures, 5 random seeds (7, 17, 29, 43, 61), and 2 expert sparsity levels (80% and 85%) on TinyStories evaluated at 2,500 recovery steps.

## Directory Structure

```
results/initialization_recovery_analysis_v1/
├── FINDINGS_REPORT.md                  # Comprehensive research findings report for advisor
├── README.md                           # Directory catalog and navigation guide
├── recovery_performance_summary.csv    # 30-experiment master results, ranks, and classifications
├── initialization_weight_statistics.csv # 45-row step-0 weight distribution metrics
├── post_mask_weight_statistics.csv     # 30-row post-mask distribution metrics matched to performance
├── seed_rankings_by_condition.csv      # Cross-condition ranking consistency for all 5 seeds
├── property_correlations.csv           # Spearman correlations between post-mask metrics and recovery gaps
└── figures/
    ├── fig1_seed_vs_performance.png
    ├── fig2_gap_distribution_summary.png
    ├── fig3_initial_weight_distributions_signed.png
    ├── fig4_initial_weight_distributions_magnitude.png
    ├── fig5_retained_vs_pruned_distributions.png
    ├── fig6_full_masked_distributions.png
    ├── fig7_property_vs_recovery_scatter.png
    ├── fig8_retained_rms_reversal_analysis.png
    ├── fig9_seed_ranking_heatmap.png
    └── fig10_absolute_vs_gap_rankings.png
```

## Key Findings Summary
1. **Seed Rankings Are Not Invariant:** Seeds do not maintain consistent recovery rankings across architectures and sparsities. Seed 17 achieves the best average gap rank (1.67) and produces candidate winners in E16 80% ($\Delta\text{CE} = -0.021140$) and E32 80% ($\Delta\text{CE} = -0.008330$), but its dense baseline is substantially weaker (mean dense rank 4.67), inflating its relative gap. Other seeds fluctuate widely (e.g., Seed 29 ranks 1st in E8 80% but 5th in E8 85%).
2. **Step-0 Initial Weights Are Identical:** All initializations follow identical theoretical Kaiming uniform distributions bounded on $[-\frac{1}{\sqrt{\text{fan\_in}}}, +\frac{1}{\sqrt{\text{fan\_in}}}]$. When normalized by $\sqrt{\text{fan\_in}}$, all 15 initializations have mean $\approx 0.000$, std $\approx 0.577$, median magnitude $\approx 0.500$, and positive fraction $\approx 50.0\%$. No outlier seeds exist at step 0.
3. **IMP Retains Higher Initial Magnitude Uniformly Across Seeds:** Applying the IMP mask retains weights with higher initial magnitudes ($2.28\times$ higher than pruned weights, retaining $\approx 49.5\%$ of initial energy at 80% sparsity). However, this selection profile is identical between winning tickets and losing tickets.
4. **Architecture-Dependent Sign Inversions:** Retained energy and RMS ratios do not universally predict recovery. In E16 at 80% sparsity, retained RMS ratio correlates positively with gap ($\rho = +0.90$, higher retained RMS associates with worse gap), whereas in E32 and E8 it correlates negatively ($\rho = -0.90$ and $-0.80$, higher retained RMS associates with better gap).

## Scripts
- **Runner script:** [`scripts/initialization_recovery_analysis_v1/run_analysis.py`](file:///users/kent/student/mweber54/moe-lth-research/moe-lth-research/scripts/initialization_recovery_analysis_v1/run_analysis.py)
  Generates all 5 CSV tables and 10 PNG figures using verified checkpoints and evaluation metrics without launching new training or GPU evaluation.

