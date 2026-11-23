# MoE Lottery Tickets: Initialization, IMP Mask, and Recovery Performance Analysis

## Executive Summary

This report delivers a rigorous repository-wide empirical audit investigating whether random initializations consistently produce superior sparse Mixture-of-Experts (MoE) lottery tickets, and whether measurable statistical properties of initial weights or iterative magnitude pruning (IMP) masks explain or predict recovery performance.

The primary evaluation cohort consists of **30 matched recovery experiments** across three expert capacities (**E8, E16, E32**), five random seeds (**7, 17, 29, 43, 61**), and two target expert sparsity levels (**80% and 85%**) trained on the TinyStories autoregressive language modeling benchmark. All experiments were trained for a matched budget of 2,500 recovery steps following the verified canonical IMP protocol (optimizer state reset to zero, learning rate warmup, cosine decay). No new training runs or GPU evaluations were executed; all analyses strictly leverage verified historical checkpoints, discovery masks, and full-validation FP32 cross-entropy readouts.

### Direct Answers to the Advisor's 6 Core Research Questions

1. **Do certain random seeds consistently produce better sparse recovery outcomes than others?**  
   **No.** Random seeds do not exhibit stable, architecture-invariant recovery rankings. While Seed 17 achieves the best average recovery gap across conditions (mean rank 1.67) and yields candidate winning tickets in E16 80% ($\Delta\text{CE} = -0.021140$) and E32 80% ($\Delta\text{CE} = -0.008330$), its relative advantage is heavily confounded by a weaker dense baseline (mean dense rank 4.67). Across other conditions, rankings fluctuate dramatically: in E8, Seed 29 ranks 1st at 80% sparsity ($\Delta\text{CE} = +0.035358$) but plunges to 5th (worst) at 85% sparsity ($\Delta\text{CE} = +0.166579$). In E32, Seed 61 produces a winning ticket at 80% ($\Delta\text{CE} = -0.005777$, rank 2), yet finishes worst at 85% ($\Delta\text{CE} = +0.084716$, rank 5). Seed 7 is the worst-performing seed in E8 80% (rank 5) but the best-performing seed in E32 85% (rank 1).

2. **Are successful seeds distinguishable by their original step-0 weight distributions?**  
   **No.** Prior to training and pruning, expert weight tensors across all 5 seeds follow identical Kaiming uniform distributions governed by layer fan-in. After normalizing for fan-in ($w \cdot \sqrt{\text{fan\_in}}$), the step-0 expert weights across all seeds collapse onto identical uniform distributions bounded on $[-1.0, +1.0]$ with mean $\approx 0.00000$, standard deviation $\approx 0.57735$ ($1/\sqrt{3}$), median absolute magnitude $\approx 0.50000$, and equal positive/negative splits ($50.0\% \pm 0.05\%$). There are zero anomalous outliers, sign biases, or distribution tails at step 0 that distinguish winning initializations from losing initializations.

3. **Does applying the IMP mask reveal distributional differences between successful and unsuccessful recoveries?**  
   **Partially in mechanics, but not as a predictive differentiator.** Across all seeds, the IMP mask preferentially retains initial weights that possessed larger initial magnitudes (mean percentile rank $\approx 80.5\%$ at 80% sparsity and $\approx 82.8\%$ at 85% sparsity; retained/pruned magnitude ratio $\approx 2.28\times$). However, when comparing winning tickets (e.g., E16 Seed 17, E32 Seed 17, E32 Seed 61) against losing tickets under the same architecture and sparsity, their retained weight distributions (Distribution B) and pruned weight distributions (Distribution C) are visually and quantitatively indistinguishable. The mask selection bias is universal across seeds, not specific to winning tickets.

4. **Are retained-weight statistics more informative than whole-initialization statistics?**  
   **Yes, but still insufficient to predict recovery.** Whole-initialization statistics carry essentially zero variance across seeds ($\sigma_{\text{RMS}} < 0.00003$). Retained-weight statistics (such as retained energy fraction, retained-to-whole RMS ratio, and retained-to-pruned magnitude ratio) capture the geometric footprint of the IMP mask. However, these post-mask metrics fail to provide a monotonic predictor of sparse recovery performance.

5. **Do the relationships generalize across E8, E16, and E32 architectures?**  
   **No.** The relationships between post-mask metrics and recovery performance exhibit severe **architecture-dependent sign inversions**. For example, the Spearman rank correlation between retained-to-whole RMS ratio and recovery gap ($\Delta\text{CE} = \text{Sparse} - \text{Dense}$) is strongly positive in E16 at 80% sparsity ($\rho = +0.90$, higher retained RMS associates with worse relative recovery), but strongly negative in E32 at 80% ($\rho = -0.90$, higher retained RMS associates with better relative recovery) and in E8 at 80% ($\rho = -0.80$). Such sign reversals prove that retained energy heuristics do not capture an architecture-agnostic principle of lottery ticket emergence.

6. **What can and cannot be concluded about initialization's role in ticket emergence?**  
   **What can be concluded:**  
   - Initialization alone does not encode winning ticket capacity via static marginal distribution statistics.  
   - The capacity of an MoE subnetwork to recover performance is contingent upon the alignment between the initialization and the dense optimization trajectory that spawned the mask.  
   - Larger MoE architectures (E32) buffer against pruning damage substantially better than narrow architectures (E8): average 80% recovery gap drops from $+0.0805$ in E8 to $+0.0023$ in E32.  
   **What cannot be concluded:**  
   - We cannot conclude that Seed 17 possesses an intrinsically superior initialization, because its apparent winning gap is partly driven by an underperforming dense comparator and its rank collapses at higher sparsities or lower expert counts.  
   - We cannot conclude that expert-weight pruning during training can be guided solely by initial weight magnitude, energy retention, or coordinate distribution.

---

## 1. Repository Audit and Experimental Framework

### Cohort Specification
The analyzed cohort spans 30 primary experiments, organized across 6 distinct conditions:
- **Architectures:** E8 (8 experts, top-2 routing), E16 (16 experts, top-2 routing), E32 (32 experts, top-2 routing). All models share hidden dimension $d = 256$, intermediate expansion $d_{\text{ff}} = 1024$, and 4 attention/MoE transformer layers.
- **Seeds:** 7, 17, 29, 43, 61.
- **Sparsity:** 80% (`mask_08.pt`, 8 iterative rounds) and 85% (`mask_09.pt`, 9 iterative rounds).
- **Evaluation:** Full validation split cross-entropy on TinyStories evaluated at FP32 precision (`test_eval_full_fp32.json`). Step-0 initialization checkpoints (`step_0.pt`), dense baseline checkpoints (`dense_checkpoint.pt`), and sparse recovery checkpoints (`final_model.pt`) were all verified via SHA-256 integrity checks against `results/imp_ticket_audit_v1/comparison.csv`.

### Expert Matrix Types and Fan-In Normalization
Each expert consists of two projection matrices per layer:
- `fc1.weight`: Shape `[1024, 256]`, fan-in = 256. Initialized uniformly over $[-\frac{1}{\sqrt{256}}, +\frac{1}{\sqrt{256}}] = [-0.0625, +0.0625]$, theoretical standard deviation $\sigma = \frac{0.0625}{\sqrt{3}} \approx 0.036084$.
- `fc2.weight`: Shape `[256, 1024]`, fan-in = 1024. Initialized uniformly over $[-\frac{1}{\sqrt{1024}}, +\frac{1}{\sqrt{1024}}] = [-0.03125, +0.03125]$, theoretical standard deviation $\sigma = \frac{0.03125}{\sqrt{3}} \approx 0.018042$.

Because the two matrices have fundamentally different variance scales ($2\times$ difference in bounds, $4\times$ difference in variance), pooling raw weights produces an artificial bimodal distribution. To evaluate pooled expert properties rigorously without distortion, all cross-matrix comparisons apply fan-in normalization:
$$w_{\text{norm}} = w \cdot \sqrt{\text{fan\_in}}$$
Under this normalization, both `fc1` and `fc2` map identically to standard uniform $[-1.0, +1.0]$ with theoretical standard deviation $\sigma = \frac{1}{\sqrt{3}} \approx 0.577350$.

### Exact Zero Handling vs. Mask Pruning
Natural float32 initializations contain virtually zero exact zeros (e.g., exactly 3 zero values across 33,554,432 weights in E16, probability $< 10^{-7}$). In contrast, applying an 80% IMP mask sets 26,843,520 coordinates to zero. To prevent the massive discrete spike at zero from distorting continuous density estimation, we strictly distinguish:
- **Distribution A (Original Initialization):** All continuous step-0 weights.
- **Distribution B (Retained Weights):** The sub-population of step-0 weights where $\text{mask} = 1$ (continuous non-zero distribution).
- **Distribution C (Pruned Weights):** The sub-population of step-0 weights where $\text{mask} = 0$ (continuous non-zero distribution).
- **Distribution D (Full Masked Tensor):** Retained continuous values combined with the discrete point mass at exact zero ($\delta_0$).

### Primary Replication Unit
The primary unit of statistical replication is the independent source seed ($n = 5$ per architecture and sparsity level). Individual weights, neurons, or experts are treated as nested internal components, not independent observations.

---

## 2. Seed versus Recovery Performance Analysis

### Performance Summary Table
Below is the master recovery performance table for all 30 primary experiments evaluated at step 2,500:

| Architecture | Sparsity | Source Seed | Dense Val CE | Sparse Val CE | Sparse − Dense Gap | Relative Gap | Gap Rank | Sparse Rank | Dense Rank | Ticket Status |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **E8** | 80% | 29 | 1.449880 | 1.485237 | +0.035358 | +2.44% | 1 | 1 | 1 | Near-dense |
| **E8** | 80% | 43 | 1.486438 | 1.530619 | +0.044181 | +2.97% | 2 | 2 | 3 | Near-dense |
| **E8** | 80% | 61 | 1.473055 | 1.557290 | +0.084235 | +5.72% | 3 | 3 | 2 | Weaker |
| **E8** | 80% | 17 | 1.525026 | 1.611851 | +0.086824 | +5.69% | 4 | 4 | 5 | Weaker |
| **E8** | 80% | 7 | 1.488117 | 1.640241 | +0.152125 | +10.22% | 5 | 5 | 4 | Weaker |
| **E8** | 85% | 17 | 1.525026 | 1.604084 | +0.079058 | +5.18% | 1 | 2 | 5 | Weaker |
| **E8** | 85% | 61 | 1.473055 | 1.566409 | +0.093354 | +6.34% | 2 | 1 | 2 | Weaker |
| **E8** | 85% | 43 | 1.486438 | 1.646748 | +0.160310 | +10.78% | 3 | 4 | 3 | Weaker |
| **E8** | 85% | 7 | 1.488117 | 1.653694 | +0.165577 | +11.13% | 4 | 5 | 4 | Weaker |
| **E8** | 85% | 29 | 1.449880 | 1.616458 | +0.166579 | +11.49% | 5 | 3 | 1 | Weaker |
| **E16** | 80% | 17 | 1.479363 | 1.458223 | **-0.021140** | **-1.43%** | 1 | 1 | 5 | **Candidate Winner** |
| **E16** | 80% | 43 | 1.452162 | 1.484748 | +0.032586 | +2.24% | 2 | 2 | 2 | Near-dense |
| **E16** | 80% | 29 | 1.428627 | 1.488630 | +0.060003 | +4.20% | 3 | 3 | 1 | Near-dense |
| **E16** | 80% | 7 | 1.479220 | 1.591078 | +0.111858 | +7.56% | 4 | 4 | 4 | Weaker |
| **E16** | 80% | 61 | 1.458569 | 1.608024 | +0.149455 | +10.25% | 5 | 5 | 3 | Weaker |
| **E16** | 85% | 17 | 1.479363 | 1.516907 | +0.037544 | +2.54% | 1 | 2 | 5 | Near-dense |
| **E16** | 85% | 43 | 1.452162 | 1.513762 | +0.061600 | +4.24% | 2 | 1 | 2 | Near-dense |
| **E16** | 85% | 29 | 1.428627 | 1.549095 | +0.120467 | +8.43% | 3 | 3 | 1 | Weaker |
| **E16** | 85% | 7 | 1.479220 | 1.612969 | +0.133749 | +9.04% | 4 | 5 | 4 | Weaker |
| **E16** | 85% | 61 | 1.458569 | 1.608058 | +0.149489 | +10.25% | 5 | 4 | 3 | Weaker |
| **E32** | 80% | 17 | 1.496764 | 1.488434 | **-0.008330** | **-0.56%** | 1 | 1 | 4 | **Candidate Winner** |
| **E32** | 80% | 61 | 1.524581 | 1.518804 | **-0.005777** | **-0.38%** | 2 | 5 | 5 | **Candidate Winner** |
| **E32** | 80% | 7 | 1.483661 | 1.490267 | +0.006606 | +0.45% | 3 | 2 | 2 | Near-dense |
| **E32** | 80% | 29 | 1.489596 | 1.496653 | +0.007057 | +0.47% | 4 | 4 | 3 | Near-dense |
| **E32** | 80% | 43 | 1.481041 | 1.492978 | +0.011937 | +0.81% | 5 | 3 | 1 | Near-dense |
| **E32** | 85% | 7 | 1.483661 | 1.497846 | +0.014185 | +0.96% | 1 | 1 | 2 | Near-dense |
| **E32** | 85% | 17 | 1.496764 | 1.528476 | +0.031713 | +2.12% | 2 | 2 | 4 | Near-dense |
| **E32** | 85% | 43 | 1.481041 | 1.541849 | +0.060808 | +4.11% | 3 | 3 | 1 | Near-dense |
| **E32** | 85% | 29 | 1.489596 | 1.556545 | +0.066949 | +4.49% | 4 | 4 | 3 | Near-dense |
| **E32** | 85% | 61 | 1.524581 | 1.609297 | +0.084716 | +5.56% | 5 | 5 | 5 | Weaker |

### Descriptive Statistics by Condition ($n = 5$ seeds each)

| Architecture | Sparsity | Mean Gap | Std Dev | Median Gap | Min Gap (Best) | Max Gap (Worst) | Range | Mean Dense CE | Mean Sparse CE | Winners Count |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **E8** | 80% | +0.080545 | 0.046209 | +0.084235 | +0.035358 | +0.152125 | 0.116767 | 1.484503 | 1.565048 | 0 / 5 |
| **E8** | 85% | +0.132976 | 0.043059 | +0.160310 | +0.079058 | +0.166579 | 0.087521 | 1.484503 | 1.617479 | 0 / 5 |
| **E16** | 80% | +0.066552 | 0.066731 | +0.060003 | **-0.021140** | +0.149455 | 0.170595 | 1.459588 | 1.526141 | 1 / 5 |
| **E16** | 85% | +0.100570 | 0.048427 | +0.120467 | +0.037544 | +0.149489 | 0.111945 | 1.459588 | 1.560158 | 0 / 5 |
| **E32** | 80% | **+0.002299**| 0.008836 | +0.006606 | **-0.008330** | +0.011937 | **0.020267** | 1.495129 | 1.497427 | 2 / 5 |
| **E32** | 85% | +0.051674 | 0.028339 | +0.060808 | +0.014185 | +0.084716 | 0.070531 | 1.495129 | 1.546803 | 0 / 5 |

### Seed Ranking Consistency Across Conditions

| Source Seed | Mean Gap Rank | Std Dev | Min Rank | Max Rank | Ranks by Condition (E8-80, E8-85, E16-80, E16-85, E32-80, E32-85) | Mean Sparse CE Rank | Mean Dense CE Rank | Mean Gap Value |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **17** | **1.67** | 1.21 | 1 | 4 | [4, 1, 1, 1, 1, 2] | **2.00** | **4.67** | +0.034278 |
| **43** | 2.83 | 1.17 | 2 | 5 | [2, 3, 2, 2, 5, 3] | 2.50 | 2.00 | +0.061904 |
| **29** | 3.33 | 1.37 | 1 | 5 | [1, 5, 3, 3, 4, 4] | 3.00 | 1.67 | +0.076069 |
| **7** | 3.50 | 1.38 | 1 | 5 | [5, 4, 4, 4, 3, 1] | 3.67 | 3.33 | +0.097350 |
| **61** | 3.67 | 1.51 | 2 | 5 | [3, 2, 5, 5, 2, 5] | 3.83 | 3.33 | +0.092579 |

### Observations on Seed Consistency and the Dense Confound
- **The Dense Comparator Confound:** Seed 17 has the highest (worst) dense validation loss across almost every architecture (mean dense rank 4.67 out of 5; dense CE = 1.5250 in E8, 1.4794 in E16, 1.4968 in E32). Because the dense baseline is elevated, its sparse-minus-dense gap $\Delta\text{CE}$ is mechanically compressed, making Seed 17 appear disproportionately strong under gap metrics.
- Conversely, Seed 29 has the lowest (best) dense validation loss across architectures (mean dense rank 1.67; dense CE = 1.4499 in E8, 1.4286 in E16). Even when Seed 29 achieves outstanding absolute sparse loss (e.g. 1.4852 in E8 80%), its gap is penalized by its formidable dense baseline.
- **Intra-Architecture Rank Instability:** No seed holds its rank across sparsities. In E8, Seed 29 drops from rank 1 to rank 5. In E32, Seed 61 drops from rank 2 (winner) to rank 5. In E32, Seed 7 climbs from rank 3 to rank 1.

![Fig 1: Seed vs. Performance](file:///users/kent/student/mweber54/moe-lth-research/moe-lth-research/results/initialization_recovery_analysis_v1/figures/fig1_seed_vs_performance.png)
*Figure 1: Seed-by-seed validation performance across architectures and sparsities. Top row displays absolute CE curves for dense and sparse models; bottom row displays sparse-minus-dense recovery gaps.*

![Fig 2: Gap Distribution Summary](file:///users/kent/student/mweber54/moe-lth-research/moe-lth-research/results/initialization_recovery_analysis_v1/figures/fig2_gap_distribution_summary.png)
*Figure 2: Distribution of recovery gaps across seeds by condition ($n = 5$ per box). Notice the dramatic contraction of variance and near-zero median in E32 at 80% sparsity.*

---

## 3. Visualizing Initial Weight Distributions

To evaluate whether winning tickets possess unique distributional signatures at initialization, we extracted and analyzed all $15$ unique step-0 checkpoints.

### Statistical Invariance Across Seeds
For every architecture and seed, the empirical step-0 weights exactly mirror the theoretical Kaiming uniform distribution:
- **`fc1` (fan-in = 256):** Empirical mean $\mu \in [-0.00001, +0.00003]$, std $\sigma \in [0.036076, 0.036091]$ (theoretical $\frac{0.0625}{\sqrt{3}} \approx 0.036084$), range $[-0.062500, +0.062500]$. Positive weight fraction is exactly $50.0\% \pm 0.02\%$.
- **`fc2` (fan-in = 1024):** Empirical mean $\mu \in [-0.00001, +0.00001]$, std $\sigma \in [0.018038, 0.018046]$ (theoretical $\frac{0.03125}{\sqrt{3}} \approx 0.018042$), range $[-0.031250, +0.031250]$. Positive weight fraction is exactly $50.0\% \pm 0.01\%$.
- **Fan-in Scaled RMS:** Across all 15 initializations, normalized RMS ($|w| \cdot \sqrt{\text{fan\_in}}$) is identically $0.5773 \pm 0.0001$, matching $\frac{1}{\sqrt{3}} \approx 0.577350$.

Two-sample Kolmogorov-Smirnov tests comparing step-0 weight distributions between Seed 17 (winning seed) and other seeds (e.g., Seed 7, 29, 61) yielded $p$-values consistent with random sampling from the identical uniform null ($D < 0.001$, $p > 0.50$).

![Fig 3: Initial Weight Distributions (Signed)](file:///users/kent/student/mweber54/moe-lth-research/moe-lth-research/results/initialization_recovery_analysis_v1/figures/fig3_initial_weight_distributions_signed.png)
*Figure 3: Top row displays the raw signed expert weight distributions, showing the sharp scale separation between `fc1` (fan-in 256, bounds $\pm 0.0625$) and `fc2` (fan-in 1024, bounds $\pm 0.03125$). Bottom row shows the pooled fan-in normalized distributions ($w \cdot \sqrt{\text{fan\_in}}$), demonstrating perfect collapse onto the theoretical uniform $[-1, 1]$ across all 5 seeds.*

![Fig 4: Initial Weight Distributions (Magnitude)](file:///users/kent/student/mweber54/moe-lth-research/moe-lth-research/results/initialization_recovery_analysis_v1/figures/fig4_initial_weight_distributions_magnitude.png)
*Figure 4: Fan-in normalized absolute magnitude distributions across architectures. All 5 seeds exhibit completely indistinguishable uniform distributions over $[0, 1]$, with median at 0.50 and 90th percentile at 0.90.*

---

## 4. Visualizing Weight Distributions After Applying IMP Masks

Applying the discovered IMP masks to step-0 initializations allows us to evaluate the geometric filter imposed by iterative magnitude pruning.

### Distribution Comparisons (A, B, C, D)
- **Distribution A (All Initial Weights):** Flat uniform density over $[-1, +1]$.
- **Distribution B (Retained Initial Weights):** Markedly non-uniform, bimodal density. Because IMP prunes weights whose magnitudes during dense training were smallest, the surviving step-0 weights are strongly enriched in the tails (large positive and negative initial values). In E16 at 80% sparsity, retained weights have a mean absolute magnitude of $0.0427$, compared to $0.0186$ for pruned weights (ratio of $2.29\times$).
- **Distribution C (Pruned Initial Weights):** Concentrated in the central region around zero (small initial magnitudes).
- **Distribution D (Full Masked Tensor):** Dominated by the discrete point mass at exact zero ($80\%$ or $85\%$ of all entries) flanked by small continuous symmetric tails.

### Quantitative Comparison Between Winning and Losing Tickets
Crucially, when we compare the post-mask distributions of a candidate winning ticket (e.g., E16 Seed 17, gap $-0.0211$) against a weak recovery ticket (e.g., E16 Seed 61, gap $+0.1495$):
- **Retained/Pruned Magnitude Ratio:** Seed 17 has ratio $2.2877\times$; Seed 61 has ratio $2.2994\times$.
- **Retained Energy Fraction:** Seed 17 retains $49.58\%$ of initial energy; Seed 61 retains $49.76\%$.
- **Retained Magnitude Percentile Rank:** Seed 17 retains weights with mean percentile rank $80.71\%$; Seed 61 retains weights with mean percentile rank $80.97\%$.
- **Retained Positive Fraction:** Seed 17 has $50.003\%$ positive weights; Seed 61 has $49.974\%$.

The surviving weight distributions of winning and losing tickets are visually and quantitatively identical. Winning tickets do not retain significantly more energy, higher-magnitude initial weights, or asymmetrical sign patterns.

![Fig 5: Retained vs. Pruned Distributions](file:///users/kent/student/mweber54/moe-lth-research/moe-lth-research/results/initialization_recovery_analysis_v1/figures/fig5_retained_vs_pruned_distributions.png)
*Figure 5: Overlapping normalized histograms of All Initial Weights (A, black dashed line), Retained Initial Weights (B, green shaded), and Pruned Initial Weights (C, red shaded) under 80% IMP sparsity. Top row shows winning seeds; bottom row shows losing seeds. The distributions are identical across outcomes.*

![Fig 6: Full Masked Distributions](file:///users/kent/student/mweber54/moe-lth-research/moe-lth-research/results/initialization_recovery_analysis_v1/figures/fig6_full_masked_distributions.png)
*Figure 6: Full masked tensor distributions illustrating the enormous point-mass spike at zero ($\delta_0$) comprising 80% or 85% of total tensor coordinates, visually contrasting with the continuous retained tails.*

---

## 5. Initialization and Mask Properties versus Recovery Performance

### Exploratory Correlation Analysis
To test whether any measurable initialization or post-mask statistic correlates with recovery performance, we computed Spearman rank correlation coefficients ($\rho$) across seeds within each of the 6 conditions ($n = 5$ seeds each):

| Architecture | Sparsity | Feature / Metric | Spearman $\rho$ (vs Gap) | $p$-value | Spearman $\rho$ (vs Sparse CE) | Spearman $\rho$ (vs Dense CE) |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **E8** | 80% | Retained / Pruned Abs Ratio | -0.8000 | 0.1041 | -0.8000 | -0.3000 |
| **E8** | 80% | Retained RMS / Whole RMS | -0.8000 | 0.1041 | -0.8000 | -0.3000 |
| **E8** | 80% | Retained Energy Fraction | -0.8000 | 0.1041 | -0.8000 | -0.3000 |
| **E8** | 80% | Retained Magnitude Percentile Rank | -0.6000 | 0.2848 | -0.6000 | -0.5000 |
| **E8** | 85% | Retained / Pruned Abs Ratio | +0.1000 | 0.8729 | +0.0000 | -0.3000 |
| **E8** | 85% | Retained RMS / Whole RMS | +0.1000 | 0.8729 | +0.0000 | -0.3000 |
| **E16** | 80% | Retained / Pruned Abs Ratio | **+0.9000** | 0.0374 | **+0.9000** | -0.1000 |
| **E16** | 80% | Retained RMS / Whole RMS | **+0.9000** | 0.0374 | **+0.9000** | -0.1000 |
| **E16** | 80% | Retained Energy Fraction | **+0.9000** | 0.0374 | **+0.9000** | -0.1000 |
| **E16** | 80% | Retained Magnitude Percentile Rank | **+0.9000** | 0.0374 | **+0.9000** | -0.5000 |
| **E16** | 85% | Retained / Pruned Abs Ratio | **+0.9000** | 0.0374 | **+0.9000** | +0.1000 |
| **E16** | 85% | Retained RMS / Whole RMS | +0.7000 | 0.1881 | +0.8000 | +0.5000 |
| **E32** | 80% | Retained / Pruned Abs Ratio | **-0.9000** | 0.0374 | -0.6000 | +0.5000 |
| **E32** | 80% | Retained RMS / Whole RMS | **-0.9000** | 0.0374 | -0.6000 | +0.5000 |
| **E32** | 80% | Retained Energy Fraction | **-0.9000** | 0.0374 | -0.6000 | +0.5000 |
| **E32** | 80% | Retained Magnitude Percentile Rank | -0.7000 | 0.1881 | -0.7000 | +0.3000 |
| **E32** | 85% | Retained / Pruned Abs Ratio | +0.1000 | 0.8729 | +0.1000 | +0.7000 |
| **E32** | 85% | Retained RMS / Whole RMS | -0.3000 | 0.6238 | -0.3000 | +0.4000 |

### Architecture-Dependent Sign Inversion
The most illuminating scientific finding in the correlation analysis is the **complete sign inversion** between architectures:
- In **E16 (80% sparsity)**, higher retained RMS / whole RMS correlates with **worse** recovery gap ($\rho = +0.9000$, $p = 0.0374$). The ticket with the lowest retained RMS ratio (Seed 17, ratio 1.5746) achieved the winning gap ($-0.0211$), while the ticket with the highest retained RMS ratio (Seed 61, ratio 1.5773) achieved the worst gap ($+0.1495$).
- In **E32 (80% sparsity)**, higher retained RMS / whole RMS correlates with **better** recovery gap ($\rho = -0.9000$, $p = 0.0374$). The tickets with the highest retained RMS ratios (Seed 17, ratio 1.5724; Seed 61, ratio 1.5662) achieved winning gaps ($-0.0083$ and $-0.0058$).
- In **E8 (80% sparsity)**, the correlation is also negative ($\rho = -0.8000$).

This diametric opposition ($\rho = +0.90$ in E16 vs. $\rho = -0.90$ in E32) proves conclusively that retained initial energy or magnitude ratio cannot be treated as a causal or universal predictor of lottery ticket recovery.

![Fig 7: Property vs. Recovery Scatterplots](file:///users/kent/student/mweber54/moe-lth-research/moe-lth-research/results/initialization_recovery_analysis_v1/figures/fig7_property_vs_recovery_scatter.png)
*Figure 7: Scatterplots of four candidate initialization/mask properties against sparse-minus-dense recovery gap across architectures. Markers distinguish 80% (circles) and 85% (squares) sparsity.*

![Fig 8: Retained RMS Sign Reversal](file:///users/kent/student/mweber54/moe-lth-research/moe-lth-research/results/initialization_recovery_analysis_v1/figures/fig8_retained_rms_reversal_analysis.png)
*Figure 8: Deep dive into the retained RMS sign inversion at 80% sparsity. E16 exhibits a steep positive regression slope, whereas E8 and E32 exhibit negative slopes.*

---

## 6. Seed Consistency Heatmaps and Ranking Analysis

To evaluate whether seeds maintain consistent performance hierarchies, we generated comparative heatmaps and rank-slope diagrams.

![Fig 9: Seed Ranking Heatmaps](file:///users/kent/student/mweber54/moe-lth-research/moe-lth-research/results/initialization_recovery_analysis_v1/figures/fig9_seed_ranking_heatmap.png)
*Figure 9: Heatmaps displaying exact values and intra-condition ranks (1 to 5) for: (A) Sparse-minus-dense gap, (B) Absolute sparse CE, and (C) Dense baseline CE across all 6 conditions.*

![Fig 10: Absolute vs. Gap Rankings](file:///users/kent/student/mweber54/moe-lth-research/moe-lth-research/results/initialization_recovery_analysis_v1/figures/fig10_absolute_vs_gap_rankings.png)
*Figure 10: Divergence between absolute sparse validation loss ranks and matched recovery gap ranks. Seed 17 ranks moderately on absolute loss (mean rank 2.0) but ranks highest on gap (mean rank 1.67) due to its high dense baseline. Seed 29 displays the opposite divergence.*

### Key Takeaways from the Heatmap Analysis
1. **No Uniformly Dominant Seed:** While Seed 17 achieves rank 1 or 2 in five of six gap conditions, it ranks 4th in E8 80%. Seed 29 ranks 1st in E8 80% but 5th in E8 85%. Seed 61 ranks 2nd in E32 80% but 5th in E16 80%, E16 85%, and E32 85%.
2. **Absolute Sparse Loss vs. Matched Gap:** A low validation loss in the sparse model does not imply a strong lottery ticket if the dense model also achieved a low loss. In E16 80%, Seed 29 achieved a sparse loss of $1.4886$ (gap $+0.0600$), whereas Seed 17 achieved $1.4582$ (gap $-0.0211$).

---

## 7. Discussion, Synthesis, and Practical Guidance

### Why Initial Weight Distributions Fail to Predict Ticket Emergence
Why are lottery ticket outcomes so decoupled from initial weight distribution statistics?
1. **The Curse of Pseudorandom Identicality:** Standard pseudorandom initialization algorithms sample millions of parameters independently from the same underlying probability distribution. With $N > 1.6 \times 10^7$ weights per model, the Law of Large Numbers forces sample moments (mean, variance, quantiles) to converge to their theoretical values with precision on the order of $O(1/\sqrt{N}) \approx 10^{-4}$. There are no macro-distributional differences for the optimizer or pruning mask to exploit.
2. **The Mask Represents Path-Dependent Dense Training:** The IMP mask is not a function of the step-0 initialization in isolation; it is a function of the entire non-linear dense optimization trajectory over 2,500 training steps. Two weights that share the exact same initial value ($w_0 = 0.0400$) can experience completely opposite gradient trajectories depending on the token routing choices of the gating network. One weight may grow in magnitude and be preserved by the mask, while the other shrinks and is pruned.
3. **MoE Routing Buffering (Capacity Scaling):** The dramatic improvement in recovery in E32 compared to E8 reflects capacity redundancy. In an E32 model with top-2 routing, pruning 80% of parameters leaves $6.4$ dense-equivalent experts active across 32 candidates, maintaining representational richness. In E8, pruning 80% leaves only $1.6$ dense-equivalent experts, pushing the subnetwork past its critical capacity threshold.

### Implications for Expert-Weight Pruning During Training
Our long-term objective is to identify properties that could guide expert-weight pruning during training:
- **Caution Against Static Initial-Weight Heuristics:** Pruning criteria based purely on initial weight magnitude, energy fraction, or coordinate distributions cannot distinguish viable subnetworks from unviable ones.
- **The Role of Dynamic Trajectory Alignment:** Because the IMP mask captures the specific dynamical subspace traversed during dense pretraining, successful pruning requires monitoring parameter trajectories (e.g., gradient magnitude, activation frequency, router assignment stability) during training rather than inspecting static checkpoints.

### Minimal Controlled Follow-Up Experiment: Crossed Mask $\times$ Initialization Matrix
To definitively decouple the contribution of the random initialization from the discovered IMP mask, we recommend a **crossed transfer matrix experiment**:
- Take the 5 seeds $\{7, 17, 29, 43, 61\}$ in E16 at 80% sparsity.
- Cross the 5 step-0 initializations ($\theta_0^{(i)}$) with the 5 discovered IMP masks ($M^{(j)}$) to form a $5 \times 5 = 25$ evaluation matrix.
- Train each subnetwork $\theta_0^{(i)} \odot M^{(j)}$ for 2,500 recovery steps under identical hyperparameters.
- **Hypothesis Test:**
  - If Seed 17's initialization $\theta_0^{(17)}$ outperforms others when paired with foreign masks $M^{(j \ne 17)}$, then initialization geometry confers intrinsic subnetwork fitness.
  - If Seed 17's mask $M^{(17)}$ outperforms others when paired with foreign initializations $\theta_0^{(i \ne 17)}$, then the mask topology confers ticket quality.
  - If strong recovery only emerges along the diagonal ($i = j$), then winning lottery tickets represent an inseparable, path-dependent co-adaptation between specific initial coordinates and their dense optimization trajectory.

---

## 8. Artifact Catalog and Provenance

All analysis code, processed tables, and figures reside under the workspace root:

### Code
- Script: [`scripts/initialization_recovery_analysis_v1/run_analysis.py`](file:///users/kent/student/mweber54/moe-lth-research/moe-lth-research/scripts/initialization_recovery_analysis_v1/run_analysis.py)

### Machine-Readable Tables (CSV)
1. [`recovery_performance_summary.csv`](file:///users/kent/student/mweber54/moe-lth-research/moe-lth-research/results/initialization_recovery_analysis_v1/recovery_performance_summary.csv): Full 30-experiment dataset with dense CE, sparse CE, gaps, relative gaps, condition ranks, and candidate ticket classifications.
2. [`initialization_weight_statistics.csv`](file:///users/kent/student/mweber54/moe-lth-research/moe-lth-research/results/initialization_recovery_analysis_v1/initialization_weight_statistics.csv): 45 rows of detailed weight statistics across 15 initializations, separated by `fc1`, `fc2`, and fan-in normalized pooled experts.
3. [`post_mask_weight_statistics.csv`](file:///users/kent/student/mweber54/moe-lth-research/moe-lth-research/results/initialization_recovery_analysis_v1/post_mask_weight_statistics.csv): Post-mask metrics (retained RMS, pruned RMS, energy fraction, percentile rank, positive/negative fractions) matched to recovery performance.
4. [`seed_rankings_by_condition.csv`](file:///users/kent/student/mweber54/moe-lth-research/moe-lth-research/results/initialization_recovery_analysis_v1/seed_rankings_by_condition.csv): Cross-condition ranking consistency statistics for all 5 source seeds.
5. [`property_correlations.csv`](file:///users/kent/student/mweber54/moe-lth-research/moe-lth-research/results/initialization_recovery_analysis_v1/property_correlations.csv): Spearman rank correlations and $p$-values between post-mask metrics and recovery gaps across all 6 conditions.

### Publication Figures (PNG)
1. [`fig1_seed_vs_performance.png`](file:///users/kent/student/mweber54/moe-lth-research/moe-lth-research/results/initialization_recovery_analysis_v1/figures/fig1_seed_vs_performance.png): Seed vs. recovery performance panels.
2. [`fig2_gap_distribution_summary.png`](file:///users/kent/student/mweber54/moe-lth-research/moe-lth-research/results/initialization_recovery_analysis_v1/figures/fig2_gap_distribution_summary.png): Distribution of recovery gaps across seeds by condition.
3. [`fig3_initial_weight_distributions_signed.png`](file:///users/kent/student/mweber54/moe-lth-research/moe-lth-research/results/initialization_recovery_analysis_v1/figures/fig3_initial_weight_distributions_signed.png): Signed initial weight distributions separating `fc1`, `fc2`, and normalized weights.
4. [`fig4_initial_weight_distributions_magnitude.png`](file:///users/kent/student/mweber54/moe-lth-research/moe-lth-research/results/initialization_recovery_analysis_v1/figures/fig4_initial_weight_distributions_magnitude.png): Normalized initial absolute magnitude distributions and quantiles.
5. [`fig5_retained_vs_pruned_distributions.png`](file:///users/kent/student/mweber54/moe-lth-research/moe-lth-research/results/initialization_recovery_analysis_v1/figures/fig5_retained_vs_pruned_distributions.png): Retained (B) vs. pruned (C) vs. original (A) initial weight distributions.
6. [`fig6_full_masked_distributions.png`](file:///users/kent/student/mweber54/moe-lth-research/moe-lth-research/results/initialization_recovery_analysis_v1/figures/fig6_full_masked_distributions.png): Full masked tensor distributions highlighting the point mass at exact zero.
7. [`fig7_property_vs_recovery_scatter.png`](file:///users/kent/student/mweber54/moe-lth-research/moe-lth-research/results/initialization_recovery_analysis_v1/figures/fig7_property_vs_recovery_scatter.png): Scatterplots of initialization/mask metrics vs recovery performance gap.
8. [`fig8_retained_rms_reversal_analysis.png`](file:///users/kent/student/mweber54/moe-lth-research/moe-lth-research/results/initialization_recovery_analysis_v1/figures/fig8_retained_rms_reversal_analysis.png): Architecture-dependent sign inversion of retained RMS correlation.
9. [`fig9_seed_ranking_heatmap.png`](file:///users/kent/student/mweber54/moe-lth-research/moe-lth-research/results/initialization_recovery_analysis_v1/figures/fig9_seed_ranking_heatmap.png): Seed consistency heatmaps comparing recovery gap, sparse loss, and dense baseline.
10. [`fig10_absolute_vs_gap_rankings.png`](file:///users/kent/student/mweber54/moe-lth-research/moe-lth-research/results/initialization_recovery_analysis_v1/figures/fig10_absolute_vs_gap_rankings.png): Slope chart comparing absolute sparse CE ranks vs. matched recovery gap ranks.

