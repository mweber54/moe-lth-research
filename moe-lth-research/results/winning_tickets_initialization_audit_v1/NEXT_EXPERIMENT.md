# Minimal Controlled Follow-up Experiment: Crossed Mask-by-Initialization Transfer

**Document Version:** 1.0 (Preregistered Design)  
**Protocol Family:** `imp_mask_initialization_transfer_v1`  
**Target Architecture:** E16 (16 experts, top-1 routing, $d_{\text{model}}=256$, expert width 1024, 4 transformer layers)  
**Sparsity:** 80% nominal expert-weight sparsity (`round_08` mask)  
**Compute Budget:** Exactly 5 recovery runs $\times$ 2,500 attempted updates ($\approx$ 3.0 GPU hours on 1 NVIDIA RTX A6000 / RTX 4090)  
**Status:** Preregistered specification only; **no new jobs launched**.

---

## 1. Scientific Motivation and Hypotheses

### The Core Unresolved Confound
Across all existing Iterative Magnitude Pruning (IMP) experiments in this repository, each learned subnetwork mask $M_s$ is evaluated strictly in conjunction with its own initial weights $W_0^{(s)}$:
$$\text{Trajectory}_s = (W_0^{(s)}, M_s, \mathcal{D}_s)$$
Consequently, **initial weight values, discovery history, and mask coordinate identity are completely confounded**. 

While the randomized control experiment ([`mask_information_v2`](file:///users/kent/student/mweber54/moe-lth-research/moe-lth-research/results/imp_mask_information_v2/results.csv)) conclusively established that mask coordinate placement matters—randomizing coordinates while preserving matrix counts or row/column degrees degraded validation CE by **$+0.1426$** and **$+0.1459$**, respectively—it could **not** determine *why* placement matters:
1. **Hypothesis 1 (Intrinsic Subnetwork Topology):** The IMP mask identifies globally favorable sparse graph topologies (e.g., specific expert-layer sparsity distributions or routing paths) that will recover effectively when applied to *any* compatible random initialization of that architecture.
2. **Hypothesis 2 (Initialization-Specific Coordinate Coupling):** The IMP mask's advantage is strictly tied to the exact numerical weight values present at step 0 in that specific initialization ($M_s \odot W_0^{(s)}$), and transferring the mask to a foreign initialization $W_0^{(s')}$ will destroy the recovery advantage.

Testing these hypotheses requires breaking the diagonal $(W_0^{(s)}, M_s)$ coupling with a minimal, controlled cross-donor intervention.

---

## 2. Experimental Design

### 2.1 Cohort and cyclic donor mapping
We utilize the existing five source seeds for the E16 architecture: $S = \{7, 17, 29, 43, 61\}$.
To avoid outcome-dependent donor selection or combinatorial explosion ($5 \times 5 = 25$ runs), we preregister a fixed cyclic permutation mapping $\pi: S \rightarrow S$:

$$\pi = (7 \rightarrow 17 \rightarrow 29 \rightarrow 43 \rightarrow 61 \rightarrow 7)$$

| Job ID | Recipient Seed $i$ | Recipient Init $W_0^{(i)}$ Hash | Donor Seed $\pi(i)$ | Donor Mask $M_{\pi(i)}$ Hash | Comparator (Own Mask) |
| :---: | :---: | :---: | :---: | :---: | :---: |
| `e16_transfer_rec7_don17` | **7** | `0fdff96d...` | **17** | `3dc09fc4...` | `e16_seed7_s0.80_own` |
| `e16_transfer_rec17_don29` | **17** | `13a7c64a...` | **29** | `34e569eb...` | `e16_seed17_s0.80_own` |
| `e16_transfer_rec29_don43` | **29** | `4df7bb53...` | **43** | `365e94b4...` | `e16_seed29_s0.80_own` |
| `e16_transfer_rec43_don61` | **43** | `c21d8b67...` | **61** | `5a073f1a...` | `e16_seed43_s0.80_own` |
| `e16_transfer_rec61_don7` | **61** | `e76fa253...` | **7** | `e19a4e23...` | `e16_seed61_s0.80_own` |

### 2.2 Reused Controls (0 additional compute)
- **Own-mask recovery controls:** The 5 original IMP round-8 recovery runs from `dataset-a-imp-r0-v1` are reused directly:
  - Seed 7: CE = `1.536968`
  - Seed 17: CE = `1.458223` (beats dense `1.479363`, gap = `−0.021140`)
  - Seed 29: CE = `1.537243`
  - Seed 43: CE = `1.488347`
  - Seed 61: CE = `1.487532`
- **Dense controls:** The 5 matched unpruned dense baselines from `dataset-a-imp-r0-v1` are reused directly.

### 2.3 Training & Optimization Protocol (Identical to Primary Audit)
- **Initial model state:** The complete recipient step-0 checkpoint ($W_0^{(i)}$) is restored.
- **Mask application:** The foreign donor mask $M_{\pi(i)}$ is applied directly to expert `fc1` and `fc2` weights in native coordinate order. Shared parameters (embeddings, attention, layer norms, router projection) remain unmasked.
- **Optimizer:** Fresh AdamW ($\text{LR}=0.0003$, $\beta_1=0.9, \beta_2=0.999$, weight decay $0.01$, gradient clipping $1.0$).
- **Precision:** FP16 mixed precision with `torch.cuda.amp.GradScaler`.
- **RNG & Stream:** Training stream consumes the exact source-specific batches reconstructed by native loader for recipient seed $i$. RNG seed reset to recipient seed $i$ at step 0.
- **Budget:** Exactly 2,500 attempted updates; no scheduler, auxiliary routing loss weight $0.1$, batch size $128 \times 128$.
- **Readout:** Model snapshots at global steps 0, 50, 250, 500, 1000, 1500, 2000, 2500 evaluated on the full 492,416-token validation set in pure FP32 with TF32 disabled.

---

## 3. Prespecified Statistical Endpoints & Falsification Criteria

### 3.1 Primary Endpoint: Paired Transfer Degradation
For each recipient seed $i \in S$, define the paired transfer difference:
$$\Delta_i = \text{CE}(M_{\pi(i)}, W_0^{(i)}) - \text{CE}(M_i, W_0^{(i)})$$
where positive $\Delta_i$ indicates that transferring a foreign mask degrades recovery performance relative to the recipient's own discovered mask.

### 3.2 Decision Rules and Hypotheses Testing

1. **Rejection of Hypothesis 1 (Intrinsic Topology Alone):**
   - If mean $\bar{\Delta} > +0.03$ CE across the 5 seeds and $\Delta_i > 0$ for at least 4 of 5 seeds, we **reject** the hypothesis that the IMP mask acts merely as a generic topological filter. The advantage requires native coordinate alignment with its discovering initialization.
2. **Rejection of Hypothesis 2 (Strict Coordinate Value Requirement):**
   - If mean $\bar{\Delta} \le +0.005$ CE (statistically indistinguishable from zero within the within-seed recovery order noise band of $\approx 0.019$ CE), we **reject** the hypothesis that the mask requires its exact original initial weights. This would prove that discovered masks are universally transferable across random initializations.
3. **Comparison with Random-Mask Baseline:**
   - Define the transfer advantage over random placement:
     $$\Gamma_i = \text{CE}(M_{\text{random}}, W_0^{(i)}) - \text{CE}(M_{\pi(i)}, W_0^{(i)})$$
   - If $\Gamma_i \approx 0.14$ CE (matching the matrix/degree control penalty), the foreign mask behaves no better than a random mask. If $0 < \bar{\Delta} \ll \bar{\Gamma}$, the foreign mask transfers partial architectural benefit but loses coordinate-specific synergy.

---

## 4. Execution Guardrails & Stopping Rules

1. **Pre-execution Verification:**
   - Byte-level SHA256 verification of all 5 recipient step-0 checkpoints and 5 donor masks against authoritative manifest ([`provenance.csv`](file:///users/kent/student/mweber54/moe-lth-research/moe-lth-research/results/imp_ticket_audit_v1/provenance.csv)).
   - Verification that recipient data loader sequence hashes match the verified primary runs.
2. **Deterministic Dispatch:**
   - Exactly one worker per available GPU using atomic lockfiles.
   - Resumption checkpoints saved at step 1000 and 2000.
3. **No Outcome-Dependent Modification:**
   - Fixed budget of 2,500 updates; no run extension, no extra seed addition, no post-hoc tuning.
4. **Execution Command:**
   ```bash
   OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 .venv/bin/python scripts/imp_mask_initialization_transfer_v1/run.py execute
   ```

