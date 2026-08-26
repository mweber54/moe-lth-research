# Phase 2: Rewind and Lottery-Ticket Tests

Run learned-mask, random-mask, random-reinitialization, and randomized-routing
controls at every configured rewind fraction.

## Fast Smoke Run

Run this after the Phase 1 smoke suite:

```powershell
python -m moe_lth.experiments.run_rewind_suite --config configs/smoke.yaml --final-checkpoint results/smoke_suite/normal/checkpoints/step_3.pt --sparsity 0.8
```

## Full Baseline Run

Run this after the Phase 1 full baseline suite:

```powershell
python -m moe_lth.experiments.run_rewind_suite --config configs/baseline.yaml --final-checkpoint results/runs/baseline_suite/normal/checkpoints/step_50000.pt --sparsity 0.8
```

Rewind artifacts are written beside the selected final checkpoint.

## RTX 3080 TinyStories Run

After completing the GPU Phase 1 suite:

```powershell
conda activate torch-gpu
python -m moe_lth.experiments.run_rewind_suite --config configs/tinystories_gpu.yaml --final-checkpoint results/tinystories_gpu_suite/normal/checkpoints/step_2500.pt --sparsity 0.5
python -m moe_lth.experiments.run_rewind_suite --config configs/tinystories_gpu.yaml --final-checkpoint results/tinystories_gpu_suite/normal/checkpoints/step_2500.pt --sparsity 0.8
```

## RTX 3080 WikiText-103 Subset Run

After completing the WikiText GPU Phase 1 suite:

```powershell
conda activate torch-gpu
python -m moe_lth.experiments.run_rewind_suite --config configs/wikitext103_gpu.yaml --final-checkpoint results/wikitext103_gpu_suite/normal/checkpoints/step_2500.pt --sparsity 0.5
python -m moe_lth.experiments.run_rewind_suite --config configs/wikitext103_gpu.yaml --final-checkpoint results/wikitext103_gpu_suite/normal/checkpoints/step_2500.pt --sparsity 0.8
```
