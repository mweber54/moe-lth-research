# Phase 4: Scaling and Robustness

The default grid covers three seeds, 4/8/16 experts, 4/8 layers, and two
load-balancing strengths. Pass `--grid-json` to run a smaller or larger grid.

```powershell
python -m moe_lth.experiments.run_robustness --config configs/baseline.yaml
```
