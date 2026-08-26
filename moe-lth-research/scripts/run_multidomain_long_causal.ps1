$ErrorActionPreference = "Stop"

Set-Location "C:\Users\User\moe-lth\moe-lth-research"
$env:PYTHONPATH = "src"

& "C:\Users\User\miniconda3\envs\thgnn\python.exe" `
  -m moe_lth.experiments.run_multidomain_causal_controls `
  --configs `
    configs\multidomain_long_gpu.yaml `
    configs\multidomain_long_gpu_seed17.yaml `
    configs\multidomain_long_gpu_seed29.yaml `
  --output-dir results\multidomain_long_causal_controls
