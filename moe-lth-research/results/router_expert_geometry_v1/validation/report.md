# Instrumentation validation

All focused checks passed: analytic alignment, sign/rotation invariance, centering,
identical geometry/functions, exact native capacity traces, retained mask counts,
fixed probe identities, modes, parameters/buffers, gradients, optimizer and batch order.
Learned masks regenerated exactly from finalized source checkpoints; deterministic
random masks regenerated exactly and retained counts match in every tensor.

Final 25-step 32E seed-7/R0/learned/50% GPU comparison: maximum parameter difference
0.0, validation-loss difference 0.0,
worker/standalone step-zero alignment difference 0.0.
Applied updates, AMP skips/scales and canonical batch hashes agree. Tolerance:
weights atol 1e-6/rtol 1e-5, loss/alignment 1e-6. The observed comparison was exact;
this does not promise bitwise determinism across devices/concurrent workloads.

Whole-callback audits at 0 and 25 (validation/enabled_v10/preservation_step*.json)
verified unchanged parameters/buffers, gradients, AdamW, GradScaler, module modes,
Python/NumPy/CPU/CUDA RNG, all batch tensors, TF32 flags and autocast states.
Diagnostics use full FP32 with TF32 disabled and restore caller flags; native
validation retains repository precision. CPU FP64 Gram eigenspaces were checked
against direct SVD. Frozen/native functions and native traces passed focused checks.

Partial smoke attempts are preserved and excluded, never resumed for production.
Final timing/storage estimates are in resource_estimate.json; initial source,
mask and model-tensor inventories are in the result root. No shared code changed.
