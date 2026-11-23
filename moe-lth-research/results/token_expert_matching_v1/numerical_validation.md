# Numerical validation

PASS: seven CPU tests cover exact identity forward, gate/acceptance conservation, token changes, independent deterministic permutations, frozen-router task gradients, shared/expert task and auxiliary paths, native recipient auxiliary decisions, corrupt/missing slot rejection, named missing/frozen parameters, within-seed draw averaging and incomplete-cohort rejection. CPU fixtures use Torch2.13; production validation uses canonical Torch2.7.0 CUDA FP16 on both hosts.

PASS on Viper and Adder: original32E IMP80 seed17 state and exact materialized batches; three native updates with and without passive capture; exact equality of model, optimizer, scaler, Python/NumPy/CPU/CUDA RNG state. Every same-state identity replay matched donor logits, auxiliary and accepted mask exactly. Shuffling changed expert/token correspondence while conservation assertions passed. Expert/shared task gradients and upstream shared auxiliary gradients remained; router parameters had no gradients.

PASS on both hosts: interrupted shuffled replay restores completed-step model/optimizer/scaler/all RNG state and data position exactly; the registered second-call reset preserves model tensors and has zero optimizer entries and fresh65536 loss scale. Missing payloads cannot fall back to native dispatch while replay is required. Payload step, batch, token hash, layer, capacity, indices and permutation bijection are checked.

Full5000-step passive donors must additionally match historical complete native CE curves, final model hash and applied/AMP counts before they become eligible for recipients. Replay identity is not required to match the native backward trajectory. Raw host results and preserved validation attempts are under validation/; CPU output is cpu_validation.log.
