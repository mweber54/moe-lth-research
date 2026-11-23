# Implementation validation

- Canonical torch 2.7.0: 21 tests passed across test_mask_information.py, test_imp.py, test_router_age_recovery.py and test_pruning.py.
- Development environment: 13 mask/IMP/pruning tests passed; 2 reporting tests passed.
- Reporting tests use synthetic observations only in pytest temporary directories. They verify nested sample counts and all five figure types. No synthetic scientific observations are written to the experiment results.
- New workflow scripts compile successfully.

Mask tests cover exact required survivor counts, independent layer/expert grouping, deterministic seeds, differing realizations, forced equality, malformed masks/names, protected weights, and zero pruned weights after AdamW with preexisting momentum and weight decay.

Numerical reproduction of a full original IMP recovery is a separate scientific execution gate, not replaced by passing unit tests. It runs before any randomized recovery.
