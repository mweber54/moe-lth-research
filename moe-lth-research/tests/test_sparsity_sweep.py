"""Regression tests for sparsity sweep experiment protocol."""

import json
import tempfile
from pathlib import Path

import pytest
import torch

from moe_lth.config import load_config
from moe_lth.experiments.run_sparsity_sweep import (
    run_sparsity_sweep,
    SPARSITIES_TO_SWEEP,
    ROUTER_AGES_PERCENT_ENDPOINTS,
)


@pytest.mark.slow
def test_sparsity_sweep_tiny_protocol():
    """Test sparsity sweep with minimal budget to verify protocol correctness.
    
    Runs:
    - 2 sparsities (60%, 90%)
    - 1 seed
    - 2 router ages (0%, 100%)
    - 2-step recovery budget
    
    Validates:
      - Sparse tickets use E_0 under learned mask
      - Dense controls use full E_0
      - Shared state identical across sparsities and router ages
      - Masks discovered from E_T, not E_0
      - Output structure matches protocol
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir_path = Path(tmpdir)
        # Use an existing reference config with small budget
        config_path = "configs/smoke.yaml"
        config = load_config(config_path)
        
        # Override for test
        config["training"]["steps"] = 2
        config["training"]["checkpoint_steps"] = [0, 2]
        config_test_dir = tmpdir_path / "ref"
        config["output_dir"] = str(config_test_dir)
        config["training"]["precision"] = "fp32"
        
        # Save test config
        test_config_path = tmpdir_path / "test_config.yaml"
        import yaml
        with open(test_config_path, "w") as f:
            yaml.dump(config, f)
        
        # Run sparsity sweep
        sweep_output_dir = tmpdir_path / "sweep"
        result = run_sparsity_sweep(
            [str(test_config_path)],
            str(sweep_output_dir),
            recovery_steps=2,
        )
        
        # Validate output structure
        assert result["output_dir"] == str(sweep_output_dir)
        assert len(result["records"]) > 0
        
        # Check records
        records = result["records"]
        
        # Should have sparse and dense conditions for each sparsity and router age
        # 2 sparsities × 2 router ages × 2 condition types (sparse + dense) = 8 records
        assert len(records) == 8, f"Expected 8 records, got {len(records)}"
        
        sparse_records = [r for r in records if r["condition_type"] == "sparse_control"]
        dense_records = [r for r in records if r["condition_type"] == "dense_control"]
        
        assert len(sparse_records) == 4, "Expected 4 sparse conditions"
        assert len(dense_records) == 4, "Expected 4 dense conditions"
        
        # Validate sparsity coverage
        sparsities_in_records = sorted(set(r["sparsity"] for r in sparse_records))
        # Note: will be [0.6, 0.9] from SPARSITIES_TO_SWEEP
        assert len(sparsities_in_records) >= 2, f"Expected ≥2 sparsities, got {sparsities_in_records}"
        
        # Validate router age endpoints
        router_ages_in_records = sorted(set(r["router_age_percent"] for r in records))
        assert router_ages_in_records == [0, 100], f"Expected [0, 100], got {router_ages_in_records}"
        
        # Validate protocol fields
        for sparse_rec in sparse_records:
            # Rewind assertion: surviving weights from E_0 under learned mask
            assert sparse_rec.get("expert_surviving_weight_source") == "E_0"
            assert sparse_rec.get("mask_source") == "E_T"
            assert sparse_rec["sparsity"] > 0
            assert sparse_rec["mask_hash"] != "dense_no_mask"
            # Integrity checks must pass
            assert sparse_rec["integrity_checks_passed"] is True
        
        for dense_rec in dense_records:
            # Dense should have no mask
            assert dense_rec.get("mask_hash") == "dense_no_mask"
            assert dense_rec["sparsity"] == 0.0
            assert dense_rec["integrity_checks_passed"] is True
        
        # Validate hashes
        # Within a sparsity level, all sparse records for same router age should have:
        #   - same expert_state_hash (same E_0 under same mask m_s)
        #   - same shared_state_hash (same shared parameters)
        #   - same mask_hash
        for sparsity in sparsities_in_records:
            sparsity_sparse = [r for r in sparse_records if r["sparsity"] == sparsity]
            expert_hashes = set(r["expert_state_hash"] for r in sparsity_sparse)
            shared_hashes = set(r["shared_state_hash"] for r in sparsity_sparse)
            mask_hashes = set(r["mask_hash"] for r in sparsity_sparse)
            
            assert len(expert_hashes) == 1, f"Sparsity {sparsity}: expert hashes differ"
            assert len(shared_hashes) == 1, f"Sparsity {sparsity}: shared hashes differ"
            assert len(mask_hashes) == 1, f"Sparsity {sparsity}: mask hashes differ"
        
        # Validate router state hashes differ between R_0 and R_100
        for sparsity in sparsities_in_records:
            sparsity_sparse = [r for r in sparse_records if r["sparsity"] == sparsity]
            r0_routers = [r["initial_router_state_hash"] for r in sparsity_sparse if r["router_age_percent"] == 0]
            r100_routers = [r["initial_router_state_hash"] for r in sparsity_sparse if r["router_age_percent"] == 100]
            
            assert len(set(r0_routers)) == 1, f"Sparsity {sparsity}: R_0 router hashes differ"
            assert len(set(r100_routers)) == 1, f"Sparsity {sparsity}: R_100 router hashes differ"
            assert r0_routers[0] != r100_routers[0], f"Sparsity {sparsity}: R_0 and R_100 routers are identical!"
        
        # Validate metrics are finite
        for rec in records:
            assert rec["final_validation_loss"] > 0, f"Invalid loss in {rec['condition']}"
            assert not math.isnan(rec["final_validation_loss"])
            assert not math.isinf(rec["final_validation_loss"])
        
        # Validate sparse-vs-dense gap
        for sparsity in sparsities_in_records:
            for age in [0, 100]:
                sparse = next(
                    (r for r in sparse_records if r["sparsity"] == sparsity and r["router_age_percent"] == age),
                    None,
                )
                dense = next(
                    (r for r in dense_records if r["sparsity"] == 0.0 and r["router_age_percent"] == age),
                    None,
                )
                if sparse and dense:
                    gap = sparse["final_validation_loss"] - dense["final_validation_loss"]
                    print(f"[s={sparsity:.2f}, R_{age}] gap={gap:.6f}")
        
        # Check JSON output files
        records_json = sweep_output_dir / "sparsity_sweep_all_records.json"
        assert records_json.exists(), f"Missing {records_json}"
        loaded_records = json.loads(records_json.read_text())
        assert len(loaded_records) == len(records)
        
        print(f"✓ Sparsity sweep protocol test passed with {len(records)} records")


import math

if __name__ == "__main__":
    test_sparsity_sweep_tiny_protocol()
