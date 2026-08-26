import tempfile
from pathlib import Path
import numpy as np
import torch
import pytest

from moe_lth.models.moe_layer import RouteTrace
from moe_lth.routing.rich_trace import RichRouteHistory, upgrade_legacy_history
from moe_lth.routing.route_history import RouteHistory

def test_rich_trace_record_save_load_hash_verify():
    # 1. Create a mock RouteTrace
    batch_size = 2
    seq_len = 3
    num_experts = 4
    top_k = 2
    num_tokens = batch_size * seq_len
    
    selected_experts = torch.randint(0, num_experts, (batch_size, seq_len))
    selected_probability = torch.rand((batch_size, seq_len))
    selected_expert_indices = torch.randint(0, num_experts, (batch_size, seq_len, top_k))
    selected_probabilities = torch.rand((batch_size, seq_len, top_k))
    entropy = torch.rand((batch_size, seq_len))
    margin = torch.rand((batch_size, seq_len))
    usage = torch.rand((num_experts,))
    dropped_fraction = torch.tensor(0.1)
    accepted_mask = torch.ones((num_tokens, top_k), dtype=torch.bool)
    # Set a few to false
    accepted_mask[0, 1] = False
    
    trace = RouteTrace(
        selected_experts=selected_experts,
        selected_probability=selected_probability,
        selected_expert_indices=selected_expert_indices,
        selected_probabilities=selected_probabilities,
        entropy=entropy,
        margin=margin,
        usage=usage,
        dropped_fraction=dropped_fraction,
        accepted_mask=accepted_mask
    )
    
    # 2. Record it to a RichRouteHistory
    history = RichRouteHistory()
    step = 1
    layer_id = 0
    history.record(step, layer_id, trace, batch_size, seq_len)
    assert history.schema_version == RichRouteHistory.TRACE_SCHEMA_VERSION
    assert history.metadata["top_k"] == top_k
    assert history.metadata["num_experts"] == num_experts
    
    # Verify hash determinism
    hash1 = history.compute_hash()
    hash2 = history.compute_hash()
    assert hash1 == hash2
    
    # 3. Verify integrity
    history.verify_integrity()
    
    # 4. Save and load
    with tempfile.TemporaryDirectory() as tmpdir:
        save_path = Path(tmpdir) / "test_rich.npz"
        history.save(save_path)
        
        loaded_history = RichRouteHistory.load(save_path)
        loaded_history.verify_integrity()
        assert loaded_history.metadata["schema_version"] == history.schema_version
        assert loaded_history.metadata["top_k"] == top_k
        
        # Check that loaded is identical
        loaded_hash = loaded_history.compute_hash()
        assert loaded_hash == hash1
        
        loaded_trace = loaded_history.get(step, layer_id)
        np.testing.assert_array_equal(loaded_trace.selected_expert_ids, trace.selected_expert_indices.reshape(num_tokens, top_k).numpy())
        
        # gate_values are converted to float16, use appropriate tolerance
        np.testing.assert_allclose(loaded_trace.gate_values, trace.selected_probabilities.reshape(num_tokens, top_k).numpy(), rtol=1e-3, atol=1e-3)
        np.testing.assert_array_equal(loaded_trace.accepted_mask, trace.accepted_mask.numpy())

def test_upgrade_legacy_history():
    old_history = RouteHistory()
    step = 1
    layer_id = 2
    batch_size = 2
    seq_len = 3
    num_tokens = batch_size * seq_len
    selected_experts = torch.randint(0, 4, (batch_size, seq_len))
    old_history.record(step, layer_id, selected_experts)
    
    rich_history = upgrade_legacy_history(old_history)
    rich_history.verify_integrity()
    
    trace = rich_history.get(step, layer_id)
    assert trace.step == step
    assert trace.layer_id == layer_id
    assert trace.selected_expert_ids.shape == (num_tokens, 1)
    
    # Check that gate_values are NaN and accepted_mask is True
    assert np.isnan(trace.gate_values).all()
    assert trace.accepted_mask.all()
    
    # Check shape
    assert trace.batch_indices.shape == (num_tokens,)
    assert trace.seq_positions.shape == (num_tokens,)
    
    # Check batch_indices correctness
    expected_batch_indices = np.repeat(np.arange(batch_size), seq_len)
    np.testing.assert_array_equal(trace.batch_indices, expected_batch_indices)
