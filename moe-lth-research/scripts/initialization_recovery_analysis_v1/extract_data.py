import csv
import json
from pathlib import Path
import torch
import numpy as np
from scipy import stats

ROOT = Path(__file__).resolve().parents[2]
comp_path = ROOT / 'results/imp_ticket_audit_v1/comparison.csv'
comp_rows = list(csv.DictReader(open(comp_path)))

print(f"Loaded {len(comp_rows)} primary recovery experiments from comparison.csv")

# Test extracting weights for one model
r0 = comp_rows[0]
arch = r0['architecture']
seed = int(r0['source_seed'])
sparsity = float(r0['sparsity'])
init_path = ROOT / r0['initialization_path']
mask_path = ROOT / r0['mask_path']

print(f"Loading {arch} seed {seed} sparsity {sparsity}...")
init_payload = torch.load(init_path, map_location='cpu', weights_only=True)
mask_payload = torch.load(mask_path, map_location='cpu', weights_only=True)

model_state = init_payload['model']

# Expert weights
fc1_weights = []
fc2_weights = []
fc1_masks = []
fc2_masks = []

for k, v in model_state.items():
    if 'moe.experts' in k and 'weight' in k:
        m = mask_payload[k].bool()
        w = v.float()
        if 'fc1' in k:
            fc1_weights.append(w)
            fc1_masks.append(m)
        elif 'fc2' in k:
            fc2_weights.append(w)
            fc2_masks.append(m)

print(f"Loaded {len(fc1_weights)} fc1 matrices, {len(fc2_weights)} fc2 matrices")
fc1_all = torch.cat([w.reshape(-1) for w in fc1_weights]).numpy()
fc2_all = torch.cat([w.reshape(-1) for w in fc2_weights]).numpy()
print(f"fc1 count: {len(fc1_all)}, mean: {fc1_all.mean():.6f}, std: {fc1_all.std():.6f}, RMS: {np.sqrt(np.mean(fc1_all**2)):.6f}")
print(f"fc2 count: {len(fc2_all)}, mean: {fc2_all.mean():.6f}, std: {fc2_all.std():.6f}, RMS: {np.sqrt(np.mean(fc2_all**2)):.6f}")
print("Expected fc1 std (uniform [-1/16, 1/16]):", 1.0 / (16.0 * np.sqrt(3)))
print("Expected fc2 std (uniform [-1/32, 1/32]):", 1.0 / (32.0 * np.sqrt(3)))
