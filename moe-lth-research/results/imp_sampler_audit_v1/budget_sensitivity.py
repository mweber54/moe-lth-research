"""Focused CPU proposal-budget sensitivity for representative failing controls."""
import importlib.util, json, math, sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
V2, V3 = HERE.parent / "imp_early_sign_reinforcement_neuron_matched_v2", HERE.parent / "imp_sign_repair_mask_routing_v3"
sys.path[:0] = [str(HERE), str(V2), str(V3)]
from common_v2 import pilot as p
spec = importlib.util.spec_from_file_location("cpu_audit_local", HERE / "audit.py")
A = importlib.util.module_from_spec(spec)
spec.loader.exec_module(A)

np, torch = p.np, p.torch
rect = A.import_rect()
sampler = A.load_sampler(HERE / "rectangle_sampler_validated.so", "rectangles_validated")
archive = json.loads((V3 / "audit_manifest.json").read_text())
controls = json.loads((V3 / "degree_masks/manifest.json").read_text())["controls"]
native = {(int(r["seed"]), int(r["draw"]), r["tensor"]): float(r["realized_distance"])
          for r in p.rows(p.PRIOR / "matrix_invariants.csv") if r["mask_type"] == "imp" and r["arm"] == "Bm"}
out = []
for seed, order, replicate in ((7, "A", 1), (29, "A", 2)):
    entry = next(x for x in archive["inputs"] if x["seed"] == seed and x["order"] == order)
    ctrl = next(x for x in controls if x["seed"] == seed and x["replicate"] == replicate)
    variant = dict(entry, base=dict(entry["base"], mask_path=ctrl["path"], mask_sha256=ctrl["mask_hash"]))
    dense = p.N._read_model_payload(Path(entry["base"]["initial_checkpoint"]))["model"]
    masks = p.N.load_masks(Path(ctrl["path"]))
    scores = torch.load(p.OUT / f"probe_{seed}_{order}.pt", weights_only=True, mmap=True)["scores"]
    matrices = {name: (dense[name].shape, rect.matrix_inputs(dense[name], *scores[name], mask)) for name, mask in masks.items()}
    for budget in (250_000, 1_000_000, 2_000_000):
        rect.PROPOSALS = budget
        pools = list(rect.collect(variant, 1, matrices, native, sampler))
        maximum = math.sqrt(sum(m.prefix[-1] for m in pools) / sum(m.denominator for m in pools))
        out.append({"seed": seed, "order": order, "label": f"degree{replicate}", "proposals_per_matrix": budget,
                    "nonempty_matrices": sum(len(m.prefix) > 1 for m in pools),
                    "rectangles": sum(len(m.prefix) - 1 for m in pools), "maximum_model_dose": maximum})
    del dense, masks, scores, matrices
(HERE / "proposal_budget_sensitivity.json").write_text(json.dumps(out, indent=2) + "\n")
print(json.dumps(out, indent=2))
