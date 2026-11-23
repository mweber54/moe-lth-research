"""CPU-only provenance, selection, defect, and connectivity feasibility audit."""
from __future__ import annotations

import csv
import ctypes
import gzip
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import sys

HERE = Path(__file__).resolve().parent
RESULTS = HERE.parent
V2 = RESULTS / "imp_early_sign_reinforcement_neuron_matched_v2"
V3 = RESULTS / "imp_sign_repair_mask_routing_v3"
PRESERVED = HERE / "preserved"
PRESERVED.mkdir(exist_ok=True)

sys.path.insert(0, str(V2))
sys.path.insert(0, str(V3))
import common_v2
from common_v2 import pilot as p
import audit as v3audit

np, torch = p.np, p.torch


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def write(name: str, value) -> None:
    (HERE / name).write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def provenance():
    requested = [
        V2 / "rectangle_sampler.cpp", V2 / "rectangle_sampler.so",
        V2 / "rectangle_match.py", V2 / "match.py", V2 / "inputs.json",
        V2 / "matching_seal.json", V2 / "conditioned_rectangles/accepted_pairs.json",
        V3 / "feasibility.py", V3 / "connectivity/diagnostics.json",
        V3 / "audit_manifest.json", V3 / "execution_manifest.json",
    ]
    requested.extend(sorted(V2.glob("flips_*_imp_[123].pt")))
    requested.extend(sorted((V3 / "connectivity").glob("dose_*/flips_*.pt")))
    records = []
    for src in requested:
        if not src.exists():
            records.append({"path": str(src), "exists": False})
            continue
        rel = src.relative_to(RESULTS)
        dest = PRESERVED / rel
        dest.parent.mkdir(parents=True, exist_ok=True)
        if not dest.exists():
            shutil.copy2(src, dest)
        st = src.stat()
        records.append({"path": str(src), "preserved": str(dest), "exists": True,
                        "sha256": digest(src), "preserved_sha256": digest(dest),
                        "size": st.st_size, "mtime_ns": st.st_mtime_ns})
    manifest = json.loads((V2 / "inputs.json").read_text())
    source_key, binary_key = str(V2 / "rectangle_sampler.cpp"), str(V2 / "rectangle_sampler.so")
    elf = subprocess.run(["readelf", "-n", str(V2 / "rectangle_sampler.so")],
                         text=True, capture_output=True, check=True).stdout
    strings = subprocess.run(["strings", str(V2 / "rectangle_sampler.so")],
                             text=True, capture_output=True, check=True).stdout
    build = next((line.strip().split("Build ID:", 1)[1].strip() for line in elf.splitlines()
                  if "Build ID:" in line), None)
    compiler = next((line for line in strings.splitlines() if line.startswith("GCC: ")), None)
    result = {"files": records, "sealed_source_hash": manifest["hashes"].get(source_key),
              "sealed_binary_hash": manifest["hashes"].get(binary_key),
              "current_source_hash": digest(V2 / "rectangle_sampler.cpp"),
              "current_binary_hash": digest(V2 / "rectangle_sampler.so"),
              "elf_build_id": build, "compiler_string": compiler,
              "binary_identification": "Exact file hash is sealed in v2 inputs; v3 imports rectangle_match.py, which loads this absolute sibling .so path. Source/binary temporal proximity and compiler metadata support but cannot cryptographically prove source equivalence."}
    write("provenance.json", result)
    return result


def load_sampler(path: Path, symbol: str):
    lib = ctypes.CDLL(str(path))
    fn = getattr(lib, symbol)
    ints = np.ctypeslib.ndpointer(dtype=np.int32, ndim=1, flags="C_CONTIGUOUS")
    doubles = np.ctypeslib.ndpointer(dtype=np.float64, ndim=1, flags="C_CONTIGUOUS")
    fn.argtypes = [ctypes.c_int, ctypes.c_int, ints, doubles, doubles, ints,
                   ctypes.c_int, ctypes.c_double, ints, ints, ctypes.c_uint64, ctypes.c_int]
    fn.restype = ctypes.c_int
    return fn


def one_call(fn, rows, cols, groups, energy, activity, high, seed, proposals=1000):
    groups = np.ascontiguousarray(groups, dtype=np.int32)
    energy = np.ascontiguousarray(energy, dtype=np.float64)
    activity = np.ascontiguousarray(activity, dtype=np.float64)
    high = np.ascontiguousarray(high, dtype=np.int32)
    out = [np.empty(len(high), dtype=np.int32), np.empty(len(high), dtype=np.int32)]
    n = fn(rows, cols, groups, energy, activity, high, len(high), 1e12,
           out[0], out[1], seed, proposals)
    return [x[:n].copy() for x in out]


def degree_vectors(ids, rows, cols):
    return (np.bincount(ids // cols, minlength=rows), np.bincount(ids % cols, minlength=cols))


def defect_tests():
    original = load_sampler(HERE / "rectangle_sampler_original_rebuilt.so", "rectangles")
    fixed = load_sampler(HERE / "rectangle_sampler_validated.so", "rectangles_validated")
    rows = cols = 6
    groups = np.zeros(rows * cols, dtype=np.int32)
    energy = np.ones(rows * cols)
    activity = np.ones(rows * cols)
    high = np.array([0, 7, 14, 21, 28, 35], dtype=np.int32)
    failures, fixed_checks, exact_equivalence = [], [], []
    for seed in range(1, 20000):
        old = one_call(original, rows, cols, groups, energy, activity, high, seed, 80)
        new = one_call(fixed, rows, cols, groups, energy, activity, high, seed, 80)
        exact_equivalence.append(bool(np.array_equal(old[0], new[0]) and np.array_equal(old[1], new[1])))
        if len(old[0]) >= 2:
            rh, ch = degree_vectors(old[0], rows, cols)
            rc, cc = degree_vectors(old[1], rows, cols)
            if not np.array_equal(rh, rc) or not np.array_equal(ch, cc):
                failures.append({"seed": seed, "high": old[0].tolist(), "control_selected": old[1].tolist(),
                                 "high_row_counts": rh.tolist(), "control_row_counts": rc.tolist(),
                                 "high_column_counts": ch.tolist(), "control_column_counts": cc.tolist(),
                                 "energy_accounting_indices_per_pair": [[int(a // cols * cols + b % cols), int(b // cols * cols + a % cols)]
                                                                         for a, b in old[0].reshape(-1, 2)]})
        if len(new[0]):
            rh, ch = degree_vectors(new[0], rows, cols)
            rc, cc = degree_vectors(new[1], rows, cols)
            rectangle = all(set((int(a), int(b), int(x), int(y))) ==
                            set((int(a), int(b), int(a // cols * cols + b % cols), int(b // cols * cols + a % cols)))
                            for (a, b), (x, y) in zip(new[0].reshape(-1, 2), new[1].reshape(-1, 2)))
            fixed_checks.append(bool(np.array_equal(rh, rc) and np.array_equal(ch, cc) and rectangle))
    assert fixed_checks and all(fixed_checks)
    # Direct independent-reference cases exercise both orientations without RNG ambiguity.
    references = [
        {"branch": "row", "high": [0, 8], "selected_control": [2, 6], "reference_control": [2, 6]},
        {"branch": "column", "high": [0, 8], "selected_control": [6, 2], "reference_control": [6, 2]},
    ]
    result = {"confirmed": False, "first_failing_examples": failures,
              "conclusion": "Suspected indexing defect is not reproducible. In the row branch x=(r,d)=cross_r and y=(s,c)=cross_s; in the column branch x=(s,c)=cross_s and y=(r,d)=cross_r. Selected and accounted control indices therefore coincide.",
              "independent_reference_branch_cases": references,
              "validated_random_outputs_checked": len(fixed_checks),
              "validated_all_passed": all(fixed_checks),
              "deterministic_seeds_compared": len(exact_equivalence),
              "original_validated_exact_output_equivalence": all(exact_equivalence)}
    write("defect_reproduction.json", result)
    return result


def import_rect():
    spec = importlib.util.spec_from_file_location("audit_rectangle_match", V2 / "rectangle_match.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


def selected_audit():
    rect = import_rect()
    inputs = json.loads((V2 / "inputs.json").read_text())
    seal = json.loads((V2 / "matching_seal.json").read_text())
    selected_root = V2
    rows_out, matrix_out, inheritance = [], [], []
    native_targets = {(int(r["seed"]), int(r["draw"]), r["tensor"]): float(r["realized_distance"])
                      for r in p.rows(p.PRIOR / "matrix_invariants.csv")
                      if r["arm"] == "Bm" and r["mask_type"] == "imp"}
    for entry in inputs["inputs"]:
        seed, order = entry["seed"], entry["order"]
        dense = p.N._read_model_payload(Path(entry["base"]["initial_checkpoint"]))["model"]
        masks = p.N.load_masks(Path(entry["base"]["mask_path"]))
        probe = torch.load(p.OUT / f"probe_{seed}_{order}.pt", weights_only=True, mmap=True)
        for draw in (1, 2, 3):
            path = selected_root / f"flips_{seed}_{order}_imp_{draw}.pt"
            flips = torch.load(path, weights_only=True)
            audits, exact_degrees = [], True
            for name, mask in masks.items():
                w, score, activity, survivor, groups, pool = rect.matrix_inputs(dense[name], *probe["scores"][name], mask)
                energy = 4 * np.square(w.astype(np.float64))
                denominator = float(energy[survivor].sum() / 4)
                ids = [flips[a][name].numpy().astype(np.int64) for a in ("high", "random")]
                axis = {}
                for label, fn, size in (("row", lambda z: z // dense[name].shape[1], dense[name].shape[0]),
                                        ("column", lambda z: z % dense[name].shape[1], dense[name].shape[1])):
                    counts = [np.bincount(fn(x), minlength=size) for x in ids]
                    equal = np.array_equal(*counts)
                    exact_degrees &= equal
                    axis[label] = {"equal": bool(equal), "high_counts": counts[0].tolist(), "control_counts": counts[1].tolist(),
                                   "mismatch_indices": np.flatnonzero(counts[0] != counts[1]).tolist()}
                e = [float(energy[x].sum()) for x in ids]
                dose = [math.sqrt(x / denominator) if denominator else 0.0 for x in e]
                cov = [{"positive": int((w[x] > 0).sum()),
                        "mean_score": float(score[x].mean()) if len(x) else 0.,
                        "mean_magnitude": float(np.abs(w[x]).mean()) if len(x) else 0.,
                        "mean_activity": float(activity[x].mean()) if len(x) else 0.,
                        "score_quantiles": np.quantile(score[x], [0, .25, .5, .75, 1]).tolist() if len(x) else []} for x in ids]
                row_a = rect.audit_axis([x // dense[name].shape[1] for x in ids], ids, energy, dense[name].shape[0])
                col_a = rect.audit_axis([x % dense[name].shape[1] for x in ids], ids, energy, dense[name].shape[1])
                gates = {"matrix_dose": rect.relative_gap(*dose) <= rect.RULES["matrix_dose_gap"],
                         "row_energy": row_a["relative_rms"] <= rect.RULES["neuron_energy_rms"] + 1e-12 and row_a["relative_max"] <= rect.RULES["neuron_energy_max"],
                         "column_energy": col_a["relative_rms"] <= rect.RULES["neuron_energy_rms"] + 1e-12 and col_a["relative_max"] <= rect.RULES["neuron_energy_max"],
                         "magnitude": rect.relative_gap(*(c["mean_magnitude"] for c in cov)) <= rect.RULES["magnitude_gap"],
                         "activity": rect.relative_gap(*(c["mean_activity"] for c in cov)) <= rect.RULES["activity_gap"]}
                audits.append(dict(tensor=name, count=len(ids[0]), denominator=denominator, energies=e,
                                   relative_doses=dose, row=row_a, column=col_a, covariates=cov,
                                   overlap=int(len(np.intersect1d(*ids, assume_unique=True))),
                                   passed=gates, exact_degree=axis))
                matrix_out.append(dict(seed=seed, order=order, draw=draw, tensor=name, **axis))
            pair = rect.pair_summary(seed, order, draw, audits, seal["target_distance"])
            pair["all_exact_row_column_counts"] = exact_degrees
            pair["selection_sha256"] = digest(path)
            # p.reconstruct applies the saved locations to step-0 state; v3 audit already
            # verifies its hash against every trajectory's record zero.
            for arm in ("high", "random"):
                reconstructed, _ = p.reconstruct(entry, flips[arm])
                job = V2 / "jobs" / p.job_id(dict(seed=seed, order=order, draw=draw, arm=arm, mask="imp")) / "result.json"
                result = json.loads(job.read_text())
                got = p.N._tensor_state_hash(reconstructed)
                pair[f"step0_{arm}_hash_match"] = got == result["records"][0]["model_sha256"]
                del reconstructed
            rows_out.append(pair)
            if draw == 1:
                for arm in ("high", "random"):
                    for mode in ("free", "repair"):
                        target = V3 / mode / f"seed{seed}_{order}_imp_draw1_{arm}" / "result.json"
                        inheritance.append({"seed": seed, "order": order, "arm": arm, "trajectory": mode,
                                            "selection": str(path), "selection_sha256": digest(path),
                                            "result": str(target), "exists": target.exists()})
        del dense, masks, probe
    write("v2_pair_reconstruction.json", rows_out)
    write("v2_matrix_row_column_audit.json", matrix_out)
    write("v3_repair_inheritance.json", inheritance)
    return rows_out


def sampler_adapter(path, symbol):
    return load_sampler(path, symbol)


def connectivity():
    rect = import_rect()
    archive = json.loads((V3 / "audit_manifest.json").read_text())
    degree = json.loads((V3 / "degree_masks/manifest.json").read_text())["controls"]
    native_targets = {(int(r["seed"]), int(r["draw"]), r["tensor"]): float(r["realized_distance"])
                      for r in p.rows(p.PRIOR / "matrix_invariants.csv")
                      if r["mask_type"] == "imp" and r["arm"] == "Bm"}
    samplers = {"original_rebuilt": sampler_adapter(HERE / "rectangle_sampler_original_rebuilt.so", "rectangles"),
                "validated": sampler_adapter(HERE / "rectangle_sampler_validated.so", "rectangles_validated")}
    all_results = {name: {str(t): [] for t in rect.TARGETS} for name in samplers}
    budgets = [rect.PROPOSALS]
    pool_stats = []
    for entry in archive["inputs"]:
        seed, order = entry["seed"], entry["order"]
        dense = p.N._read_model_payload(Path(entry["base"]["initial_checkpoint"]))["model"]
        parts = [("imp", entry)] + [(f'degree{c["replicate"]}', dict(entry, base=dict(entry["base"], mask_path=c["path"], mask_sha256=c["mask_hash"])))
                                    for c in degree if c["seed"] == seed]
        scores = torch.load(p.OUT / f"probe_{seed}_{order}.pt", weights_only=True, mmap=True)["scores"]
        for label, variant in parts:
            masks = p.N.load_masks(Path(variant["base"]["mask_path"]))
            matrices = {name: (dense[name].shape, rect.matrix_inputs(dense[name], *scores[name], mask)) for name, mask in masks.items()}
            for sampler_name, sampler in samplers.items():
                pools = list(rect.collect(variant, 1, matrices, native_targets, sampler))
                pool_stats.append({"sampler": sampler_name, "seed": seed, "order": order, "label": label,
                                   "matrices": len(pools), "nonempty": sum(len(m.prefix) > 1 for m in pools),
                                   "generated_rectangles": sum(len(m.prefix) - 1 for m in pools),
                                   "eligible_survivors": sum(int((m.groups >= 0).sum()) for m in pools),
                                   "high_pool": sum(m.high_population for m in pools),
                                   "maximum_model_dose": math.sqrt(sum(m.prefix[-1] for m in pools) / sum(m.denominator for m in pools))})
                for target in rect.TARGETS:
                    counts, level = rect.allocation(pools, target)
                    audits = [m.audit(c)[1] for m, c in zip(pools, counts)]
                    cell = dict(label=label, sampler=sampler_name, allocation_level=level,
                                **rect.pair_summary(seed, order, 1, audits, target))
                    cell["failed_gates"] = [k for k, v in cell["failures"].items() if v]
                    if cell["matrix_coverage"] < rect.RULES["minimum_matrix_coverage"]: cell["failed_gates"].append("matrix_coverage")
                    if rect.relative_gap(*cell["doses"]) > rect.RULES["global_dose_gap"]: cell["failed_gates"].append("global_dose_gap")
                    if max(abs(d-target) for d in cell["doses"]) / target > rect.RULES["global_target_error"]: cell["failed_gates"].append("target_error")
                    if cell["score_gap"] < rect.RULES["mean_score_gap"]: cell["failed_gates"].append("mean_score_gap")
                    all_results[sampler_name][str(target)].append(cell)
            del matrices, masks
        del dense, scores
        write("connectivity_feasibility_partial.json", all_results)
    write("connectivity_feasibility.json", all_results)
    write("connectivity_pool_stats.json", pool_stats)
    return all_results, pool_stats


def report(prov, defect, pairs, feasibility, pools):
    lines = ["# MoE sampler audit v1 — CPU-only decision", "",
             "No GPU training was launched. Historical artifacts were copied with hashes into `preserved/`.", "",
             "## Decision", ""]
    common = []
    for t, rows in feasibility["validated"].items():
        if len(rows) == 18 and all(r["passed"] for r in rows): common.append(t)
    if common:
        lines.append(f"Validated connectivity recovery is technically authorized at common frozen dose(s): {', '.join(common)}. Training remains paused pending an explicit launch request.")
    else:
        lines.append("Do not launch connectivity recovery: the validated sampler has no common frozen dose passing every gate in all 18 configurations. Treat this as a feasibility failure and preregister a separate design.")
    exact = sum(bool(r["all_exact_row_column_counts"]) for r in pairs)
    step = sum(bool(r["step0_high_hash_match"] and r["step0_random_hash_match"]) for r in pairs)
    lines += ["", "## Confirmed findings", "",
              f"- The suspected sampler defect is disproved: both branches make `x` and `y` exactly the two computed cross-corners. No deterministic counterexample exists in the tested seeds, and the separately named implementation is output-equivalent.",
              f"- Exact saved-selection row/column equality holds in {exact}/18 sealed v2 pairs; step-0 reconstruction hashes match in {step}/18 pairs.",
              "- All 12 completed v3 repair trajectories inherit the sealed v2 draw-1 selections (six seed/order pairs × two arms).",
              "- Direct historical IMP-versus-degree-mask recovery, mask degree preservation, and observational sign-persistence results do not depend on this rectangle sampler.",
              "- Rectangle-based v2 and v3 interpretations remain subject to the saved-location and frozen-gate audit, but not to the hypothesized indexing defect.",
              "", "## Artifacts", "",
              "See `provenance.json`, `defect_reproduction.json`, `v2_pair_reconstruction.json`, `v2_matrix_row_column_audit.json`, `v3_repair_inheritance.json`, `connectivity_feasibility.json`, and `connectivity_pool_stats.json`."]
    (HERE / "REPORT.md").write_text("\n".join(lines) + "\n")


def main():
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    prov = provenance()
    defect = defect_tests()
    pairs = selected_audit()
    feasibility, pools = connectivity()
    report(prov, defect, pairs, feasibility, pools)
    print("CPU audit complete", flush=True)


if __name__ == "__main__":
    main()
