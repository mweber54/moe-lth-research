"""Prepare and seal the exact-degree pilot follow-up using archived inputs."""
from collections import defaultdict
import ctypes
import gzip
import itertools
import json
import math

from common_v2 import OUT, PILOT, pilot as p

np, torch = p.np, p.torch
RULES = dict(targets=[.34207, .30, .25, .20, .15, .10],
             matrix_dose_gap=.05, global_dose_gap=.01, global_target_error=.05,
             neuron_energy_rms=.12, neuron_energy_max=.50, magnitude_gap=.05,
             activity_gap=.10, mean_score_gap=.10, minimum_matrix_coverage=.95,
             proposals_per_edge=160, minimum_proposals=200000)
DRAWS = (1, 2, 3)
ARMS = ("high", "random")
PAIRS = list(itertools.product(p.SEEDS, p.ORDERS, DRAWS))


def prepare():
    archive = p.read(PILOT / "inputs.json")
    inputs = [entry for entry in archive["inputs"] if entry["mask"] == "imp"]
    current = [entry for entry in p.sources() if entry["mask"] == "imp"]
    assert inputs == current and len(inputs) == 6
    hashes = {str(path): p.digest(path) for path in
              [PILOT / "inputs.json", PILOT / "probe_audit.csv", PILOT / "run.py",
               OUT / "match.py", OUT / "common_v2.py", OUT / "exchange.cpp",
               OUT / "exchange.so", OUT / "worker.py", OUT / "report.py"]}
    for path, expected in archive["hashes"].items():
        assert p.digest(path) == expected, path
        hashes[path] = expected
    calibration = defaultdict(list)
    for row in p.rows(PILOT / "probe_audit.csv"):
        calibration[int(row["seed"]), row["order"]].append(row)
    for seed in p.SEEDS:
        material = p.N._load_training_material(seed)
        for entry in (item for item in inputs if item["seed"] == seed):
            order = entry["order"]
            state, masks = p.baseline_state(entry)
            del state
            path = PILOT / f"probe_{seed}_{order}.pt"
            probe = torch.load(path, weights_only=True, mmap=True)
            assert probe["applied_updates"] == 100 and probe["max_reconstruction_error_fraction"] <= 1
            _, batches, validation = p.ordered(seed, order, material)
            assert p.N.recovery._batch_sequence_hash(validation) == entry["validation_hash"]
            identity = p.digest(path)
            batches_hash = p.N.recovery._batch_sequence_hash(batches)
            assert probe["train_hash"] == batches_hash
            audit = calibration[seed, order]
            assert set(probe["scores"]) == set(masks) == {row["tensor"] for row in audit}
            assert all(row["score_map_sha256"] == identity and row["ordered_batches_sha256"] == batches_hash for row in audit)
            hashes[str(path)] = identity
            baseline = p.reuse_baseline(entry, material)
            assert baseline == p.read(PILOT / f"baseline_{seed}_{order}_imp.json")
            dest = OUT / f"baseline_{seed}_{order}_imp.json"
            p.write(dest, baseline)
            hashes[str(dest)] = p.digest(dest)
            hashes[baseline["source"]] = baseline["sha256"]
            print("Verified inputs", seed, order, flush=True)
    jobs = [dict(seed=s, order=o, draw=d, arm=a, mask="imp") for s,o,d in PAIRS for a in ARMS]
    result = dict(inputs=inputs, jobs=jobs, hashes=hashes, rules=RULES, steps=p.STEPS,
                  seeds_namespace=OUT.name, optimizer=archive["optimizer"], router=archive["router"])
    p.write(OUT / "inputs.json", result)
    return result


def matrix_inputs(weight, score, activity, mask):
    w, r, a, survivor = (t.numpy().ravel() for t in (weight, score, activity, mask))
    eligible = np.flatnonzero((w != 0) & (a > 0) & np.isfinite(w) & np.isfinite(r) & np.isfinite(a))
    groups = np.full(len(w), -1, dtype=np.int32)
    if not len(eligible):
        return w, r, a, survivor, groups, eligible
    assert np.max(np.abs(r[eligible])) <= 1 + 1e-10
    rank = np.argsort(r[eligible], kind="stable")
    quartile = len(rank) // 4
    high = eligible[rank[-quartile:]] if quartile else eligible[:0]
    n = p.RULES["quantile_bins"]
    strata = (p.quantile_bin(np.abs(w[eligible]), n)*n + p.quantile_bin(a[eligible], n))*2 + (w[eligible]>0)
    surviving = survivor[eligible]
    groups[eligible[surviving]] = strata[surviving]
    return w, r, a, survivor, groups, high[survivor[high]]


def relative_gap(a, b):
    return abs(a-b) / max(abs(a), abs(b), 1e-30)


def swap_function():
    library = ctypes.CDLL(str(OUT / "exchange.so"))
    exchange = library.exchange
    integers = np.ctypeslib.ndpointer(dtype=np.int32, ndim=1, flags="C_CONTIGUOUS")
    doubles = np.ctypeslib.ndpointer(dtype=np.float64, ndim=1, flags="C_CONTIGUOUS")
    exchange.argtypes = [ctypes.c_int, ctypes.c_int, integers, doubles, integers,
                         ctypes.c_int, ctypes.c_uint64, ctypes.c_int]
    exchange.restype = None
    return exchange


def axis_audit(axis, ids, energy, bins):
    counts = [np.bincount(indices, minlength=bins) for indices in axis]
    assert np.array_equal(*counts)
    e = [np.bincount(indices, weights=energy[edges], minlength=bins) for indices,edges in zip(axis,ids)]
    absolute = np.abs(e[0]-e[1])
    denominator = np.maximum(*e)
    relative = np.divide(absolute, denominator, out=np.zeros_like(absolute), where=denominator>0)
    hit = counts[0] > 0
    worst = np.argsort(relative)[-5:][::-1]
    return dict(counts=counts[0].tolist(), high_energy=e[0].tolist(), random_energy=e[1].tolist(),
                relative_rms=float(np.sqrt(np.mean(relative[hit]**2))) if hit.any() else 0.,
                relative_max=float(relative.max()), absolute_max=float(absolute.max()),
                worst_relative_indices=worst.tolist(), worst_absolute_indices=np.argsort(absolute)[-5:][::-1].tolist(),
                coverage=float(hit.mean()))


def sample(inputs, shape, target, seeds, exchange):
    w, r, a, survivor, groups, high_pool = inputs
    energy = 4 * np.square(w.astype(np.float64))
    denominator = float(energy[survivor].sum()/4)
    perm = np.random.default_rng(seeds[0]).permutation(high_pool)
    cumulative = np.r_[0., np.cumsum(energy[perm])]
    desired = target**2 * denominator
    n = int(np.searchsorted(cumulative, desired))
    n = min(n, len(perm))
    if n and abs(cumulative[n-1]-desired) < abs(cumulative[n]-desired):
        n -= 1
    high = np.sort(perm[:n])
    control = np.ascontiguousarray(high, dtype=np.int32)
    exchange(*shape, groups, energy, control, n, seeds[1],
             max(RULES["minimum_proposals"], n*RULES["proposals_per_edge"]))
    ids = [high, np.sort(control.astype(np.int64))]
    assert all(len(np.unique(selected)) == n and (groups[selected] >= 0).all() for selected in ids)
    strata = [np.bincount(groups[selected], minlength=128) for selected in ids]
    assert np.array_equal(*strata)
    row = axis_audit([selected//shape[1] for selected in ids], ids, energy, shape[0])
    col = axis_audit([selected%shape[1] for selected in ids], ids, energy, shape[1])
    energies = [float(energy[selected].sum()) for selected in ids]
    doses = [math.sqrt(e/denominator) if denominator else 0. for e in energies]
    covariates = [dict(positive=int((w[selected]>0).sum()), mean_magnitude=float(np.abs(w[selected]).mean()),
                       mean_activity=float(a[selected].mean()), mean_score=float(r[selected].mean()),
                       score_quantiles=np.quantile(r[selected], [0,.25,.5,.75,1]).tolist())
                  for selected in ids] if n else [dict(positive=0, mean_magnitude=0., mean_activity=0.,
                                                       mean_score=0., score_quantiles=[]) for _ in ids]
    mag, activity = (relative_gap(*(cell[key] for cell in covariates))
                     for key in ("mean_magnitude", "mean_activity"))
    passed = dict(matrix_dose=relative_gap(*doses) <= RULES["matrix_dose_gap"],
                  row_energy=row["relative_rms"] <= RULES["neuron_energy_rms"] and row["relative_max"] <= RULES["neuron_energy_max"],
                  column_energy=col["relative_rms"] <= RULES["neuron_energy_rms"] and col["relative_max"] <= RULES["neuron_energy_max"],
                  magnitude=mag <= RULES["magnitude_gap"], activity=activity <= RULES["activity_gap"])
    return ids, dict(target=target, count=n, survivor_count=int(survivor.sum()), eligible_survivors=int((groups>=0).sum()),
                     high_pool=len(high_pool), seeds=seeds, denominator=denominator, energies=energies,
                     relative_doses=doses, strata=strata[0].tolist(), row=row, column=col,
                     overlap=int(len(np.intersect1d(*ids, assume_unique=True))), covariates=covariates,
                     magnitude_gap=mag, activity_gap=activity, passed=passed)


def pair_summary(seed, order, draw, matrices, target):
    covered = [cell for cell in matrices if cell["count"]]
    count = sum(cell["count"] for cell in covered)
    denominator = sum(cell["denominator"] for cell in matrices)
    doses = [math.sqrt(sum(cell["energies"][arm] for cell in matrices)/denominator) for arm in range(2)]
    means = {key:[sum(cell["count"]*cell["covariates"][arm][key] for cell in covered)/max(1,count)
                  for arm in range(2)] for key in ("mean_score","mean_magnitude","mean_activity")}
    failures = {gate:sum(not cell["passed"][gate] for cell in matrices) for gate in matrices[0]["passed"]}
    coverage = len(covered)/len(matrices)
    score_gap = means["mean_score"][0]-means["mean_score"][1]
    passed = (not any(failures.values()) and coverage >= RULES["minimum_matrix_coverage"]
              and relative_gap(*doses) <= RULES["global_dose_gap"]
              and max(abs(d-target) for d in doses)/target <= RULES["global_target_error"]
              and score_gap >= RULES["mean_score_gap"])
    return dict(seed=seed, order=order, draw=draw, target=target, doses=doses, passed=passed,
                failures=failures, matrix_coverage=coverage, count=count, covariate_means=means,
                score_gap=score_gap, positive=sum(cell["covariates"][0]["positive"] for cell in covered),
                overlap=sum(cell["overlap"] for cell in covered)/max(1,count),
                neuron_coverage={axis:float(np.mean([cell[axis]["coverage"] for cell in matrices]))
                                 for axis in ("row","column")},
                worst_neuron_gap={axis:max(cell[axis]["relative_max"] for cell in matrices)
                                  for axis in ("row","column")})


def match(manifest):
    native_targets = {(int(row["seed"]),int(row["draw"]),row["tensor"]):float(row["realized_distance"])
                      for row in p.rows(p.PRIOR / "matrix_invariants.csv")
                      if row["arm"] == "Bm" and row["mask_type"] == "imp"}
    exchange = swap_function()
    for target in RULES["targets"]:
        directory = OUT / f"dose_{target}"
        directory.mkdir(exist_ok=True)
        summaries, files = [], {}
        for entry in manifest["inputs"]:
            seed, order = entry["seed"], entry["order"]
            dense = p.N._read_model_payload(p.Path(entry["base"]["initial_checkpoint"]))["model"]
            masks = p.N.load_masks(p.Path(entry["base"]["mask_path"]))
            probe = torch.load(PILOT / f"probe_{seed}_{order}.pt", weights_only=True, mmap=True)
            selections = {d:{arm:{} for arm in ARMS} for d in DRAWS}
            audits = {d:[] for d in DRAWS}
            for tensor, mask in masks.items():
                weight = dense[tensor]
                matrix = matrix_inputs(weight, *probe["scores"][tensor], mask)
                for draw in DRAWS:
                    dose = target * native_targets[seed,draw,tensor] / entry["targets"][str(draw)]
                    seeds = [p.rng_seed(OUT.name, seed, order, draw, tensor, arm) for arm in ARMS]
                    ids, audit = sample(matrix, weight.shape, dose, seeds, exchange)
                    audits[draw].append(dict(tensor=tensor, **audit))
                    for arm, selected in zip(ARMS,ids):
                        selections[draw][arm][tensor] = torch.from_numpy(selected)
            for draw in DRAWS:
                summary = pair_summary(seed,order,draw,audits[draw],target)
                summaries.append(summary)
                print(json.dumps(summary), flush=True)
                filename = f"flips_{seed}_{order}_imp_{draw}.pt"
                p.save_tensor(directory / filename,selections[draw])
                path = directory / f"neurons_{seed}_{order}_{draw}.json.gz"
                with gzip.open(path, "wt", compresslevel=3) as stream:
                    json.dump(audits[draw], stream, separators=(",", ":"))
                files[filename] = p.digest(directory / filename)
                files[path.name] = p.digest(path)
            del dense, masks, probe
            p.write(directory / "summary.json", summaries)
        if len(summaries) == len(PAIRS) and all(row["passed"] for row in summaries):
            for filename in files:
                if filename.endswith(".pt"):
                    (OUT / filename).hardlink_to(directory / filename)
            p.write(OUT / "matching_seal.json", dict(target_distance=target, rules=RULES,
                    input_manifest=p.digest(OUT / "inputs.json"), summary=str(directory / "summary.json"),
                    summary_digest=p.digest(directory / "summary.json"),
                    selections={k:v for k,v in files.items() if k.endswith(".pt")},
                    matrix_audits={str(directory / k):v for k,v in files.items() if k.endswith(".gz")}))
            return
    raise SystemExit("No tested common dose met all frozen matching gates. No recoveries authorized.")



if __name__ == "__main__":
    if (OUT / "matching_seal.json").exists():
        raise SystemExit("Matching has already been sealed.")
    match(prepare())
