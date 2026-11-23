"""E32 dense AdamW probes and matched sign intervention."""
import argparse
from collections import Counter
import fcntl
import hashlib
import itertools
import json
import math
import os
from pathlib import Path
import shutil
import sys
import tempfile

sys.dont_write_bytecode = True
OUT = Path(__file__).resolve().parent
ROOT = OUT.parents[1]
os.chdir(ROOT)
PRIOR = OUT.with_name("imp_matched_sign_magnitude_recovery_v1")
PLACEMENT = OUT.with_name("imp_initial_value_placement_v1")
sys.path[:0] = [str(PRIOR), str(OUT.with_name("imp_sign_recovery_audit_v2"))]
from common import p, torch, np, rows, read
from torch.optim.optimizer import register_optimizer_step_pre_hook, register_optimizer_step_post_hook

N = p.native
N.OUT = PRIOR
SEEDS = (7, 17, 29)
MASKS = ("imp", "random1", "random2")
ORDERS = ("A", "B")
ARMS = ("high", "weak", "random")
STEPS = (0, 250, 500, 1000, 1500, 2000, 2500)
N.SNAPSHOT_STEPS = STEPS
torch.set_num_threads(2)
DEVICE = torch.device("cuda:0")
RULES = dict(probe_updates=100, quantile_bins=8, matrix_distance_rtol=.05,
             global_distance_rtol=.01, dose_factors=[1, .75, .5, .35, .25, .15, .1],
             minimum_mean_R_gap=.10, selection_seed_namespace=OUT.name,
             global_target="minimum archived achieved distance across the full cohort")


def digest(path):
    return N._sha256_file(Path(path))


def write(path, value):
    N._atomic_json(path, value)


def save_tensor(path, value):
    temp = path.with_suffix(".tmp")
    torch.save(value, temp)
    temp.replace(path)


def sources():
    previous = rows(PRIOR / "job_manifest.csv")
    placement = rows(PLACEMENT / "job_manifest.csv")
    result = []
    canonical = read(PRIOR / "canonical_stream_manifest.json")["seeds"]
    for seed, mask, order in itertools.product(SEEDS, MASKS, ORDERS):
        prior = [r for r in previous if (int(r["seed"]), r["mask_type"], r["order"], r["arm"])
                 == (seed, mask, order, "Bm")]
        if mask == "imp":
            native_rows = rows(OUT.with_name("imp_recovery_order_stability_v1") / "job_manifest.csv")
            base, = [r for r in native_rows if r["job_id"] == f"e32_seed{seed}_order_{order}_imp80"]
            base["mask_path"] = prior[0]["mask_path"]
        else:
            base, = [r for r in placement if (int(r["seed"]), r["variant"], r["order"])
                     == (seed, mask + "_original", order)]
        assert len(prior) == 3 and all(
            r["initial_checkpoint"] == base["initial_checkpoint"]
            and r["initial_checkpoint_sha256"] == base["initial_checkpoint_sha256"]
            and r["mask_path"] == base["mask_path"] and r["mask_hash"] == base["mask_sha256"]
            and int(r["cap_updates"]) == 2500 for r in prior)
        source_dir = OUT.with_name("imp_recovery_order_stability_v1") if mask == "imp" else PLACEMENT
        status = read(source_dir / "status" / (base["job_id"] + ".json"))
        candidates = list((source_dir / "jobs" / base["job_id"]).glob("attempts/*/common_evaluations.jsonl"))
        if not candidates and mask == "imp":
            candidates = list((OUT.with_name("imp_recovery_order_stability_v1") / "jobs").glob(
                f"*seed{seed}*{order}*imp*/attempts/*/common_evaluations.jsonl"))
        result.append(dict(seed=seed, mask=mask, order=order, base=base,
                           config=canonical[str(seed)]["config"],
                           validation_hash=canonical[str(seed)]["validation_batch_sequence_hash"],
                           targets={r["draw"]: float(r["realized_distance"]) for r in prior},
                           baseline_candidates=[str(x) for x in candidates], baseline_status=status))
    return result


def plan():
    manifest = OUT / "inputs.json"
    if manifest.exists():
        return read(manifest)
    inputs = sources()
    files = {Path(r["base"][k]) for r in inputs for k in
             ("initial_checkpoint", "mask_path", "variant_path") if k in r["base"]}
    files |= {PRIOR / name for name in ("job_manifest.csv", "canonical_stream_manifest.json",
                                       "recovery_order_manifest.json", "matrix_invariants.csv")}
    files |= set((PRIOR / "permutations").glob("*"))
    files |= {Path(__file__), Path(N.__file__), Path(N.recovery.__file__)}
    hashes = {str(f): digest(f) for f in sorted(files)}
    for r in inputs:
        assert hashes[r["base"]["initial_checkpoint"]] == r["base"]["initial_checkpoint_sha256"]
        if "variant_path" in r["base"]: assert hashes[r["base"]["variant_path"]] == r["base"]["variant_sha256"]
    result = dict(inputs=inputs, hashes=hashes, rules=RULES, steps=STEPS,
                  optimizer="fresh torch.optim.AdamW defaults with configured learning rate and decay",
                  router="trainable, original step-0 checkpoint", recovery_attempts=2500,
                  jobs=[dict(seed=s, mask=m, order=o, draw=d, arm=a)
                        for s,m,o,d,a in itertools.product(SEEDS,MASKS,ORDERS,(1,2,3),ARMS)],
                  baseline_jobs=[dict(seed=s,mask=m,order=o)
                                 for s,m,o in itertools.product(SEEDS,MASKS,ORDERS)])
    write(manifest, result)
    return result


def ordered(seed, order, material):
    config, canonical, validation = material
    batches = N._apply_order(canonical, N._permutations()[order]["segment_0"])
    assert len(batches) == 2500
    return config, batches, validation


class ProbeFinished(Exception):
    pass


def probe(entry, material):
    seed, order = entry["seed"], entry["order"]
    path = OUT / f"probe_{seed}_{order}.pt"
    if path.exists():
        return
    base = entry["base"]
    state = N._read_model_payload(Path(base["initial_checkpoint"]))["model"]
    masks = N.load_masks(Path(base["mask_path"]))
    config, batches, validation = ordered(seed, order, material)
    signed = {name: torch.zeros_like(state[name], device=DEVICE, dtype=torch.float64) for name in masks}
    absolute = {name: t.clone() for name, t in signed.items()}
    parameters, before = {}, {}
    count = 0
    worst = 0.
    assigned, accepted = Counter(), Counter()
    original = N.recovery._run_recovery_condition

    def pre(optimizer, args, kwargs):
        for name, param in parameters.items():
            if param.grad is not None:
                before[name] = param.detach().clone()

    @torch.no_grad()
    def post(optimizer, args, kwargs):
        nonlocal count, worst
        group, = optimizer.param_groups
        assert not group["fused"] and not group["capturable"] and not group["amsgrad"]
        beta1, beta2 = group["betas"]
        lr, wd, eps = (group[x] for x in ("lr", "weight_decay", "eps"))
        errors = []
        for name, old in before.items():
            param = parameters[name]
            moment = optimizer.state[param]
            step = int(moment["step"])
            denominator = (moment["exp_avg_sq"].sqrt() / math.sqrt(1 - beta2**step)).add_(eps)
            scale = -lr / (1 - beta1**step)
            update = (moment["exp_avg"].double() / denominator.double()).mul_(scale)
            decay = old.mul(1 - lr * wd)
            expected = decay.double() + update
            tolerance = (old.double().abs() + update.abs()) * (4 * torch.finfo(old.dtype).eps)
            errors.append(((param.double() - expected).abs() / tolerance.clamp_min(1e-30)).max())
            signed[name].add_(update)
            absolute[name].add_(update.abs())
        if errors:
            error = float(torch.stack(errors).max())
            assert math.isfinite(error) and error <= 1, ("AdamW reconstruction", error)
            worst = max(worst, error)
        before.clear()
        count += 1

    def condition(**kwargs):
        native_forward = kwargs["training_forward_callback"]
        def forward(model, tokens, step):
            if not parameters:
                parameters.update((n, v) for n, v in model.named_parameters() if n in masks)
                assert set(parameters) == set(masks)
                assert N._tensor_state_hash(model.state_dict()) == N._tensor_state_hash(state)
            output = native_forward(model, tokens, step)
            a, b = N.recovery._expert_token_counts(output)
            assigned.update(a)
            accepted.update(b)
            return output

        def after(model, optimizer, scaler, step):
            if count == 100:
                raise ProbeFinished(step)

        kwargs.update(training_forward_callback=forward, state_snapshot_callback=after,
                      interruption_checkpoint=None, diagnostic_steps=(0, 2500))
        return original(**kwargs)

    hooks = (register_optimizer_step_pre_hook(pre),
             register_optimizer_step_post_hook(post))
    N.recovery._run_recovery_condition = condition
    try:
        with tempfile.TemporaryDirectory(dir=OUT) as scratch:
            N._run_segment(dict(job_id=f"probe_{seed}_{order}", seed=seed, model_type="unpruned"),
                           Path(scratch), config, state, {}, Path(base["initial_checkpoint"]),
                           batches, validation, 0, "segment_0", DEVICE)
    except ProbeFinished as stop:
        attempts = stop.args[0]
    finally:
        N.recovery._run_recovery_condition = original
        for hook in hooks:
            hook.remove()
    assert count == 100
    scores = {n: (state[n].sign() * signed[n].cpu() / a.cpu(), a.cpu())
              for n, a in absolute.items()}
    save_tensor(path, dict(scores=scores, attempts=attempts, applied_updates=count,
                           max_reconstruction_error_fraction=worst, assigned=dict(assigned),
                           accepted=dict(accepted), train_hash=N.recovery._batch_sequence_hash(batches)))


def baseline_state(entry):
    base = entry["base"]
    if "variant_path" in base:
        return p.variant_state(base)
    masks = N.load_masks(Path(base["mask_path"]))
    state = N._masked_initial_state(Path(base["initial_checkpoint"]), masks)
    assert N._tensor_state_hash(state) == base["initial_model_state_hash"]
    assert N.recovery._mask_hash(masks) == base["mask_sha256"]
    return state, masks


def rng_seed(*parts):
    return int.from_bytes(hashlib.sha256("|".join(map(str, (OUT.name, *parts))).encode()).digest()[:8],
                          "little")


def quantile_bin(values, n):
    edges = np.quantile(values, np.arange(1, n) / n)
    return np.searchsorted(edges, values, side="right")


def pools(weight, score, activity, mask):
    w, r, a, m = (x.numpy().ravel() for x in (weight, score, activity, mask))
    eligible = np.flatnonzero((w != 0) & (a > 0) & np.isfinite(w) & np.isfinite(r) & np.isfinite(a))
    if not len(eligible):
        return [], [], 0, 0
    assert np.max(np.abs(r[eligible])) <= 1 + 1e-10
    surviving = np.flatnonzero(m[eligible])
    rank = np.argsort(r[eligible], kind="stable")
    high = np.zeros(len(eligible), dtype=bool)
    high[rank[-(len(rank)//4):]] = True if len(rank) >= 4 else False
    weak = np.zeros_like(high)
    weak[np.argsort(np.abs(r[eligible]), kind="stable")[:len(rank)//2]] = True
    n = RULES["quantile_bins"]
    group = (quantile_bin(np.abs(w[eligible]), n)*n + quantile_bin(a[eligible], n))*2 + (w[eligible]>0)
    sorted_survivors = surviving[np.argsort(group[surviving], kind="stable")]
    strata, retained = [], []
    for local in np.split(sorted_survivors, np.flatnonzero(np.diff(group[sorted_survivors]))+1):
        if not len(local):
            continue
        h, k, ids = eligible[local[high[local]]], eligible[local[weak[local]]], eligible[local]
        retained.append((int(group[local[0]]), len(ids), len(h)))
        if len(h) and len(k):
            strata.append((h, k, ids))
    return strata, retained, len(eligible), len(surviving)


def select(strata, w, squared_target, seeds):
    capacity = np.array([min(len(h), len(k)) for h,k,_ in strata], dtype=int)
    costs = np.array([4*np.mean(w[ids]**2) for _,_,ids in strata])
    possible = float(capacity @ costs)
    fraction = min(1., squared_target / possible) if possible else 0.
    exact = fraction * capacity
    counts = np.floor(exact).astype(int)
    remaining = squared_target - counts @ costs
    for j in np.argsort(counts-exact, kind="stable"):
        if counts[j] < capacity[j] and abs(remaining-costs[j]) < abs(remaining):
            remaining -= costs[j]
            counts[j] += 1
    selected = {}
    for index, arm in enumerate(ARMS):
        rng = np.random.default_rng(seeds[arm])
        parts = [rng.choice(p[index], n, replace=False) for p,n in zip(strata,counts) if n]
        selected[arm] = np.sort(np.concatenate(parts)) if parts else np.empty(0,dtype=np.int64)
    return selected, possible


def match(manifest):
    destination = OUT / "matching_seal.json"
    if destination.exists():
        return read(destination)
    targets = {(int(r["seed"]),r["mask_type"],int(r["draw"]),r["tensor"]):
               float(r["realized_distance"]) for r in rows(PRIOR / "matrix_invariants.csv") if r["arm"]=="Bm"}
    common_target = min(target for entry in manifest["inputs"] for target in entry["targets"].values())
    for factor in RULES["dose_factors"]:
        audit, global_audit, retention, files = [], [], [], {}
        for seed, order in itertools.product(SEEDS,ORDERS):
            dense = N._read_model_payload(Path(next(r["base"]["initial_checkpoint"] for r in manifest["inputs"]
                                                    if r["seed"] == seed)))["model"]
            scores = torch.load(OUT / f"probe_{seed}_{order}.pt", weights_only=True)["scores"]
            for entry in (r for r in manifest["inputs"] if (r["seed"],r["order"]) == (seed,order)):
                mask = entry["mask"]
                _, masks = baseline_state(entry)
                selections = {draw: {arm:{} for arm in ARMS} for draw in (1,2,3)}
                totals = {draw: {arm:np.zeros(4) for arm in ARMS} for draw in selections}
                for name, surviving in masks.items():
                    w = dense[name]
                    r, a = scores[name]
                    strata, retained, eligible, eligible_survivors = pools(w,r,a,surviving)
                    retention.extend(dict(seed=seed,order=order,mask=mask,tensor=name,stratum=group,
                                          survivors=n,high=n_high) for group,n,n_high in retained)
                    flat = w.numpy().ravel().astype(np.float64)
                    energy = float(np.square(flat[surviving.numpy().ravel()]).sum())
                    for draw in selections:
                        target = (factor * common_target * targets[seed,mask,draw,name]
                                  / entry["targets"][str(draw)])
                        seeds = {arm:rng_seed(seed,order,mask,draw,name,arm) for arm in ARMS}
                        picks, capacity = select(strata, flat, target**2*energy, seeds)
                        for arm, ids in picks.items():
                            selections[draw][arm][name] = torch.from_numpy(ids)
                            distance = 4*float(np.square(flat[ids]).sum())
                            selected_r = r.numpy().ravel()[ids]
                            totals[draw][arm] += (distance, energy, selected_r.sum(), len(ids))
                            nrows, ncols = w.shape
                            row_counts = np.bincount(ids//ncols, minlength=nrows)
                            col_counts = np.bincount(ids%ncols, minlength=ncols)
                            audit.append(dict(seed=seed,order=order,mask=mask,draw=draw,tensor=name,arm=arm,
                                              eligible=eligible,eligible_survivors=eligible_survivors,
                                              target=target,distance=math.sqrt(distance/energy) if energy else 0.,
                                              common_support_energy=capacity,count=len(ids),rng_seed=seeds[arm],
                                              positive=int((flat[ids]>0).sum()),
                                              mean_abs_w=float(np.abs(flat[ids]).mean()) if len(ids) else None,
                                              mean_A=float(a.numpy().ravel()[ids].mean()) if len(ids) else None,
                                              R_quantiles=np.quantile(selected_r,[0,.25,.5,.75,1]).tolist() if len(ids) else [],
                                              row_counts=row_counts.tolist(),column_counts=col_counts.tolist()))
                for draw, arms in totals.items():
                    distances = {arm:math.sqrt(d/e) for arm,(d,e,_,_) in arms.items()}
                    means = {arm:r/n if n else 0. for arm,(_,_,r,n) in arms.items()}
                    target = factor * common_target
                    global_audit.append(dict(seed=seed,order=order,mask=mask,draw=draw,target=target,
                                             distances=distances,mean_R=means,
                                             counts={arm:int(t[3]) for arm,t in arms.items()}))
                    file = OUT / f"flips_{seed}_{order}_{mask}_{draw}.pt"
                    save_tensor(file,selections[draw])
                    files[file.name] = digest(file)
                del masks
        # Compare within matched sets, including small matrices where a single flip
        # dominates distance. Zero-flip matrices must agree across all arms.
        for index in range(0,len(audit),3):
            block = audit[index:index+3]
            distances = [r["distance"] for r in block]
            passed = max(distances)-min(distances) <= RULES["matrix_distance_rtol"]*max(distances)
            for row in block:
                row["matrix_distance_passed"] = bool(passed)
        passed = (all(row["matrix_distance_passed"] for row in audit)
                  and all(max(g["distances"].values())-min(g["distances"].values())
                          <= RULES["global_distance_rtol"]*g["target"]
                          and min(g["distances"].values()) >= .95*g["target"]
                          and g["mean_R"]["high"]-max(g["mean_R"]["weak"],g["mean_R"]["random"])
                          >= RULES["minimum_mean_R_gap"] for g in global_audit))
        write(OUT / f"global_matching_{factor}.json",global_audit)
        if passed:
            write(OUT / "matrix_matching.json",audit)
            write(OUT / "retention_strata.json",retention)
            seal = dict(dose_factor=factor,target_distance=factor*common_target,
                        rules=RULES,selections=files,
                        matrix_audit=digest(OUT/"matrix_matching.json"),
                        input_manifest=digest(OUT/"inputs.json"))
            write(destination,seal)
            return seal
    raise RuntimeError("No dose met the frozen matching gates")


def read_jsonl(path):
    return [json.loads(line) for line in path.read_text().splitlines()]


def validate_losses(records, entry):
    assert [r["step"] for r in records] == list(STEPS)
    assert all(r["seed"] == entry["seed"] and r["order"] == entry["order"]
               and r["validation_stream_hash"] == entry["validation_hash"]
               and r["precision"] == "FP32" and not r["tf32"]
               and r["target_tokens"] == 492416 and math.isfinite(r["cross_entropy"]) for r in records)


def reuse_baseline(entry, material):
    base = entry["base"]
    candidate, = entry["baseline_candidates"]
    candidate = Path(candidate)
    status = entry["baseline_status"]
    assert status["status"] == "COMPLETE" and Path(status["attempt"]) == candidate.parent
    assert digest(candidate) == status["common_evaluation_sha256"]
    config,batches,_ = ordered(entry["seed"],entry["order"],material)
    segment = read(candidate.parent / "segment_0/segment_record.json")
    assert segment["optimizer"] == "fresh_AdamW" and segment["router_mode"] == "trainable"
    assert segment["sparsity"] == .8 and segment["recovery_steps"] == 2500
    assert segment["segment_training_batch_hash"] == N.recovery._batch_sequence_hash(batches)
    snapshot = N._read_model_payload(candidate.parent / "snapshots/step_0000.pt")
    assert snapshot["config"] == config
    evaluation = [r for r in read_jsonl(candidate) if r["step"] <= 2500]
    validate_losses(evaluation, entry)
    assert (N._tensor_state_hash(snapshot["model"]) == evaluation[0]["model_sha256"]
            == base["initial_model_state_hash"])
    return dict(seed=entry["seed"],mask=entry["mask"],order=entry["order"],
                source=str(candidate),sha256=digest(candidate),records=evaluation)


def job_id(job):
    return f"seed{job['seed']}_{job['order']}_{job['mask']}_draw{job['draw']}_{job['arm']}"


def reconstruct(entry, picks):
    state,masks = baseline_state(entry)
    assert set(picks) == set(masks)
    for name,ids in picks.items():
        assert ids.dtype == torch.int64 and ids.ndim == 1 and torch.equal(ids,ids.unique(sorted=True))
        assert bool(masks[name].flatten()[ids].all())
        original = state[name]
        value = original.clone()
        chosen = value.flatten()[ids]
        assert bool(torch.isfinite(chosen).all()) and bool((chosen != 0).all())
        value.flatten()[ids] = -chosen
        assert torch.equal(value.abs(),original.abs())
        assert torch.equal((value != original).flatten().nonzero().flatten(),ids)
        state[name] = value
    return state,masks


def run_job(job, entry, material, seal):
    name = job_id(job)
    prefix = OUT / "jobs" / name
    prefix.mkdir(parents=True,exist_ok=True)
    with (prefix/"lock").open("a") as lock:
        try:
            fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BlockingIOError:
            return
        endpoint = prefix / "result.json"
        if endpoint.exists():
            saved = read(endpoint)
            assert saved["job"] == job and saved["seal"] == digest(OUT/"matching_seal.json")
            validate_losses(saved["records"],entry)
            return
        file = OUT / f"flips_{job['seed']}_{job['order']}_{job['mask']}_{job['draw']}.pt"
        assert digest(file) == seal["selections"][file.name]
        picks = torch.load(file,weights_only=True)[job["arm"]]
        state,masks = reconstruct(entry,picks)
        config,batches,validation = ordered(job["seed"],job["order"],material)
        checkpoint = Path(entry["base"]["initial_checkpoint"])
        assert digest(checkpoint) == entry["base"]["initial_checkpoint_sha256"]
        attempt = prefix / "attempt"
        if attempt.exists() and not (attempt/"segment_0/interruption.pt").exists():
            attempt.rename(Path(tempfile.mkdtemp(prefix="incomplete_",dir=prefix))/"attempt")
        attempt.mkdir(exist_ok=True)
        snap = attempt / "snapshots/step_0000.pt"
        if not snap.exists():
            with N._preserved_rng():
                model = N.assemble_router_age_model(config["model"],state,str(checkpoint),masks,DEVICE)
                N._assert_model_mask(model,masks)
                assert N._tensor_state_hash(model.state_dict()) == N._tensor_state_hash(state)
                N._model_snapshot(snap,model,0,config,"sign_reinforcement_initial")
                del model
        N._run_segment(dict(job_id=name,seed=job["seed"],model_type="imp80"),attempt,config,
                       state,masks,checkpoint,batches,validation,0,"segment_0",DEVICE)
        logs = read_jsonl(attempt/"segment_0/gradient_stats/gradient_stats.jsonl")
        assert [r["step"] for r in logs] == list(range(1,2501))
        snapshots = [(step,attempt/f"snapshots/step_{step:04d}.pt","sign_reinforcement") for step in STEPS]
        evaluations = N._common_evaluate(snapshots,config,validation,job["seed"],job["order"],"imp80",
                                        name,entry["validation_hash"],DEVICE,allowed_steps=STEPS)
        validate_losses(evaluations,entry)
        assert evaluations[0]["model_sha256"] == N._tensor_state_hash(state)
        write(endpoint,dict(job=job,seal=digest(OUT/"matching_seal.json"),records=evaluations,
                            attempts=len(logs),applied_updates=sum(r["optimizer_step_applied"] for r in logs)))


def run_pipeline(stage):
    manifest = plan()
    if stage == "plan":
        return
    assert torch.__version__.startswith("2.7.0")
    for path, expected in manifest["hashes"].items():
        assert digest(path) == expected, f"Input changed: {path}"
    materials = None
    for seed in SEEDS:
        materials = N._load_training_material(seed)
        for order in ORDERS:
            entry = next(r for r in manifest["inputs"] if (r["seed"],r["order"]) == (seed,order))
            probe(entry,materials)
        del materials
    if stage == "probe":
        return
    seal = match(manifest)
    assert seal["input_manifest"] == digest(OUT/"inputs.json")
    assert seal["matrix_audit"] == digest(OUT/"matrix_matching.json")
    if stage == "match":
        return
    for seed in SEEDS:
        material = N._load_training_material(seed)
        entries = {(r["mask"],r["order"]):r for r in manifest["inputs"] if r["seed"] == seed}
        for entry in entries.values():
            baseline = reuse_baseline(entry,material)
            write(OUT/f"baseline_{seed}_{entry['order']}_{entry['mask']}.json",baseline)
        for job in manifest["jobs"]:
            if job["seed"] == seed:
                run_job(job,entries[job["mask"],job["order"]],material,seal)
        del material


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage",choices=("plan","probe","match","recover"))
    run_pipeline(parser.parse_args().stage)
