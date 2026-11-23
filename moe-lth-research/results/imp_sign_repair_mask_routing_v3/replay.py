"""Native recovery replays, fixed readouts, and a closed-half-line intervention."""
import argparse
import fcntl
import socket
from pathlib import Path
from contextlib import nullcontext
from torch.optim.optimizer import register_optimizer_step_post_hook

from audit import OUT, SOURCE, EARLY, EPSILON, READOUT, p
from common_v2 import load
from common_v2 import OUT as V2_OUT
admission_module = load("v3_v2_admission", V2_OUT/"worker.py")
admission_module.OUT = OUT
admission = admission_module.admission

torch = p.torch
N = p.N
DEVICE = p.DEVICE


class VerifiedWindow(Exception):
    pass


def selection(entry, job):
    seed,order = entry["seed"],entry["order"]
    draws = (1,2,3) if job["arm"] == "untouched" else (job["draw"],)
    files = [torch.load(SOURCE/f"flips_{seed}_{order}_imp_{d}.pt",weights_only=True) for d in draws]
    coordinates = {name:torch.cat([arm[name] for file in files for arm in file.values()]).unique(sorted=True)
                   for name in files[0]["high"]}
    picks = None if job["arm"] == "untouched" else files[0][job["arm"]]
    return coordinates,picks


def classify(x):
    return (x > EPSILON).to(torch.int8) - (x < -EPSILON).to(torch.int8)


def evaluate(model, batches):
    previous = model.training
    tf32 = torch.backends.cuda.matmul.allow_tf32,torch.backends.cudnn.allow_tf32
    total, count = 0.,0
    with N._preserved_rng(),torch.inference_mode():
        try:
            model.eval()
            torch.backends.cuda.matmul.allow_tf32 = torch.backends.cudnn.allow_tf32 = False
            for x,y in batches:
                y = y.to(DEVICE)
                logits = model(x.to(DEVICE)).logits.float()
                total += float(torch.nn.functional.cross_entropy(
                    logits.reshape(-1,logits.shape[-1]), y.reshape(-1),reduction="sum").double())
                count += y.numel()
        finally:
            model.train(previous)
            torch.backends.cuda.matmul.allow_tf32,torch.backends.cudnn.allow_tf32 = tf32
    return total/count


def replay(entry, source, material, mode, folder):
    job = source["job"]
    seed,order = job["seed"],job["order"]
    if "flip_paths" in source:
        bundles = [torch.load(path,weights_only=True) for path in source["flip_paths"]]
        coordinates = {name:torch.cat([arm[name] for bundle in bundles for arm in bundle.values()]).unique(sorted=True)
                       for name in bundles[0]["high"]}
        picks = None if job["arm"]=="untouched" else bundles[0][job["arm"]]
    else:
        coordinates,picks = selection(entry,job)
    initial,masks = p.baseline_state(entry)
    start = torch.cat([classify(initial[name].flatten()[ids]) for name,ids in coordinates.items()]).to(DEVICE)
    state = initial if picks is None else p.reconstruct(entry,picks)[0]
    config,batches,validation = p.ordered(seed,order,material)
    readout = [validation[i] for i in READOUT]
    archived = {r["step"]:r for r in source["records"]}
    checkpoints = folder/"snapshots"
    checkpoints.mkdir(exist_ok=True)
    p.save_tensor(folder/"coordinates.pt",coordinates)
    ids = {name:index.to(DEVICE) for name,index in coordinates.items()}
    projection = {}
    if mode in {"repair","new_repair"}:
        for name,index in picks.items():
            projection[name] = (index.to(DEVICE),state[name].flatten()[index].sign().to(DEVICE))
    track = {}
    params = {}
    step_now = 0
    first = torch.full(start.shape,-1,dtype=torch.int16,device=DEVICE)
    last_other = first.clone()
    hits = {name:torch.zeros_like(index,dtype=torch.int16) for name,(index,_) in projection.items()}
    bindings = []
    checkpoint = folder/"segment_0/interruption.pt"
    if checkpoint.exists():
        # Step-specific tracking is committed before the matching native checkpoint.
        resumed = torch.load(checkpoint,map_location="cpu",weights_only=False)["step"]
        saved = torch.load(folder/f"tracking_{resumed:04d}.pt",map_location=DEVICE,weights_only=True)
        first,last_other,hits,bindings = (saved[k] for k in ("first","last_other","hits","bindings"))

    def project(optimizer,args,kwargs):
        if not 1 <= step_now <= 250:
            return
        projected = []
        for name,(index,side) in projection.items():
            parameter = params[name].flatten()
            values = parameter[index]
            invalid = values * side < 0
            replacement = values.masked_fill(invalid,0)
            parameter[index] = replacement
            hits[name].add_(invalid)
            projected.append(torch.stack((invalid.sum().float(),(values-replacement).abs().sum(),
                                          (values-replacement).square().sum())))
        if projected:
            bindings.append((step_now,torch.stack(projected).sum(0).cpu().tolist()))

    native = N.recovery._run_recovery_condition
    routing_counts = dict(calls=0,verified_capacity=0)
    routing_context = nullcontext(None)
    if source.get("scaled"):
        from probes import gate_intervention
        routing_context = gate_intervention(routing_counts)
    bind = None

    def instrument(**kwargs):
        forward, callback = kwargs["training_forward_callback"],kwargs["state_snapshot_callback"]

        def train(model,tokens,step):
            nonlocal step_now
            step_now = step
            return forward(model,tokens,step)

        @torch.no_grad()
        def observe(model,optimizer,scaler,step):
            N._heartbeat(source.get("name",p.job_id(job)),folder,step,"segment_0")
            if not params:
                params.update(model.named_parameters())
                if bind is not None:
                    bind(model)
            if step <= 250:
                signs = torch.cat([classify(params[name].flatten()[index]) for name,index in ids.items()])
                if step:
                    returned = (signs == start) & (start != 0)
                    first.masked_fill_(returned & (first < 0),step)
                    last_other.masked_fill_(~returned,step)
                if step in EARLY:
                    p.save_tensor(folder/f"signs_{step:04d}.pt",signs.cpu())
                    p.write(folder/f"readout_{step:04d}.json",dict(step=step,ce=evaluate(model,readout)))
            if step % 100 == 0 or step == 250:
                if step:
                    p.save_tensor(folder/f"tracking_{step:04d}.pt",
                                  dict(first=first.cpu(),last_other=last_other.cpu(),
                                       hits={k:v.cpu() for k,v in hits.items()},bindings=bindings))
            if step == 0 or step == 250 or (mode != "free" and step in p.STEPS):
                identity = N._tensor_state_hash(model.state_dict())
                if step in archived and (mode == "free" or step == 0):
                    assert identity == archived[step]["model_sha256"],(job,step,identity,archived[step]["model_sha256"])
                if mode != "free" or step == 0:
                    N._model_snapshot(checkpoints/f"step_{step:04d}.pt",model,step,config,"sign_repair")
                if step == 0 or mode != "free":
                    p.write(folder/f"full_{step:04d}.json",dict(step=step,ce=evaluate(model,validation),model_sha256=identity))
                if step == 0 and 0 in archived:
                    assert abs(p.read(folder/"full_0000.json")["ce"]-archived[0]["cross_entropy"]) <= 1e-6
            if step == 250 and mode == "free":
                raise VerifiedWindow

        kwargs.update(training_forward_callback=train,state_snapshot_callback=observe)
        hook = register_optimizer_step_post_hook(torch.no_grad()(project))
        try:
            return native(**kwargs)
        finally:
            hook.remove()

    N.recovery._run_recovery_condition = instrument
    try:
        with routing_context as bind:
            N._run_segment(dict(job_id=source.get("name",p.job_id(job)),seed=seed,model_type="imp80"),
                           folder,config,state,masks,Path(entry["base"]["initial_checkpoint"]),
                           batches,validation,0,"segment_0",DEVICE)
    except VerifiedWindow:
        assert mode == "free"
    finally:
        N.recovery._run_recovery_condition = native
    logs = p.read_jsonl(folder/"segment_0/gradient_stats/gradient_stats.jsonl")
    applied = [r["step"] for r in logs if r["optimizer_step_applied"]]
    limit = 250 if mode == "free" else 2500
    assert [r["step"] for r in logs] == list(range(1,limit+1))
    if mode == "free":
        assert applied == [step for step in source["applied_steps"] if step <= limit]
    p.write(folder/"result.json",dict(job=job,mode=mode,attempts=limit,applied=len(applied),
            verified_steps=[0,250] if mode=="free" else [step for step in (0,) if step in archived],
            readout_stream_hash=N.recovery._batch_sequence_hash(readout),source=source.get("verified",{}),routing=routing_counts))


def worker(mode,shard):
    manifest = p.read(OUT/"audit_manifest.json")
    assert p.digest(SOURCE/"matching_seal.json") == manifest["source_seal"]
    reporter = load("sign_repair_report",OUT/"report.py")
    assert reporter.OUT == OUT
    print("Reporter",reporter.__file__,"output",reporter.OUT,flush=True)
    sources = sorted(manifest["jobs"],key=lambda item:(item["job"]["draw"],item["job"]["seed"],item["job"]["arm"]))
    entries = {(e["seed"],e["order"]):e for e in manifest["inputs"]}
    last_seed,material=None,None
    for source in sources:
        job=source["job"]
        if job["order"] != ("A","B")[shard] or (mode=="repair" and job["draw"] != 1):
            continue
        name=p.job_id(job)
        root=OUT/mode/name
        root.mkdir(parents=True,exist_ok=True)
        if (root/"result.json").is_file():
            continue
        admission(socket.gethostname())
        seed=job["seed"]
        if last_seed != seed:
            del material
            material=N._load_training_material(seed)
            last_seed=seed
        folder=root/"attempt"
        if folder.exists() and not (folder/"segment_0/interruption.pt").is_file():
            import tempfile
            folder.rename(Path(tempfile.mkdtemp(prefix="partial_",dir=root))/"attempt")
        folder.mkdir(exist_ok=True)
        print("START",mode,name,flush=True)
        replay(entries[seed,job["order"]],source,material,mode,folder)
        (folder/"result.json").replace(root/"result.json")
        with (OUT/"report.lock").open("a") as lock:
            fcntl.flock(lock,fcntl.LOCK_EX)
            reporter.main()


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode",choices=("free","repair"))
    parser.add_argument("shard",type=int,choices=(0,1))
    args=parser.parse_args()
    with (OUT/f"{args.mode}_{args.shard}.lock").open("a") as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        worker(args.mode,args.shard)
