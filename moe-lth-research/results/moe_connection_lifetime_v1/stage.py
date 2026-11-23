"""Study-local stage gates, development and prospective job entry point."""
import argparse
from datetime import datetime,timezone
import fcntl
import json
import math
import os
import sys
from pathlib import Path

from continuation import run
from forecast import score
from groups import HERE,V2,branch,torch
from prefix_state import DEV,FRESH,generate,material,location,fresh_groups

ORDERS=("A","B")


def read(path):
    return json.loads(path.read_text())


def records():
    root=HERE.parents[1]
    paths=set(HERE.glob("*.py")) | {HERE/"registered_request.md",HERE/"registration.md",HERE/"registration.sha256"}
    paths.update(root/"src/moe_lth/models"/f for f in ("experts.py","router.py","moe_layer.py","transformer.py"))
    paths.update(Path(x.__file__).resolve() for x in (branch,branch.N,branch.N.recovery,sys.modules["study_common"],sys.modules["audit_match"],torch.optim.adamw if hasattr(torch.optim,"adamw") else sys.modules[torch.optim.AdamW.__module__]))
    paths.update([Path(sys.modules["study_common"].p.__file__),V2/"input_manifest.json",V2/"sealed_inputs.json",
                  V2/"protocol.md",V2/"verify_selections.py",branch.V1/"prefix.py",
                  root/"src/moe_lth/data.py",root/"src/moe_lth/utils.py",
                  root/"src/moe_lth/pruning/router_age.py",root/"src/moe_lth/training/train.py",
                  root/"scripts/imp_recovery_order_stability_v1/shared_gpu_worker.py"])
    paths.update((root/"data/multidomain_balanced").glob("*"))
    arch=root/"results/imp_recovery_order_stability_v1"
    paths.update([arch/"canonical_stream_manifest.json",arch/"recovery_order_manifest.json"])
    paths.update((arch/"permutations").glob("*segment_0.npy"))
    return {str(p):branch.sha(p) for p in sorted(paths)}


def freeze():
    import scipy,subprocess
    assert not (HERE/"freeze.json").exists()
    assert not list((HERE/"branches").glob("*/*/*/checkpoint.pt"))
    assert not any((HERE/"groups"/f"seed{s}_{o}"/"forecast250.json").exists() for s in FRESH for o in ORDERS)
    from native_provenance import main as inventory
    inventory()
    gates=[HERE/"gradient_fixture.json",HERE/"gradient_native.json",HERE/"smoke_7_A_isolated/verified.json",
           HERE/"oracle_7_A_attempt4.log",HERE/"adder_oracle_attempt1.log",HERE/"prefix_replay.json"]
    for p in gates[-3:-1]:
        proof=json.loads(p.read_text().splitlines()[-1])
        assert all(proof["oracle_checks"].values())
    assert read(gates[2])["resume_exact"] and read(gates[-1])["exact"]
    gates.extend([HERE/"smoke_7_A_isolated/native251.json",HERE/"smoke_7_A_isolated/swapped_labels.json",
                  HERE/"native_provenance.json",HERE/"validation_domains.json"])
    assert all(read(gates[-4])["checks"].values())
    floor=read(gates[-3])["duplicate_floor_whole"]
    gates.extend([HERE/"seed_roster_audit.json",HERE/"split_audit.json"])
    assert not read(gates[-2])["prediction_definition_matches"]
    assert read(gates[-1])["identical_full_129_byte_windows"]==0
    for seed in DEV:
        for o in ORDERS:
            for t in (250,500):
                seal=check_seal(seed,o,t)
                z=read(Path(seal["forecast"]))
                floor=max(floor,z["batches"][0]["duplicate_floor"])
                gates.append(sealfile(seed,o,t))
    from importlib.util import module_from_spec, spec_from_file_location

    reporter_path = HERE / "report.py"
    spec = spec_from_file_location("connection_lifetime_report", reporter_path)
    reporter = module_from_spec(spec)
    spec.loader.exec_module(reporter)
    assert Path(reporter.__file__).resolve() == reporter_path
    for arm in ("H","R"):
        control=HERE/"random_control"/arm
        assert reporter.endpoint(control) is not None
        gates.append(control/"verified.json")
    assert scipy.__version__=="1.11.4" and torch.__version__.split("+")[0]=="2.7.0"
    branch.save_json(HERE/"freeze.json",dict(frozen_utc=datetime.now(timezone.utc).isoformat(),inputs=records(),torch=torch.__version__,scipy=scipy.__version__,
         development=DEV,prospective=FRESH,orders=ORDERS,replay_floor=floor,
         git_revision=subprocess.check_output(["git","rev-parse","HEAD"],text=True).strip(),
         evidence={str(p):branch.sha(p) for p in gates}))
    (HERE/"freeze.sha256").write_text(branch.sha(HERE/"freeze.json")+"  freeze.json\n")


def check_freeze():
    path=HERE/"freeze.json"
    assert branch.sha(path)==(HERE/"freeze.sha256").read_text().split()[0]
    assert read(path)["inputs"]==records()


def sealfile(seed,o,t):
    return HERE/"seals"/f"seed{seed}_{o}_{t}.json"


def check_seal(seed,o,t):
    rec=read(sealfile(seed,o,t))
    if rec.get("ineligible"):
        assert branch.sha(rec["audit"])==rec["audit_sha256"]
        return rec
    if seed in FRESH:
        state_record=location(seed,o)/f"state{t}.json"
        assert branch.sha(state_record)==rec["state_record_sha256"]
    assert branch.sha(rec["forecast"])==rec["forecast_sha256"] and branch.sha(rec["sets"])==rec["sets_sha256"]
    return rec


def forecast(seed,o,t):
    cell=f"seed{seed}_{o}"
    folder=HERE/"groups"/cell
    path=folder/f"forecast{t}.json"
    if sealfile(seed,o,t).exists():
        check_seal(seed,o,t)
        return
    if t==500:
        earlier=check_seal(seed,o,250)
        if earlier.get("ineligible"):
            branch.save_json(sealfile(seed,o,t),earlier)
            return
    if t==500 and seed in DEV:
        from route_replay import replay
        replay(seed,o,250)
    if not path.exists():
        if t==500 or seed in FRESH:
            if not (location(seed,o)/f"state{t}.json").exists():
                generate(seed,o,t)
        if seed in FRESH and t==250:
            if not (folder/"audit.json").exists():
                fresh_groups(seed,o)
            failed=folder/"ineligible.json"
            if failed.exists():
                branch.save_json(sealfile(seed,o,t),dict(ineligible=True,audit=str(failed),audit_sha256=branch.sha(failed),
                                                        freeze_sha256=branch.sha(HERE/"freeze.json")))
                return
        if seed in FRESH:
            from mask_audit import audit
            if not (folder/"independent_audit.json").exists():
                audit(seed,o)
        saved,cfg,batches,val,model,opt,scaler,dtype=material(seed,o,t)
        assert branch.sha(folder/"sets.pt")==read(folder/"audit.json")["sha256"]
        groups=torch.load(folder/"sets.pt",weights_only=True)
        score(saved,cfg,batches,model,opt,scaler,dtype,groups,seed,o,t,path,seed in DEV)
    from mask_audit import audit
    if not (folder/"independent_audit.json").exists():
        audit(seed,o)
    from descriptive import coordinates
    coordinates(seed,o,t)
    z=read(path)
    assert len(z["batches"])==16 and (z["seed"],z["order"],z["t"])==(seed,o,t)
    assert all(math.isfinite(z["effects"][k]["P"]) for k in ("partner","whole"))
    branch.save_json(sealfile(seed,o,t),dict(sealed_utc=datetime.now(timezone.utc).isoformat(),
            state_record_sha256=branch.sha(location(seed,o)/f"state{t}.json") if seed in FRESH or t==500 else None,
            forecast=str(path),forecast_sha256=branch.sha(path),
            sets=str(folder/"sets.pt"),sets_sha256=branch.sha(folder/"sets.pt"),
            freeze_sha256=branch.sha(HERE/"freeze.json") if seed in FRESH else None))


def main():
    p=argparse.ArgumentParser()
    p.add_argument("action",choices=("forecast","branch","freeze","replay"))
    p.add_argument("--seed",type=int,choices=DEV+FRESH)
    p.add_argument("--order",choices=ORDERS)
    p.add_argument("--t",type=int,choices=(250,500))
    p.add_argument("--arm",choices=("0","H","R","dense"))
    a=p.parse_args()
    if a.action=="freeze":
        return freeze()
    if a.action=="replay":
        generate(7,"A",250)
        branch.save_json(HERE/"prefix_replay.json",dict(exact=True,seed=7,order="A",attempts=250))
        return
    if a.seed in FRESH:
        check_freeze()
    if a.action=="forecast":
        locks=HERE/"cell_locks"
        locks.mkdir(exist_ok=True)
        with (locks/f"{a.seed}_{a.order}_{a.t}.lock").open("a") as file:
            fcntl.flock(file,fcntl.LOCK_EX)
            return forecast(a.seed,a.order,a.t)
    assert a.seed in FRESH and a.arm and (a.arm!="dense" or a.t==250)
    inventory={}
    freeze_hash=branch.sha(HERE/"freeze.json")
    for s in FRESH:
        for o in ORDERS:
            for t in (250,500):
                assert check_seal(s,o,t)["freeze_sha256"]==freeze_hash
                path=sealfile(s,o,t)
                inventory[path.name]=branch.sha(path)
    with (HERE/"all_predictions.lock").open("a") as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        manifest=HERE/"all_predictions.json"
        if manifest.exists():
            proof=read(manifest)
            assert proof["forecast_receipts"]==inventory and proof["freeze_sha256"]==freeze_hash
        else:
            assert not list((HERE/"branches").glob("*/*/*/checkpoint.pt"))
            branch.save_json(manifest,dict(sealed_utc=datetime.now(timezone.utc).isoformat(),
                         forecast_receipts=inventory,freeze_sha256=freeze_hash))
        checksum=manifest.with_suffix(".sha256")
        value=branch.sha(manifest)+"  all_predictions.json\n"
        if checksum.exists():
            assert checksum.read_text()==value
        else:
            temp=checksum.with_suffix(".tmp")
            temp.write_text(value)
            temp.replace(checksum)
    seal=check_seal(a.seed,a.order,a.t)
    dest=HERE/"branches"/f"seed{a.seed}_{a.order}"/str(a.t)/a.arm
    state=material(a.seed,a.order,a.t)
    run(a.seed,a.order,a.t,a.arm,state,dest,torch.load(seal["sets"],weights_only=True),seal["sets_sha256"])


if __name__=="__main__":
    import socket,time
    started=time.monotonic()
    done=False
    try:
        main()
        done=True
    finally:
        branch.save_json(HERE/"timing"/f"{socket.gethostname()}_{os.getpid()}_{time.time_ns()}.json",
            dict(host=socket.gethostname(),arguments=sys.argv[1:],success=done,
                 wall_seconds_after_admission=time.monotonic()-started,
                 peak_cuda_bytes=torch.cuda.max_memory_allocated() if torch.cuda.is_initialized() else 0))
