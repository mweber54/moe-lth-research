"""Run sealed degree-control comparisons and the fixed-set routing crossover."""
import argparse
import fcntl
import socket
import tempfile
from pathlib import Path

from audit import OUT, p
from common_v2 import load
from replay import N, admission, replay


def jobs(stage,manifest,repair=False):
    sources=manifest["sources"]
    for candidate in sources:
        label,entry=candidate["label"],candidate["entry"]
        seed,order=entry["seed"],entry["order"]
        initial=dict(seed=seed,order=order,mask=label,draw=1)
        if stage=="connectivity":
            if label=="imp" and manifest["target"]==.34207:
                continue
            for arm in (("high","random") if repair else ("untouched","high","random")):
                job=dict(initial,arm=arm,draw=0 if arm=="untouched" else 1)
                name=p.job_id(job)+("_constrained" if repair else "")
                yield entry,dict(job=job,name=name,flip_paths=[candidate["flips"]],records=[]),(
                    "new_repair" if repair else "new")
            continue
        both=[r["flips"] for r in sources if (r["entry"]["seed"],r["entry"]["order"])==(seed,order)]
        baseline=dict(initial,arm="untouched",draw=0)
        yield entry,dict(job=baseline,name=p.job_id(baseline),flip_paths=both,records=[],scaled=label=="scaled"),"new"
        for condition in ("native","scaled"):
            for arm in ("high","random"):
                job=dict(initial,arm=arm)
                yield entry,dict(job=job,name=f"{p.job_id(job)}_under_{condition}",
                                 flip_paths=[candidate["flips"]],records=[],scaled=condition=="scaled"),"new"


def main(stage,shard):
    repair=stage=="connectivity_repair"
    stage="connectivity" if repair else stage
    root=OUT/stage
    manifest=p.read(root/"manifest.json")
    assert manifest["stage"]==stage
    assert p.digest(root/"diagnostics.json")==manifest["diagnostics_sha256"]
    for item in manifest["sources"]:
        assert p.digest(item["entry"]["base"]["mask_path"])==item["mask_sha256"]
        for key in ("flips","diagnostic"):
            assert p.digest(item[key])==item[key+"_sha256"]
    reporter=load("v3_later_report",OUT/"later_report.py")
    assert reporter.OUT==OUT
    print("Reporter",reporter.__file__,"output",reporter.OUT,flush=True)
    last_seed,material=None,None
    for entry,source,mode in jobs(stage,manifest,repair):
        job,name=source["job"],source["name"]
        if job["order"]!=("A","B")[shard]:
            continue
        destination=root/"jobs"/name
        destination.mkdir(parents=True,exist_ok=True)
        result=destination/"result.json"
        if result.exists():
            continue
        admission(socket.gethostname())
        seed=job["seed"]
        if seed!=last_seed:
            del material
            material=N._load_training_material(seed)
            last_seed=seed
        folder=destination/"attempt"
        if folder.exists() and not (folder/"segment_0/interruption.pt").exists():
            folder.rename(Path(tempfile.mkdtemp(prefix="partial_",dir=destination))/"attempt")
        folder.mkdir(exist_ok=True)
        print("START",stage,name,flush=True)
        replay(entry,source,material,mode,folder)
        p.write(result,dict(p.read(folder/"result.json"),selection_manifest=p.digest(root/"manifest.json")))
        with (OUT/"report.lock").open("a") as lock:
            fcntl.flock(lock,fcntl.LOCK_EX)
            reporter.repair_controls() if repair else reporter.main(stage)


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage",choices=("connectivity","connectivity_repair","routing"))
    parser.add_argument("shard",type=int,choices=(0,1))
    args=parser.parse_args()
    with (OUT/f"{args.stage}_{args.shard}.lock").open("a") as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        main(args.stage,args.shard)
