"""Verify v2 artifacts and inventory the weight evidence for sign tracking."""
from pathlib import Path
import sys

OUT = Path(__file__).resolve().parent
SOURCE = OUT.with_name("imp_early_sign_reinforcement_neuron_matched_v2")
sys.path.insert(0,str(SOURCE))
from common_v2 import pilot as p

EARLY = (0,1,5,10,25,50,100,250)
EPSILON = 1e-8
READOUT = (0, 7, 15, 23)


def snapshots(folder, records):
    found = {}
    for path in sorted((folder/"snapshots").glob("step_[0-9][0-9][0-9][0-9].pt")):
        step = int(path.stem.split("_")[-1])
        found[step] = str(path)
    verified = {}
    for row in records:
        step = row["step"]
        if step not in found:
            continue
        path = Path(found[step])
        payload = p.N._read_model_payload(path)
        assert payload["step"] == step and p.N._tensor_state_hash(payload["model"]) == row["model_sha256"], path
        verified[step] = row["model_sha256"]
    return dict(paths=found,verified=verified,missing_early=[step for step in EARLY if step not in found])


def main():
    archive = p.read(SOURCE/"inputs.json")
    seal = p.read(SOURCE/"matching_seal.json")
    completion = p.read(SOURCE/"completion.json")
    assert completion["pairs"] == 18 and completion["recoveries"] == 36
    assert Path(completion["reporter"]).resolve() == SOURCE/"report.py" and Path(completion["output"]) == SOURCE
    assert p.digest(SOURCE/"inputs.json") == seal["input_manifest"]
    assert p.digest(seal["summary"]) == seal["summary_digest"]
    assert Path(p.N.__file__).resolve() in {Path(path) for path in archive["hashes"]}
    for path, expected in (archive["hashes"] | seal["matrix_audits"] |
                           {str(SOURCE/name):value for name,value in completion["artifacts"].items()} |
                           {str(SOURCE/name):value for name,value in seal["selections"].items()}).items():
        assert p.digest(path) == expected,path
    jobs = []
    for seed in p.SEEDS:
        material = p.N._load_training_material(seed)
        for entry in (r for r in archive["inputs"] if r["seed"] == seed):
            order = entry["order"]
            baseline = p.reuse_baseline(entry,material)
            assert baseline == p.read(SOURCE/f"baseline_{seed}_{order}_imp.json")
            _,batches,validation = p.ordered(seed,order,material)
            readout = [validation[i] for i in READOUT]
            probe_path = p.OUT/f"probe_{seed}_{order}.pt"
            probe = p.torch.load(probe_path,weights_only=True,mmap=True)
            assert probe["applied_updates"] == 100 and probe["train_hash"] == p.N.recovery._batch_sequence_hash(batches)
            assert p.N.recovery._batch_sequence_hash(validation) == entry["validation_hash"]
            state, masks = p.baseline_state(entry)
            base = entry["base"]
            assert all(name in state and state[name].shape == mask.shape for name,mask in masks.items())
            assert set(probe["scores"]) == set(masks)
            ordered_jobs = [job for job in archive["jobs"] if (job["seed"],job["order"]) == (seed,order)]
            variants = [(dict(seed=seed,order=order,draw=0,arm="untouched",mask="imp"),
                         Path(baseline["source"]).parent,baseline["records"],None)]
            for job in ordered_jobs:
                folder = SOURCE/"jobs"/p.job_id(job)
                result = p.read(folder/"result.json")
                assert result["job"] == job and result["seal"] == p.digest(SOURCE/"matching_seal.json")
                assert result["attempts"] == 2500
                picks = p.torch.load(SOURCE/f'flips_{seed}_{order}_imp_{job["draw"]}.pt',weights_only=True)[job["arm"]]
                expected,_ = p.reconstruct(entry,picks)
                assert result["records"][0]["model_sha256"] == p.N._tensor_state_hash(expected)
                del expected
                variants.append((job,folder/"attempt",result["records"],result["applied_updates"]))
            for job,folder,records,applied in variants:
                p.validate_losses(records,entry)
                segment = folder/"segment_0"
                logs = [r for r in p.read_jsonl(segment/"gradient_stats/gradient_stats.jsonl") if 1 <= r["step"] <= 2500]
                assert [record["step"] for record in logs] == list(range(1,2501))
                update_steps = [r["step"] for r in logs if r["optimizer_step_applied"]]
                assert applied is None or len(update_steps) == applied
                record = p.read(segment/"segment_record.json")
                assert record["optimizer"] == "fresh_AdamW" and record["router_mode"] == "trainable"
                assert record["segment_training_batch_hash"] == probe["train_hash"]
                items = snapshots(folder,records)
                jobs.append(dict(job=job,folder=str(folder),records=records,applied_steps=update_steps,
                                 source_host=p.read(folder/"heartbeat.json").get("host") if (folder/"heartbeat.json").exists() else None,
                                 **items))
                print("Audited",p.job_id(job),items["missing_early"],flush=True)
            p.write(OUT/"audit_progress.json",dict(jobs=jobs))
            del probe,state,masks
    manifest = dict(source=str(SOURCE),source_manifest=seal["input_manifest"],
                    source_seal=p.digest(SOURCE/"matching_seal.json"),inputs=archive["inputs"],
                    early_steps=EARLY,readout_indices=READOUT,near_zero=EPSILON,projection_steps=250,
                    projection_rule="Project disallowed signs to exactly zero. Zero lies in both closed half-lines.",
                    repair_draw=1,jobs=jobs)
    p.write(OUT/"audit_manifest.json",manifest)


if __name__ == "__main__":
    main()
