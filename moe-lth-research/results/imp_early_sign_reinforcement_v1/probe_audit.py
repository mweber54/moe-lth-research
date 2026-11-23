"""Summarize the verification, eligibility, and routing exposure of dense probes."""
import csv
import re
from run import OUT, SEEDS, ORDERS, N, Path, torch, digest, read


def main():
    manifest = read(OUT/"inputs.json")
    results = []
    for seed in SEEDS:
        base = next(x["base"] for x in manifest["inputs"] if x["seed"] == seed)
        initial = N._read_model_payload(Path(base["initial_checkpoint"]))["model"]
        layers = sorted({int(n.split(".")[1]) for n in initial if ".moe.experts." in n})
        for order in ORDERS:
            path = OUT/f"probe_{seed}_{order}.pt"
            probe = torch.load(path,weights_only=True,mmap=True)
            assert probe["applied_updates"] == 100 and probe["max_reconstruction_error_fraction"] <= 1
            identity = digest(path)
            for name,(score,activity) in probe["scores"].items():
                match = re.fullmatch(r"blocks\.(\d+)\.moe\.experts\.(\d+)\..+\.weight",name)
                assert match, name
                layer,expert = map(int,match.groups())
                key = f"layer_{layers.index(layer)}_expert_{expert}"
                weight = initial[name]
                eligible = weight.isfinite() & (weight != 0) & score.isfinite() & activity.isfinite() & (activity > 0)
                results.append(dict(seed=seed,order=order,tensor=name,model_block=layer,expert=expert,
                                    assigned=probe["assigned"][key],accepted=probe["accepted"][key],
                                    weights=weight.numel(),eligible=int(eligible.sum()),
                                    zero_initial=int((weight==0).sum()),
                                    zero_activity=int((activity==0).sum()),
                                    nonfinite_weight_or_activity=int((~(weight.isfinite() & activity.isfinite())).sum()),
                                    applied=probe["applied_updates"],attempts=probe["attempts"],
                                    maximum_error_fraction=probe["max_reconstruction_error_fraction"],
                                    score_map_sha256=identity,ordered_batches_sha256=probe["train_hash"]))
    with (OUT/"probe_audit.csv").open("w",newline="") as output:
        writer = csv.DictWriter(output,fieldnames=list(results[0]))
        writer.writeheader()
        writer.writerows(results)


if __name__ == "__main__":
    main()
