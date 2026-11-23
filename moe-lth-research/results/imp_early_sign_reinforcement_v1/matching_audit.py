"""Check matching constraints and quantify neuron concentration differences."""
from collections import defaultdict
import csv
import math

from run import OUT, ARMS, MASKS, read, digest, write


def top_share(counts):
    total = sum(counts)
    n = math.ceil(len(counts)/10)
    return sum(sorted(counts,reverse=True)[:n])/total if total else 0.


def main():
    seal = read(OUT/"matching_seal.json")
    source = OUT/"matrix_matching.json"
    assert digest(source) == seal["matrix_audit"]
    matrices = defaultdict(list)
    for row in read(source):
        keys = ("seed","order","mask","draw","tensor")
        matrices[tuple(row[k] for k in keys)].append(row)
    output = []
    for key, group in matrices.items():
        assert {r["arm"] for r in group} == set(ARMS)
        assert len({(r["count"],r["positive"]) for r in group}) == 1
        assert all(r["matrix_distance_passed"] for r in group)
        concentrations = {coord: {r["arm"]:top_share(r[coord+"_counts"]) for r in group}
                          for coord in ("row","column")}
        for r in group:
            assert sum(r["row_counts"]) == sum(r["column_counts"]) == r["count"]
            details = {k:r[k] for k in ("arm","target","distance","count","positive","mean_abs_w","mean_A",
                                        "R_quantiles","eligible","eligible_survivors")}
            output.append(dict(zip(keys,key),**details,
                               row_top_decile_share=concentrations["row"][r["arm"]],
                               column_top_decile_share=concentrations["column"][r["arm"]],
                               row_share_gap=max(concentrations["row"].values())-min(concentrations["row"].values()),
                               column_share_gap=max(concentrations["column"].values())-min(concentrations["column"].values())))
    with (OUT/"matching_audit.csv").open("w",newline="") as output_file:
        writer = csv.DictWriter(output_file,fieldnames=list(output[0]))
        writer.writeheader()
        writer.writerows(output)
    flagged = [r for r in output if r["arm"]=="high"
               and max(r["row_share_gap"],r["column_share_gap"]) >= .10]
    concentration = []
    for mask in MASKS:
        for arm in ARMS:
            group = [r for r in output if r["mask"] == mask and r["arm"] == arm and r["count"]]
            concentration.append(dict(mask=mask,arm=arm,
                row_top_decile_mean=sum(r["row_top_decile_share"] for r in group)/len(group),
                column_top_decile_mean=sum(r["column_top_decile_share"] for r in group)/len(group),
                mean_A_weighted=sum(r["mean_A"]*r["count"] for r in group)/sum(r["count"] for r in group),
                mean_abs_w_weighted=sum(r["mean_abs_w"]*r["count"] for r in group)/sum(r["count"] for r in group)))
    dose = read(OUT/f"global_matching_{seal['dose_factor']}.json")
    write(OUT/"matching_summary.json",dict(
        target_distance=seal["target_distance"], matrix_triplets=len(matrices),
        matrix_triplets_with_neuron_concentration_gap_ge_10pp=len(flagged),
        global_max_relative_distance_gap=max((max(g["distances"].values())-min(g["distances"].values()))
                                              /g["target"] for g in dose),
        minimum_high_R_gap=min(g["mean_R"]["high"]-max(g["mean_R"]["random"],g["mean_R"]["weak"])
                               for g in dose), concentration=concentration))
    print(f"Verified {len(matrices)} matrix triplets. {len(flagged)} had an arm range of at least "
          "10 percentage points in top-decile neuron concentration.",flush=True)


if __name__ == "__main__":
    main()
