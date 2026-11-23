"""Audit cumulative degree-matched rectangles at common model-level doses."""
from dataclasses import dataclass
import ctypes
import gzip
import json
import math

from common_v2 import OUT, PILOT, pilot as p
from match import ARMS, PAIRS, RULES, axis_audit, matrix_inputs, np, pair_summary, relative_gap, torch

DIRECT = OUT / "conditioned_rectangles"
TARGETS = [.34207, .30, .25, .20, .175, .15, .125, .10, .075, .05]
PROPOSALS = 1000000
NAMESPACE = "conditioned_rectangles_v1"


def audit_axis(indices, selected, energy, size):
    if len(selected[0]):
        return axis_audit(indices, selected, energy, size)
    return dict(counts=[0]*size, high_energy=[0.]*size, random_energy=[0.]*size,
                relative_rms=0., relative_max=0., absolute_max=0.,
                worst_relative_indices=[], worst_absolute_indices=[], coverage=0.)


@dataclass
class Matrix:
    name: str
    shape: tuple
    weights: object
    score: object
    activity: object
    groups: object
    energy: object
    denominator: float
    native_dose: float
    seed: int
    survivor_count: int
    high_population: int
    order: tuple
    prefix: object

    def total(self, count):
        return float(self.prefix[count])

    def count(self, level):
        value = (level * self.native_dose)**2 * self.denominator
        pos = min(int(np.searchsorted(self.prefix, value)), len(self.prefix)-1)
        if pos and abs(value-self.prefix[pos-1]) < abs(value-self.prefix[pos]):
            pos -= 1
        return pos

    def audit(self, count):
        ids = [np.sort(arm[:2*count].astype(np.int64)) for arm in self.order]
        n = 2*count
        assert all(len(np.unique(arm)) == n and (self.groups[arm] >= 0).all() for arm in ids)
        strata = [np.bincount(self.groups[arm], minlength=128) for arm in ids]
        assert np.array_equal(*strata)
        row = audit_axis([arm//self.shape[1] for arm in ids],ids,self.energy,self.shape[0])
        column = audit_axis([arm%self.shape[1] for arm in ids],ids,self.energy,self.shape[1])
        energies = [float(self.energy[arm].sum()) for arm in ids]
        doses = [math.sqrt(value/self.denominator) if self.denominator else 0. for value in energies]
        covariates = [dict(positive=int((self.weights[arm]>0).sum()),
                           mean_magnitude=float(np.abs(self.weights[arm]).mean()) if n else 0.,
                           mean_activity=float(self.activity[arm].mean()) if n else 0.,
                           mean_score=float(self.score[arm].mean()) if n else 0.,
                           score_quantiles=np.quantile(self.score[arm],[0,.25,.5,.75,1]).tolist() if n else [])
                      for arm in ids]
        magnitude_gap = relative_gap(*(c["mean_magnitude"] for c in covariates))
        activity_gap = relative_gap(*(c["mean_activity"] for c in covariates))
        gates = dict(matrix_dose=relative_gap(*doses) <= RULES["matrix_dose_gap"],
                     row_energy=row["relative_rms"] <= RULES["neuron_energy_rms"]+1e-12 and row["relative_max"] <= RULES["neuron_energy_max"],
                     column_energy=column["relative_rms"] <= RULES["neuron_energy_rms"]+1e-12 and column["relative_max"] <= RULES["neuron_energy_max"],
                     magnitude=magnitude_gap <= RULES["magnitude_gap"],activity=activity_gap <= RULES["activity_gap"])
        return ids, dict(tensor=self.name, count=n, seed=self.seed, denominator=self.denominator,
                         energies=energies, relative_doses=doses, native_dose_weight=self.native_dose,
                         maximum_dose=math.sqrt(float(self.prefix[-1])/self.denominator) if self.denominator else 0.,
                         row=row, column=column, covariates=covariates, strata=strata[0].tolist(),
                         overlap=int(len(np.intersect1d(*ids,assume_unique=True))), survivor_count=self.survivor_count,
                         eligible_survivors=int((self.groups >= 0).sum()), high_pool=self.high_population,
                         magnitude_gap=magnitude_gap, activity_gap=activity_gap, passed=gates)


def sampler_function():
    lib = ctypes.CDLL(str(OUT / "rectangle_sampler.so"))
    sampler = lib.rectangles
    ints = np.ctypeslib.ndpointer(dtype=np.int32,ndim=1,flags="C_CONTIGUOUS")
    floats = np.ctypeslib.ndpointer(dtype=np.float64,ndim=1,flags="C_CONTIGUOUS")
    sampler.argtypes = [ctypes.c_int,ctypes.c_int,ints,floats,floats,ints,ctypes.c_int,
                        ctypes.c_double,ints,ints,ctypes.c_uint64,ctypes.c_int]
    sampler.restype = ctypes.c_int
    return sampler


def collect(entry, draw, matrices, native_targets, sampler):
    seed, order = entry["seed"], entry["order"]
    for name, (shape, inputs) in matrices.items():
        w, score, activity, survivor, groups, high = inputs
        energy = 4*np.square(w.astype(np.float64))
        denominator = float(energy[survivor].sum()/4)
        pool = np.ascontiguousarray(high,dtype=np.int32)
        arms = tuple(np.empty(len(pool),dtype=np.int32) for _ in ARMS)
        draw_seed = p.rng_seed(OUT.name,NAMESPACE,seed,order,draw,name)
        count = sampler(*shape,groups,energy,np.ascontiguousarray(activity,dtype=np.float64),pool,
                        len(pool),TARGETS[0]**2*2*denominator,*arms,draw_seed,PROPOSALS)
        assert count % 2 == 0
        arms = tuple(arm[:count].copy() for arm in arms)
        assert bool(np.isin(arms[0],high).all())
        prefix = np.r_[0., np.cumsum(energy[arms[0]].reshape(-1,2).sum(axis=1))]
        yield Matrix(name,shape,w,score,activity,groups,energy,denominator,
                     max(native_targets[seed,draw,name]/entry["targets"][str(draw)],1e-12),
                     draw_seed,int(survivor.sum()),len(high),arms,prefix)


def allocation(matrices, target):
    denominator = sum(m.denominator for m in matrices)
    target_energy = target**2 * denominator
    if sum(m.prefix[-1] for m in matrices) <= target_energy:
        return [len(m.prefix)-1 for m in matrices], None
    low, high = 0., 1.
    while sum(m.total(m.count(high)) for m in matrices) < target_energy:
        high *= 2
    for _ in range(22):
        midpoint = (low+high)/2
        if sum(m.total(m.count(midpoint)) for m in matrices) < target_energy:
            low = midpoint
        else:
            high = midpoint
    best = min((low,high),key=lambda candidate:abs(sum(m.total(m.count(candidate)) for m in matrices)-target_energy))
    return [m.count(best) for m in matrices], best


def cohort(inputs, targets, save=False):
    sampler = sampler_function()
    native_targets = {(int(row["seed"]),int(row["draw"]),row["tensor"]):float(row["realized_distance"])
                      for row in p.rows(p.PRIOR / "matrix_invariants.csv")
                      if row["arm"] == "Bm" and row["mask_type"] == "imp"}
    summaries = {target:[] for target in targets}
    files = {}
    for entry in inputs["inputs"]:
        seed, order = entry["seed"], entry["order"]
        base = entry["base"]
        dense = p.N._read_model_payload(p.Path(base["initial_checkpoint"]))["model"]
        masks = p.N.load_masks(p.Path(base["mask_path"]))
        probe = torch.load(PILOT/f"probe_{seed}_{order}.pt",weights_only=True,mmap=True)
        matrices = {name:(dense[name].shape,matrix_inputs(dense[name],*probe["scores"][name],mask))
                    for name,mask in masks.items()}
        for draw in (1,2,3):
            pools = list(collect(entry,draw,matrices,native_targets,sampler))
            for target in targets:
                counts, allocation_level = allocation(pools,target)
                selections = {arm:{} for arm in ARMS}
                audit = []
                for matrix,count in zip(pools,counts):
                    chosen, cell = matrix.audit(count)
                    audit.append(cell)
                    if save:
                        for arm,ids in zip(ARMS,chosen):
                            selections[arm][matrix.name] = torch.from_numpy(ids)
                summary = pair_summary(seed,order,draw,audit,target)
                summary["allocation_level"] = allocation_level
                summaries[target].append(summary)
                if save:
                    assert summary["passed"], summary
                    flip = DIRECT / f"flips_{seed}_{order}_imp_{draw}.pt"
                    p.save_tensor(flip,selections)
                    path = DIRECT / f"neurons_{seed}_{order}_{draw}.json.gz"
                    with gzip.open(path,"wt",compresslevel=3) as stream:
                        json.dump(audit,stream,separators=(",",":"))
                    files[flip] = p.digest(flip)
                    files[path] = p.digest(path)
            print("Audited",seed,order,draw,
                  [(t,round(summaries[t][-1]["doses"][0],5),
                    round(summaries[t][-1]["score_gap"],4),summaries[t][-1]["passed"]) for t in targets],flush=True)
            p.write(DIRECT / ("chosen_summary.json" if save else "feasibility.json"),
                    {str(t):rows for t,rows in summaries.items()})
        del matrices, pools, dense, masks, probe
    return summaries, files


def main():
    DIRECT.mkdir(exist_ok=True)
    original = p.read(OUT / "inputs.json")
    rules = dict(RULES, targets=TARGETS, proposals=PROPOSALS,
                 seed_namespace=NAMESPACE, allocation="common model dose using whole prefix rectangles",
                 construction_row_column_gap=.12, construction_matrix_energy_gap=.015,
                 construction_magnitude_gap=.04, construction_activity_gap=.09)
    hashes = {**original["hashes"], **{str(OUT / path):p.digest(OUT / path)
              for path in ("rectangle_sampler.cpp","rectangle_sampler.so","rectangle_match.py","launch.sh")}}
    inputs = dict(original,rules=rules,hashes=hashes,seeds_namespace=NAMESPACE)
    p.write(DIRECT / "inputs.json",inputs)
    summaries, _ = cohort(inputs,TARGETS)
    passing = [target for target,rows in summaries.items() if len(rows) == len(PAIRS) and all(row["passed"] for row in rows)]
    if not passing:
        raise SystemExit("No common dose met every declared gate in all 18 pairs.")
    target = passing[0]
    chosen, files = cohort(inputs,[target],save=True)
    assert chosen[target] == summaries[target]
    for path, expected in hashes.items():
        assert p.digest(path) == expected, path
    for path in files:
        if path.suffix == ".pt":
            (OUT / path.name).hardlink_to(path)
    p.write(OUT / "inputs.json",inputs)
    summary_path = DIRECT / "accepted_pairs.json"
    p.write(summary_path,chosen[target])
    p.write(OUT / "matching_seal.json",dict(target_distance=target,rules=rules,
            input_manifest=p.digest(OUT/"inputs.json"),summary=str(summary_path),summary_digest=p.digest(summary_path),
            selections={k.name:v for k,v in files.items() if k.suffix == ".pt"},
            matrix_audits={str(k):v for k,v in files.items() if k.suffix == ".gz"}))


if __name__ == "__main__":
    main()
