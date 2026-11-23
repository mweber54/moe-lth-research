"""Construct and audit final-layer matched sets from the decision-250 prefix."""
from pathlib import Path
import hashlib
import json
import sys

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
V2 = HERE.with_name("moe_online_reinforcement_pruning_v2")
sys.path.insert(0, str(V2))
import audit_match
import branch


def scope(state):
    experts = [name for name in state if ".moe.experts." in name and name.endswith(".fc2.weight")]
    last = max(int(name.split(".")[1]) for name in experts)
    names = sorted(name for name in experts if name.startswith(f"blocks.{last}."))
    if len(names) != 32:
        raise ValueError("expected all 32 final-layer experts")
    return names


class MatchingGateFailure(ValueError):
    def __init__(self, violations, groups, rows):
        super().__init__(str(violations))
        self.violations=violations
        self.groups=groups
        self.rows=rows


def construct(state, scores, seed, order, archived=None, control=False):
    groups = {}
    rows = []
    for name in scope(state):
        nr, nc = state[name].shape
        w = state[name].numpy().astype(np.float64).ravel()
        score = scores[name]["score"].numpy().ravel()
        valid = (scores[name]["absolute_update"].numpy().ravel() > 0) & (scores[name]["initial_sign"].numpy().ravel() != 0)
        assert np.array_equal(np.isfinite(score), valid)
        pool = np.lexsort((np.arange(w.size), np.abs(w)))[:w.size//4]
        pool = pool[valid[pool]]
        random = np.sort(np.random.default_rng(audit_match.seed64(seed, order, name)).choice(pool, 13107, replace=False))
        bins = np.array_split(pool, 5)
        if archived is not None:
            assert np.array_equal(random, archived["random"][name].numpy())
        certificates=[]
        if archived is not None and not control:
            high = archived["high"][name].numpy()
        else:
            objective = score
            if control:
                entropy = hashlib.sha256(f"lifetime_v1|random2|{seed}|{order}|{name}".encode()).digest()
                objective = np.random.default_rng(int.from_bytes(entropy[:8], "big")).random(w.size)
            solved=[audit_match.flow(b,objective,np.intersect1d(random,b),nr,nc) for b in bins]
            high=np.sort(np.concatenate([indices for indices,cert in solved]))
            certificates=[cert for indices,cert in solved]
        assert len(high) == len(np.unique(high)) == len(random) and np.isin(high,pool).all()
        for b in bins:
            h, r = np.intersect1d(high,b), np.intersect1d(random,b)
            for a,z in ((h//nc,r//nc),(h%nc,r%nc)):
                assert np.array_equal(np.bincount(a), np.bincount(z))
        he, re = (float(w[idx] @ w[idx]) for idx in (high,random))
        energy_pass = abs(he-re) <= 0.05*re
        h, r = np.setdiff1d(high,random), np.setdiff1d(random,high)
        recipient = np.union1d(h%nc,r%nc)
        groups[name] = {key: torch.as_tensor(value.copy()) for key,value in
                        dict(H=high,R=random,M=np.union1d(high,random),S=recipient).items()}
        rows.append(dict(name=name, certificates=certificates, overlap=len(high)-len(h), exclusive_per_arm=len(h), union=len(high)+len(r),
                         high_energy=he, random_energy=re, energy_pass=energy_pass, score_gap=float(score[high].mean()-score[random].mean()),
                         recipient_rows=len(recipient), total_fc1_rows=nc))
    violations=[r["name"]+": energy" for r in rows if not r["energy_pass"]]
    if not control:
        if np.mean([r["score_gap"] for r in rows]) < .2:
            violations.append("score separation below 0.20")
        if sum(r["overlap"] for r in rows) > .3*len(rows)*13107:
            violations.append("overlap above 0.30")
    if violations:
        raise MatchingGateFailure(violations,groups,rows)
    return groups, rows


def development(seed, order):
    cell = f"seed{seed}_{order}"
    dest = HERE/"groups"/cell
    dest.mkdir(parents=True, exist_ok=True)
    files = json.loads((V2/"input_manifest.json").read_text())["cells"][cell]["files"]
    def read(key):
        path = Path(files[key]["path"])
        assert branch.sha(path) == files[key]["sha256"]
        return torch.load(path, map_location="cpu", weights_only=False, mmap=True)
    path = V2/"matching"/cell/"selections.pt"
    assert branch.sha(path) == json.loads((V2/"sealed_inputs.json").read_text())["cells"][cell]["selection_sha256"]
    archived = torch.load(path, map_location="cpu", weights_only=True)
    state, scores = read("checkpoint")["model"], read("scores")["scores"]
    groups, rows = construct(state,scores,seed,order,archived)
    branch.save_pt(dest/"sets.pt",groups)
    branch.save_json(dest/"audit.json",dict(cell=cell,sha256=branch.sha(dest/"sets.pt"),matrices=rows))


if __name__ == "__main__":
    for s in (7,17,29):
        for o in ("A","B"):
            development(s,o)
