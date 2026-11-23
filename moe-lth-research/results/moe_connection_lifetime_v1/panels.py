"""Identify complete original batches and their input/target source windows."""
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader

from groups import HERE, branch

ROOT = HERE.parents[1]


def source_ids(config, seed, order, t):
    file = ROOT/config["data"]["train_path"]
    data = np.frombuffer(file.read_text(encoding="utf8").encode("utf8"),dtype=np.uint8)
    length = config["data"]["seq_len"]
    size = config["training"]["batch_size"]
    n = (len(data)-1)//length
    gen = torch.Generator().manual_seed(seed)
    loader = DataLoader(range(n),batch_size=size,shuffle=True,drop_last=True,generator=gen)
    epoch = list(loader)
    perm = branch.N._permutations()[order]["segment_0"][:t]
    return data, length, [epoch[int(i)%len(epoch)].numpy() for i in perm]


def select(config,seed,order,t,batches):
    data,length,ids = source_ids(config,seed,order,t)
    distinct = {}
    for index, blocks in enumerate(ids):
        distinct.setdefault(blocks.tobytes(), index)
    candidates = list(distinct.values())
    # Two batches with adjacent source blocks share a boundary byte. Exclude
    # those pairs as well as repeated blocks from the opposite role.
    compatibility = np.zeros((len(candidates),len(candidates)),dtype=bool)
    lookup = {int(block):i for i,index in enumerate(candidates) for block in ids[index]}
    for i,index in enumerate(candidates):
        compatibility[i] = True
        for block in ids[index]:
            for adjacent in (block-1, block, block+1):
                if adjacent in lookup:
                    compatibility[i,lookup[adjacent]]=False
    assert np.array_equal(compatibility, compatibility.T)
    # Locate 16 distinct pairs while minimizing source-batch reuse. A role's
    # batch can occur in several pairs, always retaining its complete capacity
    # context. All source windows remain globally disjoint across roles.
    rng = np.random.default_rng(20261121+t)
    best = ([],[])
    for _ in range(128):
        left = []
        right = np.arange(len(candidates))
        for _ in range(16):
            choices = np.setdiff1d(np.arange(len(candidates)),left)
            retained = compatibility[choices][:,right].sum(axis=1)
            if retained.max()==0:
                break
            best_choices=choices[retained==retained.max()]
            choice=int(rng.choice(best_choices))
            left.append(choice)
            right=right[compatibility[choice,right]]
            if min(len(left),len(right)) > min(map(len,best)):
                best=(left.copy(),right.tolist())
    left,right=best
    if len(left)*len(right)<16:
        raise RuntimeError("insufficient disjoint native contexts")
    left,right=left[:16],right[:16]
    pairs=[]
    for offset in range(len(right)):
        for k,a in enumerate(left):
            b=right[(k+offset)%len(right)]
            pairs.append((candidates[a],candidates[b]))
            if len(pairs)==16:
                break
        if len(pairs)==16:
            break
    assert len(set(pairs))==16
    for i in {u for pair in pairs for u in pair}:
        window=data[ids[i][:,None]*length+np.arange(length+1)]
        x,y=batches[i]
        assert np.array_equal(window[:,:-1],x.numpy()) and np.array_equal(window[:,1:],y.numpy())
    return pairs
