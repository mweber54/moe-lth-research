"""Check whether conditioned proposals can produce enough distinct matched flips."""
import ctypes
import itertools
from common_v2 import OUT, PILOT, pilot as p
from match import matrix_inputs, np, torch


def main():
    seed, order, draw = 7, "A", 1
    entry = next(e for e in p.read(OUT / "inputs.json")["inputs"] if (e["seed"],e["order"]) == (seed,order))
    dense = p.N._read_model_payload(p.Path(entry["base"]["initial_checkpoint"]))["model"]
    masks = p.N.load_masks(p.Path(entry["base"]["mask_path"]))
    probe = torch.load(PILOT / f"probe_{seed}_{order}.pt", weights_only=True, mmap=True)
    lib = ctypes.CDLL(str(OUT / "rectangles.so"))
    sampler = lib.rectangles
    ints = np.ctypeslib.ndpointer(dtype=np.int32, ndim=1, flags="C_CONTIGUOUS")
    floats = np.ctypeslib.ndpointer(dtype=np.float64, ndim=1, flags="C_CONTIGUOUS")
    sampler.argtypes = [ctypes.c_int, ctypes.c_int, ints, floats, ints, ctypes.c_int,
                        ctypes.c_double, ints, ints, ctypes.c_uint64, ctypes.c_int]
    sampler.restype = ctypes.c_int
    totals = {}
    for name in itertools.islice(masks,32):
        weight = dense[name]
        w, score, _, surviving, groups, pool = matrix_inputs(weight,*probe["scores"][name],masks[name])
        energy = 4*np.square(w.astype(np.float64))
        pool = np.ascontiguousarray(pool,dtype=np.int32)
        hi, control = (np.empty(len(pool),dtype=np.int32) for _ in range(2))
        denominator = energy[surviving].sum()/4
        for target in (.34207,.25,.15,.10):
            count = sampler(*weight.shape,groups,energy,pool,len(pool),target**2*denominator,
                            hi,control,p.rng_seed("rectangle",seed,order,draw,name),500000)
            selections = hi[:count], control[:count]
            if count:
                for axis, extent in ((lambda a:a//weight.shape[1],weight.shape[0]), (lambda a:a%weight.shape[1],weight.shape[1])):
                    assert np.array_equal(*(np.bincount(axis(x),minlength=extent) for x in selections))
                assert np.array_equal(*(np.bincount(groups[x],minlength=128) for x in selections))
                assert all(len(np.unique(x)) == count for x in selections)
            dose = float(np.sqrt(energy[selections[0]].sum()/denominator))
            gap = float((score[selections[0]]-score[selections[1]]).sum())
            bucket=totals.setdefault(target,[])
            bucket.append((dose,count,gap))
    for target, results in totals.items():
        print(target, "minimum/median dose", min(x[0] for x in results), np.median([x[0] for x in results]),
              "gap",sum(x[2] for x in results)/max(1,sum(x[1] for x in results)),flush=True)


if __name__ == "__main__":
    main()
