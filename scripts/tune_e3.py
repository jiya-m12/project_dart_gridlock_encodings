"""Tune E3 (unbalanced penalization) weights on TRAINING instances only.
Usage: python3 scripts/tune_e3.py <set>   (set = bench | t0 | c)
Training set: same generator, sizes, amounts and filter as the named set, disjoint
seed base (make_instances.TRAIN_BASE). Criterion fixed in advance: maximise the
fraction of training instances whose E3 ground state is the true optimum.
Tie-break: smaller l2, then smaller l1. The result is used unchanged on the set.
E3 weights are in raw amount units, hence one tuning per amount scale/size."""
import os, sys, json
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
from scripts.make_instances import build_set, TRAIN_BASE
from settle.encodings import e3_unbalanced
from settle.verify import check_exact

L1 = [0.0, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 4.0]
L2 = [0.02, 0.05, 0.1, 0.2, 0.3, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0]


def training_set(name="bench"):
    return build_set(name, base=TRAIN_BASE[name], tag_prefix="train_")


def tune(name):
    train = training_set(name)
    grid = {(l1, l2): float(np.mean([check_exact(i, e3_unbalanced(i, l1, l2))["exact"] for i in train]))
            for l1 in L1 for l2 in L2}
    order = sorted(grid, key=lambda k: (-grid[k], k[1], k[0]))
    best = order[0]
    ties = [list(k) for k in order if grid[k] == grid[best]]
    out = {"set": name, "l1": best[0], "l2": best[1], "train_exact_rate": grid[best],
           "criterion": "max fraction of training instances whose E3 ground state is the true optimum",
           "tie_break": "smaller l2, then smaller l1", "tied_best": ties,
           "training_instances": [{"name": i.name, "seed": i.meta["set_seed"], "f_liq": i.meta["f_liq"]} for i in train],
           "grid_l1": L1, "grid_l2": L2, "grid": {"%g,%g" % k: v for k, v in grid.items()}}
    json.dump(out, open("results/e3_tuning_%s.json" % name, "w"), indent=1)
    return out


if __name__ == "__main__":
    for name in (sys.argv[1:] or ["bench", "t0", "c"]):
        r = tune(name)
        print(name, "chosen (l1, l2) =", (r["l1"], r["l2"]), "train exact %.3f" % r["train_exact_rate"], "ties", r["tied_best"])
