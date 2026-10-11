"""Protocol pilot on TRAINING instances (disjoint from the benchmark): which QAOA
objective to fix for the main run. Compares <H> vs CVaR_0.1 vs CVaR_0.25."""
import os, sys, json, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
from scripts.tune_e3 import training_set
from settle.encodings import e1_iqp, e2_indicator, e4_overflow_sq, e0_standard
from settle.qaoa import Problem, optimise_sweep, statevector, logical_distribution
from settle.metrics import quality
from settle import iqp

cache = iqp.load_cache_text(open("settle/iqp_data/iqp_cache.json").read())
train = training_set()
insts = train[:5] + train[20:25]
rows = []
for inst in insts:
    encs = [e1_iqp(inst, rule="neighbour", cache=cache), e2_indicator(inst, rule="neighbour"),
            e4_overflow_sq(inst, rule="neighbour")]
    e0 = e0_standard(inst, rule="neighbour")
    if e0.n_total <= 18:
        encs.append(e0)
    for enc in encs:
        pr = Problem(enc)
        for obj, a in (("mean", 1.0), ("cvar", 0.1), ("cvar", 0.25)):
            res = optimise_sweep(pr, 3, objective=obj, alpha=a)
            for p, (E, g, b, ev, conv) in res.items():
                q = quality(inst, logical_distribution(pr, statevector(pr, g, b)))
                rows.append({"inst": inst.name, "enc": enc.name, "obj": "%s%g" % (obj, a), "p": p,
                             "p_opt": q["p_opt"], "p_feas": q["p_feasible"], "enr": q["enrichment"],
                             "ratio": q["exp_ratio"]})
    print(inst.name, "done", flush=True)
json.dump(rows, open("results/pilot_objective.json", "w"), indent=0)
