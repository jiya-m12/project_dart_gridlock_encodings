"""How good is the optimiser at p = 1? Main set, CVaR_0.25, neighbour weights, E1/E2/E4.
For every instance, evaluate the p = 1 objective on a 64 x 64 grid (gamma in (0, pi],
beta in [-pi/2, pi/2); beta has period pi and (gamma, beta) -> (-gamma, -beta) gives the
same probabilities, so this covers every distinct p = 1 state up to gamma = pi), then report
  - on how many instances the best grid point has beta < 0 (the start-sign amendment), and
  - on how many the shipped p = 1 optimum (results/bench_raw.jsonl) is worse than the best
    grid point by more than 0.1% (the optimiser is local and can miss).
Output: results/p1_grid_check.json. About 1 min."""
import os, sys, json
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
from settle.instances import load_instances_json
from settle.encodings import e1_iqp, e2_indicator, e4_overflow_sq
from settle import iqp
from settle.qaoa import Problem, energy

I = load_instances_json(open("data/bench_instances.json").read())
cache = iqp.load_cache_text(open("settle/iqp_data/iqp_cache.json").read())
raw = {(r["instance"], r["encoding"]): r for r in map(json.loads, open("results/bench_raw.jsonl"))
       if r["objective"] == "cvar"}
MK = {"E1": lambda i: e1_iqp(i, "neighbour", cache=dict(cache), derive_missing=False),
      "E2": lambda i: e2_indicator(i, "neighbour"), "E4": lambda i: e4_overflow_sq(i, "neighbour")}
G = np.linspace(0.01, np.pi, 64)
B = np.linspace(-np.pi / 2, np.pi / 2, 65)[:-1]
out = {"grid": "64 x 64, gamma in [0.01, pi], beta in [-pi/2, pi/2)", "objective": "cvar", "alpha": 0.25,
       "rule": "neighbour", "set": "bench"}
for e, mk in MK.items():
    neg, miss, rel = 0, 0, []
    for i in I:
        pr = Problem(mk(i))
        E = np.array([[energy(pr, np.array([g, b]), "cvar", 0.25) for b in B] for g in G])
        gi, bi = np.unravel_index(E.argmin(), E.shape)
        neg += int(B[bi] < 0)
        ship = raw[(i.name, e)]["qaoa"]["1"]["objective_value"]
        r = (ship - E.min()) / abs(E.min())
        rel.append(float(r))
        miss += int(r > 1e-3)
    out[e] = {"n": len(I), "grid_best_beta_negative": neg, "shipped_worse_than_grid_by_0.1pct": miss,
              "largest_relative_shortfall": max(rel)}
    print(e, out[e], flush=True)
json.dump(out, open("results/p1_grid_check.json", "w"), indent=1)
