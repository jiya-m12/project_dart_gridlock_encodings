"""Penalty-weight rule study (exactness only, no QAOA). Pre-registered tight set:
ring+chords generator, amounts 1..10, degree <= 5, sizes 4 banks/6 payments and
6 banks/9 payments, liquidity f in {0.1, 0.2, 0.3}, 60 instances per cell, seeds
50000 + i (no gridlock filter). For each encoding and rule (local / neighbour /
global, gamma = 2) count instances whose encoded minimum is NOT the true optimum.
Output: results/tuning_failures.json. IQP patterns missing from the cache are derived
here and written to settle/iqp_data/iqp_cache_tuning.json (kept separate from the
cache shipped for the live demo)."""
import os, sys, json
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from settle.instances import generate
from settle.encodings import e0_standard, e1_iqp, e2_indicator, e4_overflow_sq
from settle.verify import check_exact
from settle import iqp

cache = iqp.load_cache_text(open("settle/iqp_data/iqp_cache.json").read())
ENC = {"E0": e0_standard, "E1": lambda i, rule: e1_iqp(i, rule=rule, cache=cache),
       "E2": e2_indicator, "E4": e4_overflow_sq}
rows = []
for nb, na in ((4, 6), (6, 9)):
    for f in (0.1, 0.2, 0.3):
        insts = [i for i in (generate(nb, na, 50000 + s, f_liq=f) for s in range(60)) if i is not None]
        for rule in ("local", "neighbour", "global"):
            row = {"banks": nb, "payments": na, "f_liq": f, "rule": rule, "n": len(insts)}
            for name, mk in ENC.items():
                row[name] = sum(not check_exact(i, mk(i, rule=rule))["exact"] for i in insts)
            rows.append(row)
            print(row, flush=True)
json.dump(rows, open("results/tuning_failures.json", "w"), indent=1)
iqp.save_cache(cache, "settle/iqp_data/iqp_cache_tuning.json")
