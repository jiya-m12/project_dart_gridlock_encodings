"""Qubits needed vs payment-amount resolution (analytic, no simulation).
Scaling every amount and liquidity by s leaves the feasible set unchanged, so
IQP/HO/E3/E4 qubit counts do not change; standard binary slack grows ~log2(s)
per bank. Shows why integer resolution matters for E0 on real amounts."""
import os, sys, json
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from settle.instances import Instance, gridlock_instance, load_instances_json
from settle.encodings import e0_standard, e2_indicator, e1_iqp, binding_banks
from settle import iqp

cache = iqp.load_cache_text(open("settle/iqp_data/iqp_cache.json").read())
rows = []
for inst0 in [gridlock_instance()] + load_instances_json(open("data/bench_instances.json").read())[::10]:
    for s in (1, 10, 100, 1000, 10 ** 6):
        inst = Instance(inst0.name, inst0.n_banks, [(a, b, w * s) for a, b, w in inst0.arcs],
                        [c * s for c in inst0.liquidity])
        rows.append({"instance": inst0.name, "scale": s, "payments": inst.n,
                     "E0": e0_standard(inst).n_total,
                     "E1": e1_iqp(inst, cache=cache).n_total,
                     "E2_E3_E4": e2_indicator(inst).n_total})
json.dump(rows, open("results/qubits_vs_amounts.json", "w"), indent=1)
print("%-12s %8s %4s %5s %4s %8s" % ("instance", "scale", "n", "E0", "E1", "E2/3/4"))
for r in rows:
    print("%-12s %8g %4d %5d %4d %8d" % (r["instance"], r["scale"], r["payments"], r["E0"], r["E1"], r["E2_E3_E4"]))
