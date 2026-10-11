"""Instance sets. Each set is fixed by this file (generator, seeds, filter) and was
written down BEFORE any QAOA run on it.

Common rule: ring+chords generator (settle/instances.py), degree <= 5, liquidity
C_u = ceil(f * outgoing_u) with f cycling 0.2, 0.3, 0.4; seeds base + 1000*banks + i;
keep an instance only if exact optimum > greedy RTGS queue (there is gridlock).

bench (main set, 40):  A = 4 banks/6 payments, B = 5 banks/8 payments, amounts 1..10.
t0    (added, 20):     4 banks/6 payments, amounts 1..3. Purpose: a paired comparison in
                       which the standard slack QUBO (E0) fits in <= 18 qubits. Small
                       amounts make E0 cheaper, i.e. this tier FAVOURS the baseline.
c     (added, 20):     6 banks/10 payments, amounts 1..10. Purpose: larger search space
                       (uniform P(opt) ~ 1/1024) for the slack-free/IQP encodings.
Training sets for E3 tuning use the same rules with disjoint seed bases (see tune_e3.py)."""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from settle.instances import generate, save_instances
from settle.exact import brute_force, greedy

SETS = {
    "bench": {"parts": [(4, 6, 20, "A"), (5, 8, 20, "B")], "w_max": 10, "base": 0},
    "t0": {"parts": [(4, 6, 20, "T")], "w_max": 3, "base": 200000},
    "c": {"parts": [(6, 10, 20, "C")], "w_max": 10, "base": 300000},
}
TRAIN_BASE = {"bench": 70000, "t0": 270000, "c": 370000}


def build(n_banks, n_arcs, count, tag, w_max=10, base=0):
    out, seed = [], 0
    fs = (0.2, 0.3, 0.4)
    while len(out) < count:
        f = fs[len(out) % 3]
        inst = generate(n_banks, n_arcs, base + 1000 * n_banks + seed, f_liq=f, w_max=w_max)
        seed += 1
        if inst is None:
            continue
        opt, _ = brute_force(inst)
        gv, _ = greedy(inst)
        if opt > gv:
            inst.name = "%s%02d" % (tag, len(out))
            inst.meta.update({"opt": opt, "greedy": gv, "set_seed": base + 1000 * n_banks + seed - 1})
            out.append(inst)
    return out


def build_set(name, base=None, tag_prefix=""):
    spec = SETS[name]
    b = spec["base"] if base is None else base
    out = []
    for nb, na, cnt, tag in spec["parts"]:
        out += build(nb, na, cnt, tag_prefix + tag, spec["w_max"], b)
    return out


if __name__ == "__main__":
    for name in SETS:
        insts = build_set(name)
        path = "data/%s_instances.json" % name
        save_instances(insts, path)
        print(name, len(insts), "->", path)
