import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from settle.instances import gridlock_instance, generate
from settle.exact import brute_force, milp, greedy


def test_milp_equals_brute_force():
    insts = [gridlock_instance()] + [generate(5, 8, s, f_liq=f) for s in range(10) for f in (0.2, 0.4)]
    for inst in insts:
        bf, _ = brute_force(inst)
        mv, x = milp(inst)
        assert abs(bf - mv) < 1e-9, inst.name
        assert inst.is_feasible(x)
        gv, _ = greedy(inst)
        assert gv <= bf + 1e-9


def test_degree_cap():
    for s in range(20):
        inst = generate(5, 10, s)
        assert all(inst.degree(u) <= 5 for u in range(inst.n_banks))


if __name__ == "__main__":
    test_milp_equals_brute_force(); test_degree_cap(); print("core tests OK")
