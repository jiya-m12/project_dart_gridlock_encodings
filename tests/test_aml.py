"""Public-data (AML) tier: data integrity, exact optimum at cent scale, and main.py wiring."""
import sys, os, json, itertools
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
from settle.instances import load_instances_json
from settle.exact import brute_force, milp, is_value, feasibility_mask, values
from settle.metrics import quality
from settle.aml import bs_fifo, liquidity

INSTS = load_instances_json(open("data/aml_instances.json").read())


def test_data_integrity():
    assert len(INSTS) == 22
    for i in INSTS:
        assert all(float(w).is_integer() and w > 0 for _, _, w in i.arcs)              # integer cents
        assert liquidity(i.n_banks, i.arcs, i.meta["f_liq"]) == list(i.liquidity)    # stated rule
        deg = np.zeros(i.n_banks, int)
        for a, b, _ in i.arcs:
            deg[a] += 1; deg[b] += 1
        assert deg.max() <= 5 and 6 <= i.n <= 9


def test_optimum_exact_at_cents():
    for i in INSTS:
        opt, idx = brute_force(i)
        assert opt == i.meta["opt"] and len(idx) == i.meta["n_opt_solutions"]
        assert abs(milp(i)[0] - opt) < 1e-6
        assert opt > i.meta["greedy"]                                                # gridlocked by construction


def test_one_cent_is_not_the_optimum():
    """Regression for the np.isclose(rtol=1e-5) issue: at opt ~ 1e5-1e8 cents, a selection one
    cent short of the optimum must NOT count as optimal."""
    assert list(is_value(np.array([2200425.0, 2200424.0]), 2200425.0)) == [True, False]
    i = INSTS[0]
    feas, v = feasibility_mask(i), values(i)
    px = np.zeros(2 ** i.n); px[np.flatnonzero(feas & is_value(v, i.meta["opt"]))[0]] = 1.0
    assert quality(i, px)["p_opt"] == 1.0


def test_bs_fifo_feasible():
    for i in INSTS:
        v, x = bs_fifo(i)
        assert i.is_feasible(x) and v <= i.meta["opt"]


def test_main_wiring():
    import main
    insts, cache, e3 = main.load_assets()
    encs = main.make_encodings(cache, e3)
    a = insts["AML043"]
    try:
        encs["E3"](a, "neighbour")
        raise AssertionError("E3 should not be available on the cent-scale tier")
    except main.NotAvailable:
        pass
    assert encs["E3"](insts["A00"], "neighbour").name == "E3_unbalanced"            # still fine elsewhere
    try:
        encs["E4"](a, "neighbour")
        raise AssertionError("E4 should not be available on the cent-scale tier (float64 limit)")
    except main.NotAvailable:
        pass
    assert encs["E4"](insts["A00"], "neighbour").name == "E4_overflow_sq"
    assert main.money(a, 76892) == "$768.92" and main.money(insts["A00"], 13) == "13"


if __name__ == "__main__":
    test_data_integrity(); test_optimum_exact_at_cents(); test_one_cent_is_not_the_optimum()
    test_bs_fifo_feasible(); test_main_wiring()
    print("aml tests OK")
