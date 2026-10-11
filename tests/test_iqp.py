import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
from settle.instances import gridlock_instance, load_instances_json
from settle.encodings import e1_iqp, e2_indicator
from settle.verify import check_exact, reduced_cost, check_full_enumeration
from settle import iqp

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CACHE = iqp.load_cache_text(open(os.path.join(ROOT, "settle/iqp_data/iqp_cache.json")).read())


def test_cached_polynomials_valid():
    for key, v in CACHE.items():
        d = v["d"]
        feas = [c == "1" for c in key.split(":")[1]]
        assert iqp.check(d, v["k"], feas, v["coefs"]), key


def test_gridlock_iqp_slack_2():
    e = e1_iqp(gridlock_instance(), cache=dict(CACHE), derive_missing=False)
    assert e.n_slack == 2 and e.n_total == 11


def test_bench_iqp_exact_and_full_enum():
    insts = load_instances_json(open(os.path.join(ROOT, "data/bench_instances.json")).read())
    for inst in insts:
        e = e1_iqp(inst, rule="neighbour", cache=dict(CACHE), derive_missing=False)
        assert check_exact(inst, e)["exact"], inst.name
    for inst in insts[:5]:
        e = e1_iqp(inst, cache=dict(CACHE), derive_missing=False)
        full = check_full_enumeration(e, inst, max_qubits=16)
        if full is not None:
            assert np.allclose(full, reduced_cost(e), atol=1e-6)


def test_minimal_slack_is_minimal():
    """For every cached pattern with k>0, no polynomial with k-1 slack bits exists even with |coef| <= 64."""
    for key, v in CACHE.items():
        if v["k"] == 0:
            continue
        feas = [c == "1" for c in key.split(":")[1]]
        assert iqp.derive(v["d"], feas, v["k"] - 1, coef_bound=64.0, minimise=False, time_limit=60) is None, key


if __name__ == "__main__":
    test_cached_polynomials_valid(); test_gridlock_iqp_slack_2()
    test_bench_iqp_exact_and_full_enum(); test_minimal_slack_is_minimal(); print("iqp tests OK")
