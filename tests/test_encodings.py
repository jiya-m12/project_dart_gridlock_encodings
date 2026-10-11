import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
from settle.instances import gridlock_instance, generate
from settle.encodings import e0_standard, e2_indicator, e4_overflow_sq
from settle.verify import check_exact, reduced_cost, check_full_enumeration
from settle import poly as P
from settle.qaoa import Problem, statevector, qaoa_circuit


def instances():
    out = [gridlock_instance()]
    for s in range(15):
        for f in (0.2, 0.3):
            out.append(generate(4, 6, s, f_liq=f))
    return out


def test_global_rule_exact():
    """gamma=2, global weights: provably exact for E0/E2/E4 (lambda > sum w)."""
    for inst in instances():
        for mk in (e0_standard, e2_indicator, e4_overflow_sq):
            assert check_exact(inst, mk(inst, rule="global"))["exact"], (inst.name, mk.__name__)


def test_factorised_min_matches_full_enumeration():
    for inst in instances()[1:6]:
        for mk in (e0_standard, e2_indicator, e4_overflow_sq):
            enc = mk(inst)
            full = check_full_enumeration(enc, inst, max_qubits=18)
            if full is not None:
                assert np.allclose(full, reduced_cost(enc))


def test_gridlock_resources():
    g = gridlock_instance()
    st0 = P.term_stats(P.to_ising(e0_standard(g).poly))
    st2 = P.term_stats(P.to_ising(e2_indicator(g).poly))
    assert e0_standard(g).n_total == 34 and st0["terms_by_order"][2] == 135 and st0["cnot_ladder"] == 270
    assert e2_indicator(g).n_total == 9 and st2["cnot_ladder"] == 68


def test_numpy_matches_qiskit():
    from qiskit.quantum_info import Statevector
    rng = np.random.default_rng(1)
    inst = generate(4, 6, 3, f_liq=0.2)
    for mk in (e2_indicator, e4_overflow_sq):
        pr = Problem(mk(inst))
        g, b = rng.uniform(0, 3, 2), rng.uniform(0, 3, 2)
        psi = statevector(pr, g, b)
        sv = Statevector(qaoa_circuit(pr, g, b, measure=False)).data
        assert abs(abs(np.vdot(sv, psi)) - 1) < 1e-9


if __name__ == "__main__":
    test_global_rule_exact(); test_factorised_min_matches_full_enumeration()
    test_gridlock_resources(); test_numpy_matches_qiskit(); print("encoding tests OK")
