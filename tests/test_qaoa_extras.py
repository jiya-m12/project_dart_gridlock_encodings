import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
from qiskit import QuantumCircuit
from qiskit.quantum_info import Statevector
from settle.instances import generate
from settle.encodings import e2_indicator, e4_overflow_sq, e3_unbalanced, e0_standard
from settle.qaoa import (Problem, statevector, logical_distribution, energy, packed_circuit,
                         split_counts, product_probs, cvar_from_probs)


def test_cvar_alpha1_is_mean():
    pr = Problem(e2_indicator(generate(4, 6, 3, f_liq=0.2)))
    x = np.array([0.4, 0.7, 0.3, 0.2])
    assert abs(energy(pr, x, "cvar", 1.0) - energy(pr, x, "mean")) < 1e-9


def test_cvar_on_point_mass():
    pr = Problem(e2_indicator(generate(4, 6, 3, f_liq=0.2)))
    d = pr.build_diag()
    pmf = np.zeros(len(d)); pmf[7] = 1
    assert abs(cvar_from_probs(pr, pmf, 0.25) - d[7]) < 1e-12


def test_packed_equals_separate():
    inst = generate(4, 6, 3, f_liq=0.2)
    rng = np.random.default_rng(3)
    blocks = [(Problem(e), rng.uniform(0, 2, 2), rng.uniform(0, 2, 2))
              for e in (e2_indicator(inst), e4_overflow_sq(inst), e3_unbalanced(inst, 1.5, 0.2))]
    qc, offs = packed_circuit(blocks)
    probs = Statevector(qc.remove_final_measurements(inplace=False)).probabilities_dict()
    for (pr, g, b), px in zip(blocks, split_counts(probs, blocks, offs)):
        assert np.allclose(px, logical_distribution(pr, statevector(pr, g, b)), atol=1e-12)


def test_slack_marginalisation():
    inst = generate(4, 6, 3, f_liq=0.2)
    pr = Problem(e0_standard(inst))
    g, b = [0.3], [0.5]
    psi = statevector(pr, g, b)
    px = logical_distribution(pr, psi)
    pr_full = np.abs(psi) ** 2
    idx = np.arange(len(pr_full))
    for x in (0, 5, 33):
        assert abs(px[x] - pr_full[(idx & 63) == x].sum()) < 1e-12


def test_product_probs():
    th = np.array([0.3, 1.2, 2.5, 0.9])
    qc = QuantumCircuit(4)
    for j, t in enumerate(th):
        qc.ry(t, j)
    assert np.allclose(Statevector(qc).probabilities(), product_probs(th))


def test_shots_to_optimum_uses_sampled_frequency():
    """Regression (v5 bug): on a noisy backend the shots-to-first-optimum line must come
    from the sampled frequency, not the exact noiseless P(opt)."""
    import io, contextlib
    from settle.instances import gridlock_instance
    from settle.exact import feasibility_mask, values
    import main
    inst = gridlock_instance()
    feas, v = feasibility_mask(inst), values(inst)
    opt_idx = int(np.flatnonzero(feas & (v == v[feas].max()))[0])
    px = np.zeros(2 ** inst.n); px[opt_idx] = 0.028; px[0] = 0.972      # sampled: P(opt) = 0.028
    q_exact = {"p_opt": 0.063, "p_feasible": 0.534}
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        main.report_samples(inst, px, q_exact, 3, 54, 1000)
    out = buf.getvalue()
    assert "expected shots to first optimum: 36 from this run's samples" in out, out
    assert "exact noiseless: 16" in out, out


def test_start_quadrant():
    """Regression (v6 bug): with a minimised H, |+> start and exp(-i g H) exp(-i b X),
    the good quadrant is g > 0, b < 0. (1) one qubit, H = Z: b < 0 raises P(ground);
    (2) the ramp start actually used by optimise_sweep has b < 0 and, unoptimised at p = 5,
    already gives a feasible-heavy distribution on the gridlock instance."""
    from settle import qaoa
    from settle.instances import gridlock_instance
    from settle.metrics import quality
    class One:
        n, n_logical, order = 1, 1, None
        diag = np.array([1.0, -1.0])                            # H = Z, ground state |1>
        def build_diag(self): return self.diag
    one = One()
    p_pos = abs(statevector(one, [0.4], [0.4])[1]) ** 2
    p_neg = abs(statevector(one, [0.4], [-0.4])[1]) ** 2
    assert p_neg > 0.7 > 0.3 > p_pos, (p_pos, p_neg)
    assert qaoa.BETA_START_SIGN < 0
    inst = gridlock_instance()
    pr = Problem(e2_indicator(inst, rule="neighbour"))
    p = 5; ramp = (np.arange(p) + 0.5) / p
    q = quality(inst, logical_distribution(pr, statevector(pr, 0.8 * ramp, qaoa.BETA_START_SIGN * 0.8 * (1 - ramp))))
    assert q["p_feasible"] > 0.5, q


if __name__ == "__main__":
    test_start_quadrant()
    test_cvar_alpha1_is_mean(); test_cvar_on_point_mass(); test_packed_equals_separate()
    test_slack_marginalisation(); test_product_probs(); test_shots_to_optimum_uses_sampled_frequency()
    print("qaoa extras tests OK")
